# 特征值归一化预处理实验: 对比多种归一化策略对 HPD 自编码器训练的影响
# 策略:
#   1. baseline:  无归一化 (原始数据)
#   2. trace:     X * n / tr(X), 使 tr = n (算术均值归一)
#   3. det:       X / det(X)^(1/n), 使 det = 1 (几何均值归一, Log 域居中)
#   4. maxeig:    X / λ_max(X), 使最大特征值 = 1
#   5. logcenter: log 域居中 (减去数据集 log 特征值均值)
# 公平性:
#   - 所有策略用相同随机种子初始化模型
#   - 逐 epoch 相同数据打乱顺序 + 相同 denoising 损坏
#   - 数据预载入内存, 归一化预计算
# 产出:
#   tmp/customed/eignorm_loss.csv   (epoch, loss_baseline, loss_trace, loss_det, loss_maxeig, loss_logcenter)
#   tmp/customed/eignorm_loss_curve.png
import os
import copy
import numpy as np
import torch
import scipy.io as sio
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import model as model_mod
import util

# ==================== 配置 ====================
DATA_DIR = 'D:/data/customed/customed_64'
FILE_LIST_PATH = 'D:/data/customed/train.txt'
SAVE_DIR = 'tmp/customed/saved'
CURVE_DIR = 'tmp/customed'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CURVE_DIR, exist_ok=True)

MAIN_LR_HI = 40.0
MAIN_LR_LO = 0.5
BN_LR_HI = 200.0
BN_LR_LO = 5.0
EPOCHS = int(os.environ.get('EN_EPOCHS', 100))
BATCH_SIZE = 512
SEED = 42
REC_EPS = 1e-4

# denoising 配置
USE_EIGVAL_MASK = True
N_MASK = 16
MASK_EPS = 1e-8

# smoke test: 限制样本数 (EN_NSUB=256 快速验证); 0=全量
_NSUB = int(os.environ.get('EN_NSUB', 0))

CSV_PATH = os.path.join(CURVE_DIR, 'eignorm_loss.csv')
PNG_PATH = os.path.join(CURVE_DIR, 'eignorm_loss_curve.png')
SAVE_EVERY = 10

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def cosine_lr(hi, lo, epoch, total):
    t = epoch / max(total - 1, 1)
    return lo + 0.5 * (hi - lo) * (1 + np.cos(np.pi * t))


# ==================== 归一化函数 ====================
def norm_trace(X):
    """Trace 归一化: X * n / tr(X), 使 tr(X_norm) = n。
    对 HPD 矩阵, tr > 0, 缩放保 HPD。
    """
    n = X.shape[-1]
    tr = torch.diagonal(X, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)  # [B,1,1]
    tr = tr.real  # trace of HPD is real
    return X * (n / tr)


def norm_det(X):
    """Det 归一化: X / det(X)^(1/n), 使 det(X_norm) = 1。
    等价于 Log 域减去 (tr(log X)/n) * I, 即居中 log 特征值的均值。
    用 eigh 计算 log(det) = sum(log(eig))。
    """
    n = X.shape[-1]
    with torch.no_grad():
        ev = torch.linalg.eigvalsh(X)  # [B, n], 升序
        log_det = torch.log(ev).sum(dim=-1, keepdim=True).unsqueeze(-1)  # [B,1,1]
        scale = torch.exp(log_det / n)  # det^(1/n)
    return X / scale


def norm_maxeig(X):
    """Max-eig 归一化: X / λ_max(X), 使最大特征值 = 1。"""
    with torch.no_grad():
        ev = torch.linalg.eigvalsh(X)  # [B, n], 升序
        max_ev = ev[:, -1:].unsqueeze(-1)  # [B,1,1]
    return X / max_ev


def norm_logcenter(X, mean_log_ev):
    """Log 域居中: 使每样本的 log 特征值均值 = 0 (相对于数据集均值)。
    X_norm = X / exp(mean_log_ev), 其中 mean_log_ev 是数据集的 log 特征值全局均值。
    这是全局缩放 (所有样本用同一个因子), 不改变样本间的相对关系。
    """
    scale = np.exp(mean_log_ev)
    return X / scale


# ==================== 载入数据到内存 ====================
with open(FILE_LIST_PATH) as fid:
    file_list = [ln.strip('\n').split(' ')[0].replace('\\', '/') for ln in fid]
if _NSUB > 0:
    file_list = file_list[:_NSUB]
num_samples = len(file_list)
print(f'样本数: {num_samples}, 预载入内存...', flush=True)
all_data = np.zeros((num_samples, 64, 64), dtype=np.complex128)
for i, f in enumerate(file_list):
    all_data[i] = sio.loadmat(os.path.join(DATA_DIR, f))['Y1']
all_X = torch.from_numpy(all_data).to(torch.complex128)
print('数据载入完成。', flush=True)

# ==================== 预计算各归一化版本 ====================
print('预计算归一化数据...', flush=True)

# 计算数据集全局 log 特征值均值 (用于 logcenter)
all_log_evs = []
for i in range(num_samples):
    ev = torch.linalg.eigvalsh(all_X[i:i+1])[0].numpy()
    all_log_evs.append(np.log(ev))
all_log_evs = np.concatenate(all_log_evs)
global_mean_log_ev = all_log_evs.mean()
print(f'  全局 log 特征值均值: {global_mean_log_ev:.4f}', flush=True)

# 预计算各策略的归一化数据
data_variants = {
    'baseline': all_X.clone(),
    'trace': norm_trace(all_X),
    'det': norm_det(all_X),
    'maxeig': norm_maxeig(all_X),
    'logcenter': norm_logcenter(all_X, global_mean_log_ev),
}

# 打印归一化后的统计
for name, Xv in data_variants.items():
    ev_sample = torch.linalg.eigvalsh(Xv[0:1])[0].numpy()
    tr = ev_sample.sum()
    det_log = np.log(ev_sample).sum()
    print(f'  {name:10s}: 样本0 tr={tr:.4f}, log(det)={det_log:.4f}, '
          f'eig_range=[{ev_sample.min():.2e}, {ev_sample.max():.2e}]', flush=True)

strategies = [
    {'name': 'baseline',  'label': '无归一化 (baseline)', 'color': '#888888'},
    {'name': 'trace',     'label': 'Trace 归一化 (tr=n)', 'color': '#2E86AB'},
    {'name': 'det',       'label': 'Det 归一化 (det=1)', 'color': '#D64550'},
    {'name': 'maxeig',    'label': 'Max-eig 归一化 (λmax=1)', 'color': '#F4A261'},
    {'name': 'logcenter', 'label': 'Log 域居中 (全局)', 'color': '#4CAF50'},
]


def make_masked(X, seed):
    """确定性特征值损坏 (denoising)。"""
    if not USE_EIGVAL_MASK:
        return X
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        ev, U = torch.linalg.eigh(X)
        B, D = ev.shape
        rand_perm = torch.argsort(torch.rand(B, D, generator=g), dim=1)
        mask_idx = rand_perm[:, :N_MASK]
        mask = torch.zeros(B, D, dtype=torch.bool)
        mask.scatter_(1, mask_idx, True)
        ev_new = ev.clone()
        ev_new[mask] = MASK_EPS
        Xn = torch.matmul(U * ev_new.unsqueeze(-2), U.conj().transpose(-2, -1))
    return Xn


# ==================== 初始化模型 (每个策略一个, 相同种子) ====================
print('初始化模型...', flush=True)
models = {}
for st in strategies:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    m = model_mod.HPDNetwork(use_bn=True, bn_lr=BN_LR_HI,
                             rec_params=[REC_EPS] * 24)
    m.train()
    models[st['name']] = m
print(f'{len(strategies)} 个模型初始化完成 (相同种子)。', flush=True)


def train_one_epoch(net, X_data, perm, epoch):
    """训练一个 epoch, 返回平均 loss。"""
    ep_loss, nb = 0.0, 0
    main_lr = cosine_lr(MAIN_LR_HI, MAIN_LR_LO, epoch, EPOCHS)
    bn_lr = cosine_lr(BN_LR_HI, BN_LR_LO, epoch, EPOCHS)
    for bs in range(0, num_samples, BATCH_SIZE):
        idx = perm[bs:bs + BATCH_SIZE]
        X = X_data[idx]
        X_in = make_masked(X, seed=300000 * epoch + bs)
        Y, _ = net(X_in)
        d = util.log_mat_v2(Y) - util.log_mat_v2(X)
        loss = (d.real ** 2 + d.imag ** 2).mean()
        net.zero_grad()
        loss.backward()
        net.update_para(main_lr, bn_lr=bn_lr)
        ep_loss += loss.item()
        nb += 1
    return ep_loss / nb


# ==================== 训练循环 ====================
hist = {st['name']: [] for st in strategies}

with open(CSV_PATH, 'w') as f:
    cols = 'epoch,' + ','.join(st['name'] for st in strategies)
    f.write(cols + '\n')

print(f'\n开始训练 ({EPOCHS} epochs, batch_size={BATCH_SIZE})...', flush=True)
for epoch in range(EPOCHS):
    t0 = datetime.datetime.now()
    # 所有策略用相同的 per-epoch 打乱顺序
    rng = np.random.RandomState(5678 + epoch)
    perm = rng.permutation(num_samples)

    row_losses = {}
    for st in strategies:
        name = st['name']
        loss = train_one_epoch(models[name], data_variants[name], perm, epoch)
        hist[name].append(loss)
        row_losses[name] = loss

    dt = (datetime.datetime.now() - t0).total_seconds()
    main_lr = cosine_lr(MAIN_LR_HI, MAIN_LR_LO, epoch, EPOCHS)
    losses_str = '  '.join(f"{st['name'][:4]}={row_losses[st['name']]:.5f}"
                           for st in strategies)
    print(f'epoch {epoch+1}/{EPOCHS}  {losses_str}  lr={main_lr:.2f}  time={dt:.1f}s',
          flush=True)

    # 追加 CSV
    with open(CSV_PATH, 'a') as f:
        vals = ','.join(f"{row_losses[st['name']]:.6f}" for st in strategies)
        f.write(f'{epoch+1},{vals}\n')

    # 更新对比图
    plt.figure(figsize=(14, 7))
    for st in strategies:
        ep_axis = range(1, len(hist[st['name']]) + 1)
        plt.plot(ep_axis, hist[st['name']], color=st['color'], linewidth=2,
                 label=st['label'])
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Log-Euclidean Loss', fontsize=12)
    plt.title(f'特征值归一化预处理对比 (epoch {epoch+1}/{EPOCHS}, '
              f'denoising N_MASK={N_MASK})', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 对数坐标更清晰
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=150, bbox_inches='tight')
    plt.close()

    # 定期存权重
    if (epoch + 1) % SAVE_EVERY == 0 or (epoch + 1) == EPOCHS:
        for st in strategies:
            path = os.path.join(SAVE_DIR, f"eignorm_{st['name']}.model")
            torch.save(models[st['name']], path)

# ==================== 最终汇总 ====================
print('\n===== 实验完成 =====', flush=True)
for st in strategies:
    name = st['name']
    final_loss = hist[name][-1]
    best_loss = min(hist[name])
    best_ep = hist[name].index(best_loss) + 1
    print(f"  {st['label']:30s}: final={final_loss:.6f}, best={best_loss:.6f} (ep{best_ep})",
          flush=True)

# 计算相对 baseline 的改进
baseline_final = hist['baseline'][-1]
print(f'\n相对 baseline 的改进 (final loss):', flush=True)
for st in strategies:
    name = st['name']
    if name == 'baseline':
        continue
    improvement = (baseline_final - hist[name][-1]) / baseline_final * 100
    print(f"  {st['label']:30s}: {improvement:+.2f}%", flush=True)

print(f'\n对比图: {PNG_PATH}', flush=True)
print(f'数据: {CSV_PATH}', flush=True)
