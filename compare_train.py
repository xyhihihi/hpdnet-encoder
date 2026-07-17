# 有BN vs 无BN 对比训练
# 配置: main_lr=40, bn_lr=200, 各 200 epoch, denoising 开启
# 公平性:
#   - 两模型用相同随机种子初始化 (Stiefel 权重初始化一致; BN 模型多 G=I 的 BN 层)
#   - 逐 epoch 交替训练, 相同数据打乱顺序 + 相同 denoising 损坏
#   - 数据预载入内存
# 产出:
#   - tmp/customed/saved/model_with_bn.model
#   - tmp/customed/saved/model_no_bn.model
#   - tmp/customed/compare_loss.csv  (epoch, loss_bn, loss_nobn)
#   - tmp/customed/compare_loss_curve.png  (每 epoch 更新)
import os
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

MAIN_LR = 40.0
BN_LR = 200.0
EPOCHS = int(os.environ.get('CT_EPOCHS', 200))
BATCH_SIZE = 512
SEED = 42
# smoke test 用: 限制样本数 (CT_NSUB=256 快速验证流程); 0=全量
_NSUB = int(os.environ.get('CT_NSUB', 0))

# denoising
USE_EIGVAL_MASK = True
N_MASK = 8
MASK_EPS = 1e-8

CSV_PATH = os.path.join(CURVE_DIR, 'compare_loss.csv')
PNG_PATH = os.path.join(CURVE_DIR, 'compare_loss_curve.png')
SAVE_BN = os.path.join(SAVE_DIR, 'model_with_bn.model')
SAVE_NOBN = os.path.join(SAVE_DIR, 'model_no_bn.model')
SAVE_EVERY = 5   # 每 5 epoch 存一次权重

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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


def make_masked(X, seed):
    """确定性特征值损坏 (denoising), 两模型用相同 seed 保证输入一致。"""
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


# ==================== 初始化两个模型 (相同种子) ====================
torch.manual_seed(SEED)
np.random.seed(SEED)
model_bn = model_mod.HPDNetwork(use_bn=True, bn_lr=BN_LR)

torch.manual_seed(SEED)
np.random.seed(SEED)
model_nobn = model_mod.HPDNetwork(use_bn=False)

model_bn.train()
model_nobn.train()
print('两模型初始化完成 (相同种子)。', flush=True)


def train_one_epoch(net, perm, epoch, is_bn):
    ep_loss, nb = 0.0, 0
    for bs in range(0, num_samples, BATCH_SIZE):
        idx = perm[bs:bs + BATCH_SIZE]
        X = all_X[idx]
        X_in = make_masked(X, seed=100000 * epoch + bs)  # 两模型同 seed
        Y, _ = net(X_in)
        d = util.log_mat_v2(Y) - util.log_mat_v2(X)
        loss = (d.real ** 2 + d.imag ** 2).mean()
        net.zero_grad()
        loss.backward()
        if is_bn:
            net.update_para(MAIN_LR, bn_lr=BN_LR)
        else:
            net.update_para(MAIN_LR)
        ep_loss += loss.item()
        nb += 1
    return ep_loss / nb


# ==================== 训练循环 (逐 epoch 交替) ====================
hist_bn, hist_nobn = [], []
with open(CSV_PATH, 'w') as f:
    f.write('epoch,loss_bn,loss_nobn\n')

for epoch in range(EPOCHS):
    t0 = datetime.datetime.now()
    # 两模型用相同的 per-epoch 打乱顺序
    rng = np.random.RandomState(1234 + epoch)
    perm = rng.permutation(num_samples)

    loss_bn = train_one_epoch(model_bn, perm, epoch, is_bn=True)
    loss_nobn = train_one_epoch(model_nobn, perm, epoch, is_bn=False)
    hist_bn.append(loss_bn)
    hist_nobn.append(loss_nobn)

    dt = (datetime.datetime.now() - t0).total_seconds()
    print(f'epoch {epoch+1}/{EPOCHS}  BN={loss_bn:.6f}  noBN={loss_nobn:.6f}  '
          f'(BN-noBN={loss_bn-loss_nobn:+.4f})  time={dt:.1f}s', flush=True)

    # 追加 CSV
    with open(CSV_PATH, 'a') as f:
        f.write(f'{epoch+1},{loss_bn:.6f},{loss_nobn:.6f}\n')

    # 更新对比图
    ep_axis = range(1, len(hist_bn) + 1)
    plt.figure(figsize=(11, 6))
    plt.plot(ep_axis, hist_bn, color='#D64550', linewidth=2, label='有 BN (黎曼 BatchNorm)')
    plt.plot(ep_axis, hist_nobn, color='#2E86AB', linewidth=2, label='无 BN')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Log-Euclidean Loss', fontsize=12)
    plt.title(f'有 BN vs 无 BN 训练损失对比 (main_lr={MAIN_LR}, bn_lr={BN_LR}, epoch {epoch+1}/{EPOCHS})',
              fontsize=13)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=150, bbox_inches='tight')
    plt.close()

    # 定期存权重
    if (epoch + 1) % SAVE_EVERY == 0 or (epoch + 1) == EPOCHS:
        torch.save(model_bn, SAVE_BN)
        torch.save(model_nobn, SAVE_NOBN)

# 最终存权重
torch.save(model_bn, SAVE_BN)
torch.save(model_nobn, SAVE_NOBN)
print(f'\n完成。权重: {SAVE_BN} / {SAVE_NOBN}', flush=True)
print(f'曲线: {PNG_PATH}, 数据: {CSV_PATH}', flush=True)
print(f'终点 loss: BN={hist_bn[-1]:.6f}  noBN={hist_nobn[-1]:.6f}', flush=True)
