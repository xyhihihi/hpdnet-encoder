# 学习率衰减实验: 从已收敛的 BN 模型续训, 测试衰减策略能否突破固定 LR 的平台
# 前提: compare_train.py 已跑完, model_with_bn.model 已存在
# 配置: 以 main_lr=40/bn_lr=200 收敛后的 BN 模型为起点, 对比三种衰减策略
# 产出:
#   tmp/customed/saved/decay_{tag}.model      每种策略最终权重
#   tmp/customed/decay_loss.csv               epoch, loss_fixed, loss_cosine, loss_step, loss_bnonly
#   tmp/customed/decay_loss_curve.png         衰减策略对比图 (每 epoch 更新)
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
BASE_MODEL = 'tmp/customed/saved/model_with_bn.model'   # 已收敛 BN 模型
SAVE_DIR = 'tmp/customed/saved'
CURVE_DIR = 'tmp/customed'
os.makedirs(SAVE_DIR, exist_ok=True)

EPOCHS = 200
BATCH_SIZE = 512
SEED_DATA = 9999  # 与 compare_train 不同的 seed, 保证续训数据多样性

# 固定 LR 基准 (与 compare_train 一致, 用来对比衰减效果)
BASE_MAIN_LR = 40.0
BASE_BN_LR = 200.0

# denoising
USE_EIGVAL_MASK = True
N_MASK = 8
MASK_EPS = 1e-8

CSV_PATH = os.path.join(CURVE_DIR, 'decay_loss.csv')
PNG_PATH = os.path.join(CURVE_DIR, 'decay_loss_curve.png')
SAVE_EVERY = 10

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 学习率调度 ====================
def get_lrs(strategy, epoch, total_epochs):
    """返回 (main_lr, bn_lr)。epoch: 0-indexed。"""
    t = epoch / max(total_epochs - 1, 1)  # 0..1
    name = strategy['name']
    if name == 'fixed':
        return BASE_MAIN_LR, BASE_BN_LR
    if name == 'cosine':
        # main_lr: 40 → 0.5 余弦; bn_lr: 200 → 5 余弦
        ml = 0.5 + 0.5 * (BASE_MAIN_LR - 0.5) * (1 + np.cos(np.pi * t))
        bl = 5.0 + 0.5 * (BASE_BN_LR - 5.0) * (1 + np.cos(np.pi * t))
        return ml, bl
    if name == 'step':
        # 阶梯: 前 1/3 → BASE, 中 1/3 → BASE/4, 后 1/3 → BASE/20
        thirds = total_epochs // 3
        if epoch < thirds:
            return BASE_MAIN_LR, BASE_BN_LR
        elif epoch < 2 * thirds:
            return BASE_MAIN_LR / 4, BASE_BN_LR / 4
        else:
            return BASE_MAIN_LR / 20, BASE_BN_LR / 20
    if name == 'bnonly':
        # 只衰减 bn_lr (余弦 200→5), 主 LR 固定 40
        bl = 5.0 + 0.5 * (BASE_BN_LR - 5.0) * (1 + np.cos(np.pi * t))
        return BASE_MAIN_LR, bl
    raise ValueError(name)


# ==================== 载入数据到内存 ====================
with open(FILE_LIST_PATH) as fid:
    file_list = [ln.strip('\n').split(' ')[0].replace('\\', '/') for ln in fid]
num_samples = len(file_list)
print(f'样本数: {num_samples}, 预载入内存...', flush=True)
all_data = np.zeros((num_samples, 64, 64), dtype=np.complex128)
for i, f in enumerate(file_list):
    all_data[i] = sio.loadmat(os.path.join(DATA_DIR, f))['Y1']
all_X = torch.from_numpy(all_data).to(torch.complex128)
print('数据载入完成。', flush=True)


def make_masked(X, seed):
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


# ==================== 从已收敛模型初始化各策略 ====================
print(f'载入已收敛 BN 模型: {BASE_MODEL}', flush=True)
base_model = torch.load(BASE_MODEL, weights_only=False)

strategies = [
    {'name': 'fixed',   'tag': 'fixed_40_200',   'label': '固定 LR=40,bnLR=200'},
    {'name': 'cosine',  'tag': 'cosine_40-0.5',  'label': '余弦衰减 40→0.5 / 200→5'},
    {'name': 'step',    'tag': 'step_thirds',     'label': '阶梯衰减 /1→/4→/20'},
    {'name': 'bnonly',  'tag': 'bnonly_cos',      'label': '仅 bnLR 余弦 200→5'},
]

models = {st['tag']: copy.deepcopy(base_model) for st in strategies}
for m in models.values():
    m.train()
print(f'{len(strategies)} 个策略各从已收敛模型 deepcopy 初始化完成。', flush=True)

# ==================== 预生成 per-epoch 打乱顺序 ====================
epoch_perms = [np.random.RandomState(SEED_DATA + ep).permutation(num_samples)
               for ep in range(EPOCHS)]

# ==================== 训练循环 (逐 epoch 所有策略并行推进) ====================
hist = {st['tag']: [] for st in strategies}

with open(CSV_PATH, 'w') as f:
    cols = 'epoch,' + ','.join(st['tag'] for st in strategies)
    f.write(cols + '\n')

for epoch in range(EPOCHS):
    t0 = datetime.datetime.now()
    perm = epoch_perms[epoch]
    row_losses = {}

    for st in strategies:
        tag = st['tag']
        net = models[tag]
        ml, bl = get_lrs(st, epoch, EPOCHS)

        ep_loss, nb = 0.0, 0
        for bs in range(0, num_samples, BATCH_SIZE):
            idx = perm[bs:bs + BATCH_SIZE]
            X = all_X[idx]
            X_in = make_masked(X, seed=200000 * epoch + bs)
            Y, _ = net(X_in)
            d = util.log_mat_v2(Y) - util.log_mat_v2(X)
            loss = (d.real ** 2 + d.imag ** 2).mean()
            net.zero_grad()
            loss.backward()
            net.update_para(ml, bn_lr=bl)
            ep_loss += loss.item()
            nb += 1
        avg = ep_loss / nb
        hist[tag].append(avg)
        row_losses[tag] = avg

    dt = (datetime.datetime.now() - t0).total_seconds()
    losses_str = '  '.join(f"{st['tag'].split('_')[0]}={row_losses[st['tag']]:.5f}"
                            for st in strategies)
    print(f'epoch {epoch+1}/{EPOCHS}  {losses_str}  time={dt:.1f}s', flush=True)

    with open(CSV_PATH, 'a') as f:
        vals = ','.join(f"{row_losses[st['tag']]:.6f}" for st in strategies)
        f.write(f'{epoch+1},{vals}\n')

    # 更新对比图
    colors = ['#888888', '#D64550', '#2E86AB', '#F4A261']
    plt.figure(figsize=(13, 6))
    for i, st in enumerate(strategies):
        ep_axis = range(1, len(hist[st['tag']]) + 1)
        plt.plot(ep_axis, hist[st['tag']], color=colors[i], linewidth=2, label=st['label'])
    plt.axhline(y=0.105, color='#888888', linestyle='--', alpha=0.5, label='固定LR平台(~0.105)')
    plt.xlabel('Epoch (续训)', fontsize=12)
    plt.ylabel('Log-Euclidean Loss', fontsize=12)
    plt.title(f'学习率衰减策略对比 (从已收敛 BN 模型续训, epoch {epoch+1}/{EPOCHS})',
              fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=150, bbox_inches='tight')
    plt.close()

    if (epoch + 1) % SAVE_EVERY == 0 or (epoch + 1) == EPOCHS:
        for st in strategies:
            path = os.path.join(SAVE_DIR, f"decay_{st['tag']}.model")
            torch.save(models[st['tag']], path)

# 最终存权重
for st in strategies:
    path = os.path.join(SAVE_DIR, f"decay_{st['tag']}.model")
    torch.save(models[st['tag']], path)
    print(f'  已保存: {path}  终点loss={hist[st["tag"]][-1]:.6f}', flush=True)

print(f'\n衰减实验完成。对比图: {PNG_PATH}  数据: {CSV_PATH}', flush=True)
