# 学习率策略 sweep: 从 15-epoch baseline 副本出发, 每个策略跑 10 epoch, 对比下降速度。
# 公平性保证:
#   - 每个 trial 从同一份副本 deepcopy 加载 (绝不覆盖副本)
#   - 每个 trial 用相同的 per-epoch 数据打乱顺序 (固定种子)
#   - denoising 损坏也用固定种子, 各 trial 完全一致
#   - 数据一次性载入内存, 各 trial 共用
import os
import copy
import numpy as np
import torch
import scipy.io as sio

import model as model_mod
import util

DATA_DIR = 'D:/data/customed/customed_64'
FILE_LIST_PATH = 'D:/data/customed/train.txt'
BASELINE = 'tmp/customed/saved/autoencoder_epoch15_baseline.model'

# sweep 用小配置: util 的 log/exp/rec 是逐样本 eigh (Python 循环), 速度∝样本数。
# 只比较各策略"相对下降速度", 小子集 + 固定划分即可保证公平可比。
BATCH_SIZE = 128         # 小批量, 快
TRIAL_EPOCHS = 10        # 10 epoch 看趋势与后期稳定性
N_SUBSET = 512           # 512 样本子集 (快速对比相对下降速度)
N_MASK = 8
MASK_EPS = 1e-8

# ---- 载入文件列表 + 子集数据到内存 ----
with open(FILE_LIST_PATH) as fid:
    file_list = [ln.strip('\n').split(' ')[0].replace('\\', '/') for ln in fid]
file_list = file_list[:N_SUBSET]
num_samples = len(file_list)
print(f'sweep 样本数: {num_samples} (batch={BATCH_SIZE}, epochs={TRIAL_EPOCHS}), 预载入内存...', flush=True)
all_data = np.zeros((num_samples, 64, 64), dtype=np.complex128)
for i, f in enumerate(file_list):
    all_data[i] = sio.loadmat(os.path.join(DATA_DIR, f))['Y1']
all_X = torch.from_numpy(all_data).to(torch.complex128)
print('数据载入完成。', flush=True)

# ---- 预生成每个 epoch 的打乱顺序 (所有 trial 共用, 保证公平) ----
rng = np.random.RandomState(1234)
epoch_perms = [rng.permutation(num_samples) for _ in range(TRIAL_EPOCHS)]

# baseline 参考: 加载副本, 先算一次当前 loss (未训练), 作为起点 L0
baseline_model = torch.load(BASELINE, weights_only=False)


def make_masked(X, seed):
    """确定性的特征值损坏 (denoising 输入), 各 trial 用相同 seed 保证一致。"""
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


BN_LR = 200.0   # G 的独立学习率固定 (全量已验证 200 最快且稳)


def run_trial(strategy):
    """固定 bn_lr=200, 扫主 LR (驱动 Stiefel 权重 + decoder theta)。"""
    net = copy.deepcopy(baseline_model)   # 从副本 deepcopy, 不动磁盘副本
    net.train()
    main_lr = strategy['main_lr']
    losses = []
    for ep in range(TRIAL_EPOCHS):
        perm = epoch_perms[ep]
        ep_loss, nb = 0.0, 0
        for bs in range(0, num_samples, BATCH_SIZE):
            idx = perm[bs:bs + BATCH_SIZE]
            X = all_X[idx]
            X_in = make_masked(X, seed=1000 * ep + bs)  # 确定性损坏
            Y, _ = net(X_in)
            d = util.log_mat_v2(Y) - util.log_mat_v2(X)
            loss = (d.real ** 2 + d.imag ** 2).mean()
            net.zero_grad()
            loss.backward()
            net.update_para(main_lr, bn_lr=BN_LR)   # 主 LR 扫, G 固定 200
            ep_loss += loss.item()
            nb += 1
        avg = ep_loss / nb
        losses.append(avg)
    return losses


# 固定 bn_lr=200, 扫主 LR (G 步长大幅提高后, 主 LR 最优点可能已变, 旧结论过时)
strategies = [
    {'main_lr': 0.0,   'tag': 'main_lr=0 (冻结权重)'},
    {'main_lr': 5.0,   'tag': 'main_lr=5'},
    {'main_lr': 10.0,  'tag': 'main_lr=10 (当前)'},
    {'main_lr': 20.0,  'tag': 'main_lr=20'},
    {'main_lr': 40.0,  'tag': 'main_lr=40'},
    {'main_lr': 80.0,  'tag': 'main_lr=80'},
]

# baseline 起点 loss (epoch0 之前), 用第一个 epoch perm 的首 batch 估个参考不必要, 直接看各 trial
RESULT_FILE = '_lr_sweep_result.txt'
fout = open(RESULT_FILE, 'w', encoding='utf-8')


def emit(line=''):
    print(line, flush=True)
    fout.write(line + '\n')
    fout.flush()


emit('=' * 72)
emit(f'{"策略":<20}{"e1":>8}{"e5":>8}{"e10":>8}{"总降幅":>10}{"NaN":>6}')
emit('-' * 72)
results = {}
for st in strategies:
    emit(f'... 正在跑 {st["tag"]}')
    losses = run_trial(st)
    results[st['tag']] = losses
    has_nan = any(not np.isfinite(x) for x in losses)
    total_drop = losses[0] - losses[-1]
    emit(f"{st['tag']:<20}{losses[0]:>8.4f}{losses[4]:>8.4f}{losses[-1]:>8.4f}"
         f"{total_drop:>10.4f}{str(has_nan):>6}")

emit('=' * 72)
emit('完整轨迹 (每个策略 10 epoch):')
for tag, losses in results.items():
    emit(f"  {tag:<20}: " + " ".join(f"{x:.4f}" for x in losses))

# 选出 e10 最低且无 NaN 的
valid = {t: l for t, l in results.items() if all(np.isfinite(x) for x in l)}
best = min(valid, key=lambda t: valid[t][-1])
emit(f">> 最低终点 loss 且稳定: {best}  (e10={valid[best][-1]:.4f})")
fout.close()
