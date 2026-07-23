# 流形合规性探针实验: 验证训练好的 HPD 自编码器各层输出是否严格留在 HPD 流形上
# 检测指标:
#   1. Hermitian 误差: ||X - X^H||_F / ||X||_F (相对误差, 应 ~1e-15)
#   2. 正定性: 最小特征值 min_eig > 0
#   3. 条件数: max_eig / min_eig (数值稳定性)
#   4. log(det): 流形上的"位置"指标
#   5. 特征值分布: 是否远离零 (ReEig 阈值 1e-4 是否生效)
import os
import numpy as np
import torch
import scipy.io as sio

import model as model_mod
import util

# ==================== 配置 ====================
DATA_DIR = 'D:/data/customed/customed_64'
FILE_LIST_PATH = 'D:/data/customed/test.txt'
MODEL_PATH = 'tmp/customed/saved/autoencoder_detnorm.model'
N_PROBE = 100  # 探针样本数

# ==================== Det 归一化 ====================
def norm_det(X):
    n = X.shape[-1]
    with torch.no_grad():
        ev = torch.linalg.eigvalsh(X)
        log_det = torch.log(ev).sum(dim=-1, keepdim=True).unsqueeze(-1)
        scale = torch.exp(log_det / n)
    return X / scale

# ==================== 加载模型 ====================
print(f'加载模型: {MODEL_PATH}')
net = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
net.eval()

# ==================== 加载测试数据 ====================
with open(FILE_LIST_PATH) as fid:
    file_list = [ln.strip('\n').split(' ')[0].replace('\\', '/') for ln in fid]
N_PROBE = min(N_PROBE, len(file_list))
print(f'探针样本数: {N_PROBE}')

all_data = np.zeros((N_PROBE, 64, 64), dtype=np.complex128)
for i in range(N_PROBE):
    all_data[i] = sio.loadmat(os.path.join(DATA_DIR, file_list[i]))['Y1']
all_X = torch.from_numpy(all_data).to(torch.complex128)
all_X = norm_det(all_X)  # Det 归一化 (与训练一致)

# ==================== 前向推理, 收集中间层 ====================
print('前向推理...', flush=True)
with torch.no_grad():
    Y, layer_outputs = net(all_X)

# layer_outputs 结构:
#   [0:12]  = encoder 各层输出 (64→56→49→42→36→30→25→20→16→12→9→6→4)
#   [12:24] = decoder 各层输出 (4→6→9→12→16→20→25→30→36→42→49→56→64)
enc_dims = [56, 49, 42, 36, 30, 25, 20, 16, 12, 9, 6, 4]
dec_dims = [6, 9, 12, 16, 20, 25, 30, 36, 42, 49, 56, 64]

# ==================== 探针检测函数 ====================
def probe_hpd(X, name):
    """检测一批矩阵的 HPD 流形合规性。"""
    B, n, _ = X.shape
    # 1. Hermitian 误差
    X_H = X.conj().transpose(-2, -1)
    herm_err = (X - X_H).norm(dim=(-2, -1)) / X.norm(dim=(-2, -1))  # [B]
    # 2. 特征值分析
    min_eigs = []
    max_eigs = []
    conds = []
    log_dets = []
    for i in range(B):
        Xi = 0.5 * (X[i] + X[i].conj().T)  # 对称化后取特征值
        ev = torch.linalg.eigvalsh(Xi).numpy()
        min_eigs.append(ev[0])
        max_eigs.append(ev[-1])
        conds.append(ev[-1] / max(ev[0], 1e-30))
        log_dets.append(np.sum(np.log(np.maximum(ev, 1e-30))))
    min_eigs = np.array(min_eigs)
    max_eigs = np.array(max_eigs)
    conds = np.array(conds)
    log_dets = np.array(log_dets)
    # 3. 正定性判定
    n_pos = np.sum(min_eigs > 0)
    n_neg = np.sum(min_eigs <= 0)
    # 汇总
    print(f'\n{"="*60}')
    print(f'  {name}  (dim={n}×{n}, B={B})')
    print(f'{"="*60}')
    print(f'  Hermitian 误差: mean={herm_err.mean():.2e}, max={herm_err.max():.2e}')
    print(f'  最小特征值:     mean={min_eigs.mean():.4e}, min={min_eigs.min():.4e}')
    print(f'  最大特征值:     mean={max_eigs.mean():.4e}, max={max_eigs.max():.4e}')
    print(f'  条件数:         mean={conds.mean():.2e}, max={conds.max():.2e}')
    print(f'  log(det):       mean={log_dets.mean():.4f}, std={log_dets.std():.4f}')
    print(f'  正定: {n_pos}/{B} ({100*n_pos/B:.1f}%)  非正定: {n_neg}/{B}')
    if n_neg > 0:
        neg_idx = np.where(min_eigs <= 0)[0]
        print(f'  ⚠️ 非正定样本: idx={neg_idx[:5]}, min_eig={min_eigs[neg_idx[:5]]}')
    return {
        'herm_err_mean': herm_err.mean().item(),
        'herm_err_max': herm_err.max().item(),
        'min_eig_mean': min_eigs.mean(),
        'min_eig_min': min_eigs.min(),
        'cond_mean': conds.mean(),
        'cond_max': conds.max(),
        'n_pos': n_pos,
        'n_neg': n_neg,
    }

# ==================== 逐层探针 ====================
print('\n' + '='*60)
print('  HPD 流形合规性探针实验')
print('  模型: Det归一化训练 500 epoch, loss=0.0420')
print('='*60)

# 输入
probe_hpd(all_X, '输入 (Det归一化后)')

# Encoder 各层
for i, out in enumerate(layer_outputs[:12]):
    probe_hpd(out, f'Encoder Layer {i+1} (dim={enc_dims[i]})')

# Decoder 各层
for i, out in enumerate(layer_outputs[12:]):
    probe_hpd(out, f'Decoder Layer {i+1} (dim={dec_dims[i]})')

# 最终输出
probe_hpd(Y, '最终输出 (重建)')

# ==================== 额外探针: Stiefel 权重正交性 ====================
print('\n' + '='*60)
print('  Stiefel 权重正交性检测 (W^H W = I)')
print('='*60)
for i, w in enumerate(net.weights[:12]):
    W = w.data
    WtW = W.conj().T @ W
    I = torch.eye(WtW.shape[0], dtype=WtW.dtype)
    orth_err = (WtW - I).norm().item() / I.norm().item()
    print(f'  Enc W[{i}] shape={list(W.shape)}: ||W^H W - I||/||I|| = {orth_err:.2e}')

for i, w in enumerate(net.weights[12:]):
    W = w.data
    WtW = W.conj().T @ W
    I = torch.eye(WtW.shape[0], dtype=WtW.dtype)
    orth_err = (WtW - I).norm().item() / I.norm().item()
    print(f'  Dec W[{i}] shape={list(W.shape)}: ||W^H W - I||/||I|| = {orth_err:.2e}')

# ==================== BN 偏置 G 的 HPD 性 ====================
print('\n' + '='*60)
print('  黎曼 BN 偏置 G 的 HPD 性')
print('='*60)
if net.use_bn:
    for i, bn in enumerate(net.bn_layers):
        G = bn.G.data
        G_herm = 0.5 * (G + G.conj().T)
        ev = torch.linalg.eigvalsh(G_herm).numpy()
        herm_err = (G - G.conj().T).norm().item() / G.norm().item()
        print(f'  BN[{i}] G (dim={bn.dim}): min_eig={ev.min():.4e}, '
              f'max_eig={ev.max():.4e}, herm_err={herm_err:.2e}, '
              f'PD={"✓" if ev.min() > 0 else "✗"}')

print('\n探针实验完成。')
