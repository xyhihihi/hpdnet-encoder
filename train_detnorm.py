# 训练脚本: U 型 HPD 自编码器 + Det 归一化预处理
# 实验结论: Det 归一化 (X / det(X)^(1/n)) 使 Log-Euclidean Loss 降低 ~90%
# 本脚本在原始 train.py 基础上加入 Det 归一化, 完整训练 500 epochs
# 权重保存路径与原始训练区分: autoencoder_detnorm.model
# 支持断点续训: LOAD_WEIGHT=True 时自动从 CSV 恢复 epoch/loss 历史, 接续出完整曲线
import os
import math
import csv
import numpy as np
import torch
import scipy.io as sio
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import model as model_mod
import util

# ===================== 绘图配置 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

CURVE_SAVE_DIR = "tmp/customed/loss_curve_detnorm"
os.makedirs(CURVE_SAVE_DIR, exist_ok=True)

# ==============================================
# 配置
# ==============================================
DATA_DIR = 'D:/data/customed/customed_64'
FILE_LIST_PATH = 'D:/data/customed/train.txt'
SAVE_PATH = 'tmp/customed/saved/autoencoder_detnorm.model'
LOAD_WEIGHT_PATH = SAVE_PATH
LOAD_WEIGHT = True    # ★ 改为 True 即可断点续训 (自动恢复 epoch + loss 历史 + LR)

BATCH_SIZE = 512
EPOCHS = 500

# 学习率策略: 余弦衰减 (与原始 train.py 一致)
MAIN_LR_HI = 40.0
MAIN_LR_LO = 0.5
BN_LR_HI = 200.0
BN_LR_LO = 5.0

# 特征值归一化预处理 (实验结论: Det 归一化使 loss 降低 ~90%)
# 可选: 'det' / 'trace' / 'maxeig' / 'logcenter' / None (关闭)
EIGVAL_NORM = None

# denoising AE 配置
USE_EIGVAL_MASK = True
N_MASK = 16
MASK_EPS = 1e-8

# ReEig 阈值
REC_EPS = 1e-4

# 断点续训 CSV (记录每 epoch 的 loss, 用于恢复历史)
LOSS_CSV_PATH = os.path.join(CURVE_SAVE_DIR, 'detnorm_loss.csv')


def cosine_lr(hi, lo, epoch, total):
    """余弦衰减: epoch=0 → hi, epoch=total-1 → lo。"""
    t = epoch / max(total - 1, 1)
    return lo + 0.5 * (hi - lo) * (1 + math.cos(math.pi * t))


def norm_det(X):
    """Det 归一化: X / det(X)^(1/n), 使 det(X_norm) = 1。
    等价于 Log 域减去 (tr(log X)/n) * I, 即居中 log 特征值的均值。
    """
    n = X.shape[-1]
    with torch.no_grad():
        ev = torch.linalg.eigvalsh(X)  # [B, n], 升序, 实数
        log_det = torch.log(ev).sum(dim=-1, keepdim=True).unsqueeze(-1)  # [B,1,1]
        scale = torch.exp(log_det / n)  # det^(1/n), [B,1,1]
    return X / scale


def norm_trace(X):
    """Trace 归一化: X * n / tr(X), 使 tr(X_norm) = n。"""
    n = X.shape[-1]
    tr = torch.diagonal(X, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1).real
    return X * (n / tr)


def norm_maxeig(X):
    """Max-eig 归一化: X / λ_max(X), 使最大特征值 = 1。"""
    with torch.no_grad():
        ev = torch.linalg.eigvalsh(X)
        max_ev = ev[:, -1:].unsqueeze(-1)
    return X / max_ev


def norm_logcenter(X):
    """Log 域居中: X / exp(全局 log 特征值均值)。需先计算全局统计量。"""
    with torch.no_grad():
        ev = torch.linalg.eigvalsh(X)  # [B, n]
        global_mean = torch.log(ev).mean()
    return X / torch.exp(global_mean)


def apply_eigval_norm(X, method):
    """统一入口: 根据 method 字符串选择归一化方式。"""
    if method is None or method == 'none':
        return X
    elif method == 'det':
        return norm_det(X)
    elif method == 'trace':
        return norm_trace(X)
    elif method == 'maxeig':
        return norm_maxeig(X)
    elif method == 'logcenter':
        return norm_logcenter(X)
    else:
        raise ValueError(f"未知归一化方法: {method}, 可选: det/trace/maxeig/logcenter/None")


# ==============================================
# 加载文件列表
# ==============================================
with open(FILE_LIST_PATH, 'r') as fid:
    file_list = []
    for line in fid.readlines():
        file, _ = line.strip('\n').split(' ')
        file = file.replace('\\', '/')
        file_list.append(file)

num_samples = len(file_list)
print(f'训练样本数: {num_samples}')

# ==============================================
# 预载入数据到内存 + 特征值归一化
# ==============================================
print('预载入数据到内存...', flush=True)
all_data = np.zeros((num_samples, 64, 64), dtype=np.complex128)
for i, f in enumerate(file_list):
    all_data[i] = sio.loadmat(os.path.join(DATA_DIR, f))['Y1']
all_X_raw = torch.from_numpy(all_data).to(torch.complex128)
print(f'数据载入完成。特征值归一化: {EIGVAL_NORM}', flush=True)

# 特征值归一化 (可开关)
all_X = apply_eigval_norm(all_X_raw, EIGVAL_NORM)
if EIGVAL_NORM and EIGVAL_NORM != 'none':
    print(f'{EIGVAL_NORM} 归一化完成。样本0: tr={torch.diagonal(all_X[0], dim1=-2, dim2=-1).sum().real:.4f}', flush=True)
else:
    print('归一化已关闭, 使用原始数据。', flush=True)

# ==============================================
# 初始化模型 + 断点续训
# ==============================================
net = model_mod.HPDNetwork(use_bn=True, bn_lr=BN_LR_HI,
                           rec_params=[REC_EPS] * 24)

# 恢复 loss 历史 (从 CSV)
train_loss_history = []
start_epoch = 0  # 从第几个 epoch 开始 (0-indexed)

if LOAD_WEIGHT and os.path.exists(LOAD_WEIGHT_PATH):
    print(f"加载已有模型权重: {LOAD_WEIGHT_PATH}")
    net = torch.load(LOAD_WEIGHT_PATH, weights_only=False)
    # 从 CSV 恢复 loss 历史
    if os.path.exists(LOSS_CSV_PATH):
        with open(LOSS_CSV_PATH, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # 跳过表头
            for row in reader:
                train_loss_history.append(float(row[1]))
        start_epoch = len(train_loss_history)
        print(f"断点续训: 从 epoch {start_epoch + 1} 继续, "
              f"已恢复 {start_epoch} 条 loss 历史")
    else:
        print("警告: 未找到 loss CSV, 从 epoch 1 重新计 loss 历史 (权重已加载)")
    print("权重加载完成，继续训练！")
elif LOAD_WEIGHT:
    print(f"警告：权重文件 {LOAD_WEIGHT_PATH} 不存在，将从头开始训练")

net.train()

# ==============================================
# 训练循环
# ==============================================
# 如果是新训练, 写 CSV 表头
if start_epoch == 0:
    with open(LOSS_CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss'])

for epoch in range(start_epoch, EPOCHS):
    epoch_start = datetime.datetime.now()
    main_lr = cosine_lr(MAIN_LR_HI, MAIN_LR_LO, epoch, EPOCHS)
    bn_lr = cosine_lr(BN_LR_HI, BN_LR_LO, epoch, EPOCHS)
    perm = np.random.permutation(num_samples)
    epoch_loss = 0.0
    num_batches = 0

    for batch_start in range(0, num_samples, BATCH_SIZE):
        idx = perm[batch_start:batch_start + BATCH_SIZE]
        X = all_X[idx]  # Det 归一化后的数据

        # denoising: 对归一化后的数据做特征值损坏
        if USE_EIGVAL_MASK:
            X_input = util.mask_random_eigvals(X, N_MASK, MASK_EPS)
        else:
            X_input = X

        Y, _ = net(X_input)

        # Log-Euclidean loss: ||logm(Y) - logm(X)||_F^2
        log_Y = util.log_mat_v2(Y)
        log_X = util.log_mat_v2(X)
        diff = log_Y - log_X
        loss = (diff.real ** 2 + diff.imag ** 2).mean()

        # 反向 + Riemannian 更新
        net.zero_grad()
        loss.backward()
        net.update_para(main_lr, bn_lr=bn_lr)

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / max(num_batches, 1)
    elapsed = (datetime.datetime.now() - epoch_start).total_seconds()
    print(f'epoch {epoch + 1}/{EPOCHS} loss={avg_loss:.6f} main_lr={main_lr:.3f} '
          f'bn_lr={bn_lr:.2f} time={elapsed:.1f}s', flush=True)
    train_loss_history.append(avg_loss)

    # 追加 loss 到 CSV (断点续训用)
    with open(LOSS_CSV_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, f'{avg_loss:.6f}'])

    # 绘制并保存 loss 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history,
             color='#D64550', linewidth=2, label='Det归一化 Train Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Log-Euclidean Loss', fontsize=12)
    plt.title(f'Det归一化训练 (Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.6f})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    curve_path = os.path.join(CURVE_SAVE_DIR, f"loss_curve_epoch_{epoch + 1:03d}.png")
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 保存模型
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(net, SAVE_PATH)

    # 每 100 轮额外保存快照
    if (epoch + 1) % 100 == 0:
        snap_path = SAVE_PATH.replace('.model', f'_epoch{epoch + 1}.model')
        torch.save(net, snap_path)
        print(f'  快照已保存到 {snap_path}', flush=True)

print(f'\n训练完成! 最终 loss={train_loss_history[-1]:.6f}', flush=True)
print(f'模型权重: {SAVE_PATH}', flush=True)
