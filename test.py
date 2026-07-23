# 特征提取脚本: 用训练好的 U 型 HPD 自编码器, 提取每个输入样本的 4x4 bottleneck 特征
# 输出: middlefeature0.mat (原始标签=1), middlefeature1.mat (原始标签=2)
#       同时输出每个样本的重建 Loss 并绘制曲线
import os
import numpy as np
import torch
import scipy.io as sio
import datetime

import model
import util

# ===================== 绘图全局配置 =====================
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.ion()

loss_fig, loss_ax = plt.subplots(figsize=(12, 6))
sample_loss_history = []

LOSS_CURVE_SAVE_DIR = "tmp/customed/loss_curve_test"
os.makedirs(LOSS_CURVE_SAVE_DIR, exist_ok=True)

# ==============================================
# 配置
# ==============================================
DATA_DIR = 'D:/data/customed/customed_64'
FILE_LIST_PATH = 'D:/data/customed/test.txt'
MODEL_PATH = 'tmp/customed/saved/autoencoder_epoch500.model'

# ★ 标签 1 → middlefeature0.mat, 标签 2 → middlefeature1.mat ★
SAVE_PATH_0 = 'D:/matlabcode/HPDNet-fpn/middlefeature0.mat'   # 原始标签 = 1
SAVE_PATH_1 = 'D:/matlabcode/HPDNet-fpn/middlefeature1.mat'   # 原始标签 = 2

LOSS_SAVE_PATH = 'tmp/customed/saved/test_sample_loss.mat'

BATCH_SIZE = 4096

# ==============================================
# 加载文件列表（保留标签）
# ==============================================
file_list = []
label_list = []

with open(FILE_LIST_PATH, 'r') as fid:
    for line in fid.readlines():
        parts = line.strip('\n').split(' ')
        file = parts[0].replace('\\', '/')
        label = int(parts[1])
        file_list.append(file)
        label_list.append(label)

num_samples = len(file_list)
label_array = np.array(label_list)  # shape: [N]

print(f'样本数: {num_samples}')
print(f'  原始标签=1 样本数: {np.sum(label_array == 1)} → 保存为 middlefeature0.mat')
print(f'  原始标签=2 样本数: {np.sum(label_array == 2)} → 保存为 middlefeature1.mat')

# ==============================================
# 加载训练好的自编码器
# ==============================================
model = torch.load(MODEL_PATH, map_location='cpu')
model.eval()

# ==============================================
# 推理循环
# ==============================================
X_low_list = []
all_sample_losses = []

with torch.no_grad():
    for batch_start in range(0, num_samples, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, num_samples)
        batch_files = file_list[batch_start:batch_end]
        actual_bs = len(batch_files)

        batch_data = np.zeros((actual_bs, 64, 64), dtype=np.complex128)
        for i, file in enumerate(batch_files):
            spd = sio.loadmat(os.path.join(DATA_DIR, file))['Y1']
            batch_data[i, :, :] = spd

        X = torch.from_numpy(batch_data).to(torch.complex128)
        stime = datetime.datetime.now()

        # ---------- 前向传播 ----------
        Y, layer_outputs = model(X)

        # ---------- bottleneck 特征 ----------
        X_low = layer_outputs[11]  # [B, 4, 4] HPD
        X_low_list.append(X_low.detach())

        # ---------- 逐样本 Log-Euclidean Loss ----------
        log_Y = util.log_mat_v2(Y)
        log_X = util.log_mat_v2(X)
        diff = log_Y - log_X
        per_sample_loss = (diff.real ** 2 + diff.imag ** 2).mean(dim=(-2, -1))
        per_sample_loss_np = per_sample_loss.numpy()
        all_sample_losses.append(per_sample_loss_np)

        # ---------- 实时更新 Loss 曲线 ----------
        sample_loss_history.extend(per_sample_loss_np.tolist())

        loss_ax.clear()
        loss_ax.plot(
            range(1, len(sample_loss_history) + 1),
            sample_loss_history,
            color='#E74C3C', linewidth=1.2, alpha=0.7,
            label='逐样本 Log-Euclidean Loss'
        )
        window = 50
        if len(sample_loss_history) >= 2:
            moving_avg = np.convolve(
                sample_loss_history,
                np.ones(window) / window, mode='valid'
            )
            loss_ax.plot(
                range(window, window + len(moving_avg)),
                moving_avg,
                color='#2E86AB', linewidth=2.5,
                label=f'滑动平均 (window={window})'
            )
        loss_ax.set_xlabel('样本索引', fontsize=12)
        loss_ax.set_ylabel('Log-Euclidean Loss', fontsize=12)
        loss_ax.set_title(
            f'测试集逐样本重建 Loss (已处理: {len(sample_loss_history)}/{num_samples})',
            fontsize=14
        )
        loss_ax.legend(fontsize=11)
        loss_ax.grid(True, alpha=0.3)
        loss_fig.tight_layout()
        plt.draw()
        plt.pause(0.01)

        elapsed = (datetime.datetime.now() - stime).total_seconds()
        print(f'  batch {batch_start // BATCH_SIZE + 1}: {actual_bs} samples; '
              f'batch_avg_loss={per_sample_loss_np.mean():.6f}; {elapsed:.2f}s')

# ==============================================
# 拼接全部特征和 Loss
# ==============================================
final_X_low = np.concatenate([t.numpy() for t in X_low_list], axis=0)    # [N, 4, 4]
final_sample_losses = np.concatenate(all_sample_losses, axis=0)          # [N]

print(f'\n最终特征数组 shape: {final_X_low.shape}, dtype: {final_X_low.dtype}')
print(f'总平均 Loss: {final_sample_losses.mean():.6f}')

# ==============================================
# ★ 按原始标签 1 / 2 分类（映射为 0 / 1）保存 ★
# ==============================================
mask_label1 = (label_array == 1)  # 原始标签 1 → middlefeature0
mask_label2 = (label_array == 2)  # 原始标签 2 → middlefeature1

X_low_0 = final_X_low[mask_label1]   # 原始标签=1 → middlefeature0.mat
X_low_1 = final_X_low[mask_label2]   # 原始标签=2 → middlefeature1.mat

loss_0 = final_sample_losses[mask_label1]
loss_1 = final_sample_losses[mask_label2]

files_0 = [f for f, l in zip(file_list, label_list) if l == 1]
files_1 = [f for f, l in zip(file_list, label_list) if l == 2]

print(f'\n===== 按标签分类保存（原始标签 → 文件名映射） =====')
print(f'  原始标签=1 → middlefeature0.mat: {X_low_0.shape[0]} 个样本')
print(f'    avg_loss={loss_0.mean():.6f}, max={loss_0.max():.6f}, min={loss_0.min():.6f}')
print(f'  原始标签=2 → middlefeature1.mat: {X_low_1.shape[0]} 个样本')
print(f'    avg_loss={loss_1.mean():.6f}, max={loss_1.max():.6f}, min={loss_1.min():.6f}')

# 保存 → middlefeature0.mat（原始标签=1）
os.makedirs(os.path.dirname(SAVE_PATH_0), exist_ok=True)
sio.savemat(SAVE_PATH_0, {
    'X_low': X_low_0,
    'sample_loss': loss_0,
    'mean_loss': loss_0.mean(),
    'file_list': files_0
})
print(f'  ✓ 已保存 {SAVE_PATH_0}')

# 保存 → middlefeature1.mat（原始标签=2）
os.makedirs(os.path.dirname(SAVE_PATH_1), exist_ok=True)
sio.savemat(SAVE_PATH_1, {
    'X_low': X_low_1,
    'sample_loss': loss_1,
    'mean_loss': loss_1.mean(),
    'file_list': files_1
})
print(f'  ✓ 已保存 {SAVE_PATH_1}')

# ==============================================
# 保存完整 Loss 信息（所有样本）
# ==============================================
# 同时将标签从 1/2 映射为 0/1 保存
label_mapped = np.where(label_array == 1, 0, 1)  # 1→0, 2→1

os.makedirs(os.path.dirname(LOSS_SAVE_PATH), exist_ok=True)
sio.savemat(LOSS_SAVE_PATH, {
    'sample_loss': final_sample_losses,
    'mean_loss': final_sample_losses.mean(),
    'label': label_array,               # 原始标签 1/2
    'label_mapped': label_mapped,       # 映射后标签 0/1
    'file_list': file_list
})
print(f'\n逐样本 Loss 已保存到 {LOSS_SAVE_PATH}')

# ==============================================
# 绘制最终 Loss 曲线（按原始标签 1/2 着色）
# ==============================================
plt.ioff()

# ----- 图1: 逐样本 Loss 散点图 -----
final_fig, final_ax = plt.subplots(figsize=(14, 6))

scatter_x = np.arange(1, num_samples + 1)
final_ax.scatter(
    scatter_x[mask_label1], final_sample_losses[mask_label1],
    c='#2E86AB', s=8, alpha=0.6,
    label=f'标签=1→middlefeature0 (n={mask_label1.sum()}, avg={loss_0.mean():.6f})'
)
final_ax.scatter(
    scatter_x[mask_label2], final_sample_losses[mask_label2],
    c='#E74C3C', s=8, alpha=0.6,
    label=f'标签=2→middlefeature1 (n={mask_label2.sum()}, avg={loss_1.mean():.6f})'
)

window = 50
if len(final_sample_losses) >= window:
    moving_avg = np.convolve(
        final_sample_losses, np.ones(window) / window, mode='valid'
    )
    final_ax.plot(
        range(window, window + len(moving_avg)), moving_avg,
        color='#2C3E50', linewidth=2.5, label=f'滑动平均 (window={window})'
    )

final_ax.set_xlabel('样本索引', fontsize=13)
final_ax.set_ylabel('Log-Euclidean Loss', fontsize=13)
final_ax.set_title(f'测试集逐样本重建 Loss — 按标签着色 (共 {num_samples} 个样本)', fontsize=15)
final_ax.legend(fontsize=12, loc='upper right')
final_ax.grid(True, alpha=0.3)
final_fig.tight_layout()

final_curve_path = os.path.join(LOSS_CURVE_SAVE_DIR, "test_loss_curve_final.png")
final_fig.savefig(final_curve_path, dpi=300, bbox_inches='tight')
print(f'最终 Loss 曲线已保存到 {final_curve_path}')

# ----- 图2: 按标签分组的 Loss 直方图 -----
hist_fig, hist_ax = plt.subplots(figsize=(10, 6))

hist_ax.hist(loss_0, bins=80, color='#2E86AB', alpha=0.6,
             label=f'标签=1→0 (n={len(loss_0)}, mean={loss_0.mean():.6f})', edgecolor='white')
hist_ax.hist(loss_1, bins=80, color='#E74C3C', alpha=0.6,
             label=f'标签=2→1 (n={len(loss_1)}, mean={loss_1.mean():.6f})', edgecolor='white')

hist_ax.axvline(x=loss_0.mean(), color='#2E86AB', linestyle='--', linewidth=2)
hist_ax.axvline(x=loss_1.mean(), color='#E74C3C', linestyle='--', linewidth=2)

hist_ax.set_xlabel('Log-Euclidean Loss', fontsize=13)
hist_ax.set_ylabel('样本数量', fontsize=13)
hist_ax.set_title(f'测试集 Loss 分布 — 按标签分组 (共 {num_samples} 个样本)', fontsize=15)
hist_ax.legend(fontsize=12)
hist_ax.grid(True, alpha=0.3)
hist_fig.tight_layout()

hist_save_path = os.path.join(LOSS_CURVE_SAVE_DIR, "test_loss_histogram_by_label.png")
hist_fig.savefig(hist_save_path, dpi=300, bbox_inches='tight')
print(f'Loss 直方图已保存到 {hist_save_path}')

plt.show()