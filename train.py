# 训练脚本: U 型 HPD 自编码器
# 任务: 输入 64x64 HPD 矩阵 X, 重建 Y = X, loss = Log-Euclidean 距离
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio
import datetime

import model
import util

# ==============================================
# 命令行参数
# ==============================================
parser = argparse.ArgumentParser(description='HPD 自编码器训练')
parser.add_argument('--bn', action='store_true', default=True,
                    help='启用 LieBN (Log-Euclidean Riemannian BatchNorm), 默认启用')
parser.add_argument('--no-bn', dest='bn', action='store_false',
                    help='禁用 LieBN')
args = parser.parse_args()

# ===================== 绘图全局配置 =====================
import matplotlib.pyplot as plt
plt.rcParams ['font.sans-serif'] = ['SimHei'] # 解决中文显示
plt.rcParams ['axes.unicode_minus'] = False # 解决负号显示
plt.ion () # 交互模式，实时更新画布
loss_fig, loss_ax = plt.subplots (figsize=(10, 6))
train_loss_history = [] # 记录每轮 epoch 平均 loss

CURVE_SAVE_DIR = "tmp/customed/loss_curve"
os.makedirs(CURVE_SAVE_DIR, exist_ok=True)

# ==============================================
# 配置 (沿用 Windows 路径, 本机不运行)
# ==============================================
DATA_DIR = 'D:/data/customed/customed_64'
FILE_LIST_PATH = 'D:/data/customed/train.txt'
SAVE_PATH = 'tmp/customed/saved/autoencoder.model'
LOAD_WEIGHT_PATH = SAVE_PATH
LOAD_WEIGHT = False

BATCH_SIZE = 4096
EPOCHS = 400
LR = 10

# ==============================================
# 加载文件列表 (标签不参与训练, 只取文件名)
# ==============================================
with open(FILE_LIST_PATH, 'r') as fid:
    file_list = []
    for line in fid.readlines():
        file, _ = line.strip('\n').split(' ')
        file = file.replace('\\', '/')
        file_list.append(file)

num_samples = len(file_list)
print(f'训练样本数: {num_samples}')
print(f'LieBN: {"启用" if args.bn else "禁用"}')

# ==============================================
# 初始化模型
# ==============================================
model = model.HPDNetwork(use_bn=args.bn)

if LOAD_WEIGHT and os.path.exists(LOAD_WEIGHT_PATH):
    print(f"开始加载已有模型权重: {LOAD_WEIGHT_PATH}")
    # 方式1：完整加载模型对象（和你保存方式匹配 torch.save(model)）
    model = torch.load(LOAD_WEIGHT_PATH)
    # 若你只保存state_dict，改用下面两行：
    # checkpoint = torch.load(LOAD_WEIGHT_PATH)
    # model.load_state_dict(checkpoint)
    print("权重加载完成，继续训练！")
elif LOAD_WEIGHT:
    print(f"警告：权重文件 {LOAD_WEIGHT_PATH} 不存在，将从头开始训练")

model.train()

# ==============================================
# 训练循环
# ==============================================
for epoch in range(EPOCHS):
    epoch_start = datetime.datetime.now()
    # 每个 epoch 打乱顺序
    perm = np.random.permutation(num_samples)
    epoch_loss = 0.0
    num_batches = 0

    for batch_start in range(0, num_samples, BATCH_SIZE):
        batch_files = [file_list[i] for i in perm[batch_start:batch_start + BATCH_SIZE]]
        actual_bs = len(batch_files)

        # 加载 batch 数据
        batch_data = np.zeros((actual_bs, 64, 64), dtype=np.complex128)
        for i, file in enumerate(batch_files):
            hpd = sio.loadmat(os.path.join(DATA_DIR, file))['Y1']
            batch_data[i, :, :] = hpd

        X = torch.from_numpy(batch_data).to(torch.complex128)
        X.requires_grad = False

        # 输入损坏：denoising AE，每个样本随机压 n 个特征值到 eps；目标仍是干净 X
        X_noisy = util.mask_random_eigvals(X,8,1e-8)
        # 前向
        Y, _ = model(X_noisy)

        # 前向
        # Y, _ = model(X)

        # Log-Euclidean loss: ||logm(Y) - logm(X)||_F^2, mean reduction
        log_Y = util.log_mat_v2(Y)
        log_X = util.log_mat_v2(X)
        diff = log_Y - log_X
        loss = (diff.real ** 2 + diff.imag ** 2).mean()

        # 反向 + Riemannian 更新
        model.zero_grad()
        loss.backward()
        model.update_para(LR)

        epoch_loss += loss.item()
        num_batches += 1

    # 更新 Loss
    avg_loss = epoch_loss / max(num_batches, 1)
    elapsed = (datetime.datetime.now() - epoch_start).total_seconds()
    print(f'epoch {epoch + 1}/{EPOCHS} loss={avg_loss:.6f} time={elapsed:.1f}s')
    train_loss_history.append(avg_loss)
    # 清空画布重绘
    loss_ax.clear()
    loss_ax.plot(range(1, len(train_loss_history) + 1), train_loss_history, color='#2E86AB', linewidth=2,
                 label='Train Log-Euclidean Loss')
    loss_ax.set_xlabel('Epoch', fontsize=12)
    loss_ax.set_ylabel('Average Loss', fontsize=12)
    loss_ax.set_title(f'Training Loss Curve (Current Epoch: {epoch + 1}, Loss: {avg_loss:.6f})', fontsize=14)
    loss_ax.legend(fontsize=11)
    loss_ax.grid(True, alpha=0.3)
    loss_fig.tight_layout()
    plt.draw()
    plt.pause(0.01)  # 短暂暂停刷新窗口

    curve_save_path = os.path.join(CURVE_SAVE_DIR, f"loss_curve_epoch_{epoch + 1:03d}.png")
    plt.savefig(curve_save_path, dpi=300, bbox_inches='tight')
    print(f"当前 epoch loss 曲线已保存：{curve_save_path}")

    # 保存模型 (整对象方式, 与 test.py 的 torch.load 风格一致)
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(model, SAVE_PATH)
    print(f'模型已保存到 {SAVE_PATH}')


































