# 特征值分布分析: 了解训练数据的特征值分布特征, 为归一化实验提供依据
import os
import numpy as np
import torch
import scipy.io as sio

DATA_DIR = 'D:/data/customed/customed_64'
FILE_LIST_PATH = 'D:/data/customed/train.txt'

with open(FILE_LIST_PATH) as fid:
    file_list = [ln.strip('\n').split(' ')[0].replace('\\', '/') for ln in fid]

num_samples = len(file_list)
print(f'总样本数: {num_samples}')

# 分析前 500 个样本的特征值统计
N_ANALYZE = min(500, num_samples)
print(f'分析前 {N_ANALYZE} 个样本...')

all_max_eig = []
all_min_eig = []
all_trace = []
all_det_log = []  # log(det) = sum(log(eig))
all_cond = []     # 条件数 = max_eig / min_eig
all_mean_eig = []
all_std_eig = []

for i in range(N_ANALYZE):
    hpd = sio.loadmat(os.path.join(DATA_DIR, file_list[i]))['Y1']
    X = torch.from_numpy(hpd).to(torch.complex128)
    ev = torch.linalg.eigvalsh(X).numpy()  # 升序
    all_max_eig.append(ev[-1])
    all_min_eig.append(ev[0])
    all_trace.append(ev.sum())
    all_det_log.append(np.sum(np.log(ev)))
    all_cond.append(ev[-1] / max(ev[0], 1e-30))
    all_mean_eig.append(ev.mean())
    all_std_eig.append(ev.std())

all_max_eig = np.array(all_max_eig)
all_min_eig = np.array(all_min_eig)
all_trace = np.array(all_trace)
all_det_log = np.array(all_det_log)
all_cond = np.array(all_cond)
all_mean_eig = np.array(all_mean_eig)
all_std_eig = np.array(all_std_eig)

print('\n===== 特征值分布统计 =====')
print(f'最大特征值: mean={all_max_eig.mean():.4e}, std={all_max_eig.std():.4e}, '
      f'min={all_max_eig.min():.4e}, max={all_max_eig.max():.4e}')
print(f'最小特征值: mean={all_min_eig.mean():.4e}, std={all_min_eig.std():.4e}, '
      f'min={all_min_eig.min():.4e}, max={all_min_eig.max():.4e}')
print(f'迹 (trace): mean={all_trace.mean():.4e}, std={all_trace.std():.4e}, '
      f'min={all_trace.min():.4e}, max={all_trace.max():.4e}')
print(f'log(det):   mean={all_det_log.mean():.4e}, std={all_det_log.std():.4e}, '
      f'min={all_det_log.min():.4e}, max={all_det_log.max():.4e}')
print(f'条件数:     mean={all_cond.mean():.4e}, std={all_cond.std():.4e}, '
      f'min={all_cond.min():.4e}, max={all_cond.max():.4e}, median={np.median(all_cond):.4e}')
print(f'均值特征值: mean={all_mean_eig.mean():.4e}, std={all_mean_eig.std():.4e}')
print(f'特征值标准差: mean={all_std_eig.mean():.4e}, std={all_std_eig.std():.4e}')

# 分析 trace 的变异系数 (CV)
trace_cv = all_trace.std() / all_trace.mean()
print(f'\n迹的变异系数 (CV): {trace_cv:.4f}')
print(f'log(det) 的变异系数: {all_det_log.std() / abs(all_det_log.mean()):.4f}')

# 分析归一化后的效果
print('\n===== 归一化后条件数对比 =====')
# det 归一化: X / det(X)^(1/64) → 不改变条件数
# trace 归一化: X / trace(X) → 不改变条件数
# 条件数是 scale-invariant 的
print('注意: 条件数对缩放不变, 归一化主要影响尺度而非条件数')

# 分析 log 域的分布 (Log-Euclidean 度量下)
print('\n===== Log 域统计 (与 Log-Euclidean Loss 直接相关) =====')
log_traces = all_det_log  # tr(log(X)) = log(det(X))
print(f'tr(log(X)) = log(det(X)): mean={log_traces.mean():.4f}, std={log_traces.std():.4f}')
print(f'  → 如果 std 很大, 说明样本在 Log 域的中心化程度差, 归一化有帮助')

# 分析每样本的 log 特征值分布
print('\n===== 单样本 log 特征值分析 (前 10 个样本) =====')
for i in range(min(10, N_ANALYZE)):
    hpd = sio.loadmat(os.path.join(DATA_DIR, file_list[i]))['Y1']
    X = torch.from_numpy(hpd).to(torch.complex128)
    ev = torch.linalg.eigvalsh(X).numpy()
    log_ev = np.log(ev)
    print(f'  样本{i}: log_eig range=[{log_ev.min():.3f}, {log_ev.max():.3f}], '
          f'mean={log_ev.mean():.3f}, std={log_ev.std():.3f}')
