# 快速崩溃测试: 不同样本量下 BN 何时崩溃
import os, math, numpy as np, torch, scipy.io as sio
import model as model_mod, util

DATA_DIR = 'D:/data/customed/customed_64'
FILE_LIST_PATH = 'D:/data/customed/train.txt'
BS = 512
EPOCHS = 200

with open(FILE_LIST_PATH) as f:
    fl = [l.strip().split(' ')[0].replace('\\', '/') for l in f]

def load_data(n):
    data = np.zeros((n, 64, 64), dtype=np.complex128)
    for i in range(n):
        data[i] = sio.loadmat(os.path.join(DATA_DIR, fl[i]))['Y1']
    return torch.from_numpy(data).to(torch.complex128)

def norm_det(X):
    n = X.shape[-1]
    with torch.no_grad():
        ev = torch.linalg.eigvalsh(X)
        ld = torch.log(ev).sum(-1, keepdim=True).unsqueeze(-1)
    return X / torch.exp(ld / n)

def run_test(X, label, epochs):
    n = X.shape[0]
    nb = math.ceil(n / BS)
    print(f'\n===== {label} (N={n}, {nb} batch/ep) =====', flush=True)
    torch.manual_seed(42)
    np.random.seed(42)
    net = model_mod.HPDNetwork(use_bn=True, bn_lr=200.0, rec_params=[1e-4]*24)
    net.train()
    for ep in range(epochs):
        try:
            perm = np.random.permutation(n)
            for bs in range(0, n, BS):
                idx = perm[bs:bs+BS]
                Xb = X[idx]
                Y, _ = net(Xb)
                d = util.log_mat_v2(Y) - util.log_mat_v2(Xb)
                loss = (d.real**2 + d.imag**2).mean()
                net.zero_grad()
                loss.backward()
                net.update_para(40.0, bn_lr=200.0)
            if (ep+1) % 20 == 0:
                print(f'  epoch {ep+1}: loss={loss.item():.6f}', flush=True)
        except Exception as e:
            print(f'  CRASH at epoch {ep+1}: {type(e).__name__}: {str(e)[:120]}', flush=True)
            return ep+1
    print(f'  完成 {epochs} epochs 未崩溃', flush=True)
    return None

print('加载数据...', flush=True)
X512 = load_data(512)
X1024 = load_data(1024)
X512_det = norm_det(X512)
X1024_det = norm_det(X1024)
print('数据就绪。\n', flush=True)

r1 = run_test(X512, 'N=512 无归一化', EPOCHS)
r2 = run_test(X512_det, 'N=512 Det归一化', EPOCHS)
r3 = run_test(X1024, 'N=1024 无归一化', EPOCHS)
r4 = run_test(X1024_det, 'N=1024 Det归一化', EPOCHS)

print('\n===== 汇总 =====')
print(f'N=512  无归一化:   {"ep"+str(r1)+" 崩溃" if r1 else "200ep 未崩溃"}')
print(f'N=512  Det归一化:  {"ep"+str(r2)+" 崩溃" if r2 else "200ep 未崩溃"}')
print(f'N=1024 无归一化:   {"ep"+str(r3)+" 崩溃" if r3 else "200ep 未崩溃"}')
print(f'N=1024 Det归一化:  {"ep"+str(r4)+" 崩溃" if r4 else "200ep 未崩溃"}')
