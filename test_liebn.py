"""LieBN 运行时验证脚本"""
import torch
import util
import model

print("=" * 60)
print("Test 1: exp_mat_v2 forward/backward")
print("=" * 60)

# 构造 HPD 矩阵
torch.manual_seed(42)
X = torch.randn(4, 8, 8, dtype=torch.complex128)
X = torch.matmul(X, X.conj().transpose(-2, -1)) + 0.1 * torch.eye(8, dtype=torch.complex128)

# logm -> expm 重建
log_X = util.log_mat_v2(X)
exp_log_X = util.exp_mat_v2(log_X)
recon_err = (exp_log_X - X).abs().max().item()
print(f"  logm->expm reconstruction error: {recon_err:.2e}")
assert recon_err < 1e-6, f"Reconstruction error too large: {recon_err}"
print("  PASSED")

print()
print("=" * 60)
print("Test 2: exp_mat_v2 backward (gradient check)")
print("=" * 60)

X2 = X.clone().detach()
X2.requires_grad = True
out = util.exp_mat_v2(X2)
loss = (out.real ** 2 + out.imag ** 2).mean()
loss.backward()
assert X2.grad is not None, "Gradient is None!"
print(f"  Gradient norm: {X2.grad.abs().mean().item():.2e}")
print("  PASSED")

print()
print("=" * 60)
print("Test 3: HPDLieBatchNorm forward (train mode)")
print("=" * 60)

bn = model.HPDLieBatchNorm(8)
bn.train()
out_bn = bn(X)
print(f"  Output shape: {out_bn.shape}, dtype: {out_bn.dtype}")
assert out_bn.shape == X.shape, "Shape mismatch!"
print(f"  Running mean norm after 1 batch: {bn.running_mean.abs().mean().item():.2e}")
print("  PASSED")

print()
print("=" * 60)
print("Test 4: HPDLieBatchNorm forward (eval mode)")
print("=" * 60)

bn.eval()
out_eval = bn(X)
print(f"  Output shape: {out_eval.shape}, dtype: {out_eval.dtype}")
assert out_eval.shape == X.shape, "Shape mismatch!"
print("  PASSED")

print()
print("=" * 60)
print("Test 5: HPDLieBatchNorm backward")
print("=" * 60)

bn2 = model.HPDLieBatchNorm(8)
bn2.train()
X3 = X.clone().detach()
X3.requires_grad = True
out3 = bn2(X3)
loss3 = (out3.real ** 2 + out3.imag ** 2).mean()
loss3.backward()
assert X3.grad is not None, "Input gradient is None!"
assert bn2.G_half.grad is not None, "G_half gradient is None!"
print(f"  Input grad norm: {X3.grad.abs().mean().item():.2e}")
print(f"  G_half grad norm: {bn2.G_half.grad.abs().mean().item():.2e}")
print("  PASSED")

print()
print("=" * 60)
print("Test 6: Full HPDNetwork with LieBN - forward")
print("=" * 60)

net = model.HPDNetwork(in_dim=64, use_bn=True)
net.train()
X_full = torch.randn(2, 64, 64, dtype=torch.complex128)
X_full = torch.matmul(X_full, X_full.conj().transpose(-2, -1)) + 0.01 * torch.eye(64, dtype=torch.complex128)
Y, layer_outs = net(X_full)
print(f"  Output shape: {Y.shape}")
print(f"  Encoder layers with BN: {len(net.bn_layers)}")
assert Y.shape == (2, 64, 64), "Output shape mismatch!"
print("  PASSED")

print()
print("=" * 60)
print("Test 7: Full HPDNetwork with LieBN - backward + update_para")
print("=" * 60)

net.zero_grad()
log_Y = util.log_mat_v2(Y)
log_X = util.log_mat_v2(X_full)
diff = log_Y - log_X
loss_full = (diff.real ** 2 + diff.imag ** 2).mean()
loss_full.backward()
print(f"  Loss: {loss_full.item():.6f}")

# Check that G_half has gradients
bn_grad_count = sum(1 for bn in net.bn_layers if bn is not None and bn.G_half.grad is not None)
print(f"  BN layers with G_half gradient: {bn_grad_count}/{len(net.bn_layers)}")

# update_para
net.update_para(0.01)
print("  update_para executed successfully")
print("  PASSED")

print()
print("=" * 60)
print("Test 8: HPDNetwork without LieBN (use_bn=False)")
print("=" * 60)

net_no_bn = model.HPDNetwork(in_dim=64, use_bn=False)
net_no_bn.train()
Y_no_bn, _ = net_no_bn(X_full)
loss_no_bn = (util.log_mat_v2(Y_no_bn) - util.log_mat_v2(X_full))
loss_no_bn = (loss_no_bn.real ** 2 + loss_no_bn.imag ** 2).mean()
loss_no_bn.backward()
net_no_bn.update_para(0.01)
print(f"  Loss (no BN): {loss_no_bn.item():.6f}")
print("  PASSED")

print()
print("=" * 60)
print("Test 9: Multi-epoch training simulation (3 epochs)")
print("=" * 60)

net2 = model.HPDNetwork(in_dim=64, use_bn=True)
net2.train()
losses = []
for ep in range(3):
    X_ep = torch.randn(4, 64, 64, dtype=torch.complex128)
    X_ep = torch.matmul(X_ep, X_ep.conj().transpose(-2, -1)) + 0.01 * torch.eye(64, dtype=torch.complex128)

    net2.zero_grad()
    Y_ep, _ = net2(X_ep)
    log_Y_ep = util.log_mat_v2(Y_ep)
    log_X_ep = util.log_mat_v2(X_ep)
    diff_ep = log_Y_ep - log_X_ep
    loss_ep = (diff_ep.real ** 2 + diff_ep.imag ** 2).mean()
    loss_ep.backward()
    net2.update_para(10.0)
    losses.append(loss_ep.item())
    print(f"  Epoch {ep+1}/3: loss = {loss_ep.item():.6f}")

print(f"  Losses: {[f'{l:.4f}' for l in losses]}")
print("  PASSED")

print()
print("=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
