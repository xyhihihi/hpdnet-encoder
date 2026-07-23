import math
import torch
import torch.nn.functional as F
import util as util


class HPDRiemannianBatchNorm(torch.nn.Module):
    """HPD 矩阵的黎曼 Batch Normalization (Brooks et al., NeurIPS 2019)。

    在仿射不变度量 (Affine-Invariant Metric) 下, 对一批 HPD 矩阵做:
      1. 黎曼居中 (ReCentering): 计算 batch 的 Fréchet/Karcher 均值 B,
         白化每个样本 X̄_i = B^{-1/2} X_i B^{-1/2} (把 batch 均值搬到单位阵 I)。
      2. 黎曼偏置 (ReBiasing):   搬到可学 HPD 参数 G 上 Y_i = G^{1/2} X̄_i G^{1/2}。
         G 初始化为单位阵 I ⇒ 初始 BN ≈ 恒等。
      3. running mean: 训练时用 AIM 测地线动量更新 B_run; 推理时用 B_run 居中。

    所有操作是 HPD 的同余变换 (P X P^H), 结构性保证输出仍是 HPD — 不离开流形。
    G 不走欧氏 autograd 更新, 而是在 update_para 里用 util.update_para_riemann_hpd
    做 HPD 流形上的黎曼 SGD (对标 Stiefel 权重的 util.update_para_riemann)。
    """

    def __init__(self, dim, momentum=0.1, karcher_iters=1, eps=1e-8):
        super(HPDRiemannianBatchNorm, self).__init__()
        self.dim = dim
        self.momentum = momentum
        self.karcher_iters = karcher_iters
        self.eps = eps

        # 可学 HPD 偏置 G, 初始 = I (BN 初始为恒等)。作为参数直接存 HPD 矩阵,
        # 由 update_para 中的 update_para_riemann_hpd 保证更新后仍是 HPD。
        G_init = torch.eye(dim, dtype=torch.complex128)
        self.G = torch.nn.Parameter(G_init, requires_grad=True)

        # running mean (HPD 流形上的黎曼均值), 推理用。初始 = I。
        self.register_buffer('running_mean', torch.eye(dim, dtype=torch.complex128))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, X):
        """X: (B, dim, dim) complex128 HPD → (B, dim, dim) complex128 HPD。"""
        if self.training:
            B_mean = util.hpd_karcher_mean(X, iters=self.karcher_iters)
            center = B_mean
            with torch.no_grad():
                self.num_batches_tracked += 1
                if self.num_batches_tracked == 1:
                    self.running_mean = B_mean.clone()
                else:
                    self.running_mean = util.hpd_geodesic(
                        self.running_mean, B_mean, self.momentum)
        else:
            center = self.running_mean

        center = center.detach()
        B_isqrt = util.invsqrtm_hpd(center.unsqueeze(0))[0]
        X_centered = util.hpd_congruence(B_isqrt, X)

        G_sqrt = util.sqrtm_hpd(self.G.unsqueeze(0))[0]
        Y = util.hpd_congruence(G_sqrt, X_centered)
        return Y


class HPDNetwork(torch.nn.Module):
    def __init__(self, in_dim=64, hidden_dims=None, rec_params=None,
                 activations=None, rec_N_params=None, skip_resolutions=None,
                 use_bn=True, bn_momentum=0.1, bn_karcher_iters=1, bn_lr=1.0):
        super(HPDNetwork, self).__init__()

        if hidden_dims is None:
            hidden_dims = [56, 49, 42, 36, 30, 25, 20, 16, 12, 9, 6, 4]
        num_layers = 2 * len(hidden_dims)  # 12 encoder + 12 decoder
        if rec_params is None:
            rec_params = [1e-06] * num_layers
        if activations is None:
            # encoder 12 层用 ReEig; decoder 12 层不接激活
            # decoder 用酉 U^H block_diag(X, I) U 已结构性保 HPD, 无需 ReEig 兜底
            activations = [util.rec_mat_v2] * len(hidden_dims) + [None] * len(hidden_dims)
        if rec_N_params is None:
            rec_N_params = [0] * num_layers
        if skip_resolutions is None:
            # 5 个 skip 分辨率 (关于 bottleneck 对称)
            skip_resolutions = [49, 36, 25, 16, 9]

        # 复 Stiefel 初始化: 权重形状 (n_high, n_low), 满足 W^H W = I
        # - encoder: 用 W^H X W 下行 (n_high → n_low)
        # - decoder: 用 V X V^H 上行 (n_low → n_high), V 仍是 (n_high, n_low) 列正交
        self.weights = []
        dims = [in_dim] + list(hidden_dims)  # 长度 13: dims[0]=64 ... dims[12]=4

        # 12 个 encoder 权重: W_i 形状 (dims[i], dims[i+1])
        for i in range(len(hidden_dims)):
            n_in, n_out = dims[i], dims[i + 1]
            A = torch.randn(n_in, n_out, dtype=torch.complex128)
            Q, _ = torch.linalg.qr(A)  # 约化 QR, Q 列正交
            self.weights.append(torch.nn.Parameter(Q, requires_grad=True))

        # 12 个 decoder 权重: U_k 是 n_high × n_high 方酉 (Stiefel 流形的方阵特例)
        # dec 层 k 输入 dims[12-k] (n_low), 输出 dims[11-k] (n_high)
        # 前向: Y = U^H block_diag(X, I_{n_high-n_low}) U —— 切空间 padding + 方酉共轭
        # 这一步结构性保 HPD (块对角 padding 用 I, 方酉共轭保 HPD), 不依赖 ReEig
        for k in range(len(hidden_dims)):
            n_high = dims[len(hidden_dims) - 1 - k]
            A = torch.randn(n_high, n_high, dtype=torch.complex128)
            Q, _ = torch.linalg.qr(A)  # 方阵 QR 给方酉 (Q^H Q = I)
            self.weights.append(torch.nn.Parameter(Q, requires_grad=True))

        # encoder 层索引 → 该层输出分辨率
        enc_out_res_to_idx = {dims[i + 1]: i for i in range(len(hidden_dims))}

        # 黎曼 BatchNorm (Brooks 2019): 每个 encoder 层的 ReEig 之后接一个 HPD-BN
        # G 的更新用独立绝对学习率 bn_lr (与主 LR 解耦; AIM 指数回缩需要的最优步长
        # 与 Stiefel 权重的 QR 回缩不同, 故单独调)。update_para 也可按调用覆盖 bn_lr。
        self.use_bn = use_bn
        self.bn_lr = bn_lr
        if use_bn:
            self.bn_layers = torch.nn.ModuleList([
                HPDRiemannianBatchNorm(dims[i + 1], momentum=bn_momentum,
                                       karcher_iters=bn_karcher_iters)
                for i in range(len(hidden_dims))
            ])
        else:
            self.bn_layers = None

        # Build encoder layers: down, W^H X W + (v2 或 v3)
        # v2 调用签名: rec_mat_v2(X, eps) — 用 layer['param']
        # v3 调用签名: rec_mat_v3(X, threshold, N) — 用 layer['param'] 和 layer['N']
        # 例: 奇偶层交替 v2/v3 → activations=[v2,v3,v2,v3,...], rec_N_params=[0,36,0,25,...]
        self.enc_layers = []
        for idx in range(len(hidden_dims)):
            self.enc_layers.append({
                'w': self.weights[idx],
                'activation': activations[idx],
                'param': rec_params[idx],
                'N': rec_N_params[idx],
            })

        # Build decoder layers: up, U^H block_diag(X, c·I) U + 多项式非线性 + skip; 不接 ReEig
        # c 由 pad_theta 经 softplus 保正; 初始 c=1
        # 多项式 a0I + a1Z + a2Z² 由 poly_theta (3,) 参数化, 正系数保 HPD; 初始 ≈ Z (恒等)
        self.pad_thetas = []
        self.poly_thetas = []
        init_theta = math.log(math.e - 1)  # softplus(log(e-1)) = 1.0
        init_poly = torch.tensor([-10.0, -10.0, -10.0], dtype=torch.double)  # a0≈0, a1≈1, a2≈0
        self.dec_layers = []
        for k in range(len(hidden_dims)):
            dec_idx = len(hidden_dims) + k
            dec_out_res = dims[len(hidden_dims) - 1 - k]
            skip_enc_idx = None
            if dec_out_res in skip_resolutions:
                skip_enc_idx = enc_out_res_to_idx[dec_out_res]
            pad_theta = torch.nn.Parameter(
                torch.full((1,), init_theta, dtype=torch.double, requires_grad=True)
            )
            poly_theta = torch.nn.Parameter(init_poly.clone(), requires_grad=True)
            self.pad_thetas.append(pad_theta)
            self.poly_thetas.append(poly_theta)
            self.dec_layers.append({
                'w': self.weights[dec_idx],
                'activation': activations[dec_idx],
                'param': rec_params[dec_idx],
                'N': rec_N_params[dec_idx],
                'skip_enc_idx': skip_enc_idx,
                'pad_theta': pad_theta,
                'poly_theta': poly_theta,
            })

    def _build_layer(self, X, layer):
        """下行 BiMap: W^H X W (n_high,n_high) → (n_low,n_low)。"""
        w_pc = layer['w'].contiguous()
        w = w_pc.view([1, w_pc.shape[0], w_pc.shape[1]])
        w_H = w.conj().transpose(-2, -1)

        re_part_1 = torch.matmul(torch.matmul(w_H.real, X.real), w.real)
        re_part_2 = -torch.matmul(torch.matmul(w_H.imag, X.imag), w.real)
        re_part_3 = -torch.matmul(torch.matmul(w_H.real, X.imag), w.imag)
        re_part_4 = -torch.matmul(torch.matmul(w_H.imag, X.real), w.imag)
        re_part = re_part_1 + re_part_2 + re_part_3 + re_part_4

        im_part_1 = torch.matmul(torch.matmul(w_H.real, X.real), w.imag)
        im_part_2 = -torch.matmul(torch.matmul(w_H.imag, X.imag), w.imag)
        im_part_3 = torch.matmul(torch.matmul(w_H.real, X.imag), w.real)
        im_part_4 = torch.matmul(torch.matmul(w_H.imag, X.real), w.real)
        im_part = im_part_1 + im_part_2 + im_part_3 + im_part_4

        return torch.complex(re_part, im_part)

    def _build_layer_up(self, X, layer):
        """上行 BiMap (内蕴版 可学 padding): Y = U^H block_diag(X, c·I) U。
        输入 X: (B, n_low, n_low) HPD; 输出: (B, n_high, n_high) HPD。
        U 是 (n_high, n_high) 方酉 [layer['w']]; c = softplus(layer['pad_theta']) 保正。
        block_diag(X, c·I) 是 HPD⊕HPD = HPD, 方酉共轭保 HPD—结构性保证不靠 ReEig。
        实现上先 pad X 到 (B, n_high, n_high) (左上 X 右下 c·I), 再复用 _build_layer 算 U^H X_pad U。
        """
        U = layer['w']
        pad_theta = layer['pad_theta']
        c = F.softplus(pad_theta)  # # > 0, 可学
        B, n_low, _ = X.shape
        n_high = U.shape[0]

        # block_diag(X, c·I): 先整个填 c·I, 再把左上 n_low×n_low 块覆盖成 X
        eye = torch.eye(n_high, dtype=X.dtype, device=X.device)
        X_padded = (c.to(X.dtype) * eye).unsqueeze(0).expand(B, n_high, n_high).clone()
        X_padded[:, :n_low, :n_low] = X
        # 复用 _build_layer: 它算 W^H X W, W=U (方阵) 时就是 U^H X_pad U
        return self._build_layer(X_padded, layer)

    def _apply_poly(self, Z, layer):
        """HPD-preserving 非线性: a0I + a1Z + a2Z², 正系数保 HPD, 无 eig、无重复特征值问题。
        a0=softplus(θ₀) (偏置), a1=1.0+softplus(θ₁) (线性, 至少 1), a2=softplus(θ₂) (非线性)。
        Z HPD → Z² HPD (特征值 λ→λ²>0); 正系数加权和保 HPD。
        """
        theta = layer['poly_theta']
        a0 = F.softplus(theta[0])
        a1 = 1.0 + F.softplus(theta[1])
        a2 = F.softplus(theta[2])

        B, n, _ = Z.shape
        I = torch.eye(n, dtype=Z.dtype, device=Z.device).unsqueeze(0).expand(B, n, n)
        Z2 = torch.matmul(Z, Z)  # Z²
        return a0 * I + a1 * Z + a2 * Z2

    def forward(self, input):
        X = input
        layer_outputs = []

        # Encoder: 下行 64 → 4, 每层 W^H X W + (v2 或 v3) + 黎曼 BN
        for idx, layer in enumerate(self.enc_layers):
            X = self._build_layer(X, layer)
            if layer['activation'] == util.rec_mat_v3:
                X = layer['activation'](X, layer['param'], layer['N'])
            else:
                X = layer['activation'](X, layer['param'])
            # 黎曼 BatchNorm (Brooks 2019): ReEig 后 X 已是 HPD, logm 安全
            if self.use_bn:
                X = self.bn_layers[idx](X)
            layer_outputs.append(X)

        # Decoder: 上行 4 → 64, 每层 线性 U^H block_diag(X, c·I) U → 多项式非线性 → skip 加法
        for layer in self.dec_layers:
            X = self._build_layer_up(X, layer)
            X = self._apply_poly(X, layer)  # HPD-preserving 非线性: a0I + a1Z + a2Z²
            if layer['skip_enc_idx'] is not None:
                # 加法 skip: 非线性输出加上同分辨率的 encoder 输出 (HPD + HPD = HPD)
                X = X + layer_outputs[layer['skip_enc_idx']]
            act = layer['activation']
            if act is not None:
                if act == util.rec_mat_v3:
                    X = act(X, layer['param'], layer['N'])
                else:
                    X = act(X, layer['param'])
            layer_outputs.append(X)

        return X, layer_outputs

    def update_para(self, lr, bn_lr=None):
        # 获取所有权重的梯度 (顺序: 1 → n)
        egrads = [w_p.grad.data.numpy() for w_p in self.weights]
        ws = [w_p.data.numpy() for w_p in self.weights]

        # 更新 Stiefel 权重 (顺序: n → 1; enc 矩形 Stiefel + dec 方酉都在同一复 Stiefel 流形上)
        for i in range(len(self.weights) - 1, -1, -1):
            new_w = util.update_para_riemann(ws[i], egrads[i], lr)
            self.weights[i].data.copy_(torch.from_numpy(new_w))

        # 更新 decoder 的可学 padding 参数 theta (欧氏 SGD, 不在 Stiefel 流形上)
        for theta in self.pad_thetas:
            theta.data -= lr * theta.grad.data
            theta.grad.data.zero_()

        # 更新 decoder 多项式系数 poly_theta (欧氏 SGD)
        for theta in self.poly_thetas:
            theta.data -= lr * theta.grad.data
            theta.grad.data.zero_()

        # 更新黎曼 BN 的可学 HPD 偏置 G (HPD 流形黎曼 SGD, Brooks 2019)
        # 对标 Stiefel 权重的 update_para_riemann: 欧氏梯度 → AIM 黎曼梯度 → 指数映射回缩,
        # 严格保证 G 留在 HPD 流形上。G 用独立绝对学习率 (与主 lr 解耦), 可按调用覆盖。
        if self.use_bn:
            g_lr = self.bn_lr if bn_lr is None else bn_lr
            for bn in self.bn_layers:
                if bn.G.grad is not None:
                    G_new = util.update_para_riemann_hpd(
                        bn.G.data, bn.G.grad.data, g_lr)
                    bn.G.data.copy_(G_new)
                    bn.G.grad.data.zero_()

        # 清除 Stiefel 权重梯度 (顺序: 1 → n)
        for i in range(0, len(self.weights)):
            self.weights[i].grad.data.zero_()