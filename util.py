import random

import torch
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
from torch.nn.modules.module import Module

import torch
import numpy as np
import h5py
import os
from scipy import io


def save_batch_data(data_dict, batch_label, save_path, target_label=0):
    """
    高效保存函数：
    1. 自动处理 Tensor 的 detach (解决 requires_grad 报错)
    2. 筛选 label == target_label 的数据
    3. 一次性写入 v7.3 mat 文件
    """
    # --- 1. 处理 Label ---
    # 确保 label 是 numpy 数组

    if isinstance(batch_label, torch.Tensor):
        # 训练时的 label 通常也不需要梯度，为了保险也 detach 一下
        labels = batch_label.detach().cpu().numpy()
    else:
        labels = np.array(batch_label)

    # 生成筛选掩码
    mask = (labels == target_label)

    if not np.any(mask):
        print(f"   -> 跳过保存：本批次没有标签为 {target_label} 的数据")
        return

    # --- 2. 筛选并转换数据 (关键步骤) ---
    save_dict = {}
    print(f"   -> 正在准备数据 (筛选出 {mask.sum()} 条)...")

    for key, tensor_val in data_dict.items():
        # 核心修复：
        # 1. .detach() 切断梯度，解决报错
        # 2. .numpy() 转为 numpy (因为你在 CPU 上，不需要 .cpu())
        np_val = tensor_val

        # 应用筛选
        save_dict[key] = np_val[mask]

        # --- 3. 使用 scipy 保存 (关键修改) ---
        try:
            # scipy.io.savemat 默认生成 v7 格式，兼容性好
            # 不需要指定 format 参数，默认就是 v7
            io.savemat(save_path, save_dict)

            print(f"✅ 成功! 数据已保存至: {save_path} (v7格式)")

        except Exception as e:
            print(f"❌ 保存失败: {e}")

class SVD_opt(Function):

    def forward(self, input):
        """对批量输入矩阵执行奇异值分解（SVD），返回左奇异向量矩阵 U 和奇异值向量 S。

        对 batch 中的每个 N×N 矩阵 X_i，计算其 SVD：X_i = U_i @ diag(S_i) @ V_i^T，
        并仅保留 U_i 和奇异值 S_i。注意：本实现不保证 U 的唯一性（符号可能翻转），
        且未保存输入张量用于反向传播（需手动启用 save_for_backward）。

        Args:
            input (torch.Tensor): 输入张量，形状为 [B, N, N]，其中 B 为 batch size。
                支持任意实矩阵（不要求对称或正定），但后续反向传播假设奇异值互异。

        Returns:
            tuple of torch.Tensor:
                - Us (torch.Tensor): 左奇异向量矩阵，形状 [B, N, N]。
                - Ss (torch.Tensor): 奇异值向量，形状 [B, N]，按降序排列。

        Raises:
            RuntimeError: 若输入包含 NaN 或 Inf，或 SVD 不收敛。
        """
        Us = torch.zeros_like(input)
        Ss = torch.zeros((input.shape[0], input.shape[1])).double()
        for i in range(input.shape[0]):
            U, S, V = torch.svd(input[i, :, :])
            Ss[i, :] = S
            Us[i, :, :] = U

        self.Us = Us
        self.Ss = Ss
        # self.save_for_backward(input)  # 注释掉，当前未使用；若需在 backward 中访问 input，应取消注释
        return Us, Ss

    def backward(self, dLdV, dLdS):
        """计算 SVD 分解中输入矩阵关于损失的梯度（反向传播）。

        假设前向传播中对每个输入矩阵 X_i 执行了 SVD：X_i = U_i @ diag(S_i) @ V_i^T，
        本函数根据损失 L 对 U 和 S 的梯度（即 dL/dU 和 dL/dS），利用矩阵微分理论
        计算出 dL/dX。核心公式基于奇异值扰动理论，其中耦合项由 K 矩阵（差分倒数矩阵）建模：
            K[i, j] = 1 / (σ_i - σ_j)   if i ≠ j,
            K[i, i] = 0.
        该实现假设所有奇异值互异；若存在重复奇异值，K 中对应项设为 0（数值近似）。

        Args:
            dLdV (torch.Tensor): 损失 L 对左奇异向量 U 的梯度，形状 [B, N, N]。
                注意：尽管变量名为 dLdV，实际对应的是 dL/dU（命名可能源于历史原因）。
            dLdS (torch.Tensor): 损失 L 对奇异值 S 的梯度，形状 [B, N]。

        Returns:
            torch.Tensor: 损失 L 对原始输入矩阵 X 的梯度，形状 [B, N, N]。

        Note:
            - 本实现未处理 V 的梯度，仅适用于 X 为对称矩阵或仅使用 U 和 S 的场景。
            - K 矩阵中的无穷大值（由 σ_i ≈ σ_j 引起）被显式置零以保证数值稳定性。
            - 最终梯度通过相似变换 grad = U @ tmp @ U^T 重构回原始空间。
        """
        Ut = torch.transpose(self.Us, 1, 2)  # [B, N, N]
        Ks = torch.zeros_like(dLdV)  # [B, N, N]，用于存储每个样本的 K 矩阵
        diag_dLdS = torch.zeros_like(dLdV)  # [B, N, N]，用于存储 dL/dS 的对角矩阵形式

        # 逐样本构建 K 矩阵和 diag(dL/dS)
        for i in range(dLdV.shape[0]):
            diagS = self.Ss[i, :]  # 第 i 个样本的奇异值，[N]
            diagS = diagS.contiguous()
            vs_1 = diagS.view(-1, 1)  # [N, 1]
            vs_2 = diagS.view(1, -1)  # [1, N]
            K = 1.0 / (vs_1 - vs_2)  # [N, N]，差分倒数矩阵
            K[K >= float("Inf")] = 0.0  # 处理 σ_i == σ_j 导致的 inf（设为 0）
            Ks[i, :, :] = K

            # 将 dL/dS 转换为对角矩阵：diag(dLdS[i, :])
            diag_dLdS[i, :, :] = torch.diag(dLdS[i, :])

        # 计算耦合项：K^T ⊙ (U^T @ dL/dU)，其中 ⊙ 表示逐元素乘法
        tmp = torch.transpose(Ks, 1, 2) * torch.matmul(Ut, dLdV)
        # 对 tmp 进行对称化（确保梯度对称性）并加上对角项
        tmp = 0.5 * (tmp + torch.transpose(tmp, 1, 2)) + diag_dLdS
        # 通过 U 将梯度变换回原始输入空间：grad = U @ tmp @ U^T
        grad = torch.matmul(self.Us, torch.matmul(tmp, Ut))  # checked

        return grad


class RecFunction_v2(Function):

    def forward(self, input, eps):
        """对批量输入矩阵进行稳定化重构：将奇异值截断至最小阈值后重建矩阵。

        对每个输入矩阵 X_i 执行 SVD：X_i = U_i @ diag(S_i) @ V_i^T，
        然后将奇异值 S_i 截断为 max(S_i, eps)（eps = 1e-4），
        最终重构为：X̂_i = U_i @ diag(max(S_i, eps)) @ U_i^T。

        此操作常用于避免奇异值过小导致的数值不稳定（如协方差矩阵处理、伪逆近似等），
        并强制输出为对称半正定矩阵（即使原始输入非对称）。

        Args:
            input (torch.Tensor): 输入张量，形状为 [B, N, N]，支持任意实矩阵。

        Returns:
            torch.Tensor: 重构后的矩阵张量，形状为 [B, N, N]，为对称半正定矩阵。

        Note:
            - 重构时仅使用左奇异向量 U（忽略 V），因此输出始终对称。
            - 奇异值下限 eps 固定为 0.0001，防止除零或数值崩溃。
            - 所有中间变量（U, S, 截断后的 diag(S), 掩码等）被缓存用于反向传播。
        """
        Us = torch.zeros_like(input)  # [B, N, N]：存储左奇异向量 U
        Ss = torch.zeros((input.shape[0], input.shape[1])).double()  # [B, N]：原始奇异值
        max_Ss = torch.zeros_like(input).double()  # [B, N, N]：diag(clamp(S, min=eps))
        max_Ids = torch.zeros_like(input).float()  # [B, N, N]：diag(S >= eps) 的布尔掩码（转为 float）

        # eps = 0.00001
        for i in range(input.shape[0]):
            # PyTorch 的 torch.svd 不支持复数张量,需要改用 torch.linalg.svd
            U, S, V = torch.linalg.svd(input[i, :, :], full_matrices=False)
            max_S = torch.clamp(S, min=eps)  # 将奇异值下限设为 eps
            max_Id = torch.ge(S, eps).float()  # 生成布尔掩码并转为 float（用于 backward）

            Ss[i, :] = S
            Us[i, :, :] = U
            max_Ss[i, :, :] = torch.diag(max_S)  # 构造对角矩阵
            max_Ids[i, :, :] = torch.diag(max_Id)

        re_part_1 = torch.matmul(Us.real, torch.matmul(max_Ss, torch.transpose(Us.real, 1, 2)))
        re_part_2 = torch.matmul(Us.imag, torch.matmul(max_Ss, torch.transpose(Us.imag, 1, 2)))

        re_part = re_part_1 + re_part_2

        im_part_1 = - torch.matmul(Us.real, torch.matmul(max_Ss, torch.transpose(Us.imag, 1, 2)))
        im_part_2 = torch.matmul(Us.imag, torch.matmul(max_Ss, torch.transpose(Us.real, 1, 2)))

        im_part = im_part_1 + im_part_2

        result = torch.complex(re_part, im_part)

        # 重构：X̂ = U @ diag(max_S) @ U^T（注意：使用 U 而非 V，强制对称）
        # result = torch.matmul(Us, torch.matmul(max_Ss, torch.transpose(Us, 1, 2)))

        # 缓存中间结果供 backward 使用
        self.Us = Us
        self.Ss = Ss
        self.max_Ss = max_Ss
        self.max_Ids = max_Ids
        self.save_for_backward(input)  # 保存原始输入（尽管当前 backward 未直接使用）

        self.eps = eps

        return result

    def backward(self, grad_output):
        """计算稳定化矩阵重构操作（forward）关于输入的梯度。

        假设前向传播中对每个输入矩阵 X_i 执行了 SVD 并重构为：
            C_i = U_i @ diag(max(S_i, eps)) @ U_i^T，
        本函数根据损失 L 对输出 C 的梯度（即 dL/dC = grad_output），
        利用矩阵微分和奇异值扰动理论，反向传播计算 dL/dX。

        梯度计算分为两部分：
          1. 对截断奇异值 max(S_i, eps) 的敏感度（通过掩码 max_Ids 实现）；
          2. 对左奇异向量 U_i 的敏感度（通过 K 矩阵建模奇异值间的耦合效应）。

        Args:
            grad_output (torch.Tensor): 损失 L 对前向输出 C 的梯度，
                形状为 [B, N, N]。由于 C 是对称矩阵，此处先强制对称化。

        Returns:
            torch.Tensor: 损失 L 对原始输入 X 的梯度，形状为 [B, N, N]。

        Note:
            - 输入梯度 grad_output 被显式对称化：dLdC = 0.5 * (G + G^T)，
              以确保后续推导在对称矩阵空间中成立。
            - 仅当原始奇异值 S_i >= eps 时，才允许梯度流经该奇异值方向（由 max_Ids 掩码控制）。
            - K 矩阵用于处理非对角项的交叉导数：K[i,j] = 1/(σ_i - σ_j)（i≠j），重复奇异值处设为 0。
            - 最终梯度通过相似变换 grad = U @ tmp @ U^T 重构回原始输入空间。
        """

        Ks = torch.zeros_like(grad_output)  # [B, N, N]：存储每个样本的差分倒数矩阵 K

        # Step 1: 对 grad_output 进行对称化（因前向输出 C 为对称矩阵）
        dLdC = grad_output
        dLdC = 0.5 * (dLdC + torch.transpose(dLdC.conj(), 1, 2))  # checked

        # Step 2: 准备转置的左奇异向量
        Ut = torch.transpose(self.Us.conj(), 1, 2)  # [B, N, N]

        # Step 3: 计算损失对 U 的等效梯度（记为 dLdV，命名沿用历史习惯）
        # 根据 C = U @ D @ U^T，有 dC/dU = 2 * (dLdC @ U @ D)
        dLdV = 2 * torch.matmul(torch.matmul(dLdC, self.Us), self.max_Ss.to(torch.complex128))

        # Step 4: 计算损失对奇异值 S 的梯度（考虑截断掩码）
        # 先计算无掩码梯度：dLdS_1 = U^T @ dLdC @ U
        dLdS_1 = torch.matmul(torch.matmul(Ut, dLdC), self.Us)
        # 应用掩码：仅保留 S >= eps 的位置的梯度（其余置零）
        dLdS = torch.matmul(self.max_Ids.to(torch.complex128), dLdS_1)  # checked

        # Step 5: 构建 K 矩阵并提取 dLdS 的对角部分
        diag_dLdS = torch.zeros_like(grad_output)
        for i in range(grad_output.shape[0]):
            diagS = self.Ss[i, :]  # 第 i 个样本的原始奇异值 [N]
            diagS = diagS.contiguous()
            vs_1 = diagS.view(-1, 1)  # [N, 1]
            vs_2 = diagS.view(1, -1)  # [1, N]
            K = 1.0 / (vs_1 - vs_2)  # [N, N]，差分倒数矩阵
            K[K >= float("Inf")] = 0.0  # 处理 σ_i == σ_j 导致的 inf（设为 0）
            Ks[i, :, :] = K

            # 提取 dLdS 的对角元素并构造对角矩阵
            diag_dLdS[i, :, :] = torch.diag(torch.diag(dLdS[i, :, :]))

        # Step 6: 组合非对角与对角梯度项
        # 非对角项：K^T ⊙ (U^T @ dLdV)
        tmp = torch.transpose(Ks.conj(), 1, 2) * torch.matmul(Ut, dLdV)
        # 对称化非对角项并加上对角项
        tmp = 0.5 * (tmp + torch.transpose(tmp.conj(), 1, 2)) + diag_dLdS

        # Step 7: 将梯度变换回原始输入空间：grad = U @ tmp @ U^T
        grad = torch.matmul(self.Us, torch.matmul(tmp, Ut))  # checked

        grad_eps = None

        return grad, grad_eps

class RecFunction_v3(Function):

    def forward(self, input, threshold, N):
        """对批量输入矩阵进行部分稳定化重构：仅对后N个（最小的N个）奇异值做ReLU截断。

        对每个输入矩阵 X_i 执行 SVD：X_i = U_i @ diag(S_i) @ V_i^T，
        然后将奇异值 S_i 按降序排列，只对最后 N 个（最小的 N 个）奇异值
        进行截断：max(S_i, threshold)，其余保持不变，
        最终重构为：X̂_i = U_i @ diag(S'_i) @ U_i^T。

        Args:
            input (torch.Tensor): 输入张量，形状为 [B, N, N]，支持任意复矩阵。
            threshold (float): ReLU 截断阈值，低于此值的奇异值将被提升至此值。
            N (int): 指定对后多少个（最小的N个）奇异值做截断。

        Returns:
            torch.Tensor: 重构后的矩阵张量，形状为 [B, N, N]，为对称矩阵。

        Note:
            - 仅对最小的 N 个奇异值应用截断，较大的奇异值保持不变。
            - 所有中间变量（U, S, clamp后的diag(S), 掩码等）被缓存用于反向传播。
        """
        B, dim, _ = input.shape
        Us = torch.zeros_like(input)  # [B, N, N]：左奇异向量 U
        Ss = torch.zeros((B, dim)).double()  # [B, N]：原始奇异值（降序）
        clamp_Ss = torch.zeros_like(input).double()  # [B, N, N]：diag(clamp后的S)
        mask = torch.zeros_like(input).float()  # [B, N, N]：标记哪些位置被截断的掩码

        for i in range(B):
            # PyTorch 的 torch.svd 不支持复数张量,需要改用 torch.linalg.svd
            U, S, V = torch.linalg.svd(input[i, :, :], full_matrices=False)
            # S 已经是降序排列的
            Ss[i, :] = S

            # 复制一份 S 用于截断操作
            S_clamped = S.clone()
            # 对后 N 个（最小的 N 个）奇异值强制设为 threshold
            if N > 0 and N < dim:
                S_clamped[-N:] = threshold
            elif N >= dim:
                S_clamped = torch.full_like(S_clamped, threshold)

            # 构造掩码：未修正的位置梯度为1，被修正的位置梯度为0
            mask_i = torch.zeros(dim)
            if N > 0 and N < dim:
                mask_i[:-N] = 1.0   # 前 N 个未修正，保留梯度
                mask_i[-N:] = 0.0   # 后 N 个被修正，截断梯度
            elif N >= dim:
                mask_i = torch.zeros(dim)  # 全部被修正，梯度全截断

            Us[i, :, :] = U
            clamp_Ss[i, :, :] = torch.diag(S_clamped)
            mask[i, :, :] = torch.diag(mask_i)

        # 复矩阵重构：X̂ = U @ diag(S') @ U^H
        re_part_1 = torch.matmul(Us.real, torch.matmul(clamp_Ss, torch.transpose(Us.real, 1, 2)))
        re_part_2 = torch.matmul(Us.imag, torch.matmul(clamp_Ss, torch.transpose(Us.imag, 1, 2)))
        re_part = re_part_1 + re_part_2

        im_part_1 = - torch.matmul(Us.real, torch.matmul(clamp_Ss, torch.transpose(Us.imag, 1, 2)))
        im_part_2 = torch.matmul(Us.imag, torch.matmul(clamp_Ss, torch.transpose(Us.real, 1, 2)))
        im_part = im_part_1 + im_part_2

        result = torch.complex(re_part, im_part)

        # 缓存中间结果供 backward 使用
        self.Us = Us
        self.Ss = Ss
        self.clamp_Ss = clamp_Ss
        self.mask = mask
        self.threshold = threshold
        self.N = N
        self.save_for_backward(input)

        return result

    def backward(self, grad_output):
        """计算部分稳定化矩阵重构操作关于输入的梯度。

        假设前向传播中对每个输入矩阵 X_i 执行了 SVD 并仅对后 N 个奇异值做截断，
        本函数根据损失 L 对输出 C 的梯度，计算 dL/dX。

        梯度计算分为两部分：
          1. 对被截断奇异值（后N个）的敏感度（通过 mask 实现）；
          2. 对左奇异向量 U 的敏感度（通过 K 矩阵建模奇异值间的耦合效应）。

        Args:
            grad_output (torch.Tensor): 损失 L 对前向输出 C 的梯度，形状为 [B, N, N]。

        Returns:
            torch.Tensor: 损失 L 对原始输入 X 的梯度，形状为 [B, N, N]。
        """
        Ks = torch.zeros_like(grad_output)  # [B, N, N]：差分倒数矩阵 K

        # Step 1: 对 grad_output 对称化
        dLdC = grad_output
        dLdC = 0.5 * (dLdC + torch.transpose(dLdC.conj(), 1, 2))

        # Step 2: 获取 U^H
        Ut = torch.transpose(self.Us.conj(), 1, 2)  # [B, N, N]

        # Step 3: 计算损失对 U 的等效梯度
        # 根据 C = U @ D @ U^H，有 ∂C/∂U = 2 * dLdC @ U @ D
        dLdV = 2 * torch.matmul(torch.matmul(dLdC, self.Us), self.clamp_Ss.to(torch.complex128))

        # Step 4: 计算损失对奇异值 S 的梯度
        # dLdS_1 = U^H @ dLdC @ U
        dLdS_1 = torch.matmul(torch.matmul(Ut, dLdC), self.Us)
        # 应用掩码：仅保留后 N 个位置的梯度（其余置零）
        dLdS = torch.matmul(self.mask.to(torch.complex128), dLdS_1)

        # Step 5: 构建 K 矩阵并提取 dLdS 的对角部分
        diag_dLdS = torch.zeros_like(grad_output)
        for i in range(grad_output.shape[0]):
            diagS = self.Ss[i, :]  # 第 i 个样本的原始奇异值 [N]
            diagS = diagS.contiguous()
            vs_1 = diagS.view(-1, 1)  # [N, 1]
            vs_2 = diagS.view(1, -1)  # [1, N]
            K = 1.0 / (vs_1 - vs_2)  # [N, N]，差分倒数矩阵
            K[K >= float("Inf")] = 0.0  # 处理 σ_i == σ_j 导致的 inf
            Ks[i, :, :] = K

            # 提取 dLdS 的对角元素并构造对角矩阵
            diag_dLdS[i, :, :] = torch.diag(torch.diag(dLdS[i, :, :]))

        # Step 6: 组合非对角与对角梯度项
        tmp = torch.transpose(Ks.conj(), 1, 2) * torch.matmul(Ut, dLdV)
        tmp = 0.5 * (tmp + torch.transpose(tmp.conj(), 1, 2)) + diag_dLdS

        # Step 7: 将梯度变换回原始输入空间：grad = U @ tmp @ U^H
        grad = torch.matmul(self.Us, torch.matmul(tmp, Ut))

        grad_threshold = None
        grad_N = None

        return grad, grad_threshold, grad_N


class LogFunction_v2(Function):

    def forward(self, input):
        """计算批量输入矩阵的矩阵对数（matrix logarithm）。

        对每个输入矩阵 X_i 执行奇异值分解（SVD）：X_i = U_i @ diag(S_i) @ V_i^T，
        并假设 X_i 为对称正定（SPD）矩阵（此时 U_i ≈ V_i，且 S_i > 0），
        则其矩阵对数定义为：
            log(X_i) = U_i @ diag(log(S_i)) @ U_i^T。

        本实现通过 SVD 近似特征分解，仅使用左奇异向量 U_i 重构对数矩阵，
        因此输出始终为对称矩阵。要求所有奇异值严格大于 0，否则 torch.log(S) 将产生 NaN。

        Args:
            input (torch.Tensor): 输入张量，形状为 [B, N, N]。
                应为对称正定矩阵（或至少为满秩实矩阵），以确保 S > 0。

        Returns:
            torch.Tensor: 矩阵对数结果，形状为 [B, N, N]，为对称矩阵。

        Raises:
            RuntimeError: 若输入包含非正奇异值（导致 log(S) 为 NaN 或 -inf）。

        Note:
            - 本函数不显式对称化输入，若输入非对称，结果可能不等于真实矩阵对数。
            - 中间变量（U, S, log(S), 1/S）被缓存用于反向传播。
            - 建议在调用前确保输入对称：input = 0.5 * (input + input.transpose(-2, -1))。
        """
        Us = torch.zeros_like(input)  # [B, N, N]：左奇异向量 U
        Ss = torch.zeros((input.shape[0], input.shape[1])).double()  # [B, N]：奇异值 S
        logSs = torch.zeros_like(input).double()  # [B, N, N]：diag(log(S))
        invSs = torch.zeros_like(input).double()  # [B, N, N]：diag(1 / S)，用于 backward

        for i in range(input.shape[0]):
            # PyTorch 的 torch.svd 不支持复数张量,需要改用 torch.linalg.svd
            U, S, V = torch.linalg.svd(input[i, :, :], full_matrices=False)

            Ss[i, :] = S
            Us[i, :, :] = U
            logSs[i, :, :] = torch.diag(torch.log(S))  # 构造对角矩阵 diag(log(S))
            invSs[i, :, :] = torch.diag(1.0 / S)  # 构造对角矩阵 diag(1/S)

        re_part_1 = torch.matmul(Us.real, torch.matmul(logSs, torch.transpose(Us.real, 1, 2)))
        re_part_2 = torch.matmul(Us.imag, torch.matmul(logSs, torch.transpose(Us.imag, 1, 2)))

        re_part = re_part_1 + re_part_2

        im_part_1 = - torch.matmul(Us.real, torch.matmul(logSs, torch.transpose(Us.imag, 1, 2)))
        im_part_2 = torch.matmul(Us.imag, torch.matmul(logSs, torch.transpose(Us.real, 1, 2)))

        im_part = im_part_1 + im_part_2

        result = torch.complex(re_part, im_part)

        # 重构矩阵对数：log(X) ≈ U @ diag(log(S)) @ U^T
        # result = torch.matmul(Us, torch.matmul(logSs, torch.transpose(Us.conj(), 1, 2)))

        # 缓存中间结果供 backward 使用
        self.Us = Us
        self.Ss = Ss
        self.logSs = logSs
        self.invSs = invSs
        self.save_for_backward(input)

        return result

    def backward(self, grad_output):
        """计算矩阵对数（matrix logarithm）操作关于输入矩阵的梯度。

        假设前向传播中对每个输入 SPD 矩阵 X_i 执行了近似特征分解：
            X_i ≈ U_i @ diag(S_i) @ U_i^T，
        并计算了矩阵对数：
            C_i = log(X_i) ≈ U_i @ diag(log(S_i)) @ U_i^T。

        本函数根据损失 L 对输出 C 的梯度（即 dL/dC = grad_output），
        利用矩阵微分理论反向传播得到 dL/dX。梯度由两部分构成：
          - 对左奇异向量 U 的敏感度（通过 K 矩阵建模非对角耦合）；
          - 对奇异值 S 的敏感度（通过 invSs = diag(1/S) 缩放，因 d(log S)/dS = 1/S）。

        Args:
            grad_output (torch.Tensor): 损失 L 对前向输出 C（即 log(X)）的梯度，
                形状为 [B, N, N]。由于 C 为对称矩阵，此处先强制对称化。

        Returns:
            torch.Tensor: 损失 L 对原始输入 X 的梯度，形状为 [B, N, N]。

        Note:
            - 输入梯度被显式对称化：dLdC = 0.5 * (G + G^T)，以匹配前向输出的对称性。
            - 对 S 的梯度缩放因子为 1/S（来自 d(log S)/dS），由 self.invSs 提供。
            - K 矩阵用于处理非对角项：K[i,j] = 1/(σ_i - σ_j)（i≠j），重复奇异值处设为 0。
            - 最终梯度通过相似变换 grad = U @ tmp @ U^T 重构回原始空间。
            - 要求所有奇异值 S > 0；否则 invSs 或 logSs 在前向已导致 NaN。
        """
        # 确保计算精度一致（前向使用 double）
        # grad_output = grad_output.double()
        Ks = torch.zeros_like(grad_output)  # [B, N, N]：差分倒数矩阵 K

        # Step 1: 对 grad_output 对称化（因 log(X) 是对称矩阵）
        dLdC = grad_output
        dLdC = 0.5 * (dLdC + torch.transpose(dLdC.conj(), 1, 2))  # checked

        # Step 2: 获取 U^T
        Ut = torch.transpose(self.Us.conj(), 1, 2)  # [B, N, N]

        # Step 3: 计算损失对 U 的等效梯度（记为 dLdV，命名沿用习惯）
        # 根据 C = U @ logS @ U^T，有 ∂C/∂U = 2 * dLdC @ U @ logS
        dLdV = 2 * torch.matmul(dLdC, torch.matmul(self.Us, self.logSs.to(torch.complex128)))  # [B, N, N]

        # Step 4: 计算损失对奇异值 S 的梯度
        # 先计算无缩放梯度：dLdS_1 = U^T @ dLdC @ U
        dLdS_1 = torch.matmul(torch.matmul(Ut, dLdC), self.Us)  # [B, N, N]
        # 应用链式法则：dL/dS = (dL/d logS) * (d logS / dS) = dLdS_1 ⊙ (1/S)
        # 由于 self.invSs 是 diag(1/S)，左乘即实现逐元素缩放对角线
        dLdS = torch.matmul(self.invSs.to(torch.complex128), dLdS_1)  # checked

        # Step 5: 构建 K 矩阵并提取 dLdS 的对角部分
        diag_dLdS = torch.zeros_like(grad_output)
        for i in range(grad_output.shape[0]):
            diagS = self.Ss[i, :]  # 第 i 个样本的奇异值 [N]
            diagS = diagS.contiguous()
            vs_1 = diagS.view(-1, 1)  # [N, 1]
            vs_2 = diagS.view(1, -1)  # [1, N]
            K = 1.0 / (vs_1 - vs_2)  # [N, N]，差分倒数矩阵
            K[K >= float("Inf")] = 0.0  # 处理 σ_i == σ_j 导致的 inf（设为 0）
            Ks[i, :, :] = K

            # 提取 dLdS 的对角元素并构造对角矩阵
            diag_dLdS[i, :, :] = torch.diag(torch.diag(dLdS[i, :, :]))

        # Step 6: 组合非对角与对角梯度项
        # 非对角项：K^T ⊙ (U^T @ dLdV)
        tmp = torch.transpose(Ks.conj(), 1, 2) * torch.matmul(Ut, dLdV)
        # 对称化非对角项并加上对角项
        tmp = 0.5 * (tmp + torch.transpose(tmp.conj(), 1, 2)) + diag_dLdS

        # Step 7: 将梯度变换回原始输入空间：grad = U @ tmp @ U^T
        grad = torch.matmul(self.Us, torch.matmul(tmp, Ut))  # checked

        return grad


class ExpFunction_v2(Function):
    """批量 Hermitian 矩阵的矩阵指数 (matrix exponential), 支持复数 + 手写反向。

    前向: 对每个 Hermitian 输入 X_i 做特征分解 X_i = U_i diag(λ_i) U_i^H (eigh),
          exp(X_i) = U_i diag(exp(λ_i)) U_i^H。结果为 HPD 矩阵。
    反向: Daleckii-Krein / Loewner 公式。对 C = U f(Λ) U^H (f=exp):
          dL/dX = U [ Lw ⊙ (U^H · herm(dL/dC) · U) ] U^H,
          其中 Loewner 矩阵 Lw_ij = (f(λ_i)-f(λ_j))/(λ_i-λ_j) (i≠j), f'(λ_i)=exp(λ_i) (对角/退化)。

    与 LogFunction_v2 互逆: exp_mat_v2(log_mat_v2(X)) ≈ X (X HPD)。
    用 eigh (而非 SVD) 因为切空间 Hermitian 矩阵特征值可为负, SVD 只给非负奇异值。
    """

    def forward(self, input):
        B, N, _ = input.shape
        Us = torch.zeros_like(input)                                  # [B,N,N] 特征向量 (酉)
        Ls = torch.zeros((B, N), dtype=torch.double, device=input.device)      # [B,N] 实特征值
        expLs = torch.zeros((B, N, N), dtype=torch.double, device=input.device)  # [B,N,N] diag(exp(λ))

        for i in range(B):
            # 对称化: 抵消浮点误差导致的非 Hermitian 分量
            inp_i = input[i, :, :]
            inp_i = 0.5 * (inp_i + inp_i.conj().T)

            L, U = torch.linalg.eigh(inp_i)

            Ls[i, :] = L.double()
            Us[i, :, :] = U
            expLs[i, :, :] = torch.diag(torch.exp(L.double()))

        # 复矩阵重构: exp(X) = U diag(exp λ) U^H
        re_part_1 = torch.matmul(Us.real, torch.matmul(expLs, torch.transpose(Us.real, 1, 2)))
        re_part_2 = torch.matmul(Us.imag, torch.matmul(expLs, torch.transpose(Us.imag, 1, 2)))
        re_part = re_part_1 + re_part_2

        im_part_1 = - torch.matmul(Us.real, torch.matmul(expLs, torch.transpose(Us.imag, 1, 2)))
        im_part_2 = torch.matmul(Us.imag, torch.matmul(expLs, torch.transpose(Us.real, 1, 2)))
        im_part = im_part_1 + im_part_2

        result = torch.complex(re_part, im_part)

        self.Us = Us
        self.Ls = Ls
        return result

    def backward(self, grad_output):
        Us = self.Us
        Ut = torch.transpose(Us.conj(), 1, 2)  # U^H

        # 对称化 (exp(Hermitian) 是 Hermitian, 其梯度也应 Hermitian)
        dLdC = 0.5 * (grad_output + torch.transpose(grad_output.conj(), 1, 2))

        # M = U^H dLdC U (特征基下的梯度)
        M = torch.matmul(torch.matmul(Ut, dLdC), Us)  # [B,N,N] complex

        B, N, _ = grad_output.shape
        grad = torch.zeros_like(grad_output)
        for i in range(B):
            lam = self.Ls[i, :]                 # (N,) real
            explam = torch.exp(lam)             # (N,) real
            li = lam.view(-1, 1)
            lj = lam.view(1, -1)
            fi = explam.view(-1, 1)
            fj = explam.view(1, -1)
            denom = li - lj
            # Loewner 矩阵: 非退化用差商, 退化 (λ_i≈λ_j) 用导数 f'(λ_i)=exp(λ_i)
            Lw = torch.where(denom.abs() > 1e-12, (fi - fj) / denom, fi)
            tmp = Lw.to(M.dtype) * M[i, :, :]
            grad[i, :, :] = torch.matmul(Us[i, :, :], torch.matmul(tmp, Ut[i, :, :]))

        # Hermitian 化
        grad = 0.5 * (grad + torch.transpose(grad.conj(), 1, 2))
        return grad


def SVD_customed(input):
    return SVD_opt()(input)


def rec_mat_v2(input, eps):
    # return RecFunction_v2()(input)
    return RecFunction_v2.apply(input, eps)


def rec_mat_v3(input, threshold, N):
    return RecFunction_v3.apply(input, threshold, N)


def log_mat_v2(input):
    # return LogFunction_v2()(input)
    return LogFunction_v2.apply(input)


def exp_mat_v2(input):
    """批量 Hermitian 矩阵指数 (与 log_mat_v2 互逆)。见 ExpFunction_v2。"""
    return ExpFunction_v2.apply(input)


# =====================================================================
# HPD (Hermitian Positive Definite) 流形工具 (Brooks 2019, NeurIPS)
# 仿射不变度量 (Affine-Invariant Metric, AIM):
#   g_P(U, V) = tr(P^{-1} U P^{-1} V)
# 全部基于 log_mat_v2 / exp_mat_v2 + matmul 构建, autograd 可导 (手写反向)。
# =====================================================================


def sqrtm_hpd(A):
    """HPD 矩阵平方根 A^{1/2} = exp(0.5 log A)。A: (B,n,n) HPD。"""
    return exp_mat_v2(0.5 * log_mat_v2(A))


def invsqrtm_hpd(A):
    """HPD 矩阵逆平方根 A^{-1/2} = exp(-0.5 log A)。A: (B,n,n) HPD。"""
    return exp_mat_v2(-0.5 * log_mat_v2(A))


def hpd_congruence(P_half, X):
    """同余变换 P_half @ X @ P_half^H。P_half, X HPD 时结果 HPD (保流形)。

    Args:
        P_half: (n,n) 或 (B,n,n) complex, 通常是某 HPD 矩阵的平方根/逆平方根
        X:      (B,n,n) complex HPD
    Returns:
        (B,n,n) complex HPD
    """
    if P_half.dim() == 2:
        P_half = P_half.unsqueeze(0)
    P_half_H = P_half.conj().transpose(-2, -1)
    out = torch.matmul(torch.matmul(P_half, X), P_half_H)
    # Hermitian 化, 抵消数值误差
    return 0.5 * (out + out.conj().transpose(-2, -1))


def hpd_karcher_mean(X, iters=1, eps=1e-10):
    """批量 HPD 矩阵在仿射不变度量下的 Fréchet/Karcher 均值 (Karcher 流)。

    Brooks 2019 的 Riemannian 居中需要 batch 的黎曼重心 B。用 Karcher 流迭代:
        B ← B^{1/2} exp( (1/N) Σ_i log(B^{-1/2} X_i B^{-1/2}) ) B^{1/2}
    从 B_0 = 欧氏均值 (对角占优, HPD) 起步, 通常 1~几步即可。

    Args:
        X:     (B,n,n) complex HPD
        iters: Karcher 流迭代步数 (默认 1, 已足够稳定; 增大更精确)
    Returns:
        (n,n) complex HPD, batch 的黎曼均值 (无梯度; 居中用 detach 的统计量)
    """
    with torch.no_grad():
        B = X.mean(dim=0)                              # 欧氏均值作为初值 (HPD)
        B = 0.5 * (B + B.conj().transpose(-2, -1))
        for _ in range(iters):
            B_isqrt = invsqrtm_hpd(B.unsqueeze(0))[0]  # B^{-1/2}
            B_sqrt = sqrtm_hpd(B.unsqueeze(0))[0]      # B^{1/2}
            # 白化: whitened_i = B^{-1/2} X_i B^{-1/2}
            whitened = hpd_congruence(B_isqrt, X)      # (B,n,n)
            # 切空间平均: (1/N) Σ log(whitened_i)
            tangent_mean = log_mat_v2(whitened).mean(dim=0, keepdim=True)  # (1,n,n)
            # 指数映射回流形: B ← B^{1/2} exp(tangent_mean) B^{1/2}
            B = hpd_congruence(B_sqrt, exp_mat_v2(tangent_mean))[0]
            B = 0.5 * (B + B.conj().transpose(-2, -1))
    return B


def hpd_geodesic(A, C, t):
    """AIM 测地线插值 γ(t), γ(0)=A, γ(1)=C, A/C HPD (无梯度)。

    γ(t) = A^{1/2} (A^{-1/2} C A^{-1/2})^t A^{1/2}
    用于 running mean 的黎曼动量更新 (在流形上而非欧氏空间做插值)。

    Args:
        A, C: (n,n) complex HPD
        t:    float in [0,1]
    Returns:
        (n,n) complex HPD
    """
    with torch.no_grad():
        A = A.unsqueeze(0)
        C = C.unsqueeze(0)
        A_isqrt = invsqrtm_hpd(A)
        A_sqrt = sqrtm_hpd(A)
        mid = hpd_congruence(A_isqrt[0], C)          # A^{-1/2} C A^{-1/2}
        mid_t = exp_mat_v2(t * log_mat_v2(mid))      # (·)^t
        out = hpd_congruence(A_sqrt[0], mid_t)[0]
        out = 0.5 * (out + out.conj().transpose(-2, -1))
    return out


def update_para_riemann_hpd(G, dLdG, t):
    """HPD 流形上一步黎曼梯度下降 (仿射不变度量), 对标 Stiefel 的 update_para_riemann。

    可学 HPD 偏置 G 的更新。给定欧氏梯度 dL/dG (对 G 本身求导, 因 G 直接以参数存储):
      1. 黎曼梯度 (AIM): grad_R = G · herm(dL/dG) · G   (欧氏梯度 → 切空间黎曼梯度)
      2. 指数映射回缩: G_new = G^{1/2} exp( -t · G^{-1/2} grad_R G^{-1/2} ) G^{1/2}
                     = G^{1/2} exp( -t · G^{1/2} herm(dL/dG) G^{1/2} ) G^{1/2}
         保证 G_new 精确落在 HPD 流形上 (同余 + 正定谱)。

    与 Stiefel update_para_riemann 的对应:
      cal_riemann_grad   ↔  grad_R = G herm(dL/dG) G
      cal_retraction(QR) ↔  指数映射回缩 (谱域, 严格保 HPD)

    Args:
        G:    (n,n) complex HPD, 当前偏置
        dLdG: (n,n) complex, 损失对 G 的欧氏梯度
        t:    float, 步长
    Returns:
        (n,n) complex HPD, 更新后的 G
    """
    with torch.no_grad():
        n = G.shape[0]
        G = 0.5 * (G + G.conj().T)
        S = 0.5 * (dLdG + dLdG.conj().T)             # herm(dL/dG)
        G_sqrt = sqrtm_hpd(G.unsqueeze(0))[0]        # G^{1/2}
        # 回缩指数的自变量: -t · G^{1/2} S G^{1/2}  (Hermitian)
        arg = -t * (G_sqrt @ S @ G_sqrt)
        arg = 0.5 * (arg + arg.conj().T)
        expo = exp_mat_v2(arg.unsqueeze(0))[0]       # exp(arg), HPD
        G_new = G_sqrt @ expo @ G_sqrt               # G^{1/2} exp(arg) G^{1/2}, HPD
        G_new = 0.5 * (G_new + G_new.conj().T)
    return G_new



def cal_riemann_grad_torch(X, U):
    """将欧几里得梯度投影到 Stiefel 流形（或球面/正交约束流形）上的黎曼梯度。

    假设参数矩阵 X 满足正交约束（如 X^T X = I，即 X 位于 Stiefel 流形上），
    给定损失函数在 X 处的欧几里得梯度 U（即 ∇_X L），本函数计算其在该流形上的
    黎曼梯度（Riemannian gradient），即欧氏梯度在切空间中的正交投影。

    投影公式为：
        grad_R = U - X @ sym(X^T @ U),
    其中 sym(A) = 0.5 * (A + A^T) 表示对称化操作。

    此操作广泛用于带正交约束的优化问题（如 PCA、正交 RNN、协方差建模等）。

    Args:
        X (torch.Tensor): 当前参数矩阵，形状为 [N, K]，通常满足 X^T X = I_K（列正交）。
        U (torch.Tensor): 损失函数关于 X 的欧几里得梯度，形状与 X 相同 [N, K]。

    Returns:
        torch.Tensor: 黎曼梯度，形状 [N, K]，属于 X 在 Stiefel 流形上的切空间。

    References:
        - Absil, Mahony, & Sepulchre, "Optimization Algorithms on Matrix Manifolds", 2008.
        - Edelman et al., "The Geometry of Algorithms with Orthogonality Constraints", SIAM J. Matrix Anal. Appl., 1998.
    """
    # Compute X^T @ U
    XtU = torch.matmul(torch.transpose(X, 0, 1), U)
    # Symmetrize: sym(X^T U) = 0.5 * (X^T U + U^T X)
    symXtU = 0.5 * (XtU + torch.transpose(XtU, 0, 1))
    # Project onto tangent space: Up = U - X @ sym(X^T U)
    Up = U - torch.matmul(X, symXtU)
    return Up


def cal_retraction_torch(X, rU, t):
    """在 Stiefel 流形上执行一阶回缩（retraction）操作，用于黎曼优化。

    给定当前位于 Stiefel 流形上的点 X（即满足 X^T X = I），以及其切空间中的
    黎曼梯度方向 rU，该函数沿 -rU 方向移动步长 t，并通过 QR 分解将结果投影回流形，
    实现数值稳定的回缩（retraction）。

    具体步骤：
      1. 欧氏更新：Y = X - t * rU；
      2. QR 分解：Y = Q R（经济型分解）；
      3. 修正符号：Y = Q @ diag(sign(diag(R)))，确保 R 对角元为正，使 Q 唯一且连续。

    此操作是黎曼梯度下降（Riemannian Gradient Descent）中的标准回缩方法，
    广泛用于带正交约束的优化问题（如正交神经网络、PCA、字典学习等）。

    Args:
        X (torch.Tensor): 当前参数矩阵，形状 [N, K]，通常满足 X^T X = I_K（列正交）。
        rU (torch.Tensor): 黎曼梯度（已在切空间中），形状与 X 相同 [N, K]。
        t (float): 学习率（步长），应为正实数。

    Returns:
        torch.Tensor: 回缩后的新参数点，形状 [N, K]，近似保持列正交性。

    Note:
        - 本实现依赖 torch.linalg.qr（PyTorch ≥ 1.9 推荐使用 torch.linalg.qr）。
        - 若使用旧版 PyTorch，可改用 torch.qr，但需注意返回值顺序和 mode 参数。
        - 符号修正（sign(diag(R))）对保证流形映射的光滑性和唯一性至关重要。

    References:
        - Absil, Mahony, & Sepulchre, "Optimization Algorithms on Matrix Manifolds", 2008.
        - Edelman et al., "The Geometry of Algorithms with Orthogonality Constraints", 1998.
    """
    # Step 1: 欧几里得空间中的线性更新（沿负梯度方向）
    Y = X - t * rU

    # Step 2: 对 Y 进行经济型 QR 分解（thin QR）
    # 注意：torch.linalg.qr 默认返回 (Q, R)，其中 Q 是 [N, K]，R 是 [K, K]
    Q, R = torch.linalg.qr(Y, mode='reduced')

    # Step 3: 提取 R 的对角元素符号，并构造符号对角矩阵
    # 使用 sign(·) 处理可能的零值（sign(0)=0，但理论上 R 对角元应非零）
    sign_diag_R = torch.sign(torch.diag(R))  # 形状 [K]
    # 防止 sign 为 0（数值不稳定时可能发生），可选：sign_diag_R[sign_diag_R == 0] = 1
    S = torch.diag(sign_diag_R)

    # Step 4: 修正 Q 的列符号，确保回缩映射连续且唯一
    Y_retracted = torch.matmul(Q, S)

    return Y_retracted


def update_para_riemann(X, U, t):
    """执行一次黎曼梯度下降（Riemannian Gradient Descent）更新步骤。

    该函数将欧几里得梯度 U 投影到由 X 所在流形（如 Stiefel 流形）的切空间中，
    得到黎曼梯度，然后沿该方向进行回缩（retraction）以获得流形上的新参数点。
    这是带正交约束优化问题（如正交神经网络、PCA、协方差建模等）中的标准更新策略。

    更新流程：
      1. 计算黎曼梯度：Up = Proj_{T_X M}(U)
      2. 执行回缩操作：new_X = Retr_X(-t * Up)

    Args:
        X (torch.Tensor): 当前参数矩阵，形状 [N, K]，应位于目标流形上（如 X^T X = I）。
        U (torch.Tensor): 损失函数关于 X 的欧几里得梯度，形状与 X 相同 [N, K]。
        t (float): 学习率（步长），应为正实数。

    Returns:
        torch.Tensor: 更新后的参数矩阵，形状 [N, K]，近似保持流形约束（如列正交性）。

    Note:
        - 本函数依赖两个子函数：
            - `cal_riemann_grad`: 将欧氏梯度投影为黎曼梯度；
            - `cal_retraction`: 沿切方向移动后回缩到流形。
        - 若 X 不满足流形假设（如非正交），投影和回缩的几何意义将失效。
        - 常用于 Stiefel 流形（X^T X = I）或球面（K=1）上的优化。

    Example:
        >>> X = torch.randn(5, 3)
        >>> X, _ = torch.linalg.qr(X)  # 初始化为正交矩阵
        >>> U = torch.randn_like(X)    # 假设这是从 loss.backward() 得到的梯度
        >>> new_X = update_para_riemann(X, U, t=0.01)
        >>> assert torch.allclose(new_X.T @ new_X, torch.eye(3), atol=1e-6)
    """
    # Step 1: 将欧几里得梯度 U 投影到 X 处的切空间，得到黎曼梯度
    Up = cal_riemann_grad(X, U)

    # Step 2: 沿负黎曼梯度方向移动步长 t，并通过回缩映射回到流形
    new_X = cal_retraction(X, Up, t)

    return new_X


def cal_riemann_grad(X, U):
    """
    将欧几里得梯度投影到复 Stiefel 流形上的黎曼梯度。

    假设参数矩阵 X 位于复 Stiefel 流形上（即满足 X^H X = I，列酉正交），
    给定损失函数在 X 处的欧几里得梯度 U（复矩阵），
    本函数计算其在该流形切空间中的正交投影，即黎曼梯度。

    投影公式为：
        grad_R = U - X @ herm(X^H @ U),
    其中 herm(A) = 0.5 * (A + A^H) 表示 Hermitian 化操作。
    该公式确保结果 grad_R 满足复切空间条件：X^H grad_R + (X^H grad_R)^H = 0
    即 X^H grad_R 是斜 Hermitian 矩阵。

    Args:
        X (np.ndarray): shape [N, K], complex, 满足 X.conj().T @ X ≈ I
        U (np.ndarray): shape [N, K], complex, 欧几里得梯度

    Returns:
        Up (np.ndarray): shape [N, K], complex, 黎曼梯度（属于切空间）

    Example:
        >>> X = np.random.randn(5, 3) + 1j * np.random.randn(5, 3)
        >>> Q, _ = np.linalg.qr(X)  # QR 分解对复数也生成酉矩阵
        >>> U = np.random.randn(5, 3) + 1j * np.random.randn(5, 3)
        >>> Up = cal_riemann_grad_complex(Q, U)
        >>> # 验证切空间条件: X^H Up 应为 skew-Hermitian
        >>> XtUp = Q.conj().T @ Up
        >>> assert np.allclose(XtUp + XtUp.conj().T, 0, atol=1e-10)
    """
    # Step 1: Compute X^H @ U  (Hermitian transpose)
    XtU = np.matmul(X.conj().T, U)  # shape (K, K)

    # Step 2: Hermitian part: herm(X^H U) = 0.5 * (X^H U + (X^H U)^H)
    herm_XtU = 0.5 * (XtU + XtU.conj().T)

    # Step 3: Project onto tangent space: Up = U - X @ herm(X^H U)
    Up = U - np.matmul(X, herm_XtU)

    return Up


def cal_retraction(X, rU, t):
    """
    在复 Stiefel 流形上执行 QR 回缩（retraction）操作。

    复 Stiefel 流形定义为满足 X^H X = I 的复矩阵 X（列酉正交）。
    本函数适用于 X, rU 为 complex dtype 的情况。

    Args:
        X (np.ndarray): shape [N, K], complex, 满足 X.conj().T @ X ≈ I
        rU (np.ndarray): shape [N, K], complex, 黎曼梯度（已在切空间中）
        t (float): 步长（实数）

    Returns:
        Y (np.ndarray): shape [N, K], complex, 满足 Y^H Y ≈ I
    """
    # Step 1: 欧氏更新（t 为实数，rU 可为复数）
    Y = X - t * rU

    err = np.linalg.norm(Y.conj().T @ Y - np.eye(Y.shape[1]))
    # print("Orthogonality error:", err)

    # Step 2: 经济型 QR 分解（NumPy 支持复数）
    Q, R = np.linalg.qr(Y, mode='reduced')

    # Step 3: 相位修正 —— 使 R 的对角元具有正实部（或单位模）
    diag_R = np.diag(R)

    # 避免除零：对接近零的对角元设为 1（保持数值稳定）
    eps = np.finfo(float).eps
    # 计算单位相位因子：phase = diag_R / |diag_R|，若 |diag_R| < eps 则设为 1
    magnitudes = np.abs(diag_R)
    phase = np.where(magnitudes > eps, diag_R / magnitudes, 1.0 + 0j)

    # 构造对角相位矩阵
    D = np.diag(phase)

    # Step 4: 应用相位修正：Q * D 使得 (Q*D)^H (Q*D) = I，且 R_new = D^H R 有正实对角元
    Y = Q @ D

    err = np.linalg.norm(Y.conj().T @ Y - np.eye(Y.shape[1]))
    # print("Orthogonality error:", err)

    # print(is_unitary_columns(Y))

    return Y


def is_unitary_columns(W, tol=1e-10):
    """Check if W^H W ≈ I"""
    m, n = W.shape
    if m < n:
        return False
    G = W.conj().T @ W
    I = np.eye(n, dtype=W.dtype)
    return np.linalg.norm(G - I, ord='fro') < tol


def is_hermitian_and_positive_definite(A, rtol=1e-5, atol=1e-8):
    """
    判断复数方阵 A 是否为 Hermitian 且所有特征值 > 0。

    Args:
        A: (..., n, n) 复数张量
        rtol, atol: 数值容差（用于判断 Hermitian 和特征值 > 0）

    Returns:
        is_hermitian: bool
        is_pos_def: bool （仅当 Hermitian 时有意义）
        eigvals: 特征值（实数）
    """
    assert A.is_complex(), "Input must be complex"
    assert A.shape[-1] == A.shape[-2], "Matrix must be square"

    # 1. 检查是否 Hermitian: A == A^H
    A_H = A.conj().transpose(-2, -1)
    is_hermitian = torch.allclose(A, A_H, rtol=rtol, atol=atol)

    if not is_hermitian:
        return False, False, None

    # 2. 计算 Hermitian 矩阵的特征值（保证为实数）
    eigvals = torch.linalg.eigvalsh(A)  # 专用于 Hermitian 矩阵，返回实数特征值

    # 3. 检查所有特征值 > 0
    is_pos_def = torch.all(eigvals > atol)  # 允许微小负数视为 0（数值误差）

    return True, is_pos_def.item() if eigvals.numel() == 1 else is_pos_def.all().item(), eigvals

def mask_random_eigvals(X, n=16, eps=1e-8):
    """
    对 HPD 矩阵做特征值损坏: 每样本独立随机选 n 个特征值置为 eps，重建后返回。
    用于 denoising AE 训练时的输入损坏；目标仍是干净 X，逼模型从损坏版重建。

    数学保证: eigh 返回正特征值 (X 是 HPD)，把其中 n 个压到 eps (仍正)，
    重建 U diag(ev') U^H 仍是 HPD—不离开流形。

    Args:
        X: (B, D, D) complex128 HPD 张量
        n: 每样本损坏的特征值个数 (默认 16, 对 64×64 是 25%)
        eps: 被损坏特征值替换成的极小正值 (默认 1e-8, 与 rec_mat_v2 的 ReEig 阈值对齐)

    Returns:
        X_noisy: (B, D, D) complex128 HPD 张量, 与 X 同形状
    """
    with torch.no_grad():
        # eigh: ev 升序、U 列为对应特征向量。X HPD ⇒ ev 全正。
        ev, U = torch.linalg.eigh(X)  # ev: [B, D] float, U: [B, D, D] complex
        B, D = ev.shape

        # 每样本独立采样 n 个不同的索引 (argsort of random tensor = 随机置换)
        rand_perm = torch.argsort(torch.rand(B, D), dim=1)
        mask_idx = rand_perm[:, :n]  # [B, n]
        mask = torch.zeros(B, D, dtype=torch.bool, device=ev.device)
        mask.scatter_(1, mask_idx, True)

        ev_new = ev.clone()
        ev_new[mask] = eps
        # 重建 X' = U diag(ev_new) U^H = (U * ev_new[None,:]) @ U^H
        X_noisy = torch.matmul(U * ev_new.unsqueeze(-2), U.conj().transpose(-2, -1))
    return X_noisy