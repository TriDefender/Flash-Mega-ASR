"""
AutoKernel -- Mega Kernel: Fused RMSNorm + Matmul + Residual

This kernel fuses three operations into one:
1. RMSNorm: x_norm = x / sqrt(mean(x^2) + eps) * weight
2. Matmul: y = x_norm @ W
3. Residual: output = y + residual

This eliminates:
- RMSNorm kernel launch (~2.7% of GPU time)
- Residual add kernel launch (~0.8% of GPU time)
- Intermediate memory writes (saves bandwidth)

For the Qwen3-ASR model, this pattern appears in every transformer block:
  x = x + attention(rmsnorm(x))
  x = x + mlp(rmsnorm(x))
"""

import torch
import triton
import triton.language as tl

KERNEL_TYPE = "matmul"  # Use matmul type for benchmark compatibility


@triton.jit
def fused_rmsnorm_matmul_kernel(
    X_ptr, RMS_WEIGHT_ptr, W_ptr, BIAS_ptr, RESIDUAL_ptr, OUT_ptr,
    M, N, K,
    stride_xm, stride_xn,
    stride_wk, stride_wn,
    stride_rm, stride_rn,
    stride_om, stride_on,
    eps,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    """Fused RMSNorm + Matmul + Residual kernel.
    
    Each thread block computes one tile of the output:
      OUT = RMSNorm(X) @ W + RESIDUAL
    
    where RMSNorm is applied row-wise before the matmul.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Output tile offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # RMSNorm is row-wise over the full K dimension, so compute the complete
    # sum of squares before doing the tiled matmul.
    sum_sq = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xn
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=1)

    inv_rms = tl.rsqrt(sum_sq / K + eps)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Process K dimension in tiles
    for k_start in range(0, K, BLOCK_SIZE_K):
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
        
        # Load X tile [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xn
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)
        
        # Apply RMSNorm to each row of X
        x_norm = x * inv_rms[:, None]
        
        # Load RMSNorm weight vector [BLOCK_K]
        w_ptrs = RMS_WEIGHT_ptr + offs_k
        w_mask = offs_k < K
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)
        
        # Apply weight: x_norm = x_norm * w
        x_norm = x_norm * w[None, :]
        
        # Load W tile [BLOCK_K, BLOCK_N] for matmul
        w_mat_ptrs = W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w_mat_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w_mat = tl.load(w_mat_ptrs, mask=w_mat_mask, other=0.0)
        
        # Accumulate matmul: acc += x_norm @ w_mat
        acc += tl.dot(x_norm.to(w_mat.dtype), w_mat)
    
    # Add bias if present
    if HAS_BIAS:
        bias_ptrs = BIAS_ptr + offs_n
        bias_mask = offs_n < N
        bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]
    
    # Add residual if present
    if HAS_RESIDUAL:
        res_ptrs = RESIDUAL_ptr + offs_m[:, None] * stride_rm + offs_n[None, :] * stride_rn
        res_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        res = tl.load(res_ptrs, mask=res_mask, other=0.0).to(tl.float32)
        acc = acc + res
    
    # Store output
    out_ptrs = OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(OUT_ptr.dtype.element_ty), mask=out_mask)


@triton.jit
def regular_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Regular matmul for benchmark compatibility."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        offs_k += BLOCK_SIZE_K

    c = acc.to(C_ptr.dtype.element_ty)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def kernel_fn(A: torch.Tensor, B: torch.Tensor, 
              rmsnorm_weight: torch.Tensor = None,
              bias: torch.Tensor = None,
              residual: torch.Tensor = None,
              eps: float = 1e-6) -> torch.Tensor:
    """Entry point for fused RMSNorm + Matmul + Residual.
    
    Args:
        A: Input tensor [M, K] - will be RMSNorm'd before matmul
        B: Weight tensor [K, N] - for matmul
        rmsnorm_weight: RMSNorm weight vector [K]
        bias: Optional bias vector [N]
        residual: Optional residual tensor [M, N]
        eps: RMSNorm epsilon
    
    Returns:
        Output tensor [M, N] = RMSNorm(A) @ B + bias + residual
    """
    assert A.is_cuda and B.is_cuda
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    
    # For benchmark compatibility, if no rmsnorm_weight provided,
    # just do regular matmul
    if rmsnorm_weight is None:
        # Regular matmul path
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
        
        grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
        
        regular_matmul_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        return C
    
    # Fused path
    assert rmsnorm_weight.is_cuda
    assert rmsnorm_weight.shape == (K,)
    
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = min(32, K)  # Ensure BLOCK_SIZE_K <= K
    
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    fused_rmsnorm_matmul_kernel[grid](
        A, rmsnorm_weight, B, bias, residual, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        residual.stride(0) if residual is not None else A.stride(0),
        residual.stride(1) if residual is not None else A.stride(1),
        C.stride(0), C.stride(1),
        eps,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        HAS_BIAS=bias is not None,
        HAS_RESIDUAL=residual is not None,
    )
    return C
