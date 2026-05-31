"""
Example: Integrating AutoKernel-optimized kernels into Flash-Mega-ASR.

This example shows how to use the mega-kernel to fuse RMSNorm + matmul + residual
operations in the Qwen3-ASR transformer block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the optimized kernel
from kernels.mega_fusion import kernel_fn as mega_kernel_fn


class RMSNorm(nn.Module):
    """RMS Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class OptimizedTransformerBlock(nn.Module):
    """
    Transformer block with fused RMSNorm + matmul + residual.
    
    This demonstrates how to integrate the mega-kernel into the Qwen3-ASR
    transformer block to eliminate 3 kernel launches per block.
    """
    
    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.dim = dim
        self.intermediate_size = intermediate_size
        
        # Normalization weights
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        
        # Attention projections
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        
        # MLP projections
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fused operations.
        
        Before (6 kernel launches):
          x_norm = attention_norm(x)    # Kernel 1
          q = x_norm @ wq               # Kernel 2
          k = x_norm @ wk               # Kernel 3
          v = x_norm @ wv               # Kernel 4
          attn_out = attention(q, k, v) # Kernel 5
          x = x + wo(attn_out)          # Kernel 6
          
          x_norm = ffn_norm(x)          # Kernel 7
          gate = x_norm @ gate_proj     # Kernel 8
          up = x_norm @ up_proj         # Kernel 9
          x = x + down_proj(silu(gate) * up)  # Kernel 10
        
        After (4 kernel launches with mega-kernel):
          # Fused Q/K/V projections (1 kernel)
          q = mega_kernel_fn(x, wq_weight, attention_norm.weight)
          k = mega_kernel_fn(x, wk_weight, attention_norm.weight)
          v = mega_kernel_fn(x, wv_weight, attention_norm.weight)
          
          attn_out = attention(q, k, v)  # Kernel 2
          
          # Fused output projection + residual (1 kernel)
          x = mega_kernel_fn(attn_out, wo_weight, None, x)
          
          # Fused gate/up projections (2 kernels)
          gate = mega_kernel_fn(x, gate_proj_weight, ffn_norm.weight)
          up = mega_kernel_fn(x, up_proj_weight, ffn_norm.weight)
          
          # Fused down projection + residual (1 kernel)
          x = mega_kernel_fn(silu(gate) * up, down_proj_weight, None, x)
        
        Total: 6 kernel launches (vs 10 originally)
        Savings: ~4 kernel launches per transformer block
        """
        # For now, use standard PyTorch operations
        # The mega-kernel integration requires custom weight handling
        
        # Attention block
        x_norm = self.attention_norm(x)
        q = self.wq(x_norm)
        k = self.wk(x_norm)
        v = self.wv(x_norm)
        
        # Standard attention (replace with flash attention in production)
        attn_out = F.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
            is_causal=True
        ).squeeze(0)
        
        x = x + self.wo(attn_out)
        
        # MLP block
        x_norm = self.ffn_norm(x)
        gate = F.silu(self.gate_proj(x_norm))
        up = self.up_proj(x_norm)
        x = x + self.down_proj(gate * up)
        
        return x


def benchmark_kernel_comparison():
    """
    Benchmark comparing standard vs fused operations.
    """
    import time
    
    # Setup
    device = "cuda"
    dtype = torch.bfloat16
    M, K, N = 512, 2048, 2048
    
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)
    weight = torch.randn(K, device=device, dtype=dtype)
    residual = torch.randn(M, N, device=device, dtype=dtype)
    
    # Warmup
    for _ in range(10):
        # Standard path
        x_norm = A * torch.rsqrt(A.pow(2).mean(-1, keepdim=True) + 1e-6) * weight
        C_std = torch.matmul(x_norm, B)
        C_std = C_std + residual
        
        # Fused path
        C_fused = mega_kernel_fn(A, B, rmsnorm_weight=weight, residual=residual)
    
    torch.cuda.synchronize()
    
    # Benchmark standard path
    start = time.perf_counter()
    for _ in range(100):
        x_norm = A * torch.rsqrt(A.pow(2).mean(-1, keepdim=True) + 1e-6) * weight
        C_std = torch.matmul(x_norm, B)
        C_std = C_std + residual
    torch.cuda.synchronize()
    std_time = (time.perf_counter() - start) / 100 * 1000
    
    # Benchmark fused path
    start = time.perf_counter()
    for _ in range(100):
        C_fused = mega_kernel_fn(A, B, rmsnorm_weight=weight, residual=residual)
    torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / 100 * 1000
    
    print(f"Standard path: {std_time:.2f} ms")
    print(f"Fused path:    {fused_time:.2f} ms")
    print(f"Speedup:       {std_time / fused_time:.2f}x")
    
    # Verify correctness
    max_diff = (C_std - C_fused).abs().max().item()
    print(f"Max difference: {max_diff:.6f}")


if __name__ == "__main__":
    print("=== AutoKernel Integration Example ===")
    print()
    
    if torch.cuda.is_available():
        benchmark_kernel_comparison()
    else:
        print("CUDA not available. Skipping benchmark.")
        print()
        print("To use the mega-kernel:")
        print("  from kernels.mega_fusion import kernel_fn")
        print("  output = kernel_fn(A, B, rmsnorm_weight=weight, residual=residual)")
