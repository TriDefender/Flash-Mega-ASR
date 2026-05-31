"""
Optimized Linear layer using AutoKernel's Triton kernels.

This module provides a drop-in replacement for torch.nn.Linear that uses
the optimized matmul kernel from AutoKernel.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_SKIP_LINEAR_NAMES = frozenset({"lm_head", "embed_out", "output_projection"})

# Try to import the optimized kernel
try:
    # Import from the package (kernels are in src/MegaASR/kernels/)
    from .mega_fusion import kernel_fn as mega_kernel_fn
    OPTIMIZED_KERNEL_AVAILABLE = True
    logger.info("AutoKernel optimized matmul kernel loaded successfully")
except ImportError as e:
    OPTIMIZED_KERNEL_AVAILABLE = False
    logger.warning(f"AutoKernel optimized kernel not available: {e}")


def _kernel_matmul_preserve_leading_dims(
    x: torch.Tensor,
    weight_t: torch.Tensor,
    *,
    rmsnorm_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Run the 2-D mega-kernel while preserving nn.Linear leading dimensions."""
    if x.dim() == 2:
        return mega_kernel_fn(x, weight_t, rmsnorm_weight=rmsnorm_weight, eps=eps)

    leading_shape = x.shape[:-1]
    x_2d = x.reshape(-1, x.shape[-1])
    output_2d = mega_kernel_fn(x_2d, weight_t, rmsnorm_weight=rmsnorm_weight, eps=eps)
    return output_2d.reshape(*leading_shape, weight_t.shape[1])


class OptimizedLinear(nn.Module):
    """
    Optimized Linear layer that uses AutoKernel's Triton kernel.
    
    This is a drop-in replacement for nn.Linear that achieves 97.5-97.8%
    peak compute on RTX 4060 Ti.
    
    Usage:
        # Replace nn.Linear with OptimizedLinear
        model.linear = OptimizedLinear(model.linear)
    """
    
    def __init__(self, original_linear: nn.Linear):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.weight = original_linear.weight
        self.bias = original_linear.bias
        
        # Check if we can use the optimized kernel
        self.use_optimized = (
            OPTIMIZED_KERNEL_AVAILABLE 
            and self.weight.is_cuda
            and self.weight.dtype in (torch.float16, torch.bfloat16)
        )
        
        if self.use_optimized:
            logger.debug(f"OptimizedLinear: Using Triton kernel for {self.in_features}x{self.out_features}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using optimized matmul."""
        if self.use_optimized and x.is_cuda and x.dim() >= 2:
            # Use the optimized Triton kernel
            # Note: The kernel expects A @ B, but Linear uses x @ W^T
            # So we need to transpose the weight
            weight_t = self.weight.t().contiguous()
            output = _kernel_matmul_preserve_leading_dims(x, weight_t)
            
            if self.bias is not None:
                output = output + self.bias
            
            return output
        else:
            # Fallback to standard PyTorch
            return F.linear(x, self.weight, self.bias)


class FusedRMSNormLinear(nn.Module):
    """
    Fused RMSNorm + Linear layer using AutoKernel's mega-kernel.
    
    This fuses RMSNorm and Linear into a single kernel, eliminating
    2 kernel launches per transformer block.
    
    Usage:
        # Replace RMSNorm + Linear with FusedRMSNormLinear
        norm = RMSNorm(dim)
        linear = nn.Linear(dim, out_dim)
        fused = FusedRMSNormLinear(norm, linear)
    """
    
    def __init__(self, norm: nn.Module, linear: nn.Linear):
        super().__init__()
        self.norm_weight = norm.weight if hasattr(norm, 'weight') else None
        self.eps = getattr(norm, 'eps', 1e-6)
        self.linear_weight = linear.weight
        self.linear_bias = linear.bias
        
        # Check if we can use the optimized kernel
        self.use_optimized = (
            OPTIMIZED_KERNEL_AVAILABLE 
            and self.linear_weight.is_cuda
            and self.linear_weight.dtype in (torch.float16, torch.bfloat16)
            and self.norm_weight is not None
        )
        
        if self.use_optimized:
            logger.debug("FusedRMSNormLinear: Using fused mega-kernel")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with fused RMSNorm + Linear."""
        if self.use_optimized and x.is_cuda and x.dim() >= 2:
            # Use the fused mega-kernel
            weight_t = self.linear_weight.t().contiguous()
            output = _kernel_matmul_preserve_leading_dims(
                x,
                weight_t,
                rmsnorm_weight=self.norm_weight,
                eps=self.eps,
            )
            
            if self.linear_bias is not None:
                output = output + self.linear_bias
            
            return output
        else:
            # Fallback: separate RMSNorm + Linear
            norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            x_norm = x * norm * self.norm_weight
            return F.linear(x_norm, self.linear_weight, self.linear_bias)


def _should_optimize_linear(name: str, linear: nn.Linear) -> bool:
    """Return whether a Linear layer is in the safe transformer projection set."""
    leaf_name = name.rsplit(".", 1)[-1]
    if leaf_name in _SKIP_LINEAR_NAMES:
        return False

    # Vocabulary heads are very wide and have caused illegal memory accesses in
    # generation. Keep them on PyTorch/cuBLAS; the Triton path is for internal
    # transformer projections.
    return linear.out_features <= 65536


def optimize_model(model: nn.Module, verbose: bool = False) -> nn.Module:
    """
    Replace nn.Linear layers with OptimizedLinear throughout the model.
    
    Args:
        model: The model to optimize
        verbose: Whether to log each replacement
    
    Returns:
        The model with optimized linear layers
    """
    if not OPTIMIZED_KERNEL_AVAILABLE:
        logger.warning("Optimized kernel not available. Skipping optimization.")
        return model
    
    replaced_count = 0
    skipped_count = 0
    
    def replace_linear(module: nn.Module, name: str = ""):
        nonlocal replaced_count, skipped_count
        
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            
            if isinstance(child, nn.Linear):
                if not _should_optimize_linear(full_name, child):
                    skipped_count += 1
                    if verbose:
                        logger.info(f"Skipped {full_name} for OptimizedLinear")
                    continue

                # Replace with optimized version
                setattr(module, child_name, OptimizedLinear(child))
                replaced_count += 1
                if verbose:
                    logger.info(f"Replaced {full_name} with OptimizedLinear")
            else:
                # Recursively replace in child modules
                replace_linear(child, full_name)
    
    replace_linear(model)
    logger.info(f"Optimized {replaced_count} linear layers; skipped {skipped_count}")
    
    return model


def get_optimization_status() -> dict:
    """Get the current optimization status."""
    # Check if triton is available by trying to import it
    triton_available = False
    try:
        import triton
        triton_available = True
    except ImportError:
        pass
    
    return {
        "optimized_kernel_available": OPTIMIZED_KERNEL_AVAILABLE,
        "triton_available": triton_available,
        "cuda_available": torch.cuda.is_available(),
    }
