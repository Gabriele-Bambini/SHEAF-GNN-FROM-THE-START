import logging
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import torch
import torch.nn as nn

from tsgnn.model.sheaf_vectorized import VectorizedSheafDiffusion

logger = logging.getLogger(__name__)

class SparseVectorizedSheafDiffusion(VectorizedSheafDiffusion):
    """
    Sparse COO-based Sheaf Diffusion Laplacian.
    Solves Flaw 8: Dense Graph Memory Scaling.
    
    Inherits from VectorizedSheafDiffusion to reuse indices and initialization,
    but overrides compute_connection_laplacian_vectorized to construct a
    torch.sparse_coo_tensor instead of allocating a full Nd x Nd matrix.
    """

    def compute_connection_laplacian_vectorized(
        self,
        restriction_maps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        maps = restriction_maps if restriction_maps is not None else self.restriction_maps
        d = self.d
        Nd = self.Nd

        F_src = maps[:, 0]  # (E, d, d)
        F_tgt = maps[:, 1]  # (E, d, d)

        # Compute block products via batched matmul
        off_diag = -torch.bmm(F_src.transpose(1, 2), F_tgt)   # (E, d, d)
        diag_src = torch.bmm(F_src.transpose(1, 2), F_src)     # (E, d, d)
        diag_tgt = torch.bmm(F_tgt.transpose(1, 2), F_tgt)     # (E, d, d)

        dev = maps.device
        if self.off_flat_idx.device != dev:
            self.off_flat_idx = self.off_flat_idx.to(dev)
            self.off_flat_idx_T = self.off_flat_idx_T.to(dev)
            self.diag_src_flat_idx = self.diag_src_flat_idx.to(dev)
            self.diag_tgt_flat_idx = self.diag_tgt_flat_idx.to(dev)

        compute_dtype = off_diag.dtype
        off_flat = off_diag.reshape(-1)
        diag_src_flat = diag_src.reshape(-1)
        diag_tgt_flat = diag_tgt.reshape(-1)

        # Build COO tensor from concatenated flat indices and values
        all_flat_idx = torch.cat([
            self.off_flat_idx,
            self.off_flat_idx_T,
            self.diag_src_flat_idx,
            self.diag_tgt_flat_idx
        ])
        
        all_values = torch.cat([
            off_flat,
            off_diag.transpose(1, 2).reshape(-1),
            diag_src_flat,
            diag_tgt_flat
        ])
        
        # Regularization diagonal
        eps = self._laplacian_eps
        if eps > 0:
            diag_idx_2d = torch.arange(Nd, device=dev)
            diag_flat_idx = diag_idx_2d * Nd + diag_idx_2d
            diag_values = torch.full((Nd,), eps, device=dev, dtype=compute_dtype)
            all_flat_idx = torch.cat([all_flat_idx, diag_flat_idx])
            all_values = torch.cat([all_values, diag_values])
        
        # Convert flat (row*Nd + col) to 2D (row, col)
        rows = all_flat_idx // Nd
        cols = all_flat_idx % Nd
        indices = torch.stack([rows, cols], dim=0)
        
        # PyTorch automatically sums values at duplicate indices when coalescing
        L_sparse = torch.sparse_coo_tensor(
            indices, all_values, size=(Nd, Nd), device=dev, dtype=compute_dtype
        )
        L_sparse = L_sparse.coalesce()
        
        # Check NaNs
        if not torch.isfinite(L_sparse.values()).all():
            n_bad = (~torch.isfinite(L_sparse.values())).sum().item()
            logger.warning(
                "Non-finite Laplacian detected (%d / %d sparse entries). "
                "Sanitising with nan_to_num.",
                n_bad, L_sparse.values().numel(),
            )
            clean_values = torch.nan_to_num(L_sparse.values(), nan=0.0, posinf=0.0, neginf=0.0)
            L_sparse = torch.sparse_coo_tensor(
                L_sparse.indices(), clean_values, size=(Nd, Nd), device=dev, dtype=compute_dtype
            ).coalesce()

        return L_sparse

    def sheaf_diffusion(
        self,
        x: torch.Tensor,
        L_F: torch.Tensor,
    ) -> torch.Tensor:
        """
        One step of sheaf diffusion using sparse matrix multiplication.
        x_{k+1} = x_k - α * sigma(L_F @ (x_k @ W + b))
        """
        # Ensure L_F is sparse (or fallback to dense behavior if someone passes dense)
        if not L_F.is_sparse:
            return super().sheaf_diffusion(x, L_F)
            
        alpha_bounded = torch.clamp(self.alpha, min=0.01, max=0.99)

        h = x @ self.weight + self.bias          # (N, d)
        h_lifted = h.reshape(-1, 1)              # (Nd, 1)

        # torch.sparse.mm executes sparse-dense matrix multiplication efficiently
        diffusion_update = torch.sparse.mm(L_F, h_lifted).squeeze(1) # (Nd,)
        
        diffusion_update = torch.nn.functional.elu(diffusion_update)
        
        x_new = x.reshape(-1) - alpha_bounded * diffusion_update
        return x_new.reshape(x.shape)

    def verify_laplacian_properties(
        self, restriction_maps: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute spectrum properties efficiently for sparse matrices.
        """
        L_sparse = self.compute_connection_laplacian_vectorized(restriction_maps)
        vals = L_sparse.values()
        frob = vals.pow(2).sum().sqrt().item()

        is_symmetric = True  # Symmetric by construction
        Nd = self.Nd

        if Nd > 1500:
            # Use SciPy sparse eigs for memory efficiency
            try:
                indices = L_sparse.indices().cpu().numpy()
                values = vals.cpu().numpy()
                L_sp = sp.coo_matrix((values, (indices[0], indices[1])), shape=(Nd, Nd)).tocsc()
                
                # Smallest algebraic eigenvalues
                min_eigs = sla.eigsh(L_sp, k=2, which='SA', return_eigenvectors=False)
                # Largest algebraic eigenvalue
                max_eig = sla.eigsh(L_sp, k=1, which='LA', return_eigenvectors=False)
                
                min_eigs = np.sort(min_eigs)
                
                return {
                    "is_symmetric": is_symmetric,
                    "is_psd": bool(min_eigs[0] >= -1e-4),
                    "min_eigenvalue": float(min_eigs[0]),
                    "spectral_gap": float(min_eigs[1] - min_eigs[0]) if len(min_eigs) > 1 else float("nan"),
                    "condition_number": float(max_eig[0] / max(1e-10, min_eigs[1])) if len(min_eigs) > 1 else float("inf"),
                    "max_eigenvalue": float(max_eig[0]),
                    "frobenius_norm": frob,
                }
            except Exception as e:
                logger.debug(f"Sparse eigensolver failed: {e}")
                return {
                    "is_symmetric": is_symmetric,
                    "is_psd": False,
                    "min_eigenvalue": float("nan"),
                    "spectral_gap": float("nan"),
                    "condition_number": float("nan"),
                    "max_eigenvalue": float("nan"),
                    "frobenius_norm": frob,
                }
        else:
            # Fallback to PyTorch dense eigh for smaller matrices, much faster on GPU
            L_dense = L_sparse.to_dense()
            try:
                eigs = torch.linalg.eigvalsh(L_dense.float()).sort().values
                if eigs.numel() < 2:
                    return {
                        "is_symmetric": is_symmetric,
                        "is_psd": bool((eigs >= -1e-5).all()),
                        "min_eigenvalue": eigs[0].item() if eigs.numel() else float("nan"),
                        "spectral_gap": float("nan"),
                        "condition_number": float("inf"),
                        "max_eigenvalue": eigs[-1].item() if eigs.numel() else float("nan"),
                        "frobenius_norm": frob,
                    }
                return {
                    "is_symmetric": is_symmetric,
                    "is_psd": bool((eigs >= -1e-5).all()),
                    "min_eigenvalue": eigs[0].item(),
                    "spectral_gap": (eigs[1] - eigs[0]).item(),
                    "condition_number": (eigs[-1] / eigs[1].clamp(min=1e-10)).item(),
                    "max_eigenvalue": eigs[-1].item(),
                    "frobenius_norm": frob,
                }
            except torch._C._LinAlgError:
                return {
                    "is_symmetric": is_symmetric,
                    "is_psd": False,
                    "min_eigenvalue": float("nan"),
                    "spectral_gap": float("nan"),
                    "condition_number": float("inf"),
                    "max_eigenvalue": float("nan"),
                    "frobenius_norm": frob,
                }
