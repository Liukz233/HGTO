"""CUDA Q4 tangent assembly and general sparse LU using NVIDIA cuDSS 0.8.

All numerical arrays, assembly, factorization, solve and residual certification
stay on the operator's CUDA device. Static mesh indexing and library symbolic
analysis may use the host. Hybrid execution and hybrid memory are disabled.
No CPU numerical fallback exists. The thin binding targets the pinned 0.8 ABI.
"""

from __future__ import annotations
import ctypes as ct
from functools import lru_cache
from pathlib import Path
import site, weakref
import numpy as np
import torch


@lru_cache(None)
def _library():
    candidates = [Path(p) / "nvidia/cu12/lib/libcudss.so.0" for p in site.getsitepackages()]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise ImportError("Install the gpu-sparse extra (nvidia-cudss-cu12==0.8.0.10)")
    lib = ct.CDLL(str(path))
    ptr = ct.c_void_p
    out = ct.POINTER(ptr)
    i = ct.c_int
    i64 = ct.c_int64
    signatures = {
        "cudssCreate": [out],
        "cudssDestroy": [ptr],
        "cudssConfigCreate": [out],
        "cudssConfigDestroy": [ptr],
        "cudssConfigSet": [ptr, i, ptr, ct.c_size_t],
        "cudssDataCreate": [ptr, out],
        "cudssDataDestroy": [ptr, ptr],
        "cudssSetStream": [ptr, ptr],
        "cudssMatrixCreateCsr": [out, i64, i64, i64, ptr, ptr, ptr, ptr, i, i, i, i, i, i],
        "cudssMatrixCreateDn": [out, i64, i64, i64, ptr, i, i],
        "cudssMatrixDestroy": [ptr],
        "cudssExecute": [ptr, i, ptr, ptr, ptr, ptr, ptr],
    }
    for name, args in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = i
    return lib


def _call(lib, name, *args):
    status = getattr(lib, name)(*args)
    if status:
        raise RuntimeError(f"{name} failed with cuDSS status {status}")


def _destroy(lib, device, handles):
    with torch.cuda.device(device):
        torch.cuda.synchronize(device)
        h, c, d, a, x, b = handles
        for matrix in (a, x, b):
            if matrix.value:
                lib.cudssMatrixDestroy(matrix)
        if d.value:
            lib.cudssDataDestroy(h, d)
        if c.value:
            lib.cudssConfigDestroy(c)
        if h.value:
            lib.cudssDestroy(h)


class CudaTangentSolver:
    def __init__(self, operator):
        if operator.device.type != "cuda" or operator.dtype != torch.float64:
            raise ValueError("cuDSS tangents require a CUDA FP64 operator")
        self.device = operator.device
        self.lib = _library()
        # Only mesh/connectivity metadata are transferred for this one-time map.
        nodes = operator.econn.detach().cpu().numpy()
        free = np.flatnonzero(operator.free_dof_mask.cpu().numpy())
        ids = (2 * nodes[:, :, None] + np.arange(2)).reshape(-1, 8)
        reduced = np.full(operator.n_dof, -1, dtype=np.int64)
        reduced[free] = np.arange(len(free))
        r = np.repeat(reduced[ids], 8, axis=1).reshape(-1)
        c = np.tile(reduced[ids], (1, 8)).reshape(-1)
        valid = np.flatnonzero((r >= 0) & (c >= 0))
        n = len(free)
        keys = r[valid] * n + c[valid]
        order = np.argsort(keys, kind="stable")
        unique, starts, counts = np.unique(keys[order], return_index=True, return_counts=True)
        self.gather = torch.tensor(valid[order], device=self.device)
        self.lengths = torch.tensor(counts, device=self.device)
        self.rowptr = torch.tensor(
            np.r_[0, np.cumsum(np.bincount(unique // n, minlength=n))],
            device=self.device,
            dtype=torch.int32,
        )
        self.columns = torch.tensor(unique % n, device=self.device, dtype=torch.int32)
        self.values = torch.zeros(len(unique), device=self.device, dtype=torch.float64)
        self.rhs = torch.zeros(n, device=self.device, dtype=torch.float64)
        self.solution = torch.zeros_like(self.rhs)
        self.handles = [ct.c_void_p() for _ in range(6)]
        h, cfg, data, a, x, b = self.handles
        self.analysis_done = False
        self.solves = 0
        try:
            _call(self.lib, "cudssCreate", ct.byref(h))
            _call(self.lib, "cudssConfigCreate", ct.byref(cfg))
            for key, value in [(11, 0), (15, 0), (14, 2), (24, 1)]:
                # hybrid memory=0, hybrid execution=0, host threads=2,
                # deterministic library execution=1 (not density refinement).
                setting = ct.c_int(value)
                _call(self.lib, "cudssConfigSet", cfg, key, ct.byref(setting), ct.sizeof(setting))
            _call(self.lib, "cudssDataCreate", h, ct.byref(data))
            _call(self.lib, "cudssSetStream", h, torch.cuda.current_stream(self.device).cuda_stream)
            _call(
                self.lib,
                "cudssMatrixCreateCsr",
                ct.byref(a),
                n,
                n,
                len(unique),
                self.rowptr.data_ptr(),
                None,
                self.columns.data_ptr(),
                self.values.data_ptr(),
                10,
                10,
                1,
                0,
                0,
                0,
            )
            _call(
                self.lib,
                "cudssMatrixCreateDn",
                ct.byref(x),
                n,
                1,
                n,
                self.solution.data_ptr(),
                1,
                0,
            )
            _call(self.lib, "cudssMatrixCreateDn", ct.byref(b), n, 1, n, self.rhs.data_ptr(), 1, 0)
        except Exception:
            _destroy(self.lib, self.device, self.handles)
            raise
        self._finalizer = weakref.finalize(self, _destroy, self.lib, self.device, self.handles)

    def close(self):
        self._finalizer()

    def solve(self, element_matrices, rhs, transpose=False):
        if rhs.device != self.device or rhs.dtype != torch.float64:
            raise ValueError("RHS must remain CUDA FP64")
        h, cfg, data, a, x, b = self.handles
        _call(self.lib, "cudssSetStream", h, torch.cuda.current_stream(self.device).cuda_stream)
        ke = element_matrices.transpose(-1, -2) if transpose else element_matrices
        self.values.copy_(
            torch.segment_reduce(ke.reshape(-1)[self.gather], reduce="sum", lengths=self.lengths)
        )
        self.rhs.copy_(rhs)
        if not self.analysis_done:
            _call(self.lib, "cudssExecute", h, 3, cfg, data, a, x, b)
            self.analysis_done = True
        _call(self.lib, "cudssExecute", h, 4, cfg, data, a, x, b)
        _call(self.lib, "cudssExecute", h, 1008, cfg, data, a, x, b)
        self.solves += 1
        return self.solution.clone()


def solve_cuda_tangent(operator, element_matrices, rhs, matvec, rtol, transpose=False):
    if tuple(element_matrices.shape) != (operator.n_elements, 8, 8):
        raise ValueError("Expected Q4 element tangents")
    if rhs.device.type != "cuda" or element_matrices.device != rhs.device:
        raise ValueError("No CPU state or tangent is permitted")
    if not hasattr(operator, "_cudss_tangent"):
        operator._cudss_tangent = CudaTangentSolver(operator)
    value = operator._cudss_tangent.solve(element_matrices, rhs, transpose)
    residual = torch.linalg.vector_norm(matvec(value) - rhs) / torch.linalg.vector_norm(
        rhs
    ).clamp_min(1e-30)
    ok = bool(torch.isfinite(value).all() and residual <= max(10 * float(rtol), 1e-10))
    return value, float(residual), 1, ok
