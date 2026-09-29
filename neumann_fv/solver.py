"""Compatibility checking and a self-contained projected PCG iteration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .grid import Grid2D
from .operator import FVDiffusion


class IncompatibleRightHandSide(ValueError):
    """Raised when a discrete Neumann right-hand side has nonzero total source."""
    def __init__(self, mean: float, tol: float):
        super().__init__(
            f"discrete compatibility violated: mean(b)={mean:.16e}, tolerance={tol:.16e}"
        )
        self.mean = mean
        self.tolerance = tol


@dataclass(frozen=True)
class PCGResult:
    x: np.ndarray
    b: np.ndarray
    original_b: np.ndarray
    removed_mean: float
    iterations: int
    converged: bool
    projected_initial_residual: float
    true_residual: float
    relative_true_residual: float
    solution_mean: float
    conservation_error: float
    status: str


def weighted_mean(grid: Grid2D, values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.shape != (grid.n,):
        raise ValueError(f"values must have shape ({grid.n},)")
    weights = grid.cell_volumes
    return float(np.dot(weights, values) / np.sum(weights))


def project_mean(grid: Grid2D, values: np.ndarray) -> np.ndarray:
    values = np.array(values, dtype=float, copy=True)
    return values - weighted_mean(grid, values)


def compatibility_tolerance(b: np.ndarray, atol: float, rtol: float) -> float:
    return float(atol + rtol * np.max(np.abs(b)))


def prepare_rhs(
    grid: Grid2D,
    b: np.ndarray,
    mode: Literal["reject", "project"] = "reject",
    atol: float = 1e-12,
    rtol: float = 1e-10,
) -> tuple[np.ndarray, float, float]:
    """Validate and, if explicitly requested, project a Neumann right-hand side."""
    b = np.asarray(b, dtype=float)
    if b.shape != (grid.n,):
        raise ValueError(f"b must have shape ({grid.n},)")
    if not np.all(np.isfinite(b)):
        raise ValueError("b must contain only finite values")

    mean_b = weighted_mean(grid, b)
    tol = compatibility_tolerance(b, atol, rtol)
    if abs(mean_b) > tol:
        if mode == "reject":
            raise IncompatibleRightHandSide(mean_b, tol)
        if mode != "project":
            raise ValueError("mode must be 'reject' or 'project'")

    if mode == "project":
        b_actual = b - mean_b
        removed = mean_b
    else:
        b_actual = np.array(b, dtype=float, copy=True)
        removed = 0.0
    return b_actual, removed, tol


def projected_pcg(
    op: FVDiffusion,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    *,
    rtol: float = 1e-11,
    atol: float = 1e-12,
    maxiter: int | None = None,
) -> PCGResult:
    """Projected conjugate gradients with a projected Jacobi preconditioner.

    The constraint is sum_i V_i x_i=0. Projection is applied to the residual,
    preconditioned residual, and every recurrence-generated search direction.
    No Dirichlet value is introduced, so the natural constant nullspace remains
    visible and is removed only by this scalar Lagrange constraint.
    """
    grid = op.grid
    b = np.asarray(b, dtype=float)
    if b.shape != (grid.n,):
        raise ValueError(f"b must have shape ({grid.n},)")
    if np.abs(weighted_mean(grid, b)) > compatibility_tolerance(b, atol, rtol):
        raise IncompatibleRightHandSide(
            weighted_mean(grid, b), compatibility_tolerance(b, atol, rtol)
        )

    b = project_mean(grid, b)
    x = np.zeros(grid.n) if x0 is None else np.asarray(x0, dtype=float).copy()
    if x.shape != (grid.n,):
        raise ValueError(f"x0 must have shape ({grid.n},)")
    x = project_mean(grid, x)

    if maxiter is None:
        maxiter = 10 * grid.n + 100
    if maxiter < 0:
        raise ValueError("maxiter must be nonnegative")

    pcg_diagonal = op.preconditioner_diagonal
    inv_diag = np.where(
        pcg_diagonal > 0.0,
        1.0 / pcg_diagonal,
        0.0,
    )

    r = b - op.matvec(x)
    r = project_mean(grid, r)
    initial_norm = float(np.linalg.norm(r))
    target = float(atol + rtol * np.linalg.norm(b))

    status = "converged"
    iterations = 0
    if initial_norm <= target:
        pass
    else:
        z = project_mean(grid, inv_diag * r)
        p = z.copy()
        rz = float(np.dot(r, z))
        if rz <= 0.0:
            raise RuntimeError("projected preconditioner is not positive definite")

        for iterations in range(1, maxiter + 1):
            ap = op.matvec(p)
            pap = float(np.dot(p, ap))
            if not np.isfinite(pap) or pap <= 0.0:
                status = "nonpositive curvature"
                break

            alpha = rz / pap
            x += alpha * p
            r -= alpha * ap
            r = project_mean(grid, r)
            residual_norm = float(np.linalg.norm(r))
            if residual_norm <= target:
                break

            z = project_mean(grid, inv_diag * r)
            rz_new = float(np.dot(r, z))
            if rz_new <= 0.0:
                status = "nonpositive preconditioned residual"
                break
            beta = rz_new / rz
            p = project_mean(grid, z + beta * p)
            rz = rz_new
        else:
            status = "maximum iterations reached"

    x = project_mean(grid, x)
    final_residual_vector = b - op.matvec(x)
    true_residual = float(np.linalg.norm(final_residual_vector))
    b_norm = float(np.linalg.norm(b))
    relative_true_residual = true_residual / b_norm if b_norm > 0.0 else 0.0
    face_error = op.face_fluxes(x).conservation_error(b_cell=b)
    converged = status == "converged" and true_residual <= max(
        target, 10.0 * np.finfo(float).eps * max(1.0, np.linalg.norm(b))
    )
    if not converged and status == "converged":
        status = "true residual failed final verification"

    return PCGResult(
        x=x,
        b=b,
        original_b=b.copy(),
        removed_mean=0.0,
        iterations=iterations,
        converged=converged,
        projected_initial_residual=initial_norm,
        true_residual=true_residual,
        relative_true_residual=relative_true_residual,
        solution_mean=weighted_mean(grid, x),
        conservation_error=face_error["global_abs"],
        status=status,
    )


def solve_neumann(
    op: FVDiffusion,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    *,
    rhs_mode: Literal["reject", "project"] = "reject",
    compat_atol: float = 1e-12,
    compat_rtol: float = 1e-10,
    rtol: float = 1e-11,
    atol: float = 1e-12,
    maxiter: int | None = None,
) -> PCGResult:
    """Solve a compatible singular Neumann system with a zero-mean representative."""
    b_actual, removed_mean, _ = prepare_rhs(
        op.grid,
        b,
        mode=rhs_mode,
        atol=compat_atol,
        rtol=compat_rtol,
    )
    result = projected_pcg(
        op,
        b_actual,
        x0,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
    )
    return PCGResult(
        x=result.x,
        b=b_actual,
        original_b=np.asarray(b, dtype=float),
        removed_mean=removed_mean,
        iterations=result.iterations,
        converged=result.converged,
        projected_initial_residual=result.projected_initial_residual,
        true_residual=result.true_residual,
        relative_true_residual=result.relative_true_residual,
        solution_mean=result.solution_mean,
        conservation_error=result.conservation_error,
        status=result.status,
    )
