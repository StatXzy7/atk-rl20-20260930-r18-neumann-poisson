"""Cell-centered finite-volume discretization of  -div(k grad u) = f  on a
rectangle with homogeneous Neumann (zero normal flux) boundary conditions.

Conventions
-----------
* Uniform nx x ny grid on [0, Lx] x [0, Ly]; unknowns and the coefficient k
  live at cell centers.  Grid functions are 2-D arrays of shape (ny, nx).
* The discrete operator is assembled from face fluxes.  Every interior face
  contributes exactly one flux value that is added to one neighbouring cell
  and subtracted from the other, so interior fluxes cancel pairwise by
  construction.  Boundary faces carry the prescribed zero flux and simply
  contribute nothing -- no Dirichlet ghost values are used anywhere.
* With  (A u)_ij = (1/V) * sum of fluxes leaving cell (i, j)  the discrete
  problem reads  A u = f.  A is symmetric positive semidefinite with the
  one-dimensional null space span{1}.  The discrete compatibility condition
  is  sum_ij f_ij * V = 0  and the unique representative is fixed by the
  zero-mean constraint  sum_ij u_ij * V = 0.
* The linear system is solved by a projected preconditioned conjugate
  gradient iteration (projection P = I - (1/n) 1 1^T, diagonal
  preconditioner P D^{-1} P) implemented by hand in `solve`.

See DERIVATION.md for the full derivation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Grid",
    "FVOperator",
    "SolveResult",
    "InvalidCoefficientError",
    "IncompatibleRHSError",
    "solve",
]


class InvalidCoefficientError(ValueError):
    """Raised when the diffusion coefficient is not strictly positive."""


class IncompatibleRHSError(ValueError):
    """Raised in strict mode when the right-hand side violates the discrete
    compatibility condition  sum(f) * V = 0."""


@dataclass
class Grid:
    """Uniform cell-centered grid on [0, Lx] x [0, Ly]."""

    nx: int
    ny: int
    Lx: float = 1.0
    Ly: float = 1.0

    def __post_init__(self):
        self.nx = int(self.nx)
        self.ny = int(self.ny)
        self.Lx = float(self.Lx)
        self.Ly = float(self.Ly)
        if self.nx < 2 or self.ny < 2:
            raise ValueError("need at least 2 cells in each direction")
        if not (np.isfinite(self.Lx) and np.isfinite(self.Ly)) or self.Lx <= 0.0 or self.Ly <= 0.0:
            raise ValueError("Lx and Ly must be positive and finite")
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.cell_volume = self.dx * self.dy
        self.xc = (np.arange(self.nx) + 0.5) * self.dx
        self.yc = (np.arange(self.ny) + 0.5) * self.dy

    @property
    def shape(self):
        return (self.ny, self.nx)

    @property
    def ncells(self):
        return self.nx * self.ny


class FVOperator:
    """Discrete operator  u -> -div(k grad u)  built from face fluxes."""

    def __init__(self, grid: Grid, k):
        self.grid = grid
        self.k = self._validate_k(k)
        kx, ky = self._harmonic_faces(self.k)
        # flux through a face = coeff * (u_near - u_far); coeff includes the
        # face-area / distance factor.
        self.cx = kx * (grid.dy / grid.dx)   # x-normal faces, shape (ny, nx-1)
        self.cy = ky * (grid.dx / grid.dy)   # y-normal faces, shape (ny-1, nx)
        d = np.zeros(grid.shape)
        d[:, :-1] += self.cx
        d[:, 1:] += self.cx
        d[:-1, :] += self.cy
        d[1:, :] += self.cy
        self.diag = d / grid.cell_volume     # diagonal of A (strictly positive)

    def _validate_k(self, k):
        k = np.asarray(k, dtype=float)
        if k.shape != self.grid.shape:
            raise InvalidCoefficientError(
                f"k has shape {k.shape}, expected {self.grid.shape}"
            )
        if not np.all(np.isfinite(k)):
            raise InvalidCoefficientError("k contains NaN or inf")
        if not np.all(k > 0.0):
            raise InvalidCoefficientError("k must be strictly positive in every cell")
        return k

    @staticmethod
    def _harmonic_faces(k):
        kx = 2.0 * k[:, :-1] * k[:, 1:] / (k[:, :-1] + k[:, 1:])
        ky = 2.0 * k[:-1, :] * k[1:, :] / (k[:-1, :] + k[1:, :])
        return kx, ky

    def apply(self, u):
        """Matrix-free action of A on a grid function (flux assembly)."""
        u = np.asarray(u, dtype=float).reshape(self.grid.shape)
        out = np.zeros(self.grid.shape)
        fx = self.cx * (u[:, :-1] - u[:, 1:])   # flux leaving the west cell
        out[:, :-1] += fx
        out[:, 1:] -= fx
        fy = self.cy * (u[:-1, :] - u[1:, :])    # flux leaving the south cell
        out[:-1, :] += fy
        out[1:, :] -= fy
        return out / self.grid.cell_volume

    def to_sparse(self):
        """Independent stencil assembly as a scipy CSR matrix.

        The harmonic face coefficients are recomputed from k inside explicit
        per-face loops (a separate code path from `apply`); used to
        cross-check the matrix-free implementation.
        """
        from scipy import sparse

        nx, ny = self.grid.nx, self.grid.ny
        vol = self.grid.cell_volume
        k = self.k
        rows, cols, vals = [], [], []

        def add_face(p, q, coeff):
            rows.extend((p, q, p, q))
            cols.extend((p, q, q, p))
            vals.extend((coeff, coeff, -coeff, -coeff))

        for j in range(ny):
            for i in range(nx - 1):
                kh = 2.0 * k[j, i] * k[j, i + 1] / (k[j, i] + k[j, i + 1])
                add_face(j * nx + i, j * nx + i + 1,
                         kh * self.grid.dy / self.grid.dx / vol)
        for j in range(ny - 1):
            for i in range(nx):
                kh = 2.0 * k[j, i] * k[j + 1, i] / (k[j, i] + k[j + 1, i])
                add_face(j * nx + i, (j + 1) * nx + i,
                         kh * self.grid.dx / self.grid.dy / vol)
        n = nx * ny
        return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


@dataclass
class SolveResult:
    u: np.ndarray            # zero-mean representative, shape (ny, nx)
    converged: bool
    iterations: int
    residual_norm: float     # true ||A u - b||_2, recomputed after the loop
    residual_rel: float      # relative to ||b||_2
    solution_mean: float     # volume-weighted mean of the returned u
    rhs_imbalance: float     # discrete compatibility residual sum(f) * V
    rhs_mean_removed: float  # constant c: the equation actually solved is A u = f - c
    rhs_modified: bool       # True if |c| exceeded the compatibility tolerance
    flux_imbalance_max: float  # max per-cell flux-balance error, in flux units
    boundary_flux: float     # net flux through the domain boundary (~ 0)
    equation: str            # human-readable statement of the solved equation


def solve(op: FVOperator, f, mode: str = "strict", tol: float = 1e-10,
          maxiter: int | None = None, compat_rtol: float = 1e-12) -> SolveResult:
    """Solve A u = f for the zero-mean representative by projected PCG.

    mode="strict"   : reject right-hand sides whose discrete mean exceeds
                      compat_rtol * sum(|f|) * V (IncompatibleRHSError).
    mode="project"  : solve the projected equation  A u = f - c  with
                      c = mean(f); the removed constant and the modified
                      equation are reported in the result -- the right-hand
                      side is never silently altered.
    """
    grid = op.grid
    if mode not in ("strict", "project"):
        raise ValueError("mode must be 'strict' or 'project'")
    f = np.asarray(f, dtype=float)
    if f.size != grid.ncells:
        raise ValueError(f"f has size {f.size}, expected {grid.ncells}")
    f = f.reshape(grid.shape)
    if not np.all(np.isfinite(f)):
        raise ValueError("f contains NaN or inf")

    vol = grid.cell_volume
    n = grid.ncells
    imbalance = float(f.sum() * vol)          # discrete integral of f
    scale = float(np.abs(f).sum() * vol)      # discrete integral of |f|
    c = float(f.mean())                       # constant to project out
    compatible = abs(imbalance) <= compat_rtol * scale if scale > 0.0 else True
    if mode == "strict" and not compatible:
        raise IncompatibleRHSError(
            "right-hand side violates the discrete compatibility condition: "
            f"sum(f)*V = {imbalance:.6e} (relative {abs(imbalance) / scale:.3e} "
            f"> {compat_rtol:.1e}); pure Neumann problems need a zero-mean "
            "source. Re-run with mode='project' to solve the projected "
            "equation A u = f - mean(f)."
        )

    b = f - c                                  # exactly compatible RHS
    rhs_modified = not compatible
    equation = "A u = f" if not rhs_modified else f"A u = f - ({c:.6e})"

    bnorm = float(np.linalg.norm(b))
    if bnorm == 0.0:
        u = np.zeros(grid.shape)
        return SolveResult(u, True, 0, 0.0, 0.0, float(u.mean()), imbalance,
                           c, rhs_modified, 0.0, 0.0, equation)

    if maxiter is None:
        maxiter = max(1000, 5 * n)

    def project(v):
        return v - v.mean()

    # Projected preconditioned conjugate gradient on the zero-mean subspace.
    u = np.zeros(grid.shape)
    r = b.copy()
    z = project(r / op.diag)
    p = z.copy()
    rz = float(np.sum(r * z))
    converged = False
    it = 0
    for it in range(1, maxiter + 1):
        Ap = project(op.apply(p))              # keep Ap orthogonal to constants
        pAp = float(np.sum(p * Ap))
        if not np.isfinite(pAp) or pAp <= 0.0:
            raise ArithmeticError(f"PCG breakdown: p^T A p = {pAp!r} at iteration {it}")
        alpha = rz / pAp
        u += alpha * p
        r -= alpha * Ap
        if float(np.linalg.norm(r)) <= tol * bnorm:
            converged = True
            break
        z = project(r / op.diag)               # projected diagonal preconditioner
        rz_new = float(np.sum(r * z))
        p = project(z + (rz_new / rz) * p)     # search direction stays zero-mean
        rz = rz_new

    # Honest a-posteriori diagnostics, recomputed from the final iterate.
    r_true = op.apply(u) - b
    residual_norm = float(np.linalg.norm(r_true))
    flux_residual = r_true * vol               # per-cell flux-balance error
    return SolveResult(
        u=u,
        converged=converged,
        iterations=it,
        residual_norm=residual_norm,
        residual_rel=residual_norm / bnorm,
        solution_mean=float(u.mean()),
        rhs_imbalance=imbalance,
        rhs_mean_removed=c,
        rhs_modified=rhs_modified,
        flux_imbalance_max=float(np.abs(flux_residual).max()),
        boundary_flux=float((op.apply(u) * vol).sum()),
        equation=equation,
    )
