"""Matrix-free and independently assembled finite-volume diffusion operator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .grid import Grid2D


def harmonic_mean(left: np.ndarray | float, right: np.ndarray | float) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if np.any(left <= 0.0) or np.any(right <= 0.0):
        raise ValueError("harmonic mean requires positive values")
    return 2.0 * left * right / (left + right)


def normalize_k(grid: Grid2D, k: np.ndarray | float) -> np.ndarray:
    values = np.asarray(k, dtype=float)
    if values.ndim == 0:
        values = np.full(grid.n, float(values))
    if values.shape != (grid.nx, grid.ny):
        if values.size == grid.n:
            values = values.reshape(grid.nx, grid.ny)
        else:
            raise ValueError(f"k must have shape ({grid.nx}, {grid.ny})")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("k must be finite and strictly positive on every cell")
    return values


@dataclass(frozen=True)
class FaceFluxData:
    grid: Grid2D
    k: np.ndarray
    tx: np.ndarray
    ty: np.ndarray
    fx: np.ndarray
    fy: np.ndarray

    def internal_flux_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return face-normal q and cells on the positive/negative normal sides."""
        fx = self.fx[1:-1, :]
        fy = self.fy[:, 1:-1]

        i = np.arange(1, self.grid.nx)[:, None]
        j = np.arange(self.grid.ny)[None, :]
        positive_normal_cell = i * self.grid.ny + j
        negative_normal_cell = positive_normal_cell - self.grid.ny

        i = np.arange(self.grid.nx)[:, None]
        j = np.arange(1, self.grid.ny)[None, :]
        positive_normal_cell_y = i * self.grid.ny + j
        negative_normal_cell_y = positive_normal_cell_y - 1

        flux = np.concatenate((fx.ravel(), fy.ravel()))
        positive_cells = np.concatenate(
            (positive_normal_cell.ravel(), positive_normal_cell_y.ravel())
        )
        negative_cells = np.concatenate(
            (negative_normal_cell.ravel(), negative_normal_cell_y.ravel())
        )
        return flux, positive_cells, negative_cells

    def conservation_error(self, b_cell: np.ndarray | None = None) -> dict[str, float]:
        """Global and local balance errors for the stored face fluxes.

        With homogeneous Neumann faces the boundary flux is exactly zero, so the
        global residual is sum(cell residual). It is also the cancellation error
        of all internal face contributions when ``b_cell`` is zero.
        """
        g = self.grid
        residual = np.zeros(g.n, dtype=float)
        fx_2d = self.fx.reshape(g.nx + 1, g.ny)
        fy_2d = self.fy.reshape(g.nx, g.ny + 1)
        idx = np.arange(g.nx * g.ny).reshape(g.nx, g.ny)

        if g.nx > 1:
            np.add.at(residual, idx[1:, :].ravel(), -fx_2d[1:-1, :].ravel())
            np.add.at(residual, idx[:-1, :].ravel(), fx_2d[1:-1, :].ravel())
        if g.ny > 1:
            np.add.at(residual, idx[:, 1:].ravel(), -fy_2d[:, 1:-1].ravel())
            np.add.at(residual, idx[:, :-1].ravel(), fy_2d[:, 1:-1].ravel())

        if b_cell is not None:
            residual -= self.grid.cell_volumes * np.asarray(b_cell, dtype=float)

        return {
            "global_abs": abs(float(np.sum(residual))),
            "max_abs": float(np.max(np.abs(residual))),
            "l2": float(np.linalg.norm(residual)),
        }


class FVDiffusion:
    """Cell-centered finite-volume diffusion operator on a uniform rectangle."""

    def __init__(self, grid: Grid2D, k: np.ndarray | float) -> None:
        self.grid = grid
        self.k = normalize_k(grid, k)
        g = grid

        k_xface = np.zeros((g.nx + 1, g.ny), dtype=float)
        k_yface = np.zeros((g.nx, g.ny + 1), dtype=float)
        k_xface[1:-1, :] = harmonic_mean(self.k[:-1, :], self.k[1:, :])
        k_yface[:, 1:-1] = harmonic_mean(self.k[:, :-1], self.k[:, 1:])

        self.tx = k_xface * g.areas_x / g.dx
        self.ty = k_yface * g.areas_y / g.dy

    def face_fluxes(self, u: np.ndarray) -> FaceFluxData:
        """Compute q.n=-k(grad u).n on every x/y oriented face.

        The stored array direction is lower-x to higher-x and lower-y to
        higher-y. Boundary faces are zero by the homogeneous Neumann condition.
        """
        g = self.grid
        uu = np.asarray(u, dtype=float).reshape(g.nx, g.ny)
        fx = np.zeros((g.nx + 1, g.ny), dtype=float)
        fy = np.zeros((g.nx, g.ny + 1), dtype=float)
        if g.nx > 1:
            fx[1:-1, :] = -self.tx[1:-1, :] * (uu[1:, :] - uu[:-1, :])
        if g.ny > 1:
            fy[:, 1:-1] = -self.ty[:, 1:-1] * (uu[:, 1:] - uu[:, :-1])
        return FaceFluxData(g, self.k, self.tx, self.ty, fx, fy)

    def matvec(self, u: np.ndarray) -> np.ndarray:
        """Matrix-free Au assembled from face flux divergence."""
        g = self.grid
        faces = self.face_fluxes(u)
        result = np.zeros(g.n, dtype=float)
        inv_volume = 1.0 / (g.dx * g.dy)
        fx = faces.fx
        fy = faces.fy
        idx = np.arange(g.n).reshape(g.nx, g.ny)

        if g.nx > 1:
            result[idx[1:, :].ravel()] -= fx[1:-1, :].ravel() * inv_volume
            result[idx[:-1, :].ravel()] += fx[1:-1, :].ravel() * inv_volume
        if g.ny > 1:
            result[idx[:, 1:].ravel()] -= fy[:, 1:-1].ravel() * inv_volume
            result[idx[:, :-1].ravel()] += fy[:, 1:-1].ravel() * inv_volume
        return result

    @property
    def diagonal(self) -> np.ndarray:
        g = self.grid
        diag = np.zeros(g.n, dtype=float)
        inv_volume = 1.0 / (g.dx * g.dy)
        diag_2d = diag.reshape(g.nx, g.ny)
        diag_2d[:-1, :] += self.tx[1:-1, :] * inv_volume
        diag_2d[1:, :] += self.tx[1:-1, :] * inv_volume
        diag_2d[:, :-1] += self.ty[:, 1:-1] * inv_volume
        diag_2d[:, 1:] += self.ty[:, 1:-1] * inv_volume
        return diag

    @property
    def preconditioner_diagonal(self) -> np.ndarray:
        """Diagonal used only inside projected PCG for the single-cell null case."""
        diag = self.diagonal.copy()
        if self.grid.n == 1:
            diag[0] = 1.0
        return diag

    def assemble_csr(self) -> sparse.csr_matrix:
        """Independently accumulate a sparse matrix from local face stencils."""
        g = self.grid
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        inv_volume = 1.0 / (g.dx * g.dy)

        def add(row: int, col: int, value: float) -> None:
            rows.append(int(row))
            cols.append(int(col))
            vals.append(float(value))

        for i in range(g.nx - 1):
            for j in range(g.ny):
                left = i * g.ny + j
                right = (i + 1) * g.ny + j
                transmissibility = float(self.tx[i + 1, j]) * inv_volume
                add(right, right, transmissibility)
                add(right, left, -transmissibility)
                add(left, left, transmissibility)
                add(left, right, -transmissibility)

        for i in range(g.nx):
            for j in range(g.ny - 1):
                bottom = i * g.ny + j
                top = i * g.ny + j + 1
                transmissibility = float(self.ty[i, j + 1]) * inv_volume
                add(top, top, transmissibility)
                add(top, bottom, -transmissibility)
                add(bottom, bottom, transmissibility)
                add(bottom, top, -transmissibility)

        matrix = sparse.coo_matrix(
            (np.asarray(vals, dtype=float),
             (
                 np.asarray(rows, dtype=np.int64),
                 np.asarray(cols, dtype=np.int64),
             ),),
            shape=(g.n, g.n),
        ).tocsr()
        return matrix
