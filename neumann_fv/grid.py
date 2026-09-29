"""Rectilinear cell-centered grids for two-dimensional finite-volume tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid2D:
    """A logically rectangular 2D grid with constant spacing in each axis.

    Cells are indexed in row-major array order: ``cell = i * ny + j``, where
    ``i`` is the x-direction index and ``j`` the y-direction index.
    """

    nx: int
    ny: int
    lx: float = 1.0
    ly: float = 1.0

    def __post_init__(self) -> None:
        if self.nx < 1 or self.ny < 1:
            raise ValueError("nx and ny must be positive")
        if not np.isfinite(self.lx) or not np.isfinite(self.ly):
            raise ValueError("lx and ly must be finite")
        if self.lx <= 0.0 or self.ly <= 0.0:
            raise ValueError("lx and ly must be positive")

    @property
    def n(self) -> int:
        return self.nx * self.ny

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny

    @property
    def cell_volumes(self) -> np.ndarray:
        return np.full(self.n, self.dx * self.dy, dtype=float)

    @property
    def areas_x(self) -> np.ndarray:
        """Areas of faces whose normal is in the x direction."""
        return np.full((self.nx + 1, self.ny), self.dy, dtype=float)

    @property
    def areas_y(self) -> np.ndarray:
        """Areas of faces whose normal is in the y direction."""
        return np.full((self.nx, self.ny + 1), self.dx, dtype=float)

    @property
    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        x = (np.arange(self.nx) + 0.5) * self.dx
        y = (np.arange(self.ny) + 0.5) * self.dy
        xx, yy = np.meshgrid(x, y, indexing="ij")
        return xx, yy

