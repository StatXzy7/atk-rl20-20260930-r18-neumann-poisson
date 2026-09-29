"""Manufactured solutions and quadrature for finite-volume convergence tests."""

from __future__ import annotations

import math

import numpy as np

from .grid import Grid2D


_GAUSS5_NODES = np.array([
    -0.5 * math.sqrt(5.0 + 2.0 * math.sqrt(10.0 / 7.0)) / 3.0,
    -0.5 * math.sqrt(5.0 - 2.0 * math.sqrt(10.0 / 7.0)) / 3.0,
    0.0,
    0.5 * math.sqrt(5.0 - 2.0 * math.sqrt(10.0 / 7.0)) / 3.0,
    0.5 * math.sqrt(5.0 + 2.0 * math.sqrt(10.0 / 7.0)) / 3.0,
])
_GAUSS5_WEIGHTS = np.array([
    (322.0 - 13.0 * math.sqrt(70.0)) / 900.0,
    (322.0 + 13.0 * math.sqrt(70.0)) / 900.0,
    128.0 / 225.0,
    (322.0 + 13.0 * math.sqrt(70.0)) / 900.0,
    (322.0 - 13.0 * math.sqrt(70.0)) / 900.0,
])


def smooth_u(x: np.ndarray, y: np.ndarray, lx: float = 1.0, ly: float = 1.0) -> np.ndarray:
    """Smooth field whose normal derivative vanishes on the rectangle boundary."""
    return np.cos(2.0 * np.pi * x / lx) * np.cos(2.0 * np.pi * y / ly)


def smooth_k(x: np.ndarray, y: np.ndarray, lx: float = 1.0, ly: float = 1.0) -> np.ndarray:
    return 1.0 + 0.5 * np.sin(np.pi * x / lx) * np.cos(np.pi * y / ly)


def smooth_source(
    x: np.ndarray,
    y: np.ndarray,
    lx: float = 1.0,
    ly: float = 1.0,
) -> np.ndarray:
    """f=-div(k grad u) for rectangle-aware smooth_u and smooth_k."""
    ax = 2.0 * np.pi / lx
    ay = 2.0 * np.pi / ly
    bx = np.pi / lx
    by = np.pi / ly

    cx = np.cos(ax * x)
    sx = np.sin(ax * x)
    cy = np.cos(ay * y)
    sy = np.sin(ay * y)
    u = cx * cy

    ux = -ax * sx * cy
    uy = -ay * cx * sy
    uxx = -(ax**2) * u
    uyy = -(ay**2) * u
    k = smooth_k(x, y, lx, ly)
    kx = 0.5 * bx * np.cos(bx * x) * np.cos(by * y)
    ky = -0.5 * by * np.sin(bx * x) * np.sin(by * y)
    return -(k * (uxx + uyy) + kx * ux + ky * uy)


def cell_average(
    grid: Grid2D,
    func,
    *,
    quadrature: int = 5,
) -> np.ndarray:
    """Integrate a scalar function to a cell-average value."""
    if quadrature != 5:
        raise NotImplementedError("five-point Gauss-Legendre quadrature is provided")

    x_edges = np.linspace(0.0, grid.lx, grid.nx + 1)
    y_edges = np.linspace(0.0, grid.ly, grid.ny + 1)
    result = np.zeros((grid.nx, grid.ny), dtype=float)

    for i in range(grid.nx):
        x0 = 0.5 * (x_edges[i] + x_edges[i + 1])
        x = x0 + 0.5 * grid.dx * _GAUSS5_NODES
        for j in range(grid.ny):
            y0 = 0.5 * (y_edges[j] + y_edges[j + 1])
            y = y0 + 0.5 * grid.dy * _GAUSS5_NODES
            xx, yy = np.meshgrid(x, y, indexing="ij")
            values = func(xx, yy, grid.lx, grid.ly)
            wx = _GAUSS5_WEIGHTS[:, None]
            wy = _GAUSS5_WEIGHTS[None, :]
            result[i, j] = float(np.sum(wx * wy * values) / 4.0)

    return result.ravel()


def smooth_problem_rhs(grid: Grid2D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cell-average source, sampled k, and sampled exact solution."""
    f = cell_average(grid, smooth_source)
    x, y = grid.cell_centers
    k = smooth_k(x, y, grid.lx, grid.ly).ravel()
    exact = smooth_u(x, y, grid.lx, grid.ly).ravel()
    return f, k, exact


def constrained_kkt_solve(
    matrix,
    b: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Solve [A -V; V^T 0][x; lambda]=[b;0] with dense NumPy elimination.

    This is intentionally separate from the matrix-free projected-CG path.
    """
    a = np.asarray(matrix, dtype=float)
    b = np.asarray(b, dtype=float)
    v = np.asarray(weights, dtype=float)
    n = a.shape[0]
    augmented = np.zeros((n + 1, n + 1), dtype=float)
    augmented[:n, :n] = a
    augmented[:n, n] = -v
    augmented[n, :n] = v
    rhs = np.concatenate((b, [0.0]))
    solution = np.linalg.solve(augmented, rhs)
    return solution[:n]
