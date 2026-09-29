"""Smooth manufactured solution with homogeneous Neumann boundary conditions.

u(x, y) = cos(pi x / Lx) cos(pi y / Ly) satisfies  du/dn = 0  on the whole
boundary of [0, Lx] x [0, Ly].  Two coefficient fields are provided:

* k_constant : k = k0            -> f = k0 (a^2 + b^2) u          (analytic)
* k_linear   : k = 1 + x/(2Lx) + y/(4Ly)
             -> f = k (a^2+b^2) u + kx a sin(ax) cos(by) + ky b cos(ax) sin(by)

Right-hand sides are discretized as exact cell averages (analytic for
constant k, high-order Gauss-Legendre quadrature for variable k), so the
discrete compatibility condition holds up to roundoff / quadrature error.
"""
from __future__ import annotations

import numpy as np

from fv_neumann import Grid


def exact_solution_avg(grid: Grid) -> np.ndarray:
    """Exact cell averages of cos(pi x / Lx) cos(pi y / Ly)."""
    a = np.pi / grid.Lx
    b = np.pi / grid.Ly
    xe = np.linspace(0.0, grid.Lx, grid.nx + 1)
    ye = np.linspace(0.0, grid.Ly, grid.ny + 1)
    cx = (np.sin(a * xe[1:]) - np.sin(a * xe[:-1])) / (a * grid.dx)
    cy = (np.sin(b * ye[1:]) - np.sin(b * ye[:-1])) / (b * grid.dy)
    return np.outer(cy, cx)                      # shape (ny, nx)


def k_constant(grid: Grid, k0: float = 2.0) -> np.ndarray:
    return np.full(grid.shape, float(k0))


def f_constant_k(grid: Grid, k0: float = 2.0) -> np.ndarray:
    """Analytic cell averages of f = -div(k0 grad u)."""
    a2 = (np.pi / grid.Lx) ** 2
    b2 = (np.pi / grid.Ly) ** 2
    return k0 * (a2 + b2) * exact_solution_avg(grid)


def k_linear(grid: Grid) -> np.ndarray:
    """k(x, y) = 1 + x/(2 Lx) + y/(4 Ly) sampled at cell centers (> 0)."""
    X, Y = np.meshgrid(grid.xc, grid.yc)
    return 1.0 + 0.5 * X / grid.Lx + 0.25 * Y / grid.Ly


def _f_linear_k_fun(grid: Grid):
    a = np.pi / grid.Lx
    b = np.pi / grid.Ly

    def fun(X, Y):
        k = 1.0 + 0.5 * X / grid.Lx + 0.25 * Y / grid.Ly
        u = np.cos(a * X) * np.cos(b * Y)
        # f = -div(k grad u) = k (a^2+b^2) u + kx a sin(aX) cos(bY)
        #                                   + ky b cos(aX) sin(bY)
        return (
            k * (a * a + b * b) * u
            + (0.5 / grid.Lx) * a * np.sin(a * X) * np.cos(b * Y)
            + (0.25 / grid.Ly) * b * np.cos(a * X) * np.sin(b * Y)
        )

    return fun


def cell_average(grid: Grid, fun, nq: int = 8) -> np.ndarray:
    """Cell averages of fun(x, y) by tensor Gauss-Legendre quadrature."""
    xi, wi = np.polynomial.legendre.leggauss(nq)
    xq = grid.xc[:, None] + 0.5 * grid.dx * xi[None, :]      # (nx, nq)
    yq = grid.yc[:, None] + 0.5 * grid.dy * xi[None, :]      # (ny, nq)
    X = np.broadcast_to(xq[None, None, :, :], (grid.ny, nq, grid.nx, nq))
    Y = np.broadcast_to(yq[:, :, None, None], (grid.ny, nq, grid.nx, nq))
    F = fun(X, Y)                                            # (j, q, i, p)
    return 0.25 * np.einsum("p,q,jqip->ji", wi, wi, F)


def f_linear_k(grid: Grid, nq: int = 8) -> np.ndarray:
    """Quadrature cell averages of f = -div(k grad u) for k_linear."""
    return cell_average(grid, _f_linear_k_fun(grid), nq)
