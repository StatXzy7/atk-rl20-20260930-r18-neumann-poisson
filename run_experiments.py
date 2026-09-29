#!/usr/bin/env python3
"""Reproducible numerical experiments for the pure-Neumann finite-volume solver."""

from __future__ import annotations

import json
import numpy as np

from neumann_fv import (
    FVDiffusion,
    Grid2D,
    IncompatibleRightHandSide,
    constrained_kkt_solve,
    smooth_problem_rhs,
    solve_neumann,
)
from neumann_fv.solver import prepare_rhs, weighted_mean


def result_summary(result) -> dict:
    return {
        "iterations": result.iterations,
        "converged": result.converged,
        "status": result.status,
        "true_residual": result.true_residual,
        "relative_true_residual": result.relative_true_residual,
        "solution_mean": result.solution_mean,
        "flux_conservation_error": result.conservation_error,
        "removed_mean": result.removed_mean,
    }


def discrete_round_trip() -> dict:
    grid = Grid2D(16, 12, lx=1.0, ly=1.0)
    rng = np.random.default_rng(314159)
    k = np.exp(rng.uniform(-1.5, 1.5, size=(grid.nx, grid.ny)))
    op = FVDiffusion(grid, k)
    exact = rng.normal(size=grid.n)
    exact -= weighted_mean(grid, exact)
    b = op.matvec(exact)
    result = solve_neumann(op, b, rtol=1e-12, atol=1e-13)
    matrix = op.assemble_csr()
    probe = rng.normal(size=grid.n)
    matrix_vs_free = float(np.max(np.abs(matrix @ probe - op.matvec(probe))))
    kkt = constrained_kkt_solve(matrix.toarray(), b, grid.cell_volumes)
    return {
        "grid": [grid.nx, grid.ny],
        "k_min": float(k.min()),
        "k_max": float(k.max()),
        "rhs_mean": weighted_mean(grid, b),
        "recovered_linf_error": float(np.max(np.abs(result.x - exact))),
        "matrix_vs_matrixfree_max": matrix_vs_free,
        "kkt_vs_pcg_linf": float(np.max(np.abs(kkt - result.x))),
        "kkt_true_residual": float(np.linalg.norm(matrix @ kkt - b)),
        **result_summary(result),
    }


def smooth_convergence(anisotropic: bool = False) -> list[dict]:
    rows = []
    sizes = [8, 16, 32, 64, 128]
    previous_error = None
    previous_h = None
    for n in sizes:
        grid = (
            Grid2D(n, n // 2, lx=4.0, ly=0.2)
            if anisotropic
            else Grid2D(n, n)
        )
        f, k, exact = smooth_problem_rhs(grid)
        exact -= weighted_mean(grid, exact)
        original_mean = weighted_mean(grid, f)
        b, removed, _ = prepare_rhs(grid, f, mode="project", atol=1e-12, rtol=1e-10)
        op = FVDiffusion(grid, k)
        result = solve_neumann(op, b, rtol=5e-12, atol=1e-13, maxiter=10000)
        error = np.sqrt(np.mean((result.x - exact) ** 2))
        h = max(grid.dx, grid.dy)
        observed_order = (
            np.log(previous_error / error) / np.log(previous_h / h)
            if previous_error is not None
            else None
        )
        rows.append(
            {
                "grid": [grid.nx, grid.ny],
                "dx": grid.dx,
                "dy": grid.dy,
                "h_max": h,
                "quadrature_rhs_mean_before_projection": original_mean,
                "removed_mean": removed,
                "l2_error": float(error),
                "linf_error": float(np.max(np.abs(result.x - exact))),
                "observed_order": None if observed_order is None else float(observed_order),
                **result_summary(result),
            }
        )
        previous_error = error
        previous_h = h
    return rows


def special_cases() -> dict:
    rng = np.random.default_rng(271828)

    # Checkerboard 1e8 contrast is structured and remains tractable for Jacobi CG.
    grid = Grid2D(24, 24)
    ii = np.arange(grid.nx)[:, None]
    jj = np.arange(grid.ny)[None, :]
    k = np.where((ii + jj) % 2 == 0, 1.0, 1.0e8)
    op = FVDiffusion(grid, k)
    exact = rng.normal(size=grid.n)
    exact -= exact.mean()
    b = op.matvec(exact)
    high_contrast = solve_neumann(op, b, rtol=1e-12, atol=1e-13, maxiter=20000)

    agrid = Grid2D(40, 10, lx=4.0, ly=0.1)
    ak = np.exp(rng.normal(size=(agrid.nx, agrid.ny)))
    aop = FVDiffusion(agrid, ak)
    ax = rng.normal(size=agrid.n)
    ax -= weighted_mean(agrid, ax)
    ab = aop.matvec(ax)
    anisotropic = solve_neumann(aop, ab, rtol=1e-12, atol=1e-13, maxiter=10000)

    cgrid = Grid2D(8, 8)
    cop = FVDiffusion(cgrid, np.exp(rng.normal(size=(cgrid.nx, cgrid.ny))))
    constant = solve_neumann(cop, np.zeros(cgrid.n), rtol=1e-12)

    bad_cases = {}
    for name, bad in [("zero", 0.0), ("negative", -2.0), ("nan", np.nan), ("infinite", np.inf)]:
        bk = np.ones((cgrid.nx, cgrid.ny))
        bk[0, 0] = bad
        try:
            FVDiffusion(cgrid, bk)
        except ValueError as exc:
            bad_cases[name] = str(exc)

    incompatible = np.ones(cgrid.n)
    try:
        solve_neumann(cop, incompatible, compat_atol=1e-12, compat_rtol=0.0)
    except IncompatibleRightHandSide as exc:
        incompatible_error = str(exc)

    projected = solve_neumann(
        cop,
        incompatible,
        rhs_mode="project",
        compat_atol=1e-12,
        compat_rtol=0.0,
        rtol=1e-11,
    )

    return {
        "high_contrast": {
            "grid": [grid.nx, grid.ny],
            "contrast": 1.0e8,
            "recovered_linf_error": float(np.max(np.abs(high_contrast.x - exact))),
            **result_summary(high_contrast),
        },
        "anisotropic_grid": {
            "grid": [agrid.nx, agrid.ny],
            "dx": agrid.dx,
            "dy": agrid.dy,
            "aspect_ratio": agrid.dx / agrid.dy,
            "recovered_linf_error": float(np.max(np.abs(anisotropic.x - ax))),
            **result_summary(anisotropic),
        },
        "constant_field": {
            "grid": [cgrid.nx, cgrid.ny],
            "recovered_linf_error": float(np.max(np.abs(constant.x))),
            **result_summary(constant),
        },
        "illegal_coefficients": bad_cases,
        "incompatible_rhs_rejection": incompatible_error,
        "incompatible_rhs_explicit_projection": {
            "reported_removed_mean": projected.removed_mean,
            "actual_equation_mean": weighted_mean(cgrid, projected.b),
            "true_residual": projected.true_residual,
            "iterations": projected.iterations,
        },
    }


def main() -> None:
    report = {
        "discrete_round_trip": discrete_round_trip(),
        "smooth_isotropic_convergence": smooth_convergence(False),
        "smooth_anisotropic_convergence": smooth_convergence(True),
        "special_cases": special_cases(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
