import numpy as np
import pytest

from neumann_fv import (
    FVDiffusion,
    Grid2D,
    IncompatibleRightHandSide,
    smooth_problem_rhs,
    solve_neumann,
)
from neumann_fv.solver import prepare_rhs, projected_pcg, weighted_mean


def test_discrete_manufactured_solution_round_trip():
    grid = Grid2D(7, 5, lx=1.3, ly=0.7)
    rng = np.random.default_rng(2026)
    k = np.exp(0.8 * rng.normal(size=(grid.nx, grid.ny)))
    op = FVDiffusion(grid, k)
    exact = rng.normal(size=grid.n)
    exact -= weighted_mean(grid, exact)
    b = op.matvec(exact)

    result = solve_neumann(op, b, rtol=1e-12, atol=1e-13)

    assert result.converged
    assert np.linalg.norm(op.matvec(result.x) - b, ord=np.inf) <= 1e-10
    assert abs(result.solution_mean) <= 1e-14
    assert result.conservation_error <= 1e-12
    assert np.linalg.norm(result.x - exact, ord=np.inf) <= 1e-10


def test_constant_field_is_exact_nullspace_representative():
    grid = Grid2D(4, 3)
    rng = np.random.default_rng(5)
    op = FVDiffusion(grid, np.exp(rng.normal(size=(grid.nx, grid.ny))))
    result = solve_neumann(op, np.zeros(grid.n), rtol=1e-12)

    assert result.iterations == 0
    assert np.all(result.x == 0.0)
    assert result.true_residual == 0.0
    assert np.linalg.norm(op.matvec(np.full(grid.n, 3.25))) <= 1e-13


def test_incompatible_rhs_is_rejected_not_silently_modified():
    grid = Grid2D(3, 3)
    op = FVDiffusion(grid, 1.0)
    b = np.ones(grid.n)
    with pytest.raises(IncompatibleRightHandSide) as info:
        solve_neumann(op, b)
    assert info.value.mean == pytest.approx(1.0)


def test_explicit_projection_reports_removed_mean_and_modified_equation():
    grid = Grid2D(4, 2)
    op = FVDiffusion(grid, 2.0)
    b = op.matvec(np.array([1.0, -1.0] * 4)) + 0.375
    result = solve_neumann(op, b, rhs_mode="project")

    assert result.removed_mean == pytest.approx(0.375)
    assert np.max(np.abs(result.b - (b - 0.375))) <= 1e-14
    assert abs(weighted_mean(grid, result.original_b - result.b) - 0.375) <= 1e-14
    assert np.linalg.norm(op.matvec(result.x) - result.b) <= 1e-11


def test_projected_search_directions_stay_zero_mean():
    grid = Grid2D(6, 5)
    rng = np.random.default_rng(91)
    op = FVDiffusion(grid, np.exp(rng.normal(size=(grid.nx, grid.ny))))
    exact = rng.normal(size=grid.n)
    exact -= exact.mean()
    b = op.matvec(exact)
    x0 = rng.normal(size=grid.n)
    result = projected_pcg(op, b, x0, rtol=1e-12, atol=0.0)
    assert result.converged
    assert abs(result.solution_mean) <= 1e-14


@pytest.mark.parametrize("nx,ny,lx,ly", [(8, 8, 1.0, 1.0), (16, 8, 2.0, 0.5)])
def test_smooth_mms_converges_at_second_order(nx, ny, lx, ly):
    grid = Grid2D(nx, ny, lx=lx, ly=ly)
    f, k, exact = smooth_problem_rhs(grid)
    exact = exact - exact.mean()
    projected, removed, _ = prepare_rhs(grid, f, mode="project", atol=1e-11, rtol=1e-9)
    op = FVDiffusion(grid, k)
    result = solve_neumann(op, projected, rtol=1e-12, atol=1e-13)
    h = max(grid.dx, grid.dy)
    error = np.sqrt(np.mean((result.x - exact) ** 2))

    assert abs(removed) <= 1e-8
    assert result.true_residual <= 1e-9
    assert error <= 2.0 * h**2


def test_nonfinite_rhs_rejected():
    grid = Grid2D(2, 2)
    op = FVDiffusion(grid, 1.0)
    b = np.zeros(grid.n)
    b[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        solve_neumann(op, b)


def test_single_cell_compatible_nullspace():
    grid = Grid2D(1, 1)
    op = FVDiffusion(grid, 3.0)
    result = solve_neumann(op, np.zeros(1), rtol=1e-12)
    assert result.x[0] == 0.0
    assert result.iterations == 0
    assert result.true_residual == 0.0
    assert result.conservation_error == 0.0


def test_single_cell_incompatible_rhs_rejected_and_projection_reported():
    grid = Grid2D(1, 1)
    op = FVDiffusion(grid, 3.0)
    with pytest.raises(IncompatibleRightHandSide):
        solve_neumann(op, np.array([0.5]), compat_rtol=0.0)
    result = solve_neumann(
        op,
        np.array([0.5]),
        rhs_mode="project",
        compat_rtol=0.0,
    )
    assert result.removed_mean == pytest.approx(0.5)
    assert result.b[0] == 0.0
    assert result.x[0] == 0.0


def test_projected_pcg_search_directions_cannot_drift(monkeypatch):
    grid = Grid2D(7, 6, lx=1.1, ly=0.8)
    rng = np.random.default_rng(4242)
    op = FVDiffusion(grid, np.exp(rng.normal(size=(grid.nx, grid.ny))))
    exact = rng.normal(size=grid.n)
    exact -= weighted_mean(grid, exact)
    b = op.matvec(exact)
    directions = []
    original_matvec = op.matvec

    def recording_matvec(v):
        directions.append(v.copy())
        return original_matvec(v)

    monkeypatch.setattr(op, "matvec", recording_matvec)
    result = projected_pcg(op, b, rng.normal(size=grid.n), rtol=1e-11)

    assert result.converged
    assert directions
    assert max(abs(weighted_mean(grid, p)) for p in directions) <= 1e-14
