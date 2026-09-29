"""Tests for the pure-Neumann finite-volume solver.

Every test checks physical/algebraic quantities (true residuals, solution
means, flux balances, recovery errors) directly -- never just solver flags.
"""
import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import linalg as spla

import manufactured as m
from fv_neumann import (
    FVOperator,
    Grid,
    IncompatibleRHSError,
    InvalidCoefficientError,
    solve,
)

rng = np.random.default_rng(20260930)


def maxabs(a):
    return float(np.abs(a).max())


def random_k(shape, lo=0.5, hi=2.0):
    return rng.uniform(lo, hi, size=shape)


# --------------------------------------------------------------------------
# coefficient validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_invalid_coefficient_rejected(bad):
    g = Grid(4, 3)
    k = np.ones(g.shape)
    k[1, 2] = bad
    with pytest.raises(InvalidCoefficientError):
        FVOperator(g, k)


def test_coefficient_wrong_shape_rejected():
    with pytest.raises(InvalidCoefficientError):
        FVOperator(Grid(4, 3), np.ones((3, 4) + (1,)))


# --------------------------------------------------------------------------
# operator structure: null space, symmetry, pairwise flux cancellation
# --------------------------------------------------------------------------

def test_constant_field_is_in_null_space():
    g = Grid(8, 6)
    op = FVOperator(g, random_k(g.shape))
    ones = np.ones(g.shape)
    assert np.linalg.norm(op.apply(ones)) == pytest.approx(0.0, abs=1e-13)
    A = op.to_sparse()
    assert maxabs(A @ np.ones(g.ncells)) < 1e-12


def test_sparse_assembly_symmetric_and_rank_deficient_by_one():
    g = Grid(6, 5)
    op = FVOperator(g, random_k(g.shape))
    A = op.to_sparse()
    # symmetry <=> interior face fluxes cancel pairwise
    assert (A - A.T).nnz == 0
    # row sums vanish (pure Neumann, no boundary elimination)
    assert maxabs(A @ np.ones(g.ncells)) < 1e-12
    # null space is exactly span{1}: rank n-1
    assert np.linalg.matrix_rank(A.toarray(), tol=1e-10) == g.ncells - 1


def test_matrix_free_matches_independent_sparse_assembly():
    g = Grid(9, 7, 1.0, 0.6)          # anisotropic spacing on purpose
    op = FVOperator(g, random_k(g.shape))
    A = op.to_sparse()
    for _ in range(5):
        u = rng.standard_normal(g.shape)
        assert np.allclose(op.apply(u).ravel(), A @ u.ravel(), rtol=1e-13, atol=1e-13)


def test_operator_is_symmetric_positive_semidefinite():
    g = Grid(7, 7)
    op = FVOperator(g, random_k(g.shape))
    for _ in range(5):
        u = rng.standard_normal(g.shape)
        v = rng.standard_normal(g.shape)
        assert np.sum(u * op.apply(v)) == pytest.approx(np.sum(v * op.apply(u)), rel=1e-12)
        assert np.sum(u * op.apply(u)) >= -1e-12


# --------------------------------------------------------------------------
# discrete manufactured solution: f = A u* for a zero-mean grid function
# --------------------------------------------------------------------------

@pytest.mark.parametrize("nx,ny,Lx,Ly", [(16, 16, 1.0, 1.0), (24, 12, 1.0, 0.5), (12, 24, 0.25, 1.0)])
def test_discrete_manufactured_solution_recovered(nx, ny, Lx, Ly):
    g = Grid(nx, ny, Lx, Ly)
    op = FVOperator(g, random_k(g.shape))
    u_star = rng.standard_normal(g.shape)
    u_star -= u_star.mean()                      # arbitrary zero-mean grid function
    f = op.apply(u_star)                         # right-hand side from the operator
    res = solve(op, f, tol=1e-12)
    err = maxabs(res.u - u_star) / maxabs(u_star)
    assert res.converged
    assert err < 1e-8
    assert abs(res.solution_mean) < 1e-12
    assert res.residual_rel < 1e-11
    assert res.flux_imbalance_max < 1e-10
    assert abs(res.boundary_flux) < 1e-11
    assert not res.rhs_modified


def test_constant_field_recovered_up_to_null_space():
    g = Grid(10, 8)
    op = FVOperator(g, random_k(g.shape))
    u_star = np.full(g.shape, 3.7)
    f = op.apply(u_star)
    assert maxabs(f) < 1e-13                       # constant lies in the null space
    res = solve(op, f)
    assert res.iterations == 0
    assert maxabs(res.u) == pytest.approx(0.0, abs=1e-15)
    # the representative differs from u* exactly by the mean of u*
    assert np.allclose(u_star - res.u, 3.7)


# --------------------------------------------------------------------------
# compatibility condition: rejection and reported projection
# --------------------------------------------------------------------------

def test_incompatible_rhs_rejected_in_strict_mode():
    g = Grid(8, 8)
    op = FVOperator(g, random_k(g.shape))
    f = np.ones(g.shape)                          # sum(f) V = 1 != 0
    with pytest.raises(IncompatibleRHSError, match="compatibility"):
        solve(op, f, mode="strict")


def test_projection_mode_reports_and_solves_modified_equation():
    g = Grid(12, 10)
    op = FVOperator(g, random_k(g.shape))
    u_star = rng.standard_normal(g.shape)
    u_star -= u_star.mean()
    f_compatible = op.apply(u_star)
    shift = 2.5
    f = f_compatible + shift                      # incompatible: mean(f) = shift

    res = solve(op, f, mode="project", tol=1e-12)
    # the modification is reported, not silent
    assert res.rhs_modified
    assert res.rhs_mean_removed == pytest.approx(shift, rel=1e-12)
    assert res.rhs_imbalance == pytest.approx(shift * g.ncells * g.cell_volume, rel=1e-12)
    assert "f - (" in res.equation
    # the returned field solves the *modified* equation A u = f - c
    assert maxabs(op.apply(res.u) - (f - shift)) < 1e-10 * maxabs(f)
    # and recovers the same representative as the compatible problem
    assert maxabs(res.u - u_star) < 1e-8
    # the original equation is indeed not solved (residual carries the mean)
    assert maxabs(op.apply(res.u) - f) > 1.0


def test_strict_mode_accepts_roundoff_level_imbalance():
    g = Grid(10, 10)
    op = FVOperator(g, random_k(g.shape))
    u_star = rng.standard_normal(g.shape)
    u_star -= u_star.mean()
    f = op.apply(u_star)                          # imbalance ~ 1e-17
    res = solve(op, f, mode="strict")
    assert not res.rhs_modified
    assert abs(res.rhs_mean_removed) < 1e-13


# --------------------------------------------------------------------------
# cross-check against an independently solved constrained linear system
# --------------------------------------------------------------------------

def test_matches_direct_saddle_point_solve():
    g = Grid(6, 5)
    op = FVOperator(g, random_k(g.shape))
    u_star = rng.standard_normal(g.shape)
    u_star -= u_star.mean()
    f = op.apply(u_star)
    A = op.to_sparse()
    n = g.ncells
    e = np.ones(n)
    M = sparse.bmat([[A, e[:, None]], [e[None, :], None]], format="csr")
    rhs = np.concatenate([f.ravel(), [0.0]])
    u_ref = spla.spsolve(M, rhs)[:n].reshape(g.shape)
    res = solve(op, f, tol=1e-13)
    assert maxabs(res.u - u_ref) < 1e-8


# --------------------------------------------------------------------------
# projected PCG stays out of the constant subspace
# --------------------------------------------------------------------------

def test_pcg_does_not_drift_into_constant_subspace():
    g = Grid(32, 32)
    k = 10.0 ** rng.uniform(-2.0, 2.0, size=g.shape)   # contrast 1e4
    op = FVOperator(g, k)
    u_star = rng.standard_normal(g.shape)
    u_star -= u_star.mean()
    f = op.apply(u_star)
    res = solve(op, f, tol=1e-9)
    assert res.converged
    assert abs(res.solution_mean) < 1e-10
    assert abs(res.boundary_flux) < 1e-9
    err = maxabs(res.u - u_star) / maxabs(u_star)
    assert err < 1e-6


# --------------------------------------------------------------------------
# smooth manufactured solution: second-order grid convergence
# --------------------------------------------------------------------------

def _convergence_rates(kfun, ffun, grids, mode="strict"):
    errs = []
    for n in grids:
        g = Grid(n, n)
        op = FVOperator(g, kfun(g))
        f = ffun(g)
        res = solve(op, f, mode=mode, tol=1e-11)
        assert res.converged
        u_ex = m.exact_solution_avg(g)
        u_ex -= u_ex.mean()                       # fix the same representative
        errs.append(maxabs(res.u - u_ex))
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    return errs, rates


def test_smooth_convergence_constant_k():
    _, rates = _convergence_rates(m.k_constant, m.f_constant_k, [16, 32, 64])
    assert all(r > 1.9 for r in rates)


def test_smooth_convergence_variable_k():
    _, rates = _convergence_rates(m.k_linear, m.f_linear_k, [16, 32, 64],
                                  mode="project")
    assert all(r > 1.9 for r in rates)
