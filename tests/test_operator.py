import numpy as np
import pytest

from neumann_fv import FVDiffusion, Grid2D, constrained_kkt_solve
from neumann_fv.solver import weighted_mean


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (4, 4), (5, 3)])
def test_matrixfree_matches_independent_sparse_assembly(shape):
    rng = np.random.default_rng(100 + shape[0] * 17 + shape[1])
    grid = Grid2D(*shape, lx=1.7, ly=0.9)
    k = np.exp(rng.normal(size=(grid.nx, grid.ny)))
    op = FVDiffusion(grid, k)
    matrix = op.assemble_csr()
    x = rng.normal(size=grid.n)

    if grid.n == 1:
        assert np.max(np.abs(op.matvec(x))) <= 1e-14
        assert matrix[0, 0] == pytest.approx(0.0)
    else:
        assert np.max(np.abs(op.matvec(x) - matrix @ x)) <= 1e-13
    assert np.max(np.abs(matrix @ np.ones(grid.n))) <= 1e-12
    assert np.max(np.abs(matrix.T @ np.ones(grid.n))) <= 1e-12
    assert np.abs(matrix.toarray() - matrix.T.toarray()).max() <= 1e-14


def test_internal_face_fluxes_cancel_pairwise():
    grid = Grid2D(3, 2)
    rng = np.random.default_rng(3)
    op = FVDiffusion(grid, np.exp(rng.normal(size=(grid.nx, grid.ny))))
    x = rng.normal(size=grid.n)
    flux, positive_cells, negative_cells = op.face_fluxes(x).internal_flux_vectors()
    balance = np.zeros(grid.n)
    np.add.at(balance, positive_cells, -flux)
    np.add.at(balance, negative_cells, flux)
    assert np.max(np.abs(balance / (grid.dx * grid.dy) - op.matvec(x))) <= 1e-14


def test_harmonic_mean_face_coefficients():
    grid = Grid2D(2, 1, lx=2.0, ly=1.0)
    k = np.array([[2.0], [6.0]])
    op = FVDiffusion(grid, k)
    expected_harmonic = 3.0
    assert op.tx[1, 0] == pytest.approx(expected_harmonic * grid.dy / grid.dx)


def test_independent_kkt_constrained_system_matches_matrixfree_solution():
    grid = Grid2D(3, 3, lx=1.2, ly=0.8)
    rng = np.random.default_rng(44)
    k = np.exp(rng.normal(size=(grid.nx, grid.ny)))
    op = FVDiffusion(grid, k)
    exact = rng.normal(size=grid.n)
    exact -= exact.mean()
    b = op.matvec(exact)

    x_sparse = constrained_kkt_solve(
        op.assemble_csr().toarray(),
        b,
        grid.cell_volumes,
    )
    x_free = op.matvec(x_sparse)

    assert weighted_mean(grid, x_sparse) == pytest.approx(0.0, abs=1e-13)
    assert np.linalg.norm(x_free - b) <= 1e-12
    assert np.linalg.norm(x_sparse - exact, ord=np.inf) <= 1e-11


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_illegal_coefficients_are_rejected(bad):
    grid = Grid2D(2, 2)
    k = np.ones((grid.nx, grid.ny))
    k[0, 0] = bad
    with pytest.raises(ValueError, match="strictly positive|positive"):
        FVDiffusion(grid, k)


def test_explicit_neighbor_formula_has_negative_laplacian_sign():
    grid = Grid2D(3, 2, lx=1.1, ly=0.7)
    rng = np.random.default_rng(77)
    k = np.exp(rng.normal(size=(grid.nx, grid.ny)))
    op = FVDiffusion(grid, k)
    u = rng.normal(size=grid.n)
    values = u.reshape(grid.nx, grid.ny)
    reference = np.zeros_like(values)

    def harmonic(a, b):
        return 2.0 * a * b / (a + b)

    for i in range(grid.nx):
        for j in range(grid.ny):
            integrated_laplacian_flux = 0.0
            if i > 0:
                integrated_laplacian_flux -= harmonic(k[i - 1, j], k[i, j]) * (
                    grid.dy / grid.dx
                ) * (values[i, j] - values[i - 1, j])
            if i + 1 < grid.nx:
                integrated_laplacian_flux += harmonic(k[i, j], k[i + 1, j]) * (
                    grid.dy / grid.dx
                ) * (values[i + 1, j] - values[i, j])
            if j > 0:
                integrated_laplacian_flux -= harmonic(k[i, j - 1], k[i, j]) * (
                    grid.dx / grid.dy
                ) * (values[i, j] - values[i, j - 1])
            if j + 1 < grid.ny:
                integrated_laplacian_flux += harmonic(k[i, j], k[i, j + 1]) * (
                    grid.dx / grid.dy
                ) * (values[i, j + 1] - values[i, j])
            reference[i, j] = -integrated_laplacian_flux / (grid.dx * grid.dy)

    assert np.max(np.abs(reference.ravel() - op.matvec(u))) <= 1e-13


def test_boundary_face_fluxes_are_explicitly_zero():
    grid = Grid2D(4, 3)
    rng = np.random.default_rng(88)
    op = FVDiffusion(grid, np.exp(rng.normal(size=(grid.nx, grid.ny))))
    faces = op.face_fluxes(rng.normal(size=grid.n))
    assert np.linalg.norm(faces.fx[[0, -1], :]) == 0.0
    assert np.linalg.norm(faces.fy[:, [0, -1]]) == 0.0
