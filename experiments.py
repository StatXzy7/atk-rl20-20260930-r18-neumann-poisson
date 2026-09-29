"""Reproducible experiments for the pure-Neumann FV solver.

Run:  python3 experiments.py
Every case reports the true (recomputed) residual, the solution mean, the
flux-conservation error and the iteration count -- never just a success flag.
"""
import numpy as np

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


def line(name, res, extra=""):
    print(f"  {name:<26} it={res.iterations:>5} conv={int(res.converged)} "
          f"relres={res.residual_rel:.3e} mean(u)={res.solution_mean:+.2e} "
          f"fluximb={res.flux_imbalance_max:.2e} bndflux={res.boundary_flux:+.2e} {extra}")


def hdr(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


# ---------------------------------------------------------------------------
hdr("1. Smooth manufactured solution u=cos(pi x)cos(pi y), constant k=2 -- grid convergence")
# ---------------------------------------------------------------------------
print("  f from analytic cell averages; discrete compatibility residual sum(f)*V reported per grid")
errs = []
for n in [16, 32, 64, 128, 256]:
    g = Grid(n, n)
    op = FVOperator(g, m.k_constant(g, 2.0))
    f = m.f_constant_k(g, 2.0)
    res = solve(op, f, mode="strict", tol=1e-11)
    u_ex = m.exact_solution_avg(g)
    u_ex -= u_ex.mean()
    err = maxabs(res.u - u_ex)
    errs.append(err)
    rate = np.log2(errs[-2] / errs[-1]) if len(errs) > 1 else float("nan")
    line(f"n={n:<4} err={err:.3e} p={rate:.2f}", res, f"sum(f)V={res.rhs_imbalance:+.2e}")

# ---------------------------------------------------------------------------
hdr("2. Smooth manufactured solution, variable k=1+x/2+y/4 -- grid convergence (mode='project')")
# ---------------------------------------------------------------------------
print("  f from 8-point Gauss cell averages; the (tiny) removed mean is reported, not silent")
errs = []
for n in [16, 32, 64, 128]:
    g = Grid(n, n)
    op = FVOperator(g, m.k_linear(g))
    f = m.f_linear_k(g)
    res = solve(op, f, mode="project", tol=1e-11)
    u_ex = m.exact_solution_avg(g)
    u_ex -= u_ex.mean()
    err = maxabs(res.u - u_ex)
    errs.append(err)
    rate = np.log2(errs[-2] / errs[-1]) if len(errs) > 1 else float("nan")
    line(f"n={n:<4} err={err:.3e} p={rate:.2f}", res,
         f"removed_mean={res.rhs_mean_removed:+.2e} modified={res.rhs_modified}")

# ---------------------------------------------------------------------------
hdr("3. Anisotropic grid spacing")
# ---------------------------------------------------------------------------
print("  (a) Lx=1, Ly=0.25, nx=ny  -> dx/dy = 4")
errs = []
for n in [16, 32, 64, 128]:
    g = Grid(n, n, 1.0, 0.25)
    op = FVOperator(g, m.k_constant(g, 2.0))
    res = solve(op, m.f_constant_k(g, 2.0), tol=1e-11)
    u_ex = m.exact_solution_avg(g)
    u_ex -= u_ex.mean()
    err = maxabs(res.u - u_ex)
    errs.append(err)
    rate = np.log2(errs[-2] / errs[-1]) if len(errs) > 1 else float("nan")
    line(f"nx=ny={n:<4} err={err:.3e} p={rate:.2f}", res)
print("  (b) Lx=Ly=1, nx=4*ny      -> dx/dy = 1/4")
errs = []
for ny in [8, 16, 32, 64]:
    g = Grid(4 * ny, ny)
    op = FVOperator(g, m.k_constant(g, 2.0))
    res = solve(op, m.f_constant_k(g, 2.0), tol=1e-11)
    u_ex = m.exact_solution_avg(g)
    u_ex -= u_ex.mean()
    err = maxabs(res.u - u_ex)
    errs.append(err)
    rate = np.log2(errs[-2] / errs[-1]) if len(errs) > 1 else float("nan")
    line(f"({4*ny:>3},{ny:<3}) err={err:.3e} p={rate:.2f}", res)

# ---------------------------------------------------------------------------
hdr("4. Discrete manufactured solution: f = A u* for random zero-mean u*, high-contrast k")
# ---------------------------------------------------------------------------
for n, tag, k in [
    (64, "loguniform k in [1,1e2]", 10.0 ** rng.uniform(0, 2, (64, 64))),
    (64, "loguniform k in [1,1e4]", 10.0 ** rng.uniform(0, 4, (64, 64))),
    (64, "loguniform k in [1,1e6]", 10.0 ** rng.uniform(0, 6, (64, 64))),
    (64, "checkerboard 1 / 1e6", np.where(np.indices((64, 64)).sum(0) % 2 == 0, 1.0, 1e6)),
]:
    g = Grid(n, n)
    op = FVOperator(g, k.reshape(g.shape))
    u_star = rng.standard_normal(g.shape)
    u_star -= u_star.mean()
    f = op.apply(u_star)
    res = solve(op, f, tol=1e-8)
    err = maxabs(res.u - u_star) / maxabs(u_star)
    line(f"{tag:<26} recerr={err:.2e}", res)

# ---------------------------------------------------------------------------
hdr("5. Constant field u*=3.7 (null space): f = A u* = 0")
# ---------------------------------------------------------------------------
g = Grid(32, 32)
op = FVOperator(g, 10.0 ** rng.uniform(0, 3, g.shape))
u_star = np.full(g.shape, 3.7)
f = op.apply(u_star)
print(f"  ||A u*||_inf = {maxabs(f):.3e}  (constant is in the null space)")
res = solve(op, f)
line("zero-mean representative", res, f"||u||_inf={maxabs(res.u):.2e}")
print(f"  u* - u = {maxabs(u_star - res.u):.15f} everywhere (recovered up to the null constant)")

# ---------------------------------------------------------------------------
hdr("6. Incompatible right-hand side")
# ---------------------------------------------------------------------------
g = Grid(32, 32)
op = FVOperator(g, 10.0 ** rng.uniform(0, 2, g.shape))
u_star = rng.standard_normal(g.shape)
u_star -= u_star.mean()
f = op.apply(u_star) + 1.5
print(f"  mean(f) = {f.mean():.6f}  -> discrete compatibility residual sum(f)*V = {f.sum()*g.cell_volume:.6e}")
try:
    solve(op, f, mode="strict")
    print("  strict mode: NOT rejected (BUG)")
except IncompatibleRHSError as exc:
    print(f"  strict mode rejects: {exc}")
res = solve(op, f, mode="project", tol=1e-10)
print(f"  project mode: solved equation = {res.equation}")
print(f"  removed mean c = {res.rhs_mean_removed:.12f}, rhs_modified = {res.rhs_modified}")
line("projected solve", res)
print(f"  true residual vs modified eq  ||A u-(f-c)||_inf = {maxabs(op.apply(res.u)-(f-res.rhs_mean_removed)):.3e}")
print(f"  true residual vs original eq  ||A u - f   ||_inf = {maxabs(op.apply(res.u)-f):.3e}  (= |c|, no solution exists)")
print(f"  recovery of u*                          = {maxabs(res.u-u_star):.3e}")

# ---------------------------------------------------------------------------
hdr("7. Illegal coefficients are rejected")
# ---------------------------------------------------------------------------
for tag, bad in [("k=0 in one cell", 0.0), ("k<0 in one cell", -2.0),
                 ("NaN in one cell", np.nan), ("inf in one cell", np.inf)]:
    g = Grid(8, 8)
    k = np.ones(g.shape)
    k[3, 4] = bad
    try:
        FVOperator(g, k)
        print(f"  {tag}: NOT rejected (BUG)")
    except InvalidCoefficientError as exc:
        print(f"  {tag}: InvalidCoefficientError: {exc}")
