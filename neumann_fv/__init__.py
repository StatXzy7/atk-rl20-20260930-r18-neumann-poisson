"""Cell-centered finite-volume solver for pure-Neumann Poisson equations."""

from .grid import Grid2D
from .mms import constrained_kkt_solve, smooth_problem_rhs, smooth_source, smooth_u
from .operator import FVDiffusion, FaceFluxData, harmonic_mean
from .solver import (
    IncompatibleRightHandSide,
    PCGResult,
    prepare_rhs,
    projected_pcg,
    solve_neumann,
)

__all__ = [
    "Grid2D",
    "FVDiffusion",
    "FaceFluxData",
    "harmonic_mean",
    "IncompatibleRightHandSide",
    "PCGResult",
    "prepare_rhs",
    "projected_pcg",
    "solve_neumann",
    "smooth_problem_rhs",
    "smooth_source",
    "smooth_u",
    "constrained_kkt_solve",
]
