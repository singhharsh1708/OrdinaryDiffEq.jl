module OrdinaryDiffEqSDC

import OrdinaryDiffEqCore: alg_order, isfsal,
    OrdinaryDiffEqNewtonAlgorithm,
    generic_solver_docstring,
    unwrap_alg, initialize!, perform_step!,
    OrdinaryDiffEqMutableCache, OrdinaryDiffEqConstantCache,
    @cache, alg_cache, full_cache, get_fsalfirstlast,
    constvalue, _fixup_ad, _unwrap_val
import OrdinaryDiffEqCore
import FastBroadcast: @..
import MuladdMacro: @muladd
import LinearAlgebra
using OrdinaryDiffEqNonlinearSolve: build_nlsolver, nlsolve!, nlsolvefail,
    markfirststage!, NLNewton
import ADTypes: AutoForwardDiff

using Reexport
@reexport using SciMLBase

include("sdc_tableaus.jl")
include("algorithms.jl")
include("alg_utils.jl")
include("sdc_caches.jl")
include("sdc_perform_step.jl")

export SDC

end
