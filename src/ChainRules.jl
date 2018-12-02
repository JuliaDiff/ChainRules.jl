module ChainRules

using LinearAlgebra
using Base.Broadcast: materialize, materialize!, broadcasted, Broadcasted

import NaNMath, SpecialFunctions, LinearAlgebra, LinearAlgebra.BLAS

include("chain.jl")
include("rules.jl")
include("rules/base.jl")
include("rules/linalg.jl")
include("rules/blas.jl")
include("rules/nanmath.jl")
include("rules/specialfunctions.jl")

end # module
