module ChainRules

using Cassette
using LinearAlgebra
using Base.Broadcast: materialize, materialize!, broadcasted, Broadcasted, broadcastable

import NaNMath, SpecialFunctions, LinearAlgebra, LinearAlgebra.BLAS

include("differentials.jl")
include("chain.jl")
include("rules.jl")
include("rules/base.jl")
include("rules/broadcast.jl")
include("rules/linalg.jl")
include("rules/blas.jl")
include("rules/nanmath.jl")
include("rules/specialfunctions.jl")

end # module
