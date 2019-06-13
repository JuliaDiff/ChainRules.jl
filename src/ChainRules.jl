module ChainRules

using Cassette
using LinearAlgebra
using Base.Broadcast: materialize, materialize!, broadcasted, Broadcasted, broadcastable

import NaNMath, SpecialFunctions, LinearAlgebra, LinearAlgebra.BLAS

export AbstractRule, Rule, frule, rrule

include("differentials.jl")
include("rules.jl")
include("rules/base.jl")
include("rules/array.jl")
include("rules/broadcast.jl")
include("rules/linalg/utils.jl")
include("rules/linalg/blas.jl")
include("rules/linalg/dense.jl")
include("rules/linalg/structured.jl")
include("rules/linalg/factorization.jl")
include("rules/nanmath.jl")
include("rules/specialfunctions.jl")

end # module
