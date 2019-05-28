module ChainRules

using IRTools: IRTools, IR, @dynamo, isexpr, xcall
using IRTools.MacroTools: prewalk
using LinearAlgebra
using Base.Broadcast: materialize, materialize!, broadcasted, Broadcasted, broadcastable

import NaNMath, SpecialFunctions, LinearAlgebra, LinearAlgebra.BLAS

export AbstractRule, Rule, frule, rrule

include("differentials.jl")
include("rules.jl")
include("rules/base.jl")
include("rules/broadcast.jl")
include("rules/linalg/dense.jl")
include("rules/linalg/diagonal.jl")
include("rules/linalg/symmetric.jl")
include("rules/linalg/factorization.jl")
include("rules/blas.jl")
include("rules/nanmath.jl")
include("rules/specialfunctions.jl")

include("precompile.jl")
end # module
