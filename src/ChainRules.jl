module ChainRules

using Cassette
using LinearAlgebra
using LinearAlgebra.BLAS
using Statistics
using Base.Broadcast: materialize, materialize!, broadcasted, Broadcasted, broadcastable

if VERSION < v"1.3.0-DEV.142"
    # In prior versions, the BLAS submodule also exported `dot`, which caused a conflict
    # with its parent module. To get around this, we can simply create a hard binding for
    # the one we want to use without qualification.
    import LinearAlgebra: dot
end

import NaNMath, SpecialFunctions

export AbstractRule, Rule, frule, rrule

include("differentials.jl")
include("rules.jl")
include("rules/base.jl")
include("rules/array.jl")
include("rules/broadcast.jl")
include("rules/mapreduce.jl")
include("rules/linalg/utils.jl")
include("rules/linalg/blas.jl")
include("rules/linalg/dense.jl")
include("rules/linalg/structured.jl")
include("rules/linalg/factorization.jl")
include("rules/nanmath.jl")
include("rules/specialfunctions.jl")

end # module
