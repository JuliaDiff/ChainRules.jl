module ChainRules

using Adapt: adapt
using Base.Broadcast: materialize, materialize!, broadcasted, Broadcasted, broadcastable
using ChainRulesCore
using Compat
using Distributed
using GPUArraysCore: AbstractGPUArray, AbstractGPUArrayStyle, @allowscalar
using IrrationalConstants: logtwo, logten
using LinearAlgebra
using LinearAlgebra.BLAS
using Random
using RealDot: realdot
using SparseArrays
using Statistics
using StructArrays

# Basically everything this package does is overloading these, so we make an exception
# to the normal rule of only overload via `ChainRulesCore.rrule`.
import ChainRulesCore: rrule, frule

# Experimental:
using ChainRulesCore: derivatives_given_output

# numbers that we know commute under multiplication
const CommutativeMulNumber = Union{Real,Complex}

# StructArrays
include("unzipped.jl")

include("rulesets/Core/core.jl")

include("rulesets/Base/utils.jl")
include("rulesets/Base/nondiff.jl")
include("rulesets/Base/base.jl")
include("rulesets/Base/fastmath_able.jl")
include("rulesets/Base/evalpoly.jl")
include("rulesets/Base/array.jl")
include("rulesets/Base/arraymath.jl")
include("rulesets/Base/indexing.jl")
include("rulesets/Base/sort.jl")
include("rulesets/Base/mapreduce.jl")
include("rulesets/Base/broadcast.jl")

include("rulesets/Distributed/nondiff.jl")

include("rulesets/Statistics/statistics.jl")

include("rulesets/LinearAlgebra/nondiff.jl")
include("rulesets/LinearAlgebra/utils.jl")
include("rulesets/LinearAlgebra/blas.jl")
include("rulesets/LinearAlgebra/lapack.jl")
include("rulesets/LinearAlgebra/dense.jl")
include("rulesets/LinearAlgebra/norm.jl")
include("rulesets/LinearAlgebra/matfun.jl")
include("rulesets/LinearAlgebra/structured.jl")
include("rulesets/LinearAlgebra/symmetric.jl")
include("rulesets/LinearAlgebra/factorization.jl")
include("rulesets/LinearAlgebra/uniformscaling.jl")

include("rulesets/SparseArrays/sparsematrix.jl")

include("rulesets/Random/random.jl")

end # module
