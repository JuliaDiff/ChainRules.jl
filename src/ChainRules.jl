module ChainRules

using Base.Broadcast: materialize, materialize!, broadcasted, Broadcasted, broadcastable
using Compat
using LinearAlgebra
using LinearAlgebra.BLAS
using Random
using Requires
using Statistics

# Basically everything this package does is overloading these, so we make an exception
# to the normal rule of only overload via `ChainRulesCore.rrule`.
import ChainRulesCore: rrule, frule

if VERSION < v"1.3.0-DEV.142"
    # In prior versions, the BLAS submodule also exported `dot`, which caused a conflict
    # with its parent module. To get around this, we can simply create a hard binding for
    # the one we want to use without qualification.
    import LinearAlgebra: dot
end

# numbers that we know commute under multiplication
const CommutativeMulNumber = Union{Real,Complex}

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

include("rulesets/Statistics/statistics.jl")

include("rulesets/LinearAlgebra/utils.jl")
include("rulesets/LinearAlgebra/blas.jl")
include("rulesets/LinearAlgebra/lapack.jl")
include("rulesets/LinearAlgebra/dense.jl")
include("rulesets/LinearAlgebra/norm.jl")
include("rulesets/LinearAlgebra/matfun.jl")
include("rulesets/LinearAlgebra/structured.jl")
include("rulesets/LinearAlgebra/symmetric.jl")
include("rulesets/LinearAlgebra/factorization.jl")

include("rulesets/Random/random.jl")

# Note: The following is only required because package authors sometimes do not
# declare their own rules using `ChainRulesCore.jl`. For arguably good reasons.
# So we define them here for them.
function __init__()
    @require NaNMath="77ba4419-2d1f-58cd-9bb1-8ffee604a2e3" begin
        include("rulesets/packages/NaNMath.jl")
    end

    # Note: drop SpecialFunctions dependency in next breaking release
    # https://github.com/JuliaDiff/ChainRules.jl/issues/319
    @require SpecialFunctions="276daf66-3868-5448-9aa4-cd146d93841b" begin
        if !isdefined(SpecialFunctions, :ChainRulesCore)
            include("rulesets/packages/SpecialFunctions.jl")
        end
    end
end

end # module
