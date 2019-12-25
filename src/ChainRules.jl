module ChainRules
using Reexport
@reexport using ChainRulesCore
# Basically everything this package does is overloading these, so we make an exception
# to the normal rule of only overload via `ChainRulesCore.rrule`.
import ChainRulesCore: rrule, frule

# Deal with name clashes, by defining in this module which one we mean.
const accumulate = ChainRulesCore.accumulate
const accumulate! = ChainRulesCore.accumulate!

using LinearAlgebra
using LinearAlgebra.BLAS
using Requires
using Statistics
using Base.Broadcast: materialize, materialize!, broadcasted, Broadcasted, broadcastable

if VERSION < v"1.3.0-DEV.142"
    # In prior versions, the BLAS submodule also exported `dot`, which caused a conflict
    # with its parent module. To get around this, we can simply create a hard binding for
    # the one we want to use without qualification.
    import LinearAlgebra: dot
end

include("helper_functions.jl")

include("rulesets/Base/base.jl")
include("rulesets/Base/array.jl")
include("rulesets/Base/broadcast.jl")
include("rulesets/Base/mapreduce.jl")

include("rulesets/Statistics/statistics.jl")

include("rulesets/LinearAlgebra/utils.jl")
include("rulesets/LinearAlgebra/blas.jl")
include("rulesets/LinearAlgebra/dense.jl")
include("rulesets/LinearAlgebra/structured.jl")
include("rulesets/LinearAlgebra/factorization.jl")

# Note: The following is only required because package authors sometimes do not
# declare their own rules using `ChainRulesCore.jl`. For arguably good reasons.
# So we define them here for them.
function __init__()
    @require NaNMath="77ba4419-2d1f-58cd-9bb1-8ffee604a2e3" begin
        @scalar_rule(NaNMath.sin(x), NaNMath.cos(x))
        @scalar_rule(NaNMath.cos(x), -NaNMath.sin(x))
        @scalar_rule(NaNMath.asin(x), inv(NaNMath.sqrt(1 - NaNMath.pow(x, 2))))
        @scalar_rule(NaNMath.acos(x), -inv(NaNMath.sqrt(1 - NaNMath.pow(x, 2))))
        @scalar_rule(NaNMath.acosh(x), inv(NaNMath.sqrt(NaNMath.pow(x, 2) - 1)))
        @scalar_rule(NaNMath.atanh(x), inv(1 - NaNMath.pow(x, 2)))
        @scalar_rule(NaNMath.tan(x), 1 + Ω^2)
        @scalar_rule(NaNMath.log(x), inv(x))
        @scalar_rule(NaNMath.log2(x), inv(x) / NaNMath.log(oftype(x, 2)))
        @scalar_rule(NaNMath.log10(x), inv(x) / NaNMath.log(oftype(x, 10)))
        @scalar_rule(NaNMath.log1p(x), inv(x + 1))
        @scalar_rule(NaNMath.lgamma(x), SpecialFunctions.digamma(x))
        @scalar_rule(NaNMath.sqrt(x), inv(2 * Ω))
        @scalar_rule(NaNMath.pow(x, y), (y * NaNMath.pow(x, y - 1), Ω * NaNMath.log(x)))
        @scalar_rule(NaNMath.max(x, y),
                     (ifelse((y > x) | (signbit(y) < signbit(x)), ifelse(isnan(y), One(), Zero()), ifelse(isnan(x), Zero(), One())),
                      ifelse((y > x) | (signbit(y) < signbit(x)), ifelse(isnan(y), Zero(), One()), ifelse(isnan(x), One(), Zero()))))
        @scalar_rule(NaNMath.min(x, y),
                     (ifelse((y < x) | (signbit(y) > signbit(x)), ifelse(isnan(y), One(), Zero()), ifelse(isnan(x), Zero(), One())),
                      ifelse((y < x) | (signbit(y) > signbit(x)), ifelse(isnan(y), Zero(), One()), ifelse(isnan(x), One(), Zero()))))
    end

    @require SpecialFunctions="276daf66-3868-5448-9aa4-cd146d93841b" begin
        @scalar_rule(SpecialFunctions.erf(x), (2 / sqrt(π)) * exp(-x * x))
        @scalar_rule(SpecialFunctions.erfc(x), -(2 / sqrt(π)) * exp(-x * x))
        @scalar_rule(SpecialFunctions.erfi(x), (2 / sqrt(π)) * exp(x * x))
        @scalar_rule(SpecialFunctions.digamma(x), SpecialFunctions.trigamma(x))
        @scalar_rule(SpecialFunctions.trigamma(x), SpecialFunctions.polygamma(2, x))
        @scalar_rule(SpecialFunctions.airyai(x), SpecialFunctions.airyaiprime(x))
        @scalar_rule(SpecialFunctions.airyaiprime(x), x * SpecialFunctions.airyai(x))
        @scalar_rule(SpecialFunctions.airybi(x), SpecialFunctions.airybiprime(x))
        @scalar_rule(SpecialFunctions.airybiprime(x), x * SpecialFunctions.airybi(x))
        @scalar_rule(SpecialFunctions.besselj0(x), -SpecialFunctions.besselj1(x))
        @scalar_rule(SpecialFunctions.bessely0(x), -SpecialFunctions.bessely1(x))
        @scalar_rule(SpecialFunctions.invdigamma(x), inv(SpecialFunctions.trigamma(SpecialFunctions.invdigamma(x))))
        @scalar_rule(SpecialFunctions.besselj1(x), (SpecialFunctions.besselj0(x) - SpecialFunctions.besselj(2, x)) / 2)
        @scalar_rule(SpecialFunctions.bessely1(x), (SpecialFunctions.bessely0(x) - SpecialFunctions.bessely(2, x)) / 2)
        @scalar_rule(SpecialFunctions.gamma(x), Ω * SpecialFunctions.digamma(x))
        @scalar_rule(SpecialFunctions.erfinv(x), (sqrt(π) / 2) * exp(Ω^2))
        @scalar_rule(SpecialFunctions.erfcinv(x), -(sqrt(π) / 2) * exp(Ω^2))
        @scalar_rule(SpecialFunctions.erfcx(x), (2 * x * Ω) - (2 / sqrt(π)))
        @scalar_rule(SpecialFunctions.dawson(x), 1 - (2 * x * Ω))

        # binary
        @scalar_rule(SpecialFunctions.besselj(ν, x),
                     (NaN,
                      (SpecialFunctions.besselj(ν - 1, x) -
                       SpecialFunctions.besselj(ν + 1, x)) / 2))

        @scalar_rule(SpecialFunctions.besseli(ν, x),
                     (NaN,
                      (SpecialFunctions.besseli(ν - 1, x) +
                       SpecialFunctions.besseli(ν + 1, x)) / 2))
        @scalar_rule(SpecialFunctions.bessely(ν, x),
                     (NaN,
                      (SpecialFunctions.bessely(ν - 1, x) -
                       SpecialFunctions.bessely(ν + 1, x)) / 2))

        @scalar_rule(SpecialFunctions.besselk(ν, x),
                     (NaN,
                      -(SpecialFunctions.besselk(ν - 1, x) +
                        SpecialFunctions.besselk(ν + 1, x)) / 2))

        @scalar_rule(SpecialFunctions.hankelh1(ν, x),
                     (NaN,
                      (SpecialFunctions.hankelh1(ν - 1, x) -
                       SpecialFunctions.hankelh1(ν + 1, x)) / 2))
        @scalar_rule(SpecialFunctions.hankelh2(ν, x),
                     (NaN,
                      (SpecialFunctions.hankelh2(ν - 1, x) -
                       SpecialFunctions.hankelh2(ν + 1, x)) / 2))

        @scalar_rule(SpecialFunctions.polygamma(m, x),
                     (NaN, SpecialFunctions.polygamma(m + 1, x)))

        # todo: setup for common expr
        @scalar_rule(SpecialFunctions.beta(a, b),
                     (Ω*(SpecialFunctions.digamma(a) - SpecialFunctions.digamma(a + b)),
                      Ω*(SpecialFunctions.digamma(b) - SpecialFunctions.digamma(a + b))))

        @scalar_rule(SpecialFunctions.lbeta(a, b),
                     (SpecialFunctions.digamma(a) - SpecialFunctions.digamma(a + b),
                      SpecialFunctions.digamma(b) - SpecialFunctions.digamma(a + b)))

        # Changes between SpecialFunctions 0.7 and 0.8
        if isdefined(SpecialFunctions, :lgamma)
            # actually is the absolute value of the logorithm of gamma
            @scalar_rule(SpecialFunctions.lgamma(x), SpecialFunctions.digamma(x))
        end

        if isdefined(SpecialFunctions, :logabsgamma)
            # actually is the absolute value of the logorithm of gamma, paired with sign gamma
            @scalar_rule(SpecialFunctions.logabsgamma(x), SpecialFunctions.digamma(x), Zero())
        end

        if isdefined(SpecialFunctions, :loggamma)
            @scalar_rule(SpecialFunctions.loggamma(x), SpecialFunctions.digamma(x))
        end
    end
end

end # module
