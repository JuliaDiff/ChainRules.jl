module SpecialFunctionsGlue
using ChainRulesCore
using SpecialFunctions


@scalar_rule(SpecialFunctions.lgamma(x), SpecialFunctions.digamma(x))
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

end #module
