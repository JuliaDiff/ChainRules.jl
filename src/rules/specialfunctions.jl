#####
##### `@rule`s
#####

@rule(SpecialFunctions.lgamma(x), SpecialFunctions.digamma(x))
@rule(SpecialFunctions.erf(x), (2 / sqrt(π)) * exp(-x * x))
@rule(SpecialFunctions.erfc(x), -(2 / sqrt(π)) * exp(-x * x))
@rule(SpecialFunctions.erfi(x), (2 / sqrt(π)) * exp(x * x))
@rule(SpecialFunctions.digamma(x), SpecialFunctions.trigamma(x))
@rule(SpecialFunctions.trigamma(x), SpecialFunctions.polygamma(2, x))
@rule(SpecialFunctions.airyai(x), SpecialFunctions.airyaiprime(x))
@rule(SpecialFunctions.airyaiprime(x), x * SpecialFunctions.airyai(x))
@rule(SpecialFunctions.airybi(x), SpecialFunctions.airybiprime(x))
@rule(SpecialFunctions.airybiprime(x), x * SpecialFunctions.airybi(x))
@rule(SpecialFunctions.besselj0(x), -SpecialFunctions.besselj1(x))
@rule(SpecialFunctions.bessely0(x), -SpecialFunctions.bessely1(x))
@rule(SpecialFunctions.invdigamma(x), inv(SpecialFunctions.trigamma(SpecialFunctions.invdigamma(x))))
@rule(SpecialFunctions.besselj1(x), (SpecialFunctions.besselj0(x) - SpecialFunctions.besselj(2, x)) / 2)
@rule(SpecialFunctions.bessely1(x), (SpecialFunctions.bessely0(x) - SpecialFunctions.bessely(2, x)) / 2)
@rule(SpecialFunctions.gamma(x), Ω * SpecialFunctions.digamma(x))
@rule(SpecialFunctions.erfinv(x), (sqrt(π) / 2) * exp(Ω^2))
@rule(SpecialFunctions.erfcinv(x), -(sqrt(π) / 2) * exp(Ω^2))
@rule(SpecialFunctions.erfcx(x), (2 * x * Ω) - (2 / sqrt(π)))
@rule(SpecialFunctions.dawson(x), 1 - (2 * x * Ω))
