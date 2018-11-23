#####
##### `@rule`s
#####

@rule(NaNMath.sin(x), NaNMath.cos(x))
@rule(NaNMath.cos(x), -NaNMath.sin(x))
@rule(NaNMath.asin(x), inv(NaNMath.sqrt(1 - NaNMath.pow(x, 2))))
@rule(NaNMath.acos(x), -inv(NaNMath.sqrt(1 - NaNMath.pow(x, 2))))
@rule(NaNMath.acosh(x), inv(NaNMath.sqrt(NaNMath.pow(x, 2) - 1)))
@rule(NaNMath.atanh(x), inv(1 - NaNMath.pow(x, 2)))
@rule(NaNMath.log(x), inv(x))
@rule(NaNMath.log2(x), inv(x) / NaNMath.log(2f0))
@rule(NaNMath.log10(x), inv(x) / NaNMath.log(10f0))
@rule(NaNMath.log1p(x), inv(x + 1))
@rule(NaNMath.lgamma(x), SpecialFunctions.digamma(x))
@rule(NaNMath.sqrt(x), inv(2 * Ω))
@rule(NaNMath.pow(x, y), (y * NaNMath.pow(x, y - 1), Ω * NaNMath.log(x)))
