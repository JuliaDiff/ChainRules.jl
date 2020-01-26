using ChainRulesCore

@scalar_rule(NaNMath.sin(x), NaNMath.cos(x))
@scalar_rule(NaNMath.cos(x), -NaNMath.sin(x))
@scalar_rule(NaNMath.asin(x), inv(NaNMath.sqrt(1 - NaNMath.pow(x, 2))))
@scalar_rule(NaNMath.acos(x), -inv(NaNMath.sqrt(1 - NaNMath.pow(x, 2))))
@scalar_rule(NaNMath.acosh(x), inv(NaNMath.sqrt(NaNMath.pow(x, 2) - 1)))
@scalar_rule(NaNMath.tan(x), 1 + Ω^2)
@scalar_rule(NaNMath.atanh(x), inv(1 - NaNMath.pow(x, 2)))
@scalar_rule(NaNMath.log(x), inv(x))
@scalar_rule(NaNMath.log2(x), inv(x) / NaNMath.log(oftype(x, 2)))
@scalar_rule(NaNMath.log10(x), inv(x) / NaNMath.log(oftype(x, 10)))
@scalar_rule(NaNMath.log1p(x), inv(x + 1))
@scalar_rule(NaNMath.lgamma(x), SpecialFunctions.digamma(x))
@scalar_rule(NaNMath.sqrt(x), inv(2 * Ω))
@scalar_rule(NaNMath.pow(x, y), (y * NaNMath.pow(x, y - 1), Ω * NaNMath.log(x)))
@scalar_rule(
    NaNMath.max(x, y),
    (ifelse(
        (y > x) | (signbit(y) < signbit(x)),
        ifelse(isnan(y), One(), Zero()),
        ifelse(isnan(x), Zero(), One())),
     ifelse(
        (y > x) | (signbit(y) < signbit(x)),
        ifelse(isnan(y), Zero(), One()),
        ifelse(isnan(x), One(), Zero())),
    )
)
@scalar_rule(
    NaNMath.min(x, y),
    (ifelse(
        (y < x) | (signbit(y) > signbit(x)),
        ifelse(isnan(y), One(), Zero()),
        ifelse(isnan(x), Zero(), One())),
     ifelse(
        (y < x) | (signbit(y) > signbit(x)),
        ifelse(isnan(y), Zero(), One()),
        ifelse(isnan(x), One(), Zero())),
   )
)
