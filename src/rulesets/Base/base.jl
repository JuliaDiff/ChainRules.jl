@scalar_rule(one(x), Zero())
@scalar_rule(zero(x), Zero())
@scalar_rule(abs2(x), Wirtinger(x', x))
@scalar_rule(log(x), inv(x))
@scalar_rule(log10(x), inv(x) / log(oftype(x, 10)))
@scalar_rule(log2(x), inv(x) / log(oftype(x, 2)))
@scalar_rule(log1p(x), inv(x + 1))
@scalar_rule(expm1(x), exp(x))

@scalar_rule(sin(x), cos(x))
@scalar_rule(cos(x), -sin(x))
@scalar_rule(sinpi(x), π * cospi(x))
@scalar_rule(cospi(x), -π * sinpi(x))
@scalar_rule(sind(x), (π / oftype(x, 180)) * cosd(x))
@scalar_rule(cosd(x), -(π / oftype(x, 180)) * sind(x))

@scalar_rule(asin(x), inv(sqrt(1 - x^2)))
@scalar_rule(acos(x), -inv(sqrt(1 - x^2)))
@scalar_rule(atan(x), inv(1 + x^2))
@scalar_rule(asec(x::Real), inv(abs(x) * sqrt(x^2 - 1)))
@scalar_rule(asec(x), inv(x^2 * sqrt(1 - x^-2)))
@scalar_rule(acsc(x::Real), -inv(abs(x) * sqrt(x^2 - 1)))
@scalar_rule(acsc(x), -inv(x^2 * sqrt(1 - x^-2)))
@scalar_rule(acot(x), -inv(1 + x^2))

@scalar_rule(asind(x), oftype(x, 180) / π / sqrt(1 - x^2))
@scalar_rule(acosd(x), -oftype(x, 180) / π / sqrt(1 - x^2))
@scalar_rule(atand(x), oftype(x, 180) / π / (1 + x^2))
@scalar_rule(asecd(x::Real), oftype(x, 180) / π / abs(x) / sqrt(x^2 - 1))
@scalar_rule(asecd(x), oftype(x, 180) / π / x^2 / sqrt(1 - x^-2))
@scalar_rule(acscd(x::Real), -oftype(x, 180) / π / abs(x) / sqrt(x^2 - 1))
@scalar_rule(acscd(x), -oftype(x, 180) / π / x^2 / sqrt(1 - x^-2))
@scalar_rule(acotd(x), -oftype(x, 180) / π / (1 + x^2))

@scalar_rule(sinh(x), cosh(x))
@scalar_rule(cosh(x), sinh(x))
@scalar_rule(tanh(x), sech(x)^2)
@scalar_rule(coth(x), -(csch(x)^2))

@scalar_rule(asinh(x), inv(sqrt(x^2 + 1)))
@scalar_rule(acosh(x), inv(sqrt(x^2 - 1)))
@scalar_rule(atanh(x), inv(1 - x^2))
@scalar_rule(asech(x), -inv(x * sqrt(1 - x^2)))
@scalar_rule(acsch(x::Real), -inv(abs(x) * sqrt(1 + x^2)))
@scalar_rule(acsch(x), -inv(x^2 * sqrt(1 + x^-2)))
@scalar_rule(acoth(x), inv(1 - x^2))

@scalar_rule(deg2rad(x), π / oftype(x, 180))
@scalar_rule(rad2deg(x), oftype(x, 180) / π)

@scalar_rule(conj(x), Wirtinger(Zero(), One()))
@scalar_rule(adjoint(x), Wirtinger(Zero(), One()))
@scalar_rule(transpose(x), One())

@scalar_rule(abs(x), sign(x))
@scalar_rule(rem2pi(x, r::RoundingMode), (One(), DNE()))

@scalar_rule(+(x), One())
@scalar_rule(-(x), -1)
@scalar_rule(+(x, y), (One(), One()))
@scalar_rule(-(x, y), (One(), -1))
@scalar_rule(/(x, y), (inv(y), -(x / y / y)))
@scalar_rule(\(x, y), (-(y / x / x), inv(x)))
@scalar_rule(^(x, y), (y * x^(y - 1), Ω * log(x)))

@scalar_rule(inv(x), -Ω^2)
@scalar_rule(sqrt(x), inv(2 * Ω))
@scalar_rule(cbrt(x), inv(3 * Ω^2))
@scalar_rule(exp(x), Ω)
@scalar_rule(exp2(x), Ω * log(oftype(x, 2)))
@scalar_rule(exp10(x), Ω * log(oftype(x, 10)))

@scalar_rule(tan(x), 1 + Ω^2)
@scalar_rule(sec(x), Ω * tan(x))
@scalar_rule(csc(x), -Ω * cot(x))
@scalar_rule(cot(x), -(1 + Ω^2))
@scalar_rule(tand(x), (π / oftype(x, 180)) * (1 + Ω^2))
@scalar_rule(secd(x), (π / oftype(x, 180)) * Ω * tand(x))
@scalar_rule(cscd(x), -(π / oftype(x, 180)) * Ω * cotd(x))
@scalar_rule(cotd(x), -(π / oftype(x, 180)) * (1 + Ω^2))
@scalar_rule(sech(x), -tanh(x) * Ω)
@scalar_rule(csch(x), -coth(x) * Ω)

@scalar_rule(hypot(x, y), (x / Ω, y / Ω))
@scalar_rule(sincos(x), @setup((sinx, cosx) = Ω), cosx, -sinx)
@scalar_rule(atan(y, x), @setup(u = x^2 + y^2), (x / u, -y / u))

@scalar_rule(max(x, y), @setup(gt = x > y), (gt, !gt))
@scalar_rule(min(x, y), @setup(gt = x > y), (!gt, gt))
@scalar_rule(mod(x, y), @setup((u, nan) = promote(x / y, NaN16)),
             (ifelse(isint, nan, one(u)), ifelse(isint, nan, -floor(u))))
@scalar_rule(rem(x, y), @setup((u, nan) = promote(x / y, NaN16)),
             (ifelse(isint, nan, one(u)), ifelse(isint, nan, -trunc(u))))
@scalar_rule(angle(x::Complex), @setup(u = abs2(x)), Wirtinger(-im//2 * x' / u, im//2 * x / u))
@scalar_rule(angle(x::Real), Zero())
@scalar_rule(real(x::Complex), Wirtinger(1//2, 1//2))
@scalar_rule(real(x::Real), One())
@scalar_rule(imag(x::Complex), Wirtinger(-im//2, im//2))
@scalar_rule(imag(x::Real), Zero())

# product rule requires special care for arguments where `mul` is non-commutative

frule(::typeof(*), x::Number, y::Number) = x * y, Rule((Δx, Δy) -> Δx * y + x * Δy)

rrule(::typeof(*), x::Number, y::Number) = x * y, (Rule(ΔΩ -> ΔΩ * y'), Rule(ΔΩ -> x' * ΔΩ))

frule(::typeof(identity), x) = x, Rule(identity)

rrule(::typeof(identity), x) = x, Rule(identity)
