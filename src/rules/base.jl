#####
##### `@rule`s
#####

@rule(abs2(x), x + x)
@rule(log(x), inv(x))
@rule(log10(x), inv(x) / log(10f0))
@rule(log2(x), inv(x) / log(2f0))
@rule(log1p(x), inv(x + 1))
@rule(expm1(x), exp(x))
@rule(sin(x), cos(x))
@rule(cos(x), -sin(x))
@rule(sinpi(x), π * cospi(x))
@rule(cospi(x), -π * sinpi(x))
@rule(sind(x), (π / 180f0) * cosd(x))
@rule(cosd(x), -(π / 180f0) * sind(x))
@rule(asin(x), inv(sqrt(1 - x^2)))
@rule(acos(x), -inv(sqrt(1 - x^2)))
@rule(atan(x), inv(1 + x^2))
@rule(asec(x), inv(abs(x) * sqrt(x^2 - 1)))
@rule(acsc(x), -inv(abs(x) * sqrt(x^2 - 1)))
@rule(acot(x), -inv(1 + x^2))
@rule(asind(x), 180f0 / π / sqrt(1 - x^2))
@rule(acosd(x), -180f0 / π / sqrt(1 - x^2))
@rule(atand(x), 180f0 / π / (1 + x^2))
@rule(asecd(x), 180f0 / π / abs(x) / sqrt(x^2 - 1))
@rule(acscd(x), -180f0 / π / abs(x) / sqrt(x^2 - 1))
@rule(acotd(x), -180f0 / π / (1 + x^2))
@rule(sinh(x), cosh(x))
@rule(cosh(x), sinh(x))
@rule(tanh(x), sech(x)^2)
@rule(coth(x), -(csch(x)^2))
@rule(asinh(x), inv(sqrt(x^2 + 1)))
@rule(acosh(x), inv(sqrt(x^2 - 1)))
@rule(atanh(x), inv(1 - x^2))
@rule(asech(x), -inv(x * sqrt(1 - x^2)))
@rule(acsch(x), -inv(abs(x) * sqrt(1 + x^2)))
@rule(acoth(x), inv(1 - x^2))
@rule(deg2rad(x), π / 180f0)
@rule(rad2deg(x), 180f0 / π)
@rule(conj(x), Wirtinger(Zero(), One()))
@rule(adjoint(x), Wirtinger(Zero(), One()))
@rule(transpose(x), One())
@rule(abs(x), sign(x))
@rule(rem2pi(x, r), (One(), DNE()))
@rule(sum(x), One())
@rule(+(x), One())
@rule(+(x, y), (One(), One()))
@rule(-(x, y), (One(), -one(y)))
@rule(/(x, y), (inv(y), -(x / y / y)))
@rule(\(x, y), (-(y / x / x), inv(x)))
@rule(^(x, y), (y * x^(y - 1), Ω * log(x)))
@rule(inv(x), -abs2(Ω))
@rule(sqrt(x), inv(2 * Ω))
@rule(cbrt(x), inv(3 * Ω^2))
@rule(exp(x), Ω)
@rule(exp2(x), Ω * log(2f0))
@rule(exp10(x), Ω * log(10f0))
@rule(tan(x), 1 + Ω^2)
@rule(sec(x), Ω * tan(x))
@rule(csc(x), -Ω * cot(x))
@rule(cot(x), -(1 + Ω^2))
@rule(tand(x), (π / 180f0) * (1 + Ω^2))
@rule(secd(x), (π / 180f0) * Ω * tand(x))
@rule(cscd(x), -(π / 180f0) * Ω * cotd(x))
@rule(cotd(x), -(π / 180f0) * (1 + Ω^2))
@rule(sech(x), -tanh(x) * Ω)
@rule(csch(x), -coth(x) * Ω)
@rule(hypot(x, y), (y / Ω, x / Ω))
@rule(sincos(x), @setup((sinx, cosx) = Ω), cosx, -sinx)
@rule(atan(y, x), @setup(u = hypot(x, y)), (x / u, y / u))
@rule(max(x, y), @setup(gt = x > y), (gt, !gt))
@rule(min(x, y), @setup(gt = x > y), (!gt, gt))
@rule(mod(x, y), @setup((u, nan) = promote(x / y, NaN16)),
      (ifelse(isint, nan, one(u)), ifelse(isint, nan, -floor(u))))
@rule(rem(x, y), @setup((u, nan) = promote(x / y, NaN16)),
      (ifelse(isint, nan, one(u)), ifelse(isint, nan, -trunc(u))))

#####
##### custom rules
#####

# product rule requires special care for arguments where `mul` is non-commutative

frule(::typeof(*), x, y) = x * y, (ż, ẋ, ẏ) -> add(ż, mul(ẋ, y), mul(x, ẏ))

rrule(::typeof(*), x, y) = x * y, ((x̄, z̄) -> add(x̄, mul(z̄, y')), (ȳ, z̄) -> add(ȳ, mul(x', z̄)))

#=
TODO: This partial derivative extraction should be doable without the extra
temporaries utilized here, but AFAICT such an approach is hard to write
without relying on inference hacks unless we have something akin to
https://github.com/JuliaLang/julia/issues/22129.

TODO: Handle more kinds of inputs/outputs (e.g. `Wirtinger`s) within `element_rule`.
=#
function _cast_diff(f, x)
    element_rule = u -> begin
        fu, du = frule(f, u)
        fu, materialize(du(Zero(), One()))
    end
    results = broadcast(element_rule, x)
    return first.(results), last.(results)
end

function frule(::typeof(broadcast), f, x)
    values, derivs = _cast_diff(f, x)
    return values, @chain(DNE(), cast(derivs))
end

function rrule(::typeof(broadcast), f, x)
    values, derivs = _cast_diff(f, x)
    return values, (@chain(DNE()), @chain(cast(adjoint, derivs)))
end
