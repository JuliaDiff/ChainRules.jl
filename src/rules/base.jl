#####
##### forward rules
#####

# simple `@frule`s

@frule(R → R, abs2(x), x + x)
@frule(R → R, log(x), inv(x))
@frule(R → R, log10(x), inv(x) / log(10f0))
@frule(R → R, log2(x), inv(x) / log(2f0))
@frule(R → R, log1p(x), inv(x + 1))
@frule(R → R, expm1(x), exp(x))
@frule(R → R, sin(x), cos(x))
@frule(R → R, cos(x), -sin(x))
@frule(R → R, sinpi(x), π * cospi(x))
@frule(R → R, cospi(x), -π * sinpi(x))
@frule(R → R, sind(x), (π / 180f0) * cosd(x))
@frule(R → R, cosd(x), -(π / 180f0) * sind(x))
@frule(R → R, asin(x), inv(sqrt(1 - x^2)))
@frule(R → R, acos(x), -inv(sqrt(1 - x^2)))
@frule(R → R, atan(x), inv(1 + x^2))
@frule(R → R, asec(x), inv(abs(x) * sqrt(x^2 - 1)))
@frule(R → R, acsc(x), -inv(abs(x) * sqrt(x^2 - 1)))
@frule(R → R, acot(x), -inv(1 + x^2))
@frule(R → R, asind(x), 180f0 / π / sqrt(1 - x^2))
@frule(R → R, acosd(x), -180f0 / π / sqrt(1 - x^2))
@frule(R → R, atand(x), 180f0 / π / (1 + x^2))
@frule(R → R, asecd(x), 180f0 / π / abs(x) / sqrt(x^2 - 1))
@frule(R → R, acscd(x), -180f0 / π / abs(x) / sqrt(x^2 - 1))
@frule(R → R, acotd(x), -180f0 / π / (1 + x^2))
@frule(R → R, sinh(x), cosh(x))
@frule(R → R, cosh(x), sinh(x))
@frule(R → R, tanh(x), sech(x)^2)
@frule(R → R, coth(x), -(csch(x)^2))
@frule(R → R, asinh(x), inv(sqrt(x^2 + 1)))
@frule(R → R, acosh(x), inv(sqrt(x^2 - 1)))
@frule(R → R, atanh(x), inv(1 - x^2))
@frule(R → R, asech(x), -inv(x * sqrt(1 - x^2)))
@frule(R → R, acsch(x), -inv(abs(x) * sqrt(1 + x^2)))
@frule(R → R, acoth(x), inv(1 - x^2))
@frule(R → R, deg2rad(x), π / 180f0)
@frule(R → R, rad2deg(x), 180f0 / π)

@frule(R×R → R, +(x, y), (one(x), one(y)))
@frule(R×R → R, -(x, y), (one(x), -one(y)))
@frule(R×R → R, *(x, y), (y, x))
@frule(R×R → R, /(x, y), (inv(y), -(x / y / y)))
@frule(R×R → R, \(x, y), (-(y / x / x), inv(x)))

# manually optimized `frule`s

frule(::@domain({R → R}), ::typeof(transpose), x) = (transpose(x), ẋ -> ifelse(ẋ === nothing, false, ẋ))
frule(::@domain({R → R}), ::typeof(abs), x) = (abs(x), ẋ -> ifelse(ẋ === nothing, false, signbit(x) ? ẋ : -ẋ))
frule(::@domain({R → R}), ::typeof(+), x) = (+(x), ẋ -> ifelse(ẋ === nothing, false, ẋ))
frule(::@domain({R → R}), ::typeof(-), x) = (-(x), ẋ -> ifelse(ẋ === nothing, false, -ẋ))
frule(::@domain({R → R}), ::typeof(inv), x) = (u = inv(x); (u, ẋ -> fchain(ẋ, @thunk(-abs2(u)))))
frule(::@domain({R → R}), ::typeof(sqrt), x) = (u = sqrt(x); (u, ẋ -> fchain(ẋ, @thunk(inv(2 * u)))))
frule(::@domain({R → R}), ::typeof(cbrt), x) = (u = cbrt(x); (u, ẋ -> fchain(ẋ, @thunk(inv(3 * u^2)))))
frule(::@domain({R → R}), ::typeof(exp), x) = (u = exp(x); (u, ẋ -> fchain(ẋ, @thunk(u))))
frule(::@domain({R → R}), ::typeof(exp2), x) = (u = exp2(x); (u, ẋ -> fchain(ẋ, @thunk(u * log(2f0)))))
frule(::@domain({R → R}), ::typeof(exp10), x) = (u = exp10(x); (u, ẋ -> fchain(ẋ, @thunk(u * log(10f0)))))
frule(::@domain({R → R}), ::typeof(tan), x) = (u = tan(x); (u, ẋ -> fchain(ẋ, @thunk(1 + u^2))))
frule(::@domain({R → R}), ::typeof(sec), x) = (u = sec(x); (u, ẋ -> fchain(ẋ, @thunk(u * tan(x)))))
frule(::@domain({R → R}), ::typeof(csc), x) = (u = csc(x); (u, ẋ -> fchain(ẋ, @thunk(-u * cot(x)))))
frule(::@domain({R → R}), ::typeof(cot), x) = (u = cot(x); (u, ẋ -> fchain(ẋ, @thunk(-(1 + u^2)))))
frule(::@domain({R → R}), ::typeof(tand), x) = (u = tand(x); (u, ẋ -> fchain(ẋ, @thunk((π / 180f0) * (1 + u^2)))))
frule(::@domain({R → R}), ::typeof(secd), x) = (u = secd(x); (u, ẋ -> fchain(ẋ, @thunk((π / 180f0) * u * tand(x)))))
frule(::@domain({R → R}), ::typeof(cscd), x) = (u = cscd(x); (u, ẋ -> fchain(ẋ, @thunk(-(π / 180f0) * u * cotd(x)))))
frule(::@domain({R → R}), ::typeof(cotd), x) = (u = cotd(x); (u, ẋ -> fchain(ẋ, @thunk(-(π / 180f0) * (1 + u^2)))))
frule(::@domain({R → R}), ::typeof(sech), x) = (u = sech(x); (u, ẋ -> fchain(ẋ, @thunk(-tanh(x) * u))))
frule(::@domain({R → R}), ::typeof(csch), x) = (u = csch(x); (u, ẋ -> fchain(ẋ, @thunk(-coth(x) * u))))

function frule(::@domain({R×R → R}), ::typeof(atan), y, x)
    h = hypot(y, x)
    return atan(y, x), (ẏ, ẋ) -> fchain(ẏ, @thunk(x / h), ẋ, @thunk(y / h))
end

function frule(::@domain({R×R → R}), ::typeof(hypot), x, y)
    h = hypot(x, y)
    return h, (ẋ, ẏ) -> fchain(ẋ, @thunk(x / h), ẏ, @thunk(y / h))
end

function frule(::@domain({R → R×R}), ::typeof(sincos), x)
    sinx, cosx = sincos(x)
    return (sinx, cosx),
           (ẋ -> fchain(ẋ, @thunk(cosx)),
            ẋ -> fchain(ẋ, @thunk(-sinx)))
end

function frule(::@domain({R×R → R}), ::typeof(^), x, y)
    z = x^y
    return z, (ẋ, ẏ) -> fchain(ẋ, @thunk(y * x^(y - 1)), ẏ, @thunk(z * log(x)))
end

function frule(::@domain({R×R → R}), ::typeof(mod), x, y)
    return mod(x, y), (ẋ, ẏ) -> begin
        z, nan = promote(x / y, NaN16)
        return fchain(ẋ, @thunk(ifelse(isint, nan, one(z))),
                      ẏ, @thunk(ifelse(isint, nan, -floor(z))))
    end
end

function frule(::@domain({R×R → R}), ::typeof(rem), x, y)
    return rem(x, y), (ẋ, ẏ) -> begin
        z, nan = promote(x / y, NaN16)
        return fchain(ẋ, @thunk(ifelse(isint, nan, one(z))),
                      ẏ, @thunk(ifelse(isint, nan, -trunc(z))))
    end
end

function frule(::@domain({R×_ → R}), ::typeof(rem2pi), x, r)
    return rem2pi(x, r), ẋ -> ifelse(ẋ === nothing, false, ẋ)
end

function frule(::@domain({R×R → R}), ::typeof(max), x, y)
    return max(x, y), (ẋ, ẏ) -> (gt = x > y; fchain(ẋ, @thunk(gt), ẏ, @thunk(!gt)))
end

function frule(::@domain({R×R → R}), ::typeof(min), x, y)
    return min(x, y), (ẋ, ẏ) -> (gt = x > y; fchain(ẋ, @thunk(!gt), ẏ, @thunk(gt)))
end

frule(::@domain({C → C}), ::typeof(conj), x) = conj(x), ẋ -> (false, true)

#####
##### reverse rules
#####

@rrule(R → R, sum(x), ȳ, ȳ)
@rrule(R×R → R, +(x, y), z̄, z̄, z̄)
@rrule(R×R → R, *(x, y), z̄, z̄ * y', x' * z̄)

# TODO: This partial derivative extraction should be doable without the extra
# temporaries or preallocation utilized here, but AFAICT such an approach is
# hard to write without relying on inference hacks unless we have something
# akin to https://github.com/JuliaLang/julia/issues/22129
function rrule(::@domain({_×R → R}), ::typeof(broadcast), f, x)
    f_rule = x -> begin
        y, d = frule(@domain(R → R), f, x)
        y, d(one(x))
    end
    applied_f_rule = broadcast(f_rule, x)
    values = map(first, applied_f_rule)
    derivs = map(last, applied_f_rule)
    return values, (x̄, z̄) -> rchain(x̄, @thunk(broadcasted(*, z̄, derivs)))
end
