#####
##### forward rules
#####

# simple `@frule`s

@frule(R → R, SpecialFunctions.lgamma(x), SpecialFunctions.digamma(x))
@frule(R → R, SpecialFunctions.erf(x), (2 / sqrt(π)) * exp(-x * x))
@frule(R → R, SpecialFunctions.erfc(x), -(2 / sqrt(π)) * exp(-x * x))
@frule(R → R, SpecialFunctions.erfi(x), (2 / sqrt(π)) * exp(x * x))
@frule(R → R, SpecialFunctions.digamma(x), SpecialFunctions.trigamma(x))
@frule(R → R, SpecialFunctions.trigamma(x), SpecialFunctions.polygamma(2, x))
@frule(R → R, SpecialFunctions.airyai(x), SpecialFunctions.airyaiprime(x))
@frule(R → R, SpecialFunctions.airyaiprime(x), x * SpecialFunctions.airyai(x))
@frule(R → R, SpecialFunctions.airybi(x), SpecialFunctions.airybiprime(x))
@frule(R → R, SpecialFunctions.airybiprime(x), x * SpecialFunctions.airybi(x))
@frule(R → R, SpecialFunctions.besselj0(x), -SpecialFunctions.besselj1(x))
@frule(R → R, SpecialFunctions.bessely0(x), -SpecialFunctions.bessely1(x))

@frule(R → R, SpecialFunctions.invdigamma(x),
       inv(SpecialFunctions.trigamma(SpecialFunctions.invdigamma(x))))
@frule(R → R, SpecialFunctions.besselj1(x),
       (SpecialFunctions.besselj0(x) - SpecialFunctions.besselj(2, x)) / 2)
@frule(R → R, SpecialFunctions.bessely1(x),
       (SpecialFunctions.bessely0(x) - SpecialFunctions.bessely(2, x)) / 2)

# manually optimized `frule`s

function frule(::@domain({R → R}), ::typeof(SpecialFunctions.gamma), x)
    u = SpecialFunctions.gamma(x)
    return u, ẋ -> fchain(ẋ, @thunk(u * SpecialFunctions.digamma(x)))
end

function frule(::@domain({R → R}), ::typeof(SpecialFunctions.erfinv), x)
    u = SpecialFunctions.erfinv(x)
    return u, ẋ -> fchain(ẋ, @thunk(sqrt(π) / 2) * exp(u^2))
end

function frule(::@domain({R → R}), ::typeof(SpecialFunctions.erfcinv), x)
    u = SpecialFunctions.erfcinv(x)
    return u, ẋ -> fchain(ẋ, @thunk(-(sqrt(π) / 2) * exp(u^2)))
end

function frule(::@domain({R → R}), ::typeof(SpecialFunctions.erfcx), x)
    u = SpecialFunctions.erfcx(x)
    return u, ẋ -> fchain(ẋ, @thunk((2 * x * u) - (2 / sqrt(π))))
end

function frule(::@domain({R → R}), ::typeof(SpecialFunctions.dawson), x)
    u = SpecialFunctions.dawson(x)
    return u, ẋ -> fchain(ẋ, @thunk(1 - (2 * x * u)))
end
