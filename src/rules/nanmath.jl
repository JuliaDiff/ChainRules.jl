#####
##### forward rules
#####

# simple `@frule`s

@frule(R → R, NaNMath.sin(x), NaNMath.cos(x))
@frule(R → R, NaNMath.cos(x), -NaNMath.sin(x))
@frule(R → R, NaNMath.asin(x), inv(NaNMath.sqrt(1 - NaNMath.pow(x, 2))))
@frule(R → R, NaNMath.acos(x), -inv(NaNMath.sqrt(1 - NaNMath.pow(x, 2))))
@frule(R → R, NaNMath.acosh(x), inv(NaNMath.sqrt(NaNMath.pow(x, 2) - 1)))
@frule(R → R, NaNMath.atanh(x), inv(1 - NaNMath.pow(x, 2)))
@frule(R → R, NaNMath.log(x), inv(x))
@frule(R → R, NaNMath.log2(x), inv(x) / NaNMath.log(2f0))
@frule(R → R, NaNMath.log10(x), inv(x) / NaNMath.log(10f0))
@frule(R → R, NaNMath.log1p(x), inv(x + 1))
@frule(R → R, NaNMath.lgamma(x), SpecialFunctions.digamma(x))

# manually optimized `frule`s

frule(::@domain({R → R}), ::typeof(NaNMath.sqrt), x) = (u = NaNMath.sqrt(x); (u, ẋ -> fchain(ẋ, @thunk(inv(2 * u)))))
frule(::@domain({R → R}), ::typeof(NaNMath.tan), x) = (u = NaNMath.tan(x); (u, ẋ -> fchain(ẋ, @thunk(1 + NaNMath.pow(u, 2)))))

function frule(::@domain({R×R → R}), ::typeof(NaNMath.pow), x, y)
    z = NaNMath.pow(x, y)
    return z, (ẋ, ẏ) -> fchain(ẋ, @thunk(y * NaNMath.pow(x, y - 1)), ẏ, @thunk(z * NaNMath.log(x)))
end
