# See also fastmath_able.jl for where rules are defined simple base functions
# that also have FastMath versions.

@scalar_rule one(x) zero(x)
@scalar_rule zero(x) zero(x)
@scalar_rule transpose(x) One()

# `adjoint`

frule((_, Δz), ::typeof(adjoint), z::Number) = (z', Δz')

function rrule(::typeof(adjoint), z::Number)
    adjoint_pullback(ΔΩ) = (NO_FIELDS, ΔΩ')
    return (z', adjoint_pullback)
end

# `real`

@scalar_rule real(x::Real) One()

frule((_, Δz), ::typeof(real), z::Number) = (real(z), real(Δz))

function rrule(::typeof(real), z::Number)
    # add zero(z) to embed the real number in the same number type as z
    real_pullback(ΔΩ) = (NO_FIELDS, real(ΔΩ) + zero(z))
    return (real(z), real_pullback)
end

# `imag`

@scalar_rule imag(x::Real) Zero()

frule((_, Δz), ::typeof(imag), z::Complex) = (imag(z), imag(Δz))

function rrule(::typeof(imag), z::Complex)
    imag_pullback(ΔΩ) = (NO_FIELDS, real(ΔΩ) * im)
    return (imag(z), imag_pullback)
end

# `Complex`

frule((_, Δz), ::Type{T}, z::Number) where {T<:Complex} = (T(z), Complex(Δz))
function frule((_, Δx, Δy), ::Type{T}, x::Number, y::Number) where {T<:Complex}
    return (T(x, y), Complex(Δx, Δy))
end

function rrule(::Type{T}, z::Complex) where {T<:Complex}
    Complex_pullback(ΔΩ) = (NO_FIELDS, Complex(ΔΩ))
    return (T(z), Complex_pullback)
end
function rrule(::Type{T}, x::Real) where {T<:Complex}
    Complex_pullback(ΔΩ) = (NO_FIELDS, real(ΔΩ))
    return (T(x), Complex_pullback)
end
function rrule(::Type{T}, x::Number, y::Number) where {T<:Complex}
    Complex_pullback(ΔΩ) = (NO_FIELDS, real(ΔΩ), imag(ΔΩ))
    return (T(x, y), Complex_pullback)
end

# `hypot`

@scalar_rule hypot(x::Real) sign(x)

function frule((_, Δz), ::typeof(hypot), z::Complex)
    Ω = hypot(z)
    ∂Ω = _realconjtimes(z, Δz) / ifelse(iszero(Ω), one(Ω), Ω)
    return Ω, ∂Ω
end

function rrule(::typeof(hypot), z::Complex)
    Ω = hypot(z)
    function hypot_pullback(ΔΩ)
        return (NO_FIELDS, (real(ΔΩ) / ifelse(iszero(Ω), one(Ω), Ω)) * z)
    end
    return (Ω, hypot_pullback)
end

@scalar_rule fma(x, y, z) (y, x, One())
@scalar_rule muladd(x, y::CommutativeMulNumber, z) (y, x, One())
@scalar_rule rem2pi(x, r::RoundingMode) (One(), DoesNotExist())
@scalar_rule(
    mod(x, y),
    @setup((u, nan) = promote(x / y, NaN16), isint = isinteger(x / y)),
    (ifelse(isint, nan, one(u)), ifelse(isint, nan, -floor(u))),
)

@scalar_rule deg2rad(x) π / oftype(x, 180)
@scalar_rule rad2deg(x) oftype(x, 180) / π

@scalar_rule(ldexp(x, y), (2^y, DoesNotExist()))

# Can't multiply though sqrt in acosh because of negative complex case for x
@scalar_rule acosh(x::CommutativeMulNumber) inv(sqrt(x - 1) * sqrt(x + 1))
@scalar_rule acoth(x::CommutativeMulNumber) inv(1 - x ^ 2)
@scalar_rule acsch(x::CommutativeMulNumber) -(inv(x ^ 2 * sqrt(1 + x ^ -2)))
@scalar_rule acsch(x::Real) -(inv(abs(x) * sqrt(1 + x ^ 2)))
@scalar_rule asech(x::CommutativeMulNumber) -(inv(x * sqrt(1 - x ^ 2)))
@scalar_rule asinh(x::CommutativeMulNumber) inv(sqrt(x ^ 2 + 1))
@scalar_rule atanh(x::CommutativeMulNumber) inv(1 - x ^ 2)


@scalar_rule acosd(x::CommutativeMulNumber) (-(oftype(x, 180)) / π) / sqrt(1 - x ^ 2)
@scalar_rule acotd(x::CommutativeMulNumber) (-(oftype(x, 180)) / π) / (1 + x ^ 2)
@scalar_rule acscd(x::CommutativeMulNumber) ((-(oftype(x, 180)) / π) / x ^ 2) / sqrt(1 - x ^ -2)
@scalar_rule acscd(x::Real) ((-(oftype(x, 180)) / π) / abs(x)) / sqrt(x ^ 2 - 1)
@scalar_rule asecd(x::CommutativeMulNumber) ((oftype(x, 180) / π) / x ^ 2) / sqrt(1 - x ^ -2)
@scalar_rule asecd(x::Real) ((oftype(x, 180) / π) / abs(x)) / sqrt(x ^ 2 - 1)
@scalar_rule asind(x::CommutativeMulNumber) (oftype(x, 180) / π) / sqrt(1 - x ^ 2)
@scalar_rule atand(x::CommutativeMulNumber) (oftype(x, 180) / π) / (1 + x ^ 2)

@scalar_rule cot(x::CommutativeMulNumber) -((1 + Ω ^ 2))
@scalar_rule coth(x::CommutativeMulNumber) -(csch(x) ^ 2)
@scalar_rule cotd(x::CommutativeMulNumber) -(π / oftype(x, 180)) * (1 + Ω ^ 2)
@scalar_rule csc(x::CommutativeMulNumber) -Ω * cot(x)
@scalar_rule cscd(x::CommutativeMulNumber) -(π / oftype(x, 180)) * Ω * cotd(x)
@scalar_rule csch(x::CommutativeMulNumber) -(coth(x)) * Ω
@scalar_rule sec(x::CommutativeMulNumber) Ω * tan(x)
@scalar_rule secd(x::CommutativeMulNumber) (π / oftype(x, 180)) * Ω * tand(x)
@scalar_rule sech(x::CommutativeMulNumber) -(tanh(x)) * Ω

@scalar_rule acot(x::CommutativeMulNumber) -(inv(1 + x ^ 2))
@scalar_rule acsc(x::CommutativeMulNumber) -(inv(x ^ 2 * sqrt(1 - x ^ -2)))
@scalar_rule acsc(x::Real) -(inv(abs(x) * sqrt(x ^ 2 - 1)))
@scalar_rule asec(x::CommutativeMulNumber) inv(x ^ 2 * sqrt(1 - x ^ -2))
@scalar_rule asec(x::Real) inv(abs(x) * sqrt(x ^ 2 - 1))

@scalar_rule cosd(x::CommutativeMulNumber) -(π / oftype(x, 180)) * sind(x)
@scalar_rule cospi(x::CommutativeMulNumber) -π * sinpi(x)
@scalar_rule sind(x::CommutativeMulNumber) (π / oftype(x, 180)) * cosd(x)
@scalar_rule sinpi(x::CommutativeMulNumber) π * cospi(x)
@scalar_rule tand(x::CommutativeMulNumber) (π / oftype(x, 180)) * (1 + Ω ^ 2)

@scalar_rule sinc(x::CommutativeMulNumber) cosc(x)

@scalar_rule(
    clamp(x, low, high),
    @setup(
        islow = x < low,
        ishigh = high < x,
    ),
    (!(islow | ishigh), islow, ishigh),
)
@scalar_rule x::CommutativeMulNumber \ y (-(x \ Ω), x \ one(y))

function frule((_, ẏ), ::typeof(identity), x)
    return (x, ẏ)
end

function rrule(::typeof(identity), x)
    function identity_pullback(ȳ)
        return (NO_FIELDS, ȳ)
    end
    return (x, identity_pullback)
end

