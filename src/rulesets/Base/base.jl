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
@scalar_rule muladd(x, y, z) (y, x, One())
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
@scalar_rule acosh(x) inv(sqrt(x - 1) * sqrt(x + 1))
@scalar_rule acoth(x) inv(1 - x ^ 2)
@scalar_rule acsch(x) -(inv(x ^ 2 * sqrt(1 + x ^ -2)))
@scalar_rule acsch(x::Real) -(inv(abs(x) * sqrt(1 + x ^ 2)))
@scalar_rule asech(x) -(inv(x * sqrt(1 - x ^ 2)))
@scalar_rule asinh(x) inv(sqrt(x ^ 2 + 1))
@scalar_rule atanh(x) inv(1 - x ^ 2)


@scalar_rule acosd(x) (-(oftype(x, 180)) / π) / sqrt(1 - x ^ 2)
@scalar_rule acotd(x) (-(oftype(x, 180)) / π) / (1 + x ^ 2)
@scalar_rule acscd(x) ((-(oftype(x, 180)) / π) / x ^ 2) / sqrt(1 - x ^ -2)
@scalar_rule acscd(x::Real) ((-(oftype(x, 180)) / π) / abs(x)) / sqrt(x ^ 2 - 1)
@scalar_rule asecd(x) ((oftype(x, 180) / π) / x ^ 2) / sqrt(1 - x ^ -2)
@scalar_rule asecd(x::Real) ((oftype(x, 180) / π) / abs(x)) / sqrt(x ^ 2 - 1)
@scalar_rule asind(x) (oftype(x, 180) / π) / sqrt(1 - x ^ 2)
@scalar_rule atand(x) (oftype(x, 180) / π) / (1 + x ^ 2)

@scalar_rule cot(x) -((1 + Ω ^ 2))
@scalar_rule coth(x) -(csch(x) ^ 2)
@scalar_rule cotd(x) -(π / oftype(x, 180)) * (1 + Ω ^ 2)
@scalar_rule csc(x) -Ω * cot(x)
@scalar_rule cscd(x) -(π / oftype(x, 180)) * Ω * cotd(x)
@scalar_rule csch(x) -(coth(x)) * Ω
@scalar_rule sec(x) Ω * tan(x)
@scalar_rule secd(x) (π / oftype(x, 180)) * Ω * tand(x)
@scalar_rule sech(x) -(tanh(x)) * Ω

@scalar_rule acot(x) -(inv(1 + x ^ 2))
@scalar_rule acsc(x) -(inv(x ^ 2 * sqrt(1 - x ^ -2)))
@scalar_rule acsc(x::Real) -(inv(abs(x) * sqrt(x ^ 2 - 1)))
@scalar_rule asec(x) inv(x ^ 2 * sqrt(1 - x ^ -2))
@scalar_rule asec(x::Real) inv(abs(x) * sqrt(x ^ 2 - 1))

@scalar_rule cosd(x) -(π / oftype(x, 180)) * sind(x)
@scalar_rule cospi(x) -π * sinpi(x)
@scalar_rule sind(x) (π / oftype(x, 180)) * cosd(x)
@scalar_rule sinpi(x) π * cospi(x)
@scalar_rule tand(x) (π / oftype(x, 180)) * (1 + Ω ^ 2)

@scalar_rule(
    clamp(x, low, high),
    @setup(
        islow = x < low,
        ishigh = high < x,
    ),
    (!(islow | ishigh), islow, ishigh),
)
@scalar_rule x \ y (-(Ω / x), one(y) / x)

function frule((_, ẏ), ::typeof(identity), x)
    return (x, ẏ)
end

function rrule(::typeof(identity), x)
    function identity_pullback(ȳ)
        return (NO_FIELDS, ȳ)
    end
    return (x, identity_pullback)
end

#####
##### `evalpoly`
#####

if VERSION ≥ v"1.4"
    function frule((_, Δx, Δp), ::typeof(evalpoly), x, p)
        N = length(p)
        @inbounds y = p[N]
        Δy = Δp[N]
        @inbounds for i in (N - 1):-1:1
            Δy = muladd(Δx, y, muladd(x, Δy, Δp[i]))
            y = muladd(x, y, p[i])
        end
        return y, Δy
    end

    function rrule(::typeof(evalpoly), x, p)
        y, ys = _evalpoly_intermediates(x, p)
        function evalpoly_pullback(Δy)
            ∂x, ∂p = _evalpoly_back(x, p, ys, Δy)
            return NO_FIELDS, ∂x, ∂p
        end
        return y, evalpoly_pullback
    end

    # evalpoly but storing intermediates
    function _evalpoly_intermediates(x, p::Tuple)
        return if @generated
            N = length(p.parameters)
            exs = []
            vars = []
            ex = :(p[$N])
            for i in 1:(N - 1)
                yi = Symbol("y", i)
                push!(vars, yi)
                push!(exs, :($yi = $ex))
                ex = :(muladd(x, $yi, p[$(N - i)]))
            end
            push!(exs, :(y = $ex))
            Expr(:block, exs..., :(y, ($(vars...),)))
        else
            _evalpoly_intermediates_fallback(x, p)
        end
    end
    function _evalpoly_intermediates_fallback(x, p::Tuple)
        N = length(p)
        y = p[N]
        ys = (y, ntuple(N - 2) do i
            return y = muladd(x, y, p[N - i])
        end...)
        y = muladd(x, y, p[1])
        return y, ys
    end
    function _evalpoly_intermediates(x, p)
        N = length(p)
        @inbounds yn = one(x) * p[N]
        ys = similar(p, typeof(yn), N - 1)
        @inbounds ys[1] = yn
        @inbounds for i in 2:(N - 1)
            ys[i] = muladd(x, ys[i - 1], p[N - i + 1])
        end
        @inbounds y = muladd(x, ys[N - 1], p[1])
        return y, ys
    end

    # TODO: Handle following cases
    #     1) x is a UniformScaling, pᵢ is a matrix
    #     2) x is a matrix, pᵢ is a UniformScaling
    @inline _evalpoly_backx(x, yi, ∂yi) = ∂yi * yi'
    @inline _evalpoly_backx(x, yi, ∂x, ∂yi) = muladd(∂yi, yi', ∂x)
    @inline _evalpoly_backx(x::Number, yi, ∂yi) = conj(dot(∂yi, yi))
    @inline _evalpoly_backx(x::Number, yi, ∂x, ∂yi) = _evalpoly_backx(x, yi, ∂yi) + ∂x

    @inline _evalpoly_backp(pi, ∂yi) = ∂yi

    function _evalpoly_back(x, p::Tuple, ys, Δy)
        return if @generated
            exs = []
            vars = []
            N = length(p.parameters)
            for i in 2:(N - 1)
                ∂pi = Symbol("∂p", i)
                push!(vars, ∂pi)
                push!(exs, :(∂x = _evalpoly_backx(x, ys[$(N - i)], ∂x, ∂yi)))
                push!(exs, :($∂pi = _evalpoly_backp(p[$i], ∂yi)))
                push!(exs, :(∂yi = x′ * ∂yi))
            end
            push!(vars, :(_evalpoly_backp(p[$N], ∂yi))) # ∂pN
            Expr(
                :block,
                :(x′ = x'),
                :(∂yi = Δy),
                :(∂p1 = _evalpoly_backp(p[1], ∂yi)),
                :(∂x = _evalpoly_backx(x, ys[$(N - 1)], ∂yi)),
                :(∂yi = x′ * ∂yi),
                exs...,
                :(∂p = (∂p1, $(vars...))),
                :(∂x, Composite{typeof(p),typeof(∂p)}(∂p)),
            )
        else
            _evalpoly_back_fallback(x, p, ys, Δy)
        end
    end
    function _evalpoly_back_fallback(x, p::Tuple, ys, Δy)
        x′ = x'
        ∂yi = Δy
        N = length(p)
        ∂p1 = _evalpoly_backp(p[1], ∂yi)
        ∂x = _evalpoly_backx(x, ys[N - 1], ∂yi)
        ∂yi = x′ * ∂yi
        ∂p = (
            ∂p1,
            ntuple(N - 2) do i
                ∂x = _evalpoly_backx(x, ys[N-i-1], ∂x, ∂yi)
                ∂pi = _evalpoly_backp(p[i+1], ∂yi)
                ∂yi = x′ * ∂yi
                return ∂pi
            end...,
            _evalpoly_backp(p[N], ∂yi), # ∂pN
        )
        return ∂x, Composite{typeof(p),typeof(∂p)}(∂p)
    end
    function _evalpoly_back(x, p, ys, Δy)
        x′ = x'
        ∂yi = one(x′) * Δy
        N = length(p)
        @inbounds ∂p1 = _evalpoly_backp(p[1], ∂yi)
        ∂p = similar(p, typeof(∂p1))
        @inbounds begin
            ∂x = _evalpoly_backx(x, ys[N - 1], ∂yi)
            ∂yi = x′ * ∂yi
            ∂p[1] = ∂p1
            for i in 2:(N - 1)
                ∂x = _evalpoly_backx(x, ys[N - i], ∂x, ∂yi)
                ∂p[i] = _evalpoly_backp(p[i], ∂yi)
                ∂yi = x′ * ∂yi
            end
            ∂p[N] = _evalpoly_backp(p[N], ∂yi)
        end
        return ∂x, ∂p
    end
end
