# See also fastmath_able.jl for where rules are defined simple base functions
# that also have FastMath versions.

@scalar_rule one(x) Zero()
@scalar_rule zero(x) Zero()
@scalar_rule adjoint(x::Real) One()
@scalar_rule transpose(x) One()
@scalar_rule imag(x::Real) Zero()
@scalar_rule hypot(x::Real) sign(x)


@scalar_rule fma(x, y, z) (y, x, One())
@scalar_rule muladd(x, y, z) (y, x, One())
@scalar_rule real(x::Real) One()
@scalar_rule rem2pi(x, r::RoundingMode) (One(), DoesNotExist())
@scalar_rule(
    mod(x, y),
    @setup((u, nan) = promote(x / y, NaN16), isint = isinteger(x / y)),
    (ifelse(isint, nan, one(u)), ifelse(isint, nan, -floor(u))),
)


@scalar_rule deg2rad(x) π / oftype(x, 180)
@scalar_rule rad2deg(x) oftype(x, 180) / π


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

@scalar_rule x \ y (-((y / x) / x), inv(x))

function frule((_, ẏ), ::typeof(identity), x)
    return (x, ẏ)
end

function rrule(::typeof(identity), x)
    function identity_pullback(ȳ)
        return (NO_FIELDS, ȳ)
    end
    return (x, identity_pullback)
end

function rrule(::typeof(identity), x::Tuple)
    # `identity(::Tuple)` returns multiple outputs;because that is how we think of
    # returning a tuple, so its pullback needs to accept multiple inputs.
    # `identity(::Tuple)` has one input, so its pullback should return 1 matching output
    # see https://github.com/JuliaDiff/ChainRulesCore.jl/issues/152
    function identity_pullback(ȳs...)
        return (NO_FIELDS, Composite{typeof(x)}(ȳs...))
    end
    return x, identity_pullback
end

#####
##### `evalpoly`
#####

if VERSION ≥ v"1.4"
    function frule((_, Δx, Δp), ::typeof(evalpoly), x, p)
        q = _evalpoly_dxcoef(p)
        return evalpoly(x, p), evalpoly(x, Δp) + evalpoly(x, q) * Δx
    end

    function rrule(::typeof(evalpoly), x, p)
        function evalpoly_pullback(Δy)
            ∂x = @thunk _evalpoly_backx(Δy, x, p)
            ∂p = @thunk _evalpoly_backp(Δy, x, p)
            return NO_FIELDS, ∂x, ∂p
        end
        return evalpoly(x, p), evalpoly_pullback
    end

    function _evalpoly_dxcoef(p::Tuple)
        return if @generated
            N = length(p.parameters)
            exs = ntuple(i -> :($i * p[$(i + 1)]), N - 1)
            :($(exs...),)
        else # fallback in case code generation not possible
            ntuple(i -> i * p[i + 1], length(p) - 1)
        end
    end
    function _evalpoly_dxcoef(p::AbstractVector)
        N = length(p)
        @inbounds q = (1:(N - 1)) .* view(p, 2:N)
        return q
    end

    # TODO: Handle when x is a UniformScaling, p is a matrix
    function _evalpoly_backx(Δy, x, p)
        q = _evalpoly_dxcoef(p)
        return evalpoly(x, q)' * Δy
    end

    # This is a geometric progression, that is ∂p = Δy * (I, x, x², …, xⁿ)'
    # TODO: Handle when x is a matrix, p is a UniformScaling
    function _evalpoly_backp(Δy, x, p::Tuple)
        x′ = x'
        ∂p = if @generated
            N = length(p.parameters)
            exs = []
            ∂pis = []
            a = :(Δy)
            for i in 1:(N - 1)
                ∂pi = Symbol("∂p", i)
                push!(∂pis, ∂pi)
                push!(exs, :($∂pi = $a))
                a = :($∂pi * x′)
            end
            ∂pN = Symbol("∂p", N)
            push!(exs, :($∂pN = $a))
            push!(∂pis, ∂pN)
            Expr(:block, exs..., :($(∂pis...),))
        else # fallback in case code generation not possible
            N = length(p)
            ∂pi = Δy
            ntuple(N) do i
                i == 1 && return Δy
                return ∂pi *= x′
            end
        end
        return Composite{typeof(p)}(∂p...)
    end
    function _evalpoly_backp(Δy, x, p::AbstractVector)
        ∂p = similar(p, typeof(Δy * x))
        N = length(∂p)
        x′ = x'
        ∂p[1] = Δy
        for i in 2:N
            @inbounds ∂p[i] = ∂p[i - 1] * x′
        end
        return ∂p
    end
end
