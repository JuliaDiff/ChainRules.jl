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

    function _evalpoly_intermediates(x, p::Tuple)
        return if @generated
            N = length(p.parameters)
            defs = []
            vars = []
            ex = :(p[$N])
            for i in (N - 1):-1:1
                yi = Symbol("y", i + 1)
                push!(vars, yi)
                push!(defs, :($yi = $ex))
                ex = :(muladd(x, $yi, p[$i]))
            end
            push!(defs, :(y1 = $ex))
            Expr(:block, defs..., :(y1, ($(vars...), y1)))
        else # fallback when can't generate code
            N = length(p)
            y = one(x) * p[N]
            ntuple(N) do i
                i == 1 && return y
                return y = muladd(x, y, p[N - i + 1])
            end
            y, ys
        end
    end
    function _evalpoly_intermediates(x, p)
        N = length(p)
        @inbounds yn = one(x) * p[N]
        ys = similar(p, typeof(yn))
        @inbounds ys[1] = yn
        @inbounds for i in 2:N
            ys[i] = muladd(x, ys[i - 1], p[N - i + 1])
        end
        @inbounds y = ys[N]
        return y, ys
    end

    # TODO: Handle following cases
    #     1) x is a UniformScaling, pᵢ is a matrix
    #     2) x is a matrix, pᵢ is a UniformScaling
    function _evalpoly_back(x, p::Tuple, ys, Δy)
        x′ = x'
        ∂x = zero(Δy)
        ∂p = if @generated
            defs = []
            vars = []
            ex = :(Δy)
            N = length(p.parameters)
            for i in 1:(N - 1)
                ∂yi = Symbol("∂y", i)
                push!(vars, ∂yi)
                push!(defs, :($∂yi = $ex))
                push!(defs, :(∂x = muladd($∂yi, ys[$(N - i)]', ∂x)))
                ex = :(x′ * $∂yi)
            end
            Expr(:block, defs..., :($(vars...), $ex))
        else # fallback when can't generate code
            ∂yi = Δy
            N = length(p)
            ntuple(N) do i
                i == 1 && return ∂yi
                ∂x = muladd(∂yi, ys[N - i + 1]', ∂x)
                return ∂yi = x′ * ∂yi
            end
        end
        return ∂x, Composite{typeof(p),typeof(∂p)}(∂p)
    end
    function _evalpoly_back(x, p, ys, Δy)
        x′ = x'
        ∂p1 = one(x′) * Δy
        ∂p = similar(p, typeof(∂p1))
        @inbounds ∂p[1] = ∂p1
        ∂x = zero(Δy)
        N = length(p)
        @inbounds for i in 1:(N - 1)
            ∂x = muladd(∂p[i], ys[N - i]', ∂x)
            ∂p[i + 1] = x′ * ∂p[i]
        end
        return ∂x, ∂p
    end
end
