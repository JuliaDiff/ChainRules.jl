# See also fastmath_able.jl for where rules are defined simple base functions
# that also have FastMath versions.

@scalar_rule copysign(y, x) (ifelse(signbit(x)!=signbit(y), -one(y), +one(y)), NoTangent())
@scalar_rule transpose(x) true

# `zero`

function frule((_, _), ::typeof(zero), x)
    return (zero(x), ZeroTangent())
end

function rrule(::typeof(zero), x)
    zero_pullback = Returns((NoTangent(), ZeroTangent()))
    return (zero(x), zero_pullback)
end

# `one`

function frule((_, _), ::typeof(one), x)
    return (one(x), ZeroTangent())
end

function rrule(::typeof(one), x)
    one_pullback = Returns((NoTangent(), ZeroTangent()))
    return (one(x), one_pullback)
end


function ChainRulesCore.frule((_, ȯbj, _, ẋ), ::typeof(setfield!), obj, field, x)
    ȯbj::MutableTangent
    y = setfield!(obj, field, x)
    ẏ = setproperty!(ȯbj, field, ẋ)
    return y, ẏ
end

# `adjoint`

frule((_, Δz), ::typeof(adjoint), z::Number) = (z', Δz')

function rrule(::typeof(adjoint), z::Number)
    adjoint_pullback(ΔΩ) = (NoTangent(), ΔΩ')
    return (z', adjoint_pullback)
end

# `real`

@scalar_rule real(x::Real) true

frule((_, Δz), ::typeof(real), z::Number) = (real(z), real(Δz))

function rrule(::typeof(real), z::Number)
    # add zero(z) to embed the real number in the same number type as z
    real_pullback(ΔΩ) = (NoTangent(), real(ΔΩ) + zero(z))
    return (real(z), real_pullback)
end

# Conversions to Float

@scalar_rule float(x) true
@scalar_rule Float64(x::Real) true
@scalar_rule Float32(x::Real) true
@scalar_rule AbstractFloat(x::Real) true

# `imag`

@scalar_rule imag(x::Real) ZeroTangent()

frule((_, Δz), ::typeof(imag), z::Complex) = (imag(z), imag(Δz))

function rrule(::typeof(imag), z::Complex)
    imag_pullback(ΔΩ) = (NoTangent(), real(ΔΩ) * im)
    return (imag(z), imag_pullback)
end

# `Complex`

frule((_, Δz), ::Type{T}, z::Number) where {T<:Complex} = (T(z), Complex(Δz))
function frule((_, Δx, Δy), ::Type{T}, x::Number, y::Number) where {T<:Complex}
    return (T(x, y), Complex(Δx, Δy))
end

function rrule(::Type{T}, z::Complex) where {T<:Complex}
    Complex_pullback(ΔΩ) = (NoTangent(), Complex(ΔΩ))
    return (T(z), Complex_pullback)
end
function rrule(::Type{T}, x::Real) where {T<:Complex}
    Complex_pullback(ΔΩ) = (NoTangent(), real(ΔΩ))
    return (T(x), Complex_pullback)
end
function rrule(::Type{T}, x::Number, y::Number) where {T<:Complex}
    project_x = ProjectTo(x)
    project_y = ProjectTo(y)

    function Complex_pullback(Ω̄)
        ΔΩ = unthunk(Ω̄)
        return (NoTangent(), project_x(real(ΔΩ)), project_y(imag(ΔΩ)))
    end
    return (T(x, y), Complex_pullback)
end

@scalar_rule complex(x) true

# `hypot`

@scalar_rule hypot(x::Real) sign(x)

function frule((_, Δz), ::typeof(hypot), z::Complex)
    Ω = hypot(z)
    ∂Ω = realdot(z, Δz) / ifelse(iszero(Ω), one(Ω), Ω)
    return Ω, ∂Ω
end

function rrule(::typeof(hypot), z::Complex)
    Ω = hypot(z)
    function hypot_pullback(ΔΩ)
        return (NoTangent(), (real(ΔΩ) / ifelse(iszero(Ω), one(Ω), Ω)) * z)
    end
    return (Ω, hypot_pullback)
end

@scalar_rule fma(x, y, z) (y, x, true)
@scalar_rule muladd(x, y, z) (y, x, true)
@scalar_rule muladd(x::Union{Number, ZeroTangent}, y::Union{Number, ZeroTangent}, z::Union{Number, ZeroTangent}) (y, x, true)
@scalar_rule rem2pi(x, r::RoundingMode) (true, NoTangent())
@scalar_rule(
    mod(x, y),
    @setup((u, nan) = promote(x / y, NaN16), isint = isinteger(x / y)),
    (ifelse(isint, nan, one(u)), ifelse(isint, nan, -floor(u))),
)

@scalar_rule deg2rad(x) deg2rad(one(x))
@scalar_rule rad2deg(x) rad2deg(one(x))

@scalar_rule(ldexp(x, y), (2^y, NoTangent()))

# Can't multiply though sqrt in acosh because of negative complex case for x
@scalar_rule acosh(x) inv(sqrt(x - 1) * sqrt(x + 1))
@scalar_rule acoth(x) inv(1 - x ^ 2)
@scalar_rule acsch(x) -(inv(x ^ 2 * sqrt(1 + x ^ -2)))
@scalar_rule acsch(x::Real) -(inv(abs(x) * sqrt(1 + x ^ 2)))
@scalar_rule asech(x) -(inv(x * sqrt(1 - x ^ 2)))
@scalar_rule asinh(x) inv(sqrt(x ^ 2 + 1))
@scalar_rule atanh(x) inv(1 - x ^ 2)


@scalar_rule acosd(x) -inv(deg2rad(sqrt(1 - x ^ 2)))
@scalar_rule acotd(x) -inv(deg2rad(1 + x ^ 2))
@scalar_rule acscd(x) -inv(deg2rad(x^2 * sqrt(1 - x ^ -2)))
@scalar_rule acscd(x::Real) -inv(deg2rad(abs(x) * sqrt(x ^ 2 - 1)))
@scalar_rule asecd(x) inv(deg2rad(x ^ 2 * sqrt(1 - x ^ -2)))
@scalar_rule asecd(x::Real) inv(deg2rad(abs(x) * sqrt(x ^ 2 - 1)))
@scalar_rule asind(x) inv(deg2rad(sqrt(1 - x ^ 2)))
@scalar_rule atand(x) inv(deg2rad(1 + x ^ 2))

@scalar_rule cot(x) -((1 + Ω ^ 2))
@scalar_rule coth(x) -(csch(x) ^ 2)
@scalar_rule cotd(x) -deg2rad(1 + Ω ^ 2)
@scalar_rule csc(x) -Ω * cot(x)
@scalar_rule cscd(x) -deg2rad(Ω * cotd(x))
@scalar_rule csch(x) -(coth(x)) * Ω
@scalar_rule sec(x) Ω * tan(x)
@scalar_rule secd(x) deg2rad(Ω * tand(x))
@scalar_rule sech(x) -(tanh(x)) * Ω

@scalar_rule acot(x) -(inv(1 + x ^ 2))
@scalar_rule acsc(x) -(inv(x ^ 2 * sqrt(1 - x ^ -2)))
@scalar_rule acsc(x::Real) -(inv(abs(x) * sqrt(x ^ 2 - 1)))
@scalar_rule asec(x) inv(x ^ 2 * sqrt(1 - x ^ -2))
@scalar_rule asec(x::Real) inv(abs(x) * sqrt(x ^ 2 - 1))

@scalar_rule cosd(x) -deg2rad(sind(x))
@scalar_rule cospi(x) -π * sinpi(x)
@scalar_rule sind(x) deg2rad(cosd(x))
@scalar_rule sinpi(x) π * cospi(x)
@scalar_rule tand(x) deg2rad(1 + Ω ^ 2)

@scalar_rule sinc(x) cosc(x)

# the position of the minus sign below warrants the correct type for π  
@scalar_rule sincospi(x) @setup((sinpix, cospix) = Ω) (π * cospix)  (π * (-sinpix))

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
        return (NoTangent(), ȳ)
    end
    return (x, identity_pullback)
end

ChainRulesCore.derivatives_given_output(Ω, ::typeof(identity), x) = tuple(tuple(true))

# rounding related,
# we use `zero` rather than `ZeroTangent()` for scalar, and avoids issues with map etc
@scalar_rule round(x) zero(x)
@scalar_rule floor(x) zero(x)
@scalar_rule ceil(x) zero(x)

# `literal_pow`
# This is mostly handled by AD; it's a micro-optimisation to provide a gradient for x*x*x
# Note that rules for `^` are defined in the fastmath_able.jl

function frule((_, _, Δx, _), ::typeof(Base.literal_pow), ::typeof(^), x::Real, ::Val{2})
    return x * x, 2 * x * Δx
end
function frule((_, _, Δx, _), ::typeof(Base.literal_pow), ::typeof(^), x::Real, ::Val{3})
    x2 = x * x
    return x2 * x, 3 * x2 * Δx
end

function rrule(::typeof(Base.literal_pow), ::typeof(^), x::Real, ::Val{2})
    square_pullback(dy) = (NoTangent(), NoTangent(), ProjectTo(x)(2 * x * dy), NoTangent())
    return x * x, square_pullback
end
function rrule(::typeof(Base.literal_pow), ::typeof(^), x::Real, ::Val{3})
    x2 = x * x
    cube_pullback(dy) = (NoTangent(), NoTangent(), ProjectTo(x)(3 * x2 * dy), NoTangent())
    return x2 * x, cube_pullback
end

#####
##### `map`
#####

# Ideally reverse mode should always iterate in reverse order. For `map` and broadcasting
# this may matter with a stateful `f`, but in general their order isn't guaranteed anyway,
# so it's unclear how much effort should be spent on that. But `map` on Tuples normally
# gets unrolled, so perhaps it does guarantee order, and reversing it should be cheap too.

function rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(map), f::F, xs::Tuple...) where {F}
    length_y = minimum(length, xs)
    hobbits = ntuple(length_y) do i
        args = getindex.(xs, i)
        rrule_via_ad(config, f, args...)
    end
    y = map(first, hobbits)
    num_xs = Val(length(xs))
    paddings = map(x -> ntuple(Returns(NoTangent()), (length(x) - length_y)), xs)
    all(isempty, paddings) || @error """map(f, xs::Tuple...) does not allow mistmatched lengths!
        But its `rrule` does; when JuliaLang/julia #42216 is fixed this warning should be removed."""
    function map_pullback(dy_raw)
        dy = unthunk(dy_raw)
        # We want to call the pullbacks in `rrule_via_ad` in reverse sequence to the forward pass:
        backevals = ntuple(length_y) do i
            rev_i = length_y - i + 1
            last(hobbits[rev_i])(dy[rev_i])
        end |> reverse
        # This df doesn't infer, could test Base.issingletontype(F), but it's not the only inference problem.
        df = ProjectTo(f)(sum(first, backevals))
        # Now unzip that. Because `map` like `zip` should when any `x` stops, some `dx`s may need padding.
        # Although in fact, `map(+, (1,2), (3,4,5))` is an error... https://github.com/JuliaLang/julia/issues/42216
        dxs = ntuple(num_xs) do k
            dx_short = map(bv -> bv[k+1], backevals)
            ProjectTo(xs[k])((dx_short..., paddings[k]...))  # ProjectTo makes the Tangent for us
        end
        return (NoTangent(), df, dxs...)
    end
    map_back(dy::AbstractZero) = (NoTangent(), NoTangent(), ntuple(Returns(NoTangent()), num_xs)...)
    return y, map_pullback
end

#####
##### `task_local_storage`
#####

# Called by `@allowscalar` from GPUArrays

ChainRules.@non_differentiable task_local_storage(key::Any)
ChainRules.@non_differentiable task_local_storage(key::Any, value::Any)

function rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(task_local_storage), body::Function, key, value)
    y, back = task_local_storage(key, value) do
        rrule_via_ad(config, body)
    end
    function task_local_storage_pullback(dy)
        dbody = only(back(dy))
        return (NoTangent(), dbody, NoTangent(), NoTangent())
    end
    return y, task_local_storage_pullback
end

####
#### merge
####
# need to work around inability to return closures from generated functions
struct MergePullback{T1,T2} end
(this::MergePullback)(dy::AbstractThunk) = this(unthunk(dy))
(::MergePullback)(x::AbstractZero) = (NoTangent(), x, x)
@generated function (::MergePullback{T1,T2})(
    dy::Tangent
) where {F1,T1<:NamedTuple{F1},F2,T2<:NamedTuple{F2}}
    _getproperty_kwexpr(key) = :($key = getproperty(dy, $(Meta.quot(key))))
    quote
        dnt1 = Tangent{T1}(; $(map(_getproperty_kwexpr, setdiff(F1, F2))...))
        dnt2 = Tangent{T2}(; $(map(_getproperty_kwexpr, F2)...))
        return (NoTangent(), dnt1, dnt2)
    end
end

function rrule(::typeof(merge), nt1::T1, nt2::T2) where {T1<:NamedTuple,T2<:NamedTuple}
    y = merge(nt1, nt2)
    return y, MergePullback{T1,T2}()
end
