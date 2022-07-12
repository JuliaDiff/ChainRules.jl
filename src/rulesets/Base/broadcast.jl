using Base.Broadcast: Broadcast, broadcasted, Broadcasted
const RCR = RuleConfig{>:HasReverseMode}

function rrule(::typeof(copy), bc::Broadcasted)
    uncopy(Δ) = (NoTangent(), Δ)
    return copy(bc), uncopy
end

# Skip AD'ing through the axis computation
function rrule(::typeof(Broadcast.instantiate), bc::Broadcasted)
    uninstantiate(Δ) = (NoTangent(), Δ)
    return Broadcast.instantiate(bc), uninstantiate
end

_print(args...) = nothing # println(join(args, " ")) # 

#####
##### Split broadcasting
#####

# For `z = g.(f.(xs))`, this finds `y = f.(x)` eagerly because the rules for either `f` or `g` may need it,
# and we don't know whether re-computing `y` is cheap. 
# (We could check `f` first like `sum(f, x)` does, but checking whether `g` needs `y` is tricky.)

function rrule(cfg::RCR, ::typeof(broadcasted), f::F, args::Vararg{Any,N}) where {F,N}
    T = Broadcast.combine_eltypes(f, args)
    if T === Bool  # TODO use nondifftype here
        # 1: Trivial case: non-differentiable output, e.g. `x .> 0`
        _print("split_bc_trivial", f)
        bc_trivial_back(_) = (NoTangent(), NoTangent(), ntuple(Returns(ZeroTangent()), length(args))...)
        return f.(args...), bc_trivial_back
    elseif T <: Number && may_bc_derivatives(T, f, args...)
        # 2: Fast path: use arguments & result to find derivatives.
        return split_bc_derivatives(f, args...)
    elseif T <: Number && may_bc_forwards(cfg, f, args...)
        # 3: Future path: use `frule_via_ad`?
        return split_bc_forwards(cfg, f, args...)
    else
        # 4: Slow path: collect all the pullbacks & apply them later.
        return split_bc_pullbacks(cfg, f, args...)
    end
end

# Path 2: This is roughly what `derivatives_given_output` is designed for, should be fast.

function may_bc_derivatives(::Type{T}, f::F, args::Vararg{Any,N}) where {T,F,N}
    TΔ = Core.Compiler._return_type(derivatives_given_output, Tuple{T, F, map(_eltype, args)...})
    return isconcretetype(TΔ)
end

_eltype(x) = eltype(x)  # ... but try harder to avoid `eltype(Broadcast.broadcasted(+, [1,2,3], 4.5)) == Any`:
_eltype(bc::Broadcast.Broadcasted) = Broadcast.combine_eltypes(bc.f, bc.args)

function split_bc_derivatives(f::F, arg) where {F}
    _print("split_bc_derivative", f)
    ys = f.(arg)
    function bc_one_back(dys)  # For f.(x) we do not need StructArrays / unzip at all
        delta = broadcast(unthunk(dys), ys, arg) do dy, y, a
            das = only(derivatives_given_output(y, f, a))
            dy * conj(only(das))  # possibly this * should be made nan-safe.
        end
        return (NoTangent(), NoTangent(), ProjectTo(arg)(delta))
    end
    bc_one_back(z::AbstractZero) = (NoTangent(), NoTangent(), z)
    return ys, bc_one_back
end
function split_bc_derivatives(f::F, args::Vararg{Any,N}) where {F,N}
    _print("split_bc_derivatives", f, N)
    ys = f.(args...)
    function bc_many_back(dys)
        deltas = tuplecast(unthunk(dys), ys, args...) do dy, y, as...
            das = only(derivatives_given_output(y, f, as...))
            map(da -> dy * conj(da), das)  # possibly this * should be made nan-safe.
        end
        dargs = map(unbroadcast, args, deltas)  # ideally sum in unbroadcast could be part of tuplecast?
        return (NoTangent(), NoTangent(), dargs...)
    end
    bc_many_back(z::AbstractZero) = (NoTangent(), NoTangent(), map(Returns(z), args)...)
    return ys, bc_many_back
end

# Path 3: Use forward mode, or an `frule` if one exists.
# To allow `args...` we need either chunked forward mode, with `adot::Tuple` perhaps:
#   https://github.com/JuliaDiff/ChainRulesCore.jl/issues/92
#   https://github.com/JuliaDiff/Diffractor.jl/pull/54
# Or else we need to call the `f` multiple times, and maybe that's OK: 
# We do know that `f` doesn't have parameters, so maybe it's pure enough,
# and split broadcasting may anyway change N^2 executions into N, e.g. `g.(v ./ f.(v'))`.
# We don't know `f` is cheap, but `split_bc_pullbacks` tends to be very slow.

function may_bc_forwards(cfg::C, f::F, args::Vararg{Any,N}) where {C,F,N}
    Base.issingletontype(F) || return false
    N==1 || return false  # Could weaken this to 1 differentiable
    cfg isa RuleConfig{>:HasForwardsMode} && return true  # allows frule_via_ad
    TA = map(_eltype, args)
    TF = Core.Compiler._return_type(frule, Tuple{C, Tuple{NoTangent, TA...}, F, TA...})
    return isconcretetype(TF) && TF <: Tuple
end

split_bc_forwards(cfg::RuleConfig{>:HasForwardsMode}, f::F, arg) where {F} = split_bc_inner(frule_via_ad, cfg, f, arg)
split_bc_forwards(cfg::RuleConfig, f::F, arg) where {F} = split_bc_inner(frule, cfg, f, arg)
function split_bc_inner(frule_fun::R, cfg::RuleConfig, f::F, arg) where {R,F}
    _print("split_bc_forwards", frule_fun, f)
    ys, ydots = tuplecast(arg) do a
        frule_fun(cfg, (NoTangent(), one(a)), f, a)
    end
    function back_forwards(dys)
        delta = broadcast(ydots, unthunk(dys), arg) do ydot, dy, a
            ProjectTo(a)(conj(ydot) * dy)  # possibly this * should be made nan-safe.
        end
        return (NoTangent(), NoTangent(), ProjectTo(arg)(delta))
    end
    back_forwards(z::AbstractZero) = (NoTangent(), NoTangent(), z)
    return ys, back_forwards
end

# Path 4: The most generic, save all the pullbacks. Can be 1000x slower.
# Since broadcast makes no guarantee about order of calls, and un-fusing
# can change the number of calls, don't bother to try to reverse the iteration.

function split_bc_pullbacks(cfg::RCR, f::F, args::Vararg{Any,N}) where {F,N}
    _print("split_bc_generic", f, N)
    ys3, backs = tuplecast(args...) do a...
        rrule_via_ad(cfg, f, a...)
    end
    function back_generic(dys)
        deltas = tuplecast(backs, unthunk(dys)) do back, dy  # (could be map, sizes match)
            map(unthunk, back(dy))
        end
        dargs = map(unbroadcast, args, Base.tail(deltas))
        df = ProjectTo(f)(sum(first(deltas)))
        return (NoTangent(), df, dargs...)
    end
    back_generic(z::AbstractZero) = (NoTangent(), NoTangent(), map(Returns(z), args)...)
    return ys3, back_generic
end

# Don't run broadcasting on scalars
function rrule(cfg::RCR, ::typeof(broadcasted), f::F, args::Number...) where {F}
    _print("split_bc_scalar", f)
    z, back = rrule_via_ad(cfg, f, args...)
    return z, dz -> (NoTangent(), back(dz)...)
end

#####
##### Fused broadcasting
#####

# For certain cheap operations we can easily allow fused broadcast; the forward pass may be run twice.
# These all have `RuleConfig{>:HasReverseMode}` only for dispatch, to beat the split rule above.
# Accept `x::Broadcasted` because they produce it; can't dispatch on eltype but `x` is assumed to contain `Number`s.

const NumericOrBroadcast = Union{Number, AbstractArray{<:Number}, NTuple{<:Any,Number}, Broadcast.Broadcasted}

##### Arithmetic: +, -, *, ^2, /

function rrule(::RCR, ::typeof(broadcasted), ::typeof(+), xs::NumericOrBroadcast...)
    _print("plus", length(xs))
    function bc_plus_back(dy_raw)
        dy = unthunk(dy_raw)
        return (NoTangent(), NoTangent(), map(x -> unbroadcast(x, dy), xs)...)  # no copies, this may return dx2 === dx3
    end
    return broadcasted(+, xs...), bc_plus_back
end

function rrule(::RCR, ::typeof(broadcasted), ::typeof(-), x::NumericOrBroadcast, y::NumericOrBroadcast)
    _print("minus 2")
    function bc_minus_back(dz_raw)
        dz = unthunk(dz_raw)
        return (NoTangent(), NoTangent(), @thunk(unbroadcast(x, dz)), @thunk(-unbroadcast(y, dz)))
    end
    return broadcasted(-, x, y), bc_minus_back
end

function rrule(::RCR, ::typeof(broadcasted), ::typeof(-), x::NumericOrBroadcast)
    _print("minus 1")
    bc_minus_back(dy) = (NoTangent(), NoTangent(), @thunk -unthunk(dy))
    return broadcasted(-, x), bc_minus_back
end

function rrule(::RCR, ::typeof(broadcasted), ::typeof(*), x::NumericOrBroadcast, y::NumericOrBroadcast)
    _print("times")
    function bc_times_back(Δraw)
        Δ = unthunk(Δraw)
        return (NoTangent(), NoTangent(), _back_star(x, y, Δ), _back_star(y, x, Δ))
    end
    return broadcasted(*, x, y), bc_times_back
end
_back_star(x, y, Δ) = @thunk unbroadcast(x, Δ .* conj.(y))  # this case probably isn't better than generic
_back_star(x::Number, y, Δ) = @thunk LinearAlgebra.dot(y, Δ)  # ... but this is why the rule exists
_back_star(x::Bool, y, Δ) = NoTangent()
_back_star(x::Complex{Bool}, y, Δ) = NoTangent()  # e.g. for fun.(im.*x)

#=
# This works, but not sure it improves any benchmarks.
function rrule(cfg::RCR, ::typeof(broadcasted), ::typeof(*), x::NumericOrBroadcast, y::NumericOrBroadcast, zs::NumericOrBroadcast...)
    _print("times", 2 + length(zs))
    xy, back1 = rrule(cfg, broadcasted, *, x, y)
    xyz, back2 = rrule(cfg, broadcasted, *, xy, zs...)
    function bc_times3_back(dxyz)
        _, _, dxy, dzs... = back2(dxyz)
        _, _, dx, dy = back1(dxy)
        return (NoTangent(), NoTangent(), dx, dy, dzs...)
    end
    xyz, bc_times3_back
end
=#

function rrule(::RCR, ::typeof(broadcasted), ::typeof(Base.literal_pow), ::typeof(^), x::NumericOrBroadcast, ::Val{2})
    _print("square")
    function bc_square_back(dy_raw)
        dx = @thunk ProjectTo(x)(2 .* unthunk(dy_raw) .* conj.(x))
        return (NoTangent(), NoTangent(), NoTangent(), dx, NoTangent())
    end
    return broadcasted(Base.literal_pow, ^, x, Val(2)), bc_square_back
end

function rrule(::RCR, ::typeof(broadcasted), ::typeof(/), x::NumericOrBroadcast, y::Number)
    _print("divide")
    # z = broadcast(/, x, y)
    z = broadcasted(/, x, y)
    function bc_divide_back(dz_raw)
        dz = unthunk(dz_raw)
        dx = @thunk unbroadcast(x, dz ./ conj.(y))
        # dy = @thunk -LinearAlgebra.dot(z, dz) / conj(y)  # the reason to be eager is to allow dot here
        dy = @thunk -sum(Broadcast.instantiate(broadcasted(*, broadcasted(conj, z), dz))) / conj(y)  # complete sum is fast
        return (NoTangent(), NoTangent(), dx, dy)
    end
    return z, bc_divide_back
end

# For the same functions, send accidental broadcasting over numbers directly to `rrule`.
# (Could perhaps move all to @scalar_rule?)

function _prepend_zero((y, back))
    extra_back(dy) = (NoTangent(), back(dy)...)
    return y, extra_back
end

rrule(::RCR, ::typeof(broadcasted), ::typeof(+), args::Number...) = rrule(+, args...) |> _prepend_zero
rrule(::RCR, ::typeof(broadcasted), ::typeof(-), x::Number, y::Number) = rrule(-, x, y) |> _prepend_zero
rrule(::RCR, ::typeof(broadcasted), ::typeof(-), x::Number) = rrule(-, x) |> _prepend_zero
rrule(::RCR, ::typeof(broadcasted), ::typeof(*), x::Number, y::Number) = rrule(*, x, y) |> _prepend_zero
rrule(::RCR, ::typeof(broadcasted), ::typeof(Base.literal_pow), ::typeof(^), x::Number, ::Val{2}) =
    rrule(Base.literal_pow, ^, x, Val(2)) |> _prepend_zero
rrule(::RCR, ::typeof(broadcasted), ::typeof(/), x::Number, y::Number) = rrule(/, x, y) |> _prepend_zero

##### Identity, number types

rrule(::RCR, ::typeof(broadcasted), ::typeof(identity), x::NumericOrBroadcast) = rrule(identity, x) |> _prepend_zero
rrule(::RCR, ::typeof(broadcasted), ::typeof(identity), x::Number) = rrule(identity, x) |> _prepend_zero  # ambiguity

function rrule(::RCR, ::typeof(broadcasted), ::Type{T}, x::NumericOrBroadcast) where {T<:Number}
    _print("bc type", T)
    bc_type_back(dz) = (NoTangent(), NoTangent(), @thunk(unbroadcast(x, unthunk(dz))))
    return broadcasted(T, x), bc_type_back
end
rrule(::RCR, ::typeof(broadcasted), ::Type{T}, x::Number) where {T<:Number} = rrule(T, x) |> _prepend_zero

function rrule(::RCR, ::typeof(broadcasted), ::typeof(float), x::NumericOrBroadcast)
    _print("bc float")
    bc_float_back(dz) = (NoTangent(), NoTangent(), @thunk(unbroadcast(x, unthunk(dz))))
    return broadcasted(float, x), bc_float_back
end
rrule(::RCR, ::typeof(broadcasted), ::typeof(float), x::Number) = rrule(float, x) |> _prepend_zero

##### Complex: conj, real, imag

for conj in [:conj, :adjoint]  # identical as we know eltype <: Number
    @eval begin
        function rrule(::RCR, ::typeof(broadcasted), ::typeof($conj), x::NumericOrBroadcast)
            bc_conj_back(dx) = (NoTangent(), NoTangent(), conj(unthunk(dx)))
            return broadcasted($conj, x), bc_conj_back
        end
        rrule(::RCR, ::typeof(broadcasted), ::typeof($conj), x::Number) = rrule($conj, x) |> _prepend_zero
        rrule(::RCR, ::typeof(broadcasted), ::typeof($conj), x::AbstractArray{<:Real}) = rrule(identity, x) |> _prepend_zero
        # This `AbstractArray{<:Real}` rule won't catch `conj.(x.+1)` with lazy `.+` rule.
        # Could upgrade to infer eltype of the `Broadcasted`?
    end
end

function rrule(::RCR, ::typeof(broadcasted), ::typeof(real), x::NumericOrBroadcast)
    _print("real")
    bc_real_back(dz) = (NoTangent(), NoTangent(), @thunk(real(unthunk(dz))))
    return broadcasted(real, x), bc_real_back
end
rrule(::RCR, ::typeof(broadcasted), ::typeof(real), x::Number) = rrule(real, x) |> _prepend_zero
rrule(::RCR, ::typeof(broadcasted), ::typeof(real), x::AbstractArray{<:Real}) = rrule(identity, x) |> _prepend_zero

function rrule(::RCR, ::typeof(broadcasted), ::typeof(imag), x::NumericOrBroadcast)
    _print("imag")
    bc_imag_back(dz) = (NoTangent(), NoTangent(), @thunk(im .* real.(unthunk(dz))))
    return broadcasted(imag, x), bc_imag_back
end
rrule(::RCR, ::typeof(broadcasted), ::typeof(imag), x::Number) = rrule(imag, x) |> _prepend_zero
function rrule(::RCR, ::typeof(broadcasted), ::typeof(imag), x::AbstractArray{<:Real})
    _print("imag(real)")
    bc_imag_back_2(dz) = (NoTangent(), NoTangent(), ZeroTangent())
    return broadcasted(imag, x), bc_imag_back_2
end

function rrule(::RCR, ::typeof(broadcasted), ::typeof(complex), x::NumericOrBroadcast)
    _print("bc complex")
    bc_complex_back(dz) = (NoTangent(), NoTangent(), @thunk(unbroadcast(x, unthunk(dz))))
    return broadcasted(complex, x), bc_complex_back
end
rrule(::RCR, ::typeof(broadcasted), ::typeof(complex), x::Number) = rrule(complex, x) |> _prepend_zero

#####
##### Shape fixing
#####

# When sizes disagree, broadcasting gradient uses `unbroadcast` to reduce to correct shape.
# It's sometimes a little wasteful to allocate a too-large `dx`, but difficult to make more efficient.

function unbroadcast(x::Base.AbstractArrayOrBroadcasted, dx)
    N = ndims(dx)
    if length(x) == length(dx)
        ProjectTo(x)(dx)  # handles trivial reshapes, offsets, structured matrices, row vectors
    else
        dims = ntuple(d -> get(size(x), d, 1) == 1 ? d : N+1, N)  # hack to get type-stable `dims`
        ProjectTo(x)(sum(dx; dims))
    end
end
unbroadcast(x::Base.AbstractArrayOrBroadcasted, dx::AbstractZero) = dx

unbroadcast(x::T, dx) where {T<:Tuple{Any}} = ProjectTo(x)(Tangent{T}(sum(dx)))
function unbroadcast(x::T, dx) where {T<:Tuple{Vararg{Any,N}}} where {N}
    val = if length(x) == length(dx)
        dx
    else
        sum(dx; dims=2:ndims(dx))
    end
    return ProjectTo(x)(NTuple{length(x)}(val)) # Tangent
end

unbroadcast(f::Function, df) = sum(df)
unbroadcast(x::Number, dx) = ProjectTo(x)(sum(dx))
unbroadcast(x::Base.RefValue, dx) = ProjectTo(x)(Ref(sum(dx)))

unbroadcast(::Bool, dx) = NoTangent()
unbroadcast(::AbstractArray{Bool}, dx) = NoTangent()
unbroadcast(::AbstractArray{Bool}, dx::AbstractZero) = dx  # ambiguity
unbroadcast(::Val, dx) = NoTangent()

function unbroadcast(x, dx)
    p = ProjectTo(x)
    if dx isa AbstractZero || p isa ProjectTo{<:AbstractZero}
        return NoTangent()
    end
    b = Broadcast.broadcastable(x)
    if b isa Ref  # then x is scalar under broadcast
        return p(sum(dx))
    else
        error("don't know how to handle broadcast gradient for x::$(typeof(x))")
    end
end

#####
##### For testing
#####

function rrule(cfg::RCR, ::typeof(copy∘broadcasted), f, args...)
    y, back = rrule(cfg, broadcasted, f, args...)
    return _maybe_copy(y), back
end

_maybe_copy(y) = copy(y)
_maybe_copy(y::Tuple) = y
