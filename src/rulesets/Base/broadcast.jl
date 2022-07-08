using Base.Broadcast: Broadcast, broadcasted, Broadcasted
const RCR = RuleConfig{>:HasReverseMode}

rrule(::typeof(copy), bc::Broadcasted) = copy(bc), Δ -> (NoTangent(), Δ)

# Skip AD'ing through the axis computation
function rrule(::typeof(Broadcast.instantiate), bc::Broadcasted)
    uninstantiate(Δ) = Core.tuple(NoTangent(), Δ)
    return Broadcast.instantiate(bc), uninstantiate
end

_print(args...) = nothing # println(join(args, " "))

#####
##### Split broadcasting
#####

function rrule(cfg::RCR, ::typeof(broadcasted), f::F, args::Vararg{Any,N}) where {F,N}
 # = split_bc_rule(cfg, f, args...)
 # function split_bc_rule(cfg::RCR, f::F, args::Vararg{Any,N}) where {F,N}
    T = Broadcast.combine_eltypes(f, args)
    TΔ = Core.Compiler._return_type(derivatives_given_output, Tuple{T, F, map(eltype, args)...})
    if T === Bool
        # 1: Trivial case: non-differentiable output, e.g. `x .> 0`
        _print("split_bc_rule 1 ", f)
        back_1(_) = ntuple(Returns(ZeroTangent()), length(args)+2)
        return f.(args...), back_1
    elseif T <: Number && isconcretetype(TΔ)
        # 2: Fast path: just broadcast, and use arguments & result to find derivatives.
        _print("split_bc_rule 2", f, N)
        ys = f.(args...)
        function back_2_one(dys)  # For f.(x) we do not need StructArrays / unzip at all
            delta = broadcast(unthunk(dys), ys, args...) do dy, y, a
                das = only(derivatives_given_output(y, f, a))
                dy * conj(only(das))  # possibly this * should be made nan-safe.
            end
            (NoTangent(), NoTangent(), ProjectTo(only(args))(delta))
        end
        back_2_one(z::AbstractZero) = (NoTangent(), NoTangent(), z)
        function back_2_many(dys)
            deltas = tuplecast(unthunk(dys), ys, args...) do dy, y, as...
                das = only(derivatives_given_output(y, f, as...))
                map(da -> dy * conj(da), das)
            end
            dargs = map(unbroadcast, args, deltas)  # ideally sum in unbroadcast could be part of tuplecast?
            (NoTangent(), NoTangent(), dargs...)
        end
        back_2_many(z::AbstractZero) = (NoTangent(), NoTangent(), map(Returns(z), args)...)
        return ys, N==1 ? back_2_one : back_2_many
    else
        _print("split_bc_rule 3", f, N)
        # 3: Slow path: collect all the pullbacks & apply them later.
        # (Since broadcast makes no guarantee about order of calls, and un-fusing 
        # can change the number of calls, don't bother to try to reverse the iteration.)
        ys3, backs = tuplecast(args...) do a...
            rrule_via_ad(cfg, f, a...)
        end
        function back_3(dys)
            deltas = tuplecast(backs, unthunk(dys)) do back, dy  # could be map, sizes match
                map(unthunk, back(dy))
            end
            dargs = map(unbroadcast, args, Base.tail(deltas))
            (NoTangent(), ProjectTo(f)(sum(first(deltas))), dargs...)
        end
        back_3(z::AbstractZero) = (NoTangent(), NoTangent(), map(Returns(z), args)...)
        return ys3, back_3
    end
end

# Don't run broadcasting on scalars
function rrule(cfg::RCR, ::typeof(broadcasted), f::F, args::Number...) where {F}
# function split_bc_rule(cfg::RCR, f::F, args::Number...) where {F}
    _print("split_bc_rule scalar", f)
    z, back = rrule_via_ad(cfg, f, args...)
    return z, dz -> (NoTangent(), back(dz)...)
end

# using StructArrays
#
# function tuplecast(f::F, args...) where {F}
#     T = Broadcast.combine_eltypes(f, args)
#     if isconcretetype(T)
#         T <: Tuple || throw(ArgumentError("tuplecast(f, args) only works on functions returning a tuple."))
#     end
#     bc = Broadcast.instantiate(Broadcast.broadcasted(f, args...))
#     StructArrays.components(StructArray(bc))
# end

#####
##### Fused broadcasting
#####

# For certain cheap operations we can easily allow fused broadcast.
# These all have `RuleConfig{>:HasReverseMode}` as otherwise the split rule matches first & they are not used.
# They accept `Broadcasted` because they produce it; it has no eltype but is assumed to contain `Number`s.
const NumericOrBroadcast = Union{Number, AbstractArray{<:Number}, NTuple{<:Any,Number}, Broadcast.Broadcasted}

function rrule(::RCR, ::typeof(broadcasted), ::typeof(+), xs::NumericOrBroadcast...)
    _print("plus", length(xs))
    function bc_plus_back(dy_raw)
        dy = unthunk(dy_raw)
        (NoTangent(), NoTangent(), map(x -> unbroadcast(x, dy), xs)...)
    end
    return broadcasted(+, xs...), bc_plus_back
end

function rrule(::RCR, ::typeof(broadcasted), ::typeof(-), x::NumericOrBroadcast, y::NumericOrBroadcast)
    _print("minus 2")
    bc_minus_back(Δraw) = let Δ = unthunk(Δraw)
        (NoTangent(), NoTangent(), @thunk(unbroadcast(x, Δ)), @thunk(-unbroadcast(y, Δ)))
    end
    return broadcasted(-, x, y), bc_minus_back
end

function rrule(::RCR, ::typeof(broadcasted), ::typeof(-), x::NumericOrBroadcast)
    _print("minus 1")
    bc_minus_back(dy) = (NoTangent(), NoTangent(), @thunk -unthunk(dy))
    return broadcasted(-, x), bc_minus_back
end

using LinearAlgebra: dot

function rrule(::RCR, ::typeof(broadcasted), ::typeof(*), x::NumericOrBroadcast, y::NumericOrBroadcast)
    _print("times")
    function bc_times_back(Δraw)
        Δ = unthunk(Δraw)
        (NoTangent(), NoTangent(), _back_star(x, y, Δ), _back_star(y, x, Δ))
    end
    return broadcasted(*, x, y), bc_times_back
end
_back_star(x, y, Δ) = @thunk unbroadcast(x, Δ .* conj.(y))
_back_star(x::Number, y, Δ) = @thunk dot(y, Δ)
_back_star(x::Bool, y, Δ) = NoTangent()
_back_star(x::Complex{Bool}, y, Δ) = NoTangent()  # e.g. for fun.(im.*x)

# TODO check what happens for A * B * C

function rrule(::RCR, ::typeof(broadcasted), ::typeof(Base.literal_pow), ::typeof(^), x::NumericOrBroadcast, ::Val{2})
    _print("square")
    function bc_square_back(dy_raw)
        dx = @thunk ProjectTo(x)(2 .* unthunk(dy_raw) .* conj.(x))
        (NoTangent(), NoTangent(), NoTangent(), dx, NoTangent())
    end
    return broadcasted(Base.literal_pow, ^, x, Val(2)), bc_square_back
end

function rrule(::RCR, ::typeof(broadcasted), ::typeof(/), x::NumericOrBroadcast, y::Number)
    _print("divide")
    z = broadcast(/, x, y)
    function bc_divide_back(Δraw)
        Δ = unthunk(Δraw)
        dx = @thunk unbroadcast(x, Δ ./ conj.(y))
        dy = @thunk -dot(z, Δ) / (conj(y))  # the reason to be eager is to allow dot here
        (NoTangent(), NoTangent(), dx, dy)
    end
    return z, bc_divide_back
end

# For the same functions, send accidental broadcasting over numbers directly to `rrule`.
# Could perhaps move all to @scalar_rule?

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

# A few more cheap functions

rrule(::RCR, ::typeof(broadcasted), ::typeof(identity), x::NumericOrBroadcast) = rrule(identity, x) |> _prepend_zero
rrule(::RCR, ::typeof(broadcasted), ::typeof(identity), x::Number) = rrule(identity, x) |> _prepend_zero  # ambiguity

function rrule(::RCR, ::typeof(broadcasted), ::typeof(conj), x::NumericOrBroadcast)
    bc_conj_back(dx) = (NoTangent(), NoTangent(), conj(unthunk(dx)))
    return broadcasted(conj, x), bc_conj_back
end
rrule(::RCR, ::typeof(broadcasted), ::typeof(conj), x::Number) = rrule(conj, x) |> _prepend_zero
rrule(::RCR, ::typeof(broadcasted), ::typeof(conj), x::AbstractArray{<:Real}) = rrule(identity, x) |> _prepend_zero

# TODO real, imag

#####
##### Shape fixing
#####

# Reverse mode broadcasting uses `unbroadcast` to reduce to correct shape:

function unbroadcast(x::Base.AbstractArrayOrBroadcasted, dx)
    N = ndims(dx)
    if length(x) == length(dx)
        ProjectTo(x)(dx)  # handles trivial reshapes, offsets, structured matrices, row vectors
    else
        dims = ntuple(d -> get(size(x), d, 1) == 1 ? d : N+1, N)  # hack to get type-stable `dims`
        ProjectTo(x)(sum(dx; dims))  # ideally this sum might be thunked?
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
    ProjectTo(x)(NTuple{length(x)}(val)) # Tangent
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
