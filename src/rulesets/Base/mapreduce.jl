#####
##### `sum(x)`
#####

function frule((_, ẋ), ::typeof(sum), x::Tuple)
    return sum(x), sum(ẋ)
end
function frule((_, ẋ), ::typeof(sum), x; dims=:)
    return sum(x; dims=dims), sum(ẋ; dims=dims)
end

function rrule(::typeof(sum), x::AbstractArray; dims=:)
    project = ProjectTo(x)
    y = sum(x; dims=dims)
    function sum_pullback(dy_raw)
        dy = unthunk(dy_raw)
        x_thunk = InplaceableThunk(
            # Protect `dy` from broadcasting, for when `x` is an array of arrays:
            dx -> dx .+= (dims isa Colon ? Ref(dy) : dy),
            @thunk project(_unsum(x, dy, dims))  # `_unsum` handles Ref internally
        )
        return (NoTangent(), x_thunk)
    end
    return y, sum_pullback
end

# This broadcasts `dy` to the shape of `x`, and should preserve e.g. CuArrays, StaticArrays.
# Ideally this would only need `typeof(x)` not `x`, but `similar` only has a suitable method
# when `eltype(x) == eltype(dy)`, which isn't guaranteed.
_unsum(x, dy, dims) = broadcast(last∘tuple, x, dy)
_unsum(x, dy, ::Colon) = broadcast(last∘tuple, x, Ref(dy))

# Allow for second derivatives of `sum`, by writing rules for `_unsum`:

function frule((_, _, dydot, _), ::typeof(_unsum), x, dy, dims)
    return _unsum(x, dy, dims), _unsum(x, dydot, dims)
end

function rrule(::typeof(_unsum), x, dy, dims)
    z = _unsum(x, dy, dims)
    _unsum_pullback(dz) = (NoTangent(), NoTangent(), sum(unthunk(dz); dims=dims), NoTangent())
    return z, _unsum_pullback
end

#####
##### `sum(f, x)`
#####

# Can't map over Adjoint/Transpose Vector
function rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(sum),
    f,
    xs::Union{Adjoint{<:Number,<:AbstractVector},Transpose{<:Number,<:AbstractVector}};
    kwargs...
)
    op = xs isa Adjoint ? adjoint : transpose
    # since summing a vector we don't need to worry about dims which simplifies adjointing
    vector = parent(xs)
    y, vector_sum_pb = rrule(config, sum, f, vector; kwargs...)
    function covector_sum_pb(ȳ)
        s̄um, f̄, v̄ = vector_sum_pb(ȳ)
        return s̄um, f̄, op(v̄)
    end

    return y, covector_sum_pb
end

function rrule(
    config::RuleConfig{>:HasReverseMode}, ::typeof(sum), f, xs::AbstractArray; dims=:
)
    fx_and_pullbacks = map(x->rrule_via_ad(config, f, x), xs)
    y = sum(first, fx_and_pullbacks; dims=dims)

    pullbacks = last.(fx_and_pullbacks)

    project = ProjectTo(xs)

    function sum_pullback(ȳ)
        call(f, x) = f(x)
        # if dims is :, then need only left-handed only broadcast
        broadcast_ȳ = dims isa Colon  ? (ȳ,) : ȳ
        f̄_and_x̄s = call.(pullbacks, broadcast_ȳ)
        # no point thunking as most of work is in f̄_and_x̄s which we need to compute for both
        f̄ = if fieldcount(typeof(f)) === 0 # Then don't need to worry about derivative wrt f
            NoTangent()
        else
            sum(first, f̄_and_x̄s)
        end
        x̄s = map(unthunk ∘ last, f̄_and_x̄s) # project does not support receiving InplaceableThunks
        return NoTangent(), f̄, project(x̄s)
    end
    return y, sum_pullback
end

# https://github.com/JuliaDiff/ChainRules.jl/issues/522
# The rule above assumes `f` is callable. Arrays are not, this came up when summing
# arrays with weights in StatsBase
@opt_out ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(sum),
    x::AbstractArray,
    y::AbstractArray;
    dims=:
)

function frule(
    (_, _, Δx),
    ::typeof(sum),
    ::typeof(abs2),
    x::AbstractArray{T};
    dims=:,
) where {T<:Union{Real,Complex}}
    ẋ = unthunk(Δx)
    y = sum(abs2, x; dims=dims)
    ∂y = if dims isa Colon
        2 * realdot(x, ẋ)
    elseif VERSION ≥ v"1.2" # multi-iterator mapreduce introduced in v1.2
        mapreduce(+, x, ẋ; dims=dims) do xi, dxi
            2 * realdot(xi, dxi)
        end
    else
        2 * sum(realdot.(x, ẋ); dims=dims)
    end
    return y, ∂y
end

function rrule(
    ::typeof(sum),
    ::typeof(abs2),
    x::AbstractArray{T};
    dims=:,
) where {T<:Union{Real,Complex}}
    y = sum(abs2, x; dims=dims)
    function sum_abs2_pullback(ȳ)
        x_thunk = InplaceableThunk(
            dx -> dx .+= 2 .* real.(ȳ) .* x,
            @thunk(2 .* real.(ȳ) .* x),
        )
        return (NoTangent(), NoTangent(), x_thunk)
    end
    return y, sum_abs2_pullback
end

# Fix dispatch for this pidgeon-hole optimization,
# Rules with RuleConfig dispatch with priority over without (regardless of other args).
# and if we don't specify what do do for one that HasReverseMode then it is ambigious
for Config in (RuleConfig, RuleConfig{>:HasReverseMode})
    @eval function rrule(
        ::$Config, ::typeof(sum), ::typeof(abs2), x::AbstractArray{T}; dims=:,
    ) where {T<:Union{Real,Complex}}
        return rrule(sum, abs2, x; dims=dims)
    end
end

#####
##### `cumsum`
#####

function frule((_, xdot), ::typeof(cumsum), x::AbstractArray; dims::Integer)
    return cumsum(x; dims=dims), cumsum(xdot; dims=dims)
end
frule(tang, ::typeof(cumsum), x::AbstractVector) = frule(tang, cumsum, x; dims=1)

function frule((_, ydot, xdot), ::typeof(cumsum!), y::AbstractArray, x::AbstractArray; dims::Integer)
    return cumsum!(y, x; dims=dims), cumsum!(ydot, xdot; dims=dims)
end
frule(t, ::typeof(cumsum!), y::AbstractVector, x::AbstractVector) = frule(t, cumsum!, y, x; dims=1)

function rrule(::typeof(cumsum), x::AbstractArray; dims::Integer)
    project = ProjectTo(x)
    function cumsum_pullback(dy)
        step1 = reverse(unthunk(dy); dims=dims)
        if ChainRulesCore.is_inplaceable_destination(step1) && VERSION >= v"1.6"
            step2 = cumsum!(step1, step1; dims=dims)
            step3 = reverse!(step2; dims=dims)
        else
            step2 = cumsum(step1; dims=dims)
            step3 = reverse(step2; dims=dims)
        end
        return (NoTangent(), project(step3))
    end
    return cumsum(x; dims=dims), cumsum_pullback
end
rrule(::typeof(cumsum), x::AbstractVector) = rrule(cumsum, x; dims=1)

#####
##### `prod`
#####

function rrule(::typeof(prod), x::AbstractArray{T}; dims=:) where {T<:CommutativeMulNumber}
    y = prod(x; dims=dims)
    project_x = ProjectTo(x)
    # vald = dims isa Colon ? nothing : dims isa Integer ? Val(Int(dims)) : Val(Tuple(dims))
    function prod_pullback(ȳ)
        dy = unthunk(ȳ)
        x_thunk = InplaceableThunk(
            # In-place versions -- same branching
            dx -> if dims === (:)
                ∇prod!(dx, x, dy, y)
            elseif any(iszero, x)
                vald = dims isa Colon ? nothing : dims isa Integer ? Val(Int(dims)) : Val(Tuple(dims))
                ∇prod_dims!(dx, vald, x, dy, y)
            else
                dx .+= conj.(y ./ x) .* dy
            end,
            # Out-of-place versions
            @thunk project_x(if dims === (:)
                ∇prod(x, dy, y)
            elseif any(iszero, x)  # Then, and only then, will ./x lead to NaN
                vald = dims isa Colon ? nothing : dims isa Integer ? Val(Int(dims)) : Val(Tuple(dims))
                ∇prod_dims(vald, x, dy, y)  # val(Int(dims)) is about 2x faster than Val(Tuple(dims))
            else
                conj.(y ./ x) .* dy
            end)
        )
        return (NoTangent(), x_thunk)
    end
    return y, prod_pullback
end

function ∇prod_dims(vald::Val{dims}, x, dy, y=prod(x; dims=dims)) where {dims}
    T = promote_type(eltype(x), eltype(dy))
    dx = fill!(similar(x, T, axes(x)), zero(T))
    ∇prod_dims!(dx, vald, x, dy, y)
    return dx
end

function ∇prod_dims!(dx, ::Val{dims}, x, dy, y) where {dims}
    iters = ntuple(d -> d in dims ? tuple(:) : axes(x,d), ndims(x))  # Without Val(dims) this is a serious type instability
    @inbounds for ind in Iterators.product(iters...)
        jay = map(i -> i isa Colon ? 1 : i, ind)
        @views ∇prod!(dx[ind...], x[ind...], dy[jay...], y[jay...])
    end
    return dx
end

function ∇prod(x, dy::Number=1, y::Number=prod(x))
    T = promote_type(eltype(x), eltype(dy))
    dx = fill!(similar(x, T, axes(x)), zero(T)) # axes(x) makes MArray on StaticArrays, Array for structured matrices
    ∇prod!(dx, x, dy, y)
    return dx
end

function ∇prod!(dx, x, dy::Number=1, y::Number=prod(x))
    numzero = iszero(y) ? count(iszero, x) : 0
    if numzero == 0  # This can happen while y==0, if there are several small xs
        dx .+= conj.(y ./ x) .* dy
    elseif numzero == 1
        ∇prod_one_zero!(dx, x, dy)
    else
        # numzero > 1, then all first derivatives are zero
    end
    return dx
end

function ∇prod_one_zero!(dx, x, dy::Number=1)  # Assumes exactly one x is zero
    i_zero = 0
    p_rest = one(promote_type(eltype(x), typeof(dy)))
    for i in eachindex(x)
        xi = @inbounds x[i]
        p_rest *= ifelse(iszero(xi), one(xi), conj(xi))
        i_zero = ifelse(iszero(xi), i, i_zero)
    end
    dx[i_zero] += p_rest * dy
    return
end

#####
##### `cumprod`
#####

function rrule(::typeof(cumprod), x::AbstractVector{<:Real}; dims::Integer=1)
    y = cumprod(x; dims=dims)  # does nothing unless dims == 1
    project_x = ProjectTo(x)
    function cumprod_pullback_1(dy_raw)
        dy = unthunk(dy_raw)
        dx_thunk = InplaceableThunk(
            dx -> if dims == 1
                ∇cumprod!(dx, x, dy, y)
            else
                dx .+= dy
            end
            ,
            @thunk project_x(if dims == 1
                ∇cumprod(x, dy, y)
            else
                dy
            end)
            )
        return (NoTangent(), dx_thunk)
    end
    return y, cumprod_pullback_1
end

function rrule(::typeof(cumprod), x::AbstractArray{<:Real}; dims::Integer)
    y = cumprod(x; dims=dims)
    project_x = ProjectTo(x)
    function cumprod_pullback_2(dy_raw)
        dy = unthunk(dy_raw)
        dx_thunk = InplaceableThunk(
            dx -> if dims <= ndims(x)
                vald = Val(Int(dims))
                ∇cumprod_dim!(dx, vald, x, dy, y)
            else
                dx .+= dy
            end
            ,
            @thunk project_x(if dims <= ndims(x)
                vald = Val(Int(dims))
                ∇cumprod_dim(vald, x, dy, y)
            else
                dy
            end)
            )
        return (NoTangent(), dx_thunk)
    end
    return y, cumprod_pullback_2
end

function ∇cumprod_dim(vald::Val{dim}, x::AbstractArray, dy=fill!(zero(x),1), y=cumprod(x; dims=dim)) where {dim}
     T = promote_type(eltype(x), eltype(dy))
     dx = fill!(similar(x, T, axes(x)), zero(T))
     ∇cumprod_dim!(dx, vald, x, dy, y)
     return dx
 end

@inline function ∇cumprod_dim!(dx::AbstractArray, ::Val{dim}, x::AbstractArray, dy, y) where {dim}
    iters = ntuple(k -> k==dim ? Ref(:) : axes(x,k), ndims(x))
    for ind in Iterators.product(iters...)
        @views ∇cumprod!(dx[ind...], x[ind...], dy[ind...], y[ind...])
    end
    return dx
end

function ∇cumprod(x::AbstractVector, dy=one(x), y=cumprod(x))
    T = promote_type(eltype(x), eltype(dy))  # really needs to allow dy * y / x
    dx = fill!(similar(x, T, axes(x)), zero(T))  # axes(x) makes MArray on StaticArrays, Array for structured matrices
    ∇cumprod!(dx, x, dy, y)
    return dx
end

@inline function ∇cumprod!(dx::AbstractVector, x::AbstractVector, dy, y)
    lo, hi = firstindex(x), lastindex(x)
    z = something(findfirst(iszero, x), hi+1)
    acc = zero(eltype(dy))
    @inbounds for k in z-1:-1:lo
        acc += y[k] * dy[k]
        dx[k] += acc / x[k]
    end
    @inbounds if z != hi+1
        yk = z==1 ? one(eltype(y)) : y[z-1]  # will be prod(x[j] for j=1:k if j!=z)
        dx[z] += yk * dy[z]
        for k in (z+1):hi
            yk *= x[k]
            dx[z] += yk * dy[k]
        end
    end
    return dx
end

#####
##### `foldl`
#####

# `foldl` guarantees to execute `f` in order, left to right. So it makes sense even when
# this `f` is stateful, in which case the gradient must be calculated in the reverse order. 

# The implementation aims to be efficient for both tuples and arrays, although using accumulate
# to carry intermediate results along creates arrays of tuples which could be avoided; using a
# loop can be a few times faster. Note also that it does not return a gradient for `init`.

function rrule(
        config::RuleConfig{>:HasReverseMode}, ::typeof(foldl), op::G, x::Union{AbstractArray, Tuple};
        init=_InitialValue()
    ) where {G}
    list, start = if init === _InitialValue()
        _drop1(x), first(x)
    else
        # Case with init keyword is simpler to understand first!
        _reshape1(x, :), init  # (vec is for Julia 1.0, accumulate is fussy)
    end
    hobbits = accumulate(list; init=(start, nothing)) do (a,_), b
        # Here `a` is what we would normally cary forward, and `_` ignores
        # the previous iteration's pullback function (needed later),
        # while `b` is the fresh input from `list` as usual.
        c, back = rrule_via_ad(config, op, a, b)  # LHS is just documentation here!
        # We don't really need to store every `c`, last one is `foldl` output.
        # (The name, BTW, is because "there and back again" is the subtitle of Tolkien's book.)
    end
    y = first(last(hobbits))
    axe = axes(x)
    project = ProjectTo(x)
    function unfoldl(dy)
        trio = accumulate(_reverse1(hobbits); init=(0, dy, 0)) do (_, dc, _), (_, back)
            ds, da, db = back(dc)
            # Don't need to store every `da`, need one for the next iteration + maybe last
        end
        dop = sum(first, trio)
        dx = map(last, _reverse1(trio))
        if init === _InitialValue()
            # `hobbits` is one short
            dx = _vcat1(trio[end][2], dx)
        end
        return (NoTangent(), dop, project(_reshape1(dx, axe)))
    end
    return y, unfoldl
end


#####
##### Iterator-or-Tuple functions
#####

# This zoo of underscore functions helps `foldl` & `accumulate` handle both tuples and arrays,
# and also provides some alternatives for versions of Julia where iterators weren't supported.
# Inspired by `Base._reverse`, used in defn of `foldr`.

# To support 2nd derivatives, some may need their own gradient rules. And _drop1 should perhaps
# be replaced by _peel1 like Iterators.peel

if VERSION >= v"1.6"
    _reverse1(x) = Iterators.reverse(x)
    _drop1(x) = Iterators.drop(x, 1)
    _zip2(x, y) = zip(x, y)  # for `accumulate`, below
else
    # Old versions don't support accumulate(::itr), nor multi-dim reverse
    _reverse1(x) = reverse(vec(x))
    _drop1(x) = vec(x)[2:end]
    _zip2(x, y) = collect(zip(x, y))
end
_reverse1(x::Tuple) = reverse(x)
_drop1(x::Tuple) = Base.tail(x)
_zip2(x::Tuple{Vararg{Any,N}}, y::Tuple{Vararg{Any,N}}) where N = ntuple(i -> (x[i],y[i]), N)

struct _InitialValue end  # Old versions don't have `Base._InitialValue`

_vcat1(x, ys::AbstractVector) = vcat(x, ys)
_vcat1(x::AbstractArray, ys::AbstractVector) = vcat([x], ys)
_vcat1(x, ys::Tuple) = (x, ys...)

_reshape1(x::AbstractArray, axe) = reshape(x, axe)
_reshape1(x::Tuple, axe) = x

_no_tuple_tangent(dx::Tangent) = ChainRulesCore.backing(dx)
_no_tuple_tangent(dx) = dx


#####
##### `accumulate`
#####

# Like `foldl` this by definition works in order, so it makes sense to allow stateful `f`.

function rrule(
        config::RuleConfig{>:HasReverseMode}, ::typeof(accumulate), op::G, x::Union{AbstractArray, Tuple}; 
        init=_InitialValue(), dims=nothing
    ) where {G}
    isnothing(dims) || dims == 1 && x isa Base.AbstractVecOrTuple || throw(
        "accumulate(op, x; dims) is not currently supported by ChainRules, sorry"
        # It's not supported by AD either, so no point calling back, and no regression:
        # gradient(x -> sum(accumulate(/, x, dims=1)), rand(3,4)) 
        # ERROR: Mutating arrays is not supported
    )
    list, start = if init === _InitialValue()
        _drop1(x), first(x)
    else
        x, init
    end
    hobbits = accumulate(list; init = (start, nothing)) do (a, _), b
        c, back = rrule_via_ad(config, op, a, b)
    end
    y = map(first, hobbits)
    if init === _InitialValue()
        # `hobbits` is one short, and first one doesn't invoke `op`
        y = _vcat1(first(x), y)
    end
    axe = axes(x)
    project = ProjectTo(x)
    function decumulate(dy)
        dy_plain = _no_tuple_tangent(unthunk(dy))
        rev_list = if init === _InitialValue()
            if VERSION >= v"1.6"
                # Here we rely on `zip` to stop early. Begin explicit with _reverse1(_drop1(...))
                # gets "no method matching iterate(::Base.Iterators.Reverse{Base.Iterators.Drop{Array{"
                _zip2(_reverse1(hobbits), _reverse1(dy_plain))
            else
                # However, on 1.0 and some others, zip does not stop early. But since accumulate
                # also doesn't work on iterators, `_drop1` doesn't make one, so this should work:
                _zip2(_reverse1(hobbits), _reverse1(_drop1(dy_plain)))
                # What an awful tangle.
            end
        else
            _zip2(_reverse1(hobbits), _reverse1(dy_plain))
        end
        trio = accumulate(rev_list; init=(0, ZeroTangent(), 0)) do (_, dc, _), ((_, back), dz)
            ds, da, db = back(dc + dz)
            # Don't need to store every 'da', but need for next iteration, and the last one.
        end
        dop = sum(first, trio)
        dx = map(last, _reverse1(trio))
        if init == _InitialValue()
            # `hobbits` is one short, and the first one is weird
            dx = _vcat1(trio[end][2] + dy_plain[1], dx)
        end
        return (NoTangent(), dop, project(_reshape1(dx, axe)))
    end
    return _reshape1(y, axe), decumulate
end
