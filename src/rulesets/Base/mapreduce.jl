#####
##### `sum(x)`
#####

function frule((_, ẋ), ::typeof(sum), x::Tuple)
    return sum(x), sum(ẋ)
end

function frule((_, ẋ), ::typeof(sum), x; dims=:)
    return sum(x; dims=dims), sum(ẋ; dims=dims)
end

function frule((_, ẏ, ẋ), ::typeof(sum!), y::AbstractArray, x::AbstractArray)
    return sum!(y, x), sum!(ẏ, ẋ)
end

function rrule(::typeof(sum), x::Tuple)
    project = ProjectTo(x)
    len = Val(length(x))
    function sum_pullback(dy_raw)
        dy = unthunk(dy_raw)
        dx = dy isa AbstractZero ? dy : ntuple(Returns(dy), len)
        return (NoTangent(), project(dx))
    end
    return sum(x), sum_pullback
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

function rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(sum), f::F, xs::Tuple) where {F}
    fxs, unmap = rrule(config, map, f, xs)
    y, unsum = rrule(config, sum, fxs)
    function sum_pullback_f(dy)
        _, dfxs = unsum(dy)
        _, df, dxs = unmap(dfxs)
        (NoTangent(), df, dxs)
    end
    y, sum_pullback_f
end

function rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(sum),
    f::F,
    xs::AbstractArray{T};
    dims = :,
) where {F,T}
    project = ProjectTo(xs)

    if _uses_input_only(f, T)
        # Then we can compute the forward pass as usual, save nothing but `xs`:
        function sum_pullback_f1(dy)
            dxs = broadcast(unthunk(dy), xs) do dyₖ, xᵢ
                ∂yₖ∂xᵢ = only(only(derivatives_given_output(nothing, f, xᵢ)))
                dyₖ * conj(∂yₖ∂xᵢ)
            end
            return (NoTangent(), NoTangent(), project(dxs))
        end
        return sum(f, xs; dims), sum_pullback_f1
    end

    # (There is an intermediate case, where `derivatives_given_output` needs to
    # see `f.(xs)` but we don't need the pullbacks. Not implemented at present.)

    # In the general case, we need to save all the pullbacks:
    # (Here `map` or `broadcast` would fail for adjoint vectors.)
    fx_and_pullbacks = [rrule_via_ad(config, f, xᵢ) for xᵢ in xs]
    y = sum(first, fx_and_pullbacks; dims)

    function sum_pullback_f2(dy)
        # For arrays of arrays, we ought to protect the element against broadcasting:
        broadcast_dy = dims isa Colon ? Ref(unthunk(dy)) : unthunk(dy)
        if Base.issingletontype(F)
            # Then at least `f` has no gradient. 
            # Broadcasting here gets the shape right with or without `dims` keyword.
            dxs = broadcast(fx_and_pullbacks, broadcast_dy) do (_, pbᵢ), dyₖ
                unthunk(last(pbᵢ(dyₖ)))
            end
            return (NoTangent(), NoTangent(), project(dxs))

        else
            # Most general case. If `f` were stateful, we would need to reverse the order
            # of iteration here, but since this function makes no guarantee, even the primal
            # result is then ill-defined.
            df_and_dxs = broadcast(fx_and_pullbacks, broadcast_dy) do (_, pbᵢ), dyₖ
                pbᵢ(dyₖ)
            end
            df = sum(first, df_and_dxs)
            dxs = map(unthunk ∘ last, df_and_dxs)
            return (NoTangent(), df, project(dxs))
        end
    end
    return y, sum_pullback_f2
end

"""
    _uses_input_only(f, xT::Type)

Returns `true` if it can prove that `derivatives_given_output` will work using only the input
of the given type. Thus there is no need to store the output `y = f(x::xT)`, allowing us to take
a fast path in the `rrule` for `sum(f, xs)`.

Works by seeing if the result of `derivatives_given_output(nothing, f, x)` can be inferred.
The method of `derivatives_given_output` usually comes from `@scalar_rule`.
"""
function _uses_input_only(f::F, ::Type{xT}) where {F,xT}
    gT = Core.Compiler._return_type(derivatives_given_output, Tuple{Nothing, F, xT})
    # Here we must check `<: Number`, to avoid this, the one rule which can return the `nothing`:
    # ChainRules.derivatives_given_output("anything", exp, 1) == (("anything",),)
    return isconcretetype(gT) && gT <: Tuple{Tuple{Number}}
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
    else
        mapreduce(+, x, ẋ; dims=dims) do xi, dxi
            2 * realdot(xi, dxi)
        end
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
    return cumsum(x; dims), cumsum(xdot; dims)
end
frule(tang, ::typeof(cumsum), x::AbstractVector) = frule(tang, cumsum, x; dims=1)

function frule((_, ydot, xdot), ::typeof(cumsum!), y::AbstractArray, x::AbstractArray; dims::Integer)
    return cumsum!(y, x; dims), cumsum!(ydot, xdot; dims)
end
frule(t, ::typeof(cumsum!), y::AbstractVector, x::AbstractVector) = frule(t, cumsum!, y, x; dims=1)

function rrule(::typeof(cumsum), x::AbstractArray{T,N}; dims::Integer) where {T,N}
    project = ProjectTo(x)
    function cumsum_pullback(dy)
        if dims > N  # trivial case, for which reverse fails
            return (NoTangent(), project(unthunk(dy)))
        end
        step1 = reverse(unthunk(dy); dims=dims)
        if ChainRulesCore.is_inplaceable_destination(step1)
            step2 = cumsum!(step1, step1; dims)
            step3 = reverse!(step2; dims)
        else
            step2 = cumsum(step1; dims)
            step3 = reverse(step2; dims)
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
∇prod_dims(::Val, x, dy::AbstractZero, y=0) = dy

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
∇prod(x, dy::AbstractZero, y::Number=0) = dy

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
∇cumprod_dim(vald::Val, x::AbstractArray, dy::AbstractZero, y=0) = dy

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
∇cumprod(x::AbstractVector, dy::AbstractZero, y=0) = dy

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
##### `mapfoldl(f, g, ::Tuple)`
#####

using Base: mapfoldl_impl

# For tuples there should be no harm in handling `map` first.
# This will also catch `mapreduce`.

function rrule(
        cfg::RuleConfig{>:HasReverseMode}, ::typeof(mapfoldl_impl), f::F, op::G, init, x::Tuple;
    ) where {F,G}
    y, backmap = rrule(cfg, map, f, x)
    z, backred = rrule(cfg, Base.mapfoldl_impl, identity, op, init, y)
    function mapfoldl_pullback_tuple(dz)
        _, _, dop, dinit, dy = backred(dz)
        _, df, dx = backmap(dy)
        return (NoTangent(), df, dop, dinit, dx)
    end
    return z, mapfoldl_pullback_tuple
end

#####
##### `foldl(f, ::Tuple)`
#####

# `foldl` guarantees to execute `f` in order, left to right. So it makes sense even when
# this `f` is stateful, in which case the gradient must be calculated in the reverse order.

# The rule is attached to `Base.mapfoldl_impl` because this gets the `init` keyword as an argument,
# which is handled below. For tuples, `reduce` also comes here.

function rrule(
        config::RuleConfig{>:HasReverseMode},
        ::typeof(Base.mapfoldl_impl),
        ::typeof(identity),
        op::G, 
        init::Base._InitialValue, 
        x::Tuple;
    ) where {G}
    hobbits = accumulate(Base.tail(x); init=(first(x), nothing)) do (a, _), b
        # Here `a` is what we would normally cary forward, and `_` ignores
        # the previous iteration's pullback function (needed later),
        # while `b` is the fresh input from `list` as usual.
        c, back = rrule_via_ad(config, op, a, b)
        # We don't really need to store every `c`, last one is `foldl` output.
        # (The name, BTW, is because "there and back again" is the subtitle of Tolkien's book.)
    end
    y = first(last(hobbits))
    project = ProjectTo(x)
    function foldl_pullback_tuple(dy)
        trio = accumulate(reverse(hobbits); init=(0, dy, 0)) do (_, dc, _), (_, back)
            ds, da, db = back(dc)
            # Don't need to store every `da`, need one for the next iteration + the last.
        end
        dop = sum(first, trio)
        dx = (trio[end][2], reverse(map(last, trio))...)
        return (NoTangent(), NoTangent(), ProjectTo(op)(dop), NoTangent(), project(dx))
    end
    return y, foldl_pullback_tuple
end

function rrule(
        config::RuleConfig{>:HasReverseMode},
        ::typeof(Base.mapfoldl_impl),
        ::typeof(identity),
        op::G, 
        init, 
        x::Tuple;
    ) where {G}
    # Trivial case handled here to avoid ambiguities (and necc. because of Base.tail below)
    foldl_pullback_empty(dy) = (NoTangent(), NoTangent(), NoTangent(), dy, NoTangent())
    isempty(x) && return init, foldl_pullback_empty
    
    # Treat `init` by simply appending it to the `x`:
    y, back = rrule(config, Base.mapfoldl_impl, identity, op, Base._InitialValue(), (init, x...))
    project_x = ProjectTo(x)
    project_in = ProjectTo(init)
    function foldl_pullback_tuple_init(dy)
        _, _, dop, _, dxplus = back(dy)
        return (NoTangent(), NoTangent(), dop, project_in(first(dxplus)), project_x(Base.tail(dxplus)))
    end
    return y, foldl_pullback_tuple_init
end

#####
##### `foldl(f, ::Array)`
#####

# The implementation was originally for both tuples and arrays, although using accumulate
# to carry intermediate results along creates arrays of tuples which could be avoided.
# Using a loop can be a few times faster, this should be replaced:
# https://github.com/FluxML/Zygote.jl/issues/644#issuecomment-628762305

# Note also that it does not return a gradient for `init`, now marked `@not_implemented`.

function rrule(
        config::RuleConfig{>:HasReverseMode}, ::typeof(Base.mapfoldl_impl), ::typeof(identity), op::G, init, x::Union{AbstractArray, Tuple};
    ) where {G}
    start, list = if init === Base._InitialValue()
        Iterators.peel(x)
    else
        # Case with init keyword is simpler to understand first!
        init, x
    end
    hobbits = accumulate(list; init=(start, nothing)) do (a, _), b
        c, back = rrule_via_ad(config, op, a, b)
    end
    y = first(last(hobbits))
    axe = axes(x)
    project = ProjectTo(x)
    function unfoldl(dy)
        trio = accumulate(Iterators.reverse(hobbits); init=(0, dy, 0)) do (_, dc, _), (_, back)
            ds, da, db = back(dc)
        end
        dop = sum(first, trio)
        dx = map(last, Iterators.reverse(trio))
        if init === Base._InitialValue()  # `hobbits` is one short
            dx = _vcat1(trio[end][2], dx)
        end
        d_init = @not_implemented "gradient for foldl does not at present include init, sorry"
        return (NoTangent(), NoTangent(), dop, d_init, project(reshape(dx, axe)))
    end
    return y, unfoldl
end

_vcat1(x, ys::AbstractVector) = vcat(x, ys)
_vcat1(x::AbstractArray, ys::AbstractVector) = vcat([x], ys)

#####
##### `accumulate`
#####

# Like `foldl` this by definition works in order, so it makes sense to allow stateful `f`.

# Also like `foldl`, the version with a keyword `init` can't easily be given a gradient.
# Move it down to: `_accumulate!(op, B, A::AbstractVector, dims::Nothing, init::Nothing)`

function rrule(
        config::RuleConfig{>:HasReverseMode},
        ::typeof(Base._accumulate!), 
        op::G, y::AbstractVector, 
        x::AbstractVector, 
        dims::Nothing, 
        init,
    ) where {G}

    start, list = if init === nothing
        Iterators.peel(x)
    else
        something(init), x
    end
    hobbits = accumulate(list; init = (start, nothing)) do (a, _), b
        c, back = rrule_via_ad(config, op, a, b)
    end
    if init === nothing
        # `hobbits` is one short, and first one doesn't invoke `op`
        y[1] = first(x)
        map!(first, @view(y[2:end]), hobbits)
    else
        map!(first, y, hobbits)
    end
    axe = axes(x)
    project = ProjectTo(x)
    function decumulate(dy)
        dy_plain = unthunk(dy)
        rev_list = zip(Iterators.reverse(hobbits), Iterators.reverse(dy_plain))
        # Here we rely on `zip` to stop early when init === nothing. Begin explicit with Iterators.reverse(Iterators.drop(..., 1))
        # gets "no method matching iterate(::Base.Iterators.Reverse{Base.Iterators.Drop{Array{"
        trio = accumulate(rev_list; init=(0, ZeroTangent(), 0)) do (_, dc, _), ((_, back), dz)
            ds, da, db = back(dc + dz)
            # Don't need to store every 'da', but need for next iteration, and the last one.
        end
        dop = sum(first, trio)
        dx = map(last, Iterators.reverse(trio))
        if init == nothing
            # `hobbits` is one short, and the first one is weird
            dx = _vcat1(trio[end][2] + dy_plain[1], dx)
        end
        dy = @not_implemented "no gradient for `B` in `accumulate!(f, B, A)`, the rule intends to support `accumulate` only"
        d_init_not = @not_implemented "gradient for accumulate does not at present include init, sorry"
        d_init = init === nothing ? NoTangent() : Tangent{typeof(init)}(; value = d_init_not)
        return (NoTangent(), dop, dy, project(reshape(dx, axe)), NoTangent(), d_init)
    end
    return reshape(y, axe), decumulate
end
