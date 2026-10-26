# Int rather than Int64/Integer is intentional
function ChainRulesCore.frule((_, Δ, _), ::typeof(getfield), strct, sym::Union{Int,Symbol})
    return (getfield(strct, sym), isa(Δ, NoTangent) ? NoTangent() : getproperty(Δ, sym))
end

function ChainRulesCore.frule((_, Δ, _, _), ::typeof(getfield), strct, sym::Union{Int,Symbol}, inbounds)
    return (getfield(strct, sym, inbounds), isa(Δ, NoTangent) ? NoTangent() : getproperty(Δ, sym))
end

"for a given tuple type, returns a Val{N} where N is the length of the tuple"
_tuple_N(::Type{<:Tuple{Vararg{Any, N}}}) where {N} = Val(N)

function rrule(::typeof(getindex), x::T, i::Integer) where {T<:Tuple}
    function getindex_back_1(dy)
        dx = ntuple(j -> j == i ? dy : NoTangent(), _tuple_N(T))
        return (NoTangent(), Tangent{T}(dx...), NoTangent())
    end
    return x[i], getindex_back_1
end

# Special case for tuples of only numbers
function rrule(::typeof(getindex), x::T, i::Integer) where {T<:NTuple{<:Any,<:Number}}
    function getindex_back_2(dy_raw)
        dy = unthunk(dy_raw)
        dx = ntuple(j -> j == i ? dy : zero(dy), _tuple_N(T))
        return (NoTangent(), Tangent{T}(dx...), NoTangent())
    end
    return x[i], getindex_back_2
end

# Note Zygote has getindex(::Tuple, ::UnitRange) separately from getindex(::Tuple, ::AbstractVector),
# whether that's more efficient has not been investigated here.
# https://github.com/FluxML/Zygote.jl/blob/master/src/lib/lib.jl#L125-L142
function rrule(::typeof(getindex), x::T, inds) where {T<:Tuple}  # e.g. ranges, not type-stable
    function getindex_back_3(dy_raw)
        dy = unthunk(dy_raw)
        dx = ntuple(Returns(NoTangent()), _tuple_N(T))
        for (dyi, i) in zip(dy, inds)
            dx = Base.setindex(dx, dyi + dx[i], i)
        end
        return (NoTangent(), Tangent{T}(dx...), NoTangent())
    end
    return x[inds], getindex_back_3
end

function rrule(::typeof(getindex), x::Tuple, ::Colon)
    getindex_back_4(dy) = (NoTangent(), dy, NoTangent())
    return x, getindex_back_4
end

#####
##### getindex(::AbstractArray)
#####

function frule((_, ẋ), ::typeof(getindex), x::AbstractArray, inds...)
    return x[inds...], ẋ[inds...]
end

function rrule(::typeof(getindex), x::AbstractArray, inds...)
    nots = map(Returns(NoTangent()), inds)
    getindex_pullback(dy) = (NoTangent(), thunked_∇getindex(x, dy, inds...), nots...)
    getindex_pullback(z::AbstractZero) = (NoTangent(), z, nots...)
    return x[inds...], getindex_pullback
end

function thunked_∇getindex(x, dy, inds...)
    return InplaceableThunk(
        dx -> ∇getindex!(dx, unthunk(dy), Base.to_indices(x, inds)...),
        @thunk(∇getindex(x, unthunk(dy), inds...)),
    )
end

"""
    ∇getindex(x, dy, inds...)

For the `rrule` of `y = x[inds...]`, this function is roughly
`setindex(zero(x), dy, inds...)`, returning the array `dx`.
Differentiable. Includes `ProjectTo(x)(dx)`.
"""
function ∇getindex(x::AbstractArray{T,N}, dy, inds...) where {T,N}
    # `to_indices` removes any logical indexing, colons, CartesianIndex etc,
    # leaving just Int / AbstractVector of Int
    plain_inds = Base.to_indices(x, inds)
    dx = if plain_inds isa NTuple{N, Int} && T<:Number
        # scalar indexing
        OneElement(dy, plain_inds, axes(x))
    else  # some from slicing (potentially noncontigous)
        dx = _setindex_zero(x, dy, plain_inds...)
        ∇getindex!(dx, dy, plain_inds...)
    end
    return ProjectTo(x)(dx)  # since we have x, may as well do this inside, not in rules
end
∇getindex(x::AbstractArray, z::AbstractZero, inds...) = z

"""
    OneElement(val, ind, axes) <: AbstractArray

Extremely simple `struct` used for the gradient of scalar `getindex`.
"""
struct OneElement{T,N,I,A} <: AbstractArray{T,N}
  val::T
  ind::I
  axes::A
  OneElement(val::T, ind::I, axes::A) where {T<:Number, I<:NTuple{N,Int}, A<:NTuple{N,AbstractUnitRange}} where {N} = new{T,N,I,A}(val, ind, axes)
end
Base.size(A::OneElement) = map(length, A.axes)
Base.axes(A::OneElement) = A.axes
Base.getindex(A::OneElement{T,N}, i::Vararg{Int,N}) where {T,N} = ifelse(i==A.ind, A.val, zero(T))

function ChainRulesCore.add!!(xs::AbstractArray{<:Any,N}, oe::OneElement{<:Any,N}) where {N}
    if !ChainRulesCore.is_inplaceable_destination(xs)
        xs = collect(xs)
    end
    xs[oe.ind...] += oe.val
    return xs
end

Base.:(+)(xs::AbstractArray, oe::OneElement) = add!!(copy(xs), oe)
Base.:(+)(oe::OneElement, xs::AbstractArray) = +(xs, oe)
Base.:(+)(oe1::OneElement, oe2::OneElement) = +(collect(oe1), oe2)

"""
    _setindex_zero(x, dy, inds...)

This returns roughly `dx = zero(x)`, except that this is guaranteed to be mutable via `similar`,
and its element type is wide enough to allow `setindex!(dx, dy, inds...)`, which is exactly what
`∇getindex` does next.

It's unfortunate to close over `x`, but `similar(typeof(x), axes(x))` doesn't
allow `eltype(dy)`, nor does it work for many structured matrices.
"""
_setindex_zero(x::AbstractArray{<:Number}, dy, inds::Integer...) = fill!(similar(x, typeof(dy), axes(x)), false)
_setindex_zero(x::AbstractArray{<:Number}, dy, inds...) = fill!(similar(x, eltype(dy), axes(x)), false)
function _setindex_zero(x::AbstractArray, dy, inds::Integer...)
    # This allows for types which don't define zero (like Vector) and types whose zero special (like Tangent),
    # but always makes an abstract type. TODO: make it infer concrete type for e.g. vectors of SVectors
    T = Union{typeof(dy), ZeroTangent}
    return fill!(similar(x, T, axes(x)), ZeroTangent())
end
function _setindex_zero(x::AbstractArray, dy, inds...)
    T = Union{eltype(dy), ZeroTangent}
    return fill!(similar(x, T, axes(x)), ZeroTangent())
end
ChainRules.@non_differentiable _setindex_zero(x::AbstractArray, dy::Any, inds::Any...)

function ∇getindex!(dx::AbstractArray, dy, inds::Integer...)
    @views dx[inds...] += dy
    return dx
end
function ∇getindex!(dx::AbstractArray, dy, inds...)
    view(dx, inds...) .+= dy
    return dx
end

# Allow for second derivatives, by writing rules for `∇getindex`:

function frule((_, _, dẏ), ::typeof(∇getindex), x, dy, inds...)
    return ∇getindex(x, dy, inds...), ∇getindex(x, dẏ, inds...)
end

function rrule(::typeof(∇getindex), x, dy, inds...)
    z = ∇getindex(x, dy, inds...)
    function ∇getindex_pullback(dz)
        d2y = getindex(unthunk(dz), inds...)
        nots = map(Returns(NoTangent()), inds)
        return (NoTangent(), NoTangent(), ProjectTo(dy)(d2y), nots...)
    end
    return z, ∇getindex_pullback
end

# Indexing with repeated indices on a GPU will lead ∇getindex to have race conditions & wrong answers.
# To avoid this, copy everything back to the CPU.
# But don't do that for indices which are known to be unique, e.g. `A[1, 2:3, :]` the colon gives Base.Slice:

function ∇getindex!(dx::AbstractGPUArray, dy, inds::Integer...)
    view(dx, inds...) .+= Ref(dy)
    return dx
end
function ∇getindex!(dx::AbstractGPUArray, dy, inds::Union{Integer, AbstractUnitRange, Base.Slice}...)
    view(dx, inds...) .+= dy
    return dx
end
function ∇getindex!(dx::AbstractGPUArray, dy, inds...)
    dx_cpu = adapt(Array, dx)
    view(dx_cpu, adapt(Array, inds)...) .+= adapt(Array, dy)
    copyto!(dx, dx_cpu)
    return dx
end

#####
##### view
#####

function frule((_, ẋ), ::typeof(view), x::AbstractArray, inds...)
    return view(x, inds...), view(ẋ, inds...)
end

function rrule(::typeof(view), x::AbstractArray, inds...)
    nots = map(Returns(NoTangent()), inds)
    view_pullback(dy) = (NoTangent(), thunked_∇getindex(x, dy, inds...), nots...)
    view_pullback(z::AbstractZero) = (NoTangent(), z, nots...)
    return view(x, inds...), view_pullback
end

function rrule(::typeof(view), x::AbstractArray, i::Integer, jkl::Integer...)
    # This case returns a zero-dim array, unlike getindex. So we fool ∇getindex:
    function view_pullback_0(dy)
        nots = map(Returns(NoTangent()), (i, jkl...))
        return (NoTangent(), thunked_∇getindex(x, dy, i:i, jkl...), nots...)
    end
    return view(x, i, jkl...), view_pullback_0
end

#####
##### setindex!
#####

function frule((_, ẋ, v̇), ::typeof(setindex!), x::AbstractArray, v, inds...)
    return setindex!(x, v, inds...), setindex!(ẋ, v̇, inds...)
end

#####
##### unsafe_getindex
#####

# This is called by e.g. `iterate(1:0.1:2)`,
# and fixes https://github.com/FluxML/Zygote.jl/issues/1247
# Only needs to accept AbstractRange, but AbstractVector makes testing easier.

function frule((_, ẋ), ::typeof(Base.unsafe_getindex), x::AbstractVector, i::Integer)
    return Base.unsafe_getindex(x, i), getindex(ẋ, i)
end

function rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(Base.unsafe_getindex), x::AbstractVector, i::Integer)
    return rrule_via_ad(cfg, getindex, x, i)
end

#####
##### `eachslice` and friends
#####

function rrule(::typeof(eachrow), x::AbstractVecOrMat)
    allrows(dy) = (NoTangent(), ∇eachslice(unthunk(dy), x, Val(1)))
    return collect(eachrow(x)), allrows
end

function rrule(::typeof(eachcol), x::AbstractVecOrMat)
    allcols(dy) = (NoTangent(), ∇eachslice(unthunk(dy), x, Val(2)))
    return collect(eachcol(x)), allcols
end

function rrule(::typeof(eachslice), x::AbstractArray; dims)
    y = collect(eachslice(x; dims=dims))
    @assert length(dims) == 1 """That's amazing, after many years JuliaLang/julia#32310
        actually landed. Sadly, the gradient rule for `eachslice` is unable to handle this
        case right now, please make an issue at https://github.com/JuliaDiff/ChainRules.jl"""
    dim = only(dims)
    allslices(dy) = (NoTangent(), ∇eachslice(unthunk(dy), x, Val(dim)))
    return y, allslices
end

# Using Val(dim) here is worth a factor of 2 in this, on Julia 1.8-
# @btime rrule(eachcol, $([1 2; 3 4]))[2]($([[10, 20], [30, 40]]))
function ∇eachslice(dys_raw, x::AbstractArray, vd::Val{dim}) where {dim}
    dys = unthunk.(unthunk(dys_raw))
    i1 = findfirst(dy -> dy isa AbstractArray, dys)
    if i1 === nothing  # all slices are Zero!
        return _zero_fill!(similar(x, float(eltype(x)), axes(x)))
    end

    T = Base.promote_eltype(dys...)
    # The whole point of this gradient is that we can allocate one `dx` array:
    dx = similar(x, T, axes(x))
    for i in axes(x, dim)
        slice = selectdim(dx, dim, i)
        dy = dys[i]
        if dy isa AbstractZero
            _zero_fill!(slice)  # Avoids this: copyto!([1,2,3], ZeroTangent()) == [0,2,3]
        else
            copyto!(slice, dy)
        end
    end
    return ProjectTo(x)(dx)
end
∇eachslice(dys::AbstractZero, x::AbstractArray, vd::Val{dim}) where {dim} = dys

_zero_fill!(dx::AbstractArray) = fill!(dx, zero(eltype(dx)))

function rrule(::typeof(∇eachslice), dys, x, vd::Val)
    function ∇∇eachslice(dz_raw)
        dz = unthunk(dz_raw)
        # eachslice(dz; dims=_val(vd)) does not make @code_warntype happy
        iter = vd == Val(1) ? eachrow(dz) : vd == Val(2) ? eachcol(dz) : eachslice(dz; dims=_val(vd))
        return (NoTangent(), collect(iter), NoTangent(), NoTangent())
    end
    return ∇eachslice(dys, x, vd), ∇∇eachslice
end

# These rules help with testing, and won't hurt:
# They are correct as we always `collect` the primal result as we need that
# information for the reverse pass
ChainRules.rrule(::typeof(collect∘eachrow), x) = rrule(eachrow, x)
ChainRules.rrule(::typeof(collect∘eachcol), x) = rrule(eachcol, x)
ChainRules.rrule(::typeof(collect∘eachslice), x; dims) = rrule(eachslice, x; dims=dims)
