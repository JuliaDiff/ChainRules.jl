#####
##### getindex(::Tuple)
#####

function frule((_, ẋ), ::typeof(getindex), x::Tuple, i::Integer)
    return x[i], ẋ[i]
end

function frule((_, ẋ), ::typeof(getindex), x::Tuple, i)
    y = x[i]
    return y, Tangent{typeof(y)}(ẋ[i]...)
end

"for a given typle type, returns a Val{N} where N is the length of the tuple"
_tuple_N(::Type{<:Tuple{Vararg{<:Any, N}}}) where {N} = Val(N)

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
##### getindex
#####

function frule((_, ẋ), ::typeof(getindex), x::AbstractArray, inds...)
    return x[inds...], ẋ[inds...]
end

function rrule(::typeof(getindex), x::Array{<:Number}, inds...)
    # removes any logical indexing, CartesianIndex etc
    # leaving us just with a tuple of Int, Arrays of Int and Ranges of Int
    plain_inds = Base.to_indices(x, inds)
    y = getindex(x, plain_inds...)
    function getindex_pullback(ȳ)
        function getindex_add!(Δ)
            # this a optimizes away for simple cases
            for (ȳ_ii, ii) in zip(ȳ, Iterators.product(plain_inds...))
                Δ[ii...] += ȳ_ii
            end
            return Δ
        end

        x̄ = InplaceableThunk(
            getindex_add!,
            @thunk(getindex_add!(zero(x))),
        )
        īnds = broadcast(Returns(NoTangent()), inds)
        return (NoTangent(), x̄, īnds...)
    end

    return y, getindex_pullback
end

#####
##### first, tail
#####

function frule((_, ẋ), ::typeof(first), x::Tuple)
    return first(x), first(ẋ)
end

function rrule(::typeof(first), x::T) where {T<:Tuple}
    first_back(dy) = (NoTangent(), Tangent{T}(ntuple(j -> j == 1 ? dy : NoTangent(), _tuple_N(T))...))
    return first(x), first_back
end

function frule((_, ẋ), ::typeof(Base.tail), x::Tuple)
    y = Base.tail(x)
    return y, Tangent{typeof(y)}(Base.tail(Tuple(ẋ))...)
end

function rrule(::typeof(Base.tail), x::T) where {T<:Tuple}
    tail_pullback(dy) = (NoTangent(), Tangent{T}(NoTangent(), dy...))
    return Base.tail(x), tail_pullback
end

#####
##### view
#####

function frule((_, ẋ), ::typeof(view), x::AbstractArray, inds...)
    return view(x, inds...), view(ẋ, inds...)
end

#####
##### setindex!
#####

function frule((_, ẋ, v̇), ::typeof(setindex!), x::AbstractArray, v, inds...)
    return setindex!(x, v, inds...), setindex!(ẋ, v̇, inds...)
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
    dys = unthunk(dys_raw)
    i1 = findfirst(dy -> dy isa AbstractArray, dys)
    if i1 === nothing  # all slices are Zero!
        return _zero_fill!(similar(x, float(eltype(x)), axes(x)))
    end
    T = promote_type(eltype(dys[i1]), eltype(x))
    # The whole point of this gradient is that we can allocate one `dx` array:
    dx = similar(x, T, axes(x))
    for i in axes(x, dim)
        slice = selectdim(dx, dim, i)
        if dys[i] isa AbstractZero
            _zero_fill!(slice)  # Avoids this: copyto!([1,2,3], ZeroTangent()) == [0,2,3]
        else
            copyto!(slice, dys[i])
        end
    end
    return ProjectTo(x)(dx)
end
∇eachslice(dys::AbstractZero, x::AbstractArray, vd::Val{dim}) where {dim} = dys

_zero_fill!(dx::AbstractArray{<:Number}) = fill!(dx, zero(eltype(dx)))
_zero_fill!(dx::AbstractArray) = map!(zero, dx, dx)

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
