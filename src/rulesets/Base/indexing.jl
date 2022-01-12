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
    dim = if length(dims) == 1
        only(dims)
    else
        throw(ArgumentError("""That's amazing, after many years JuliaLang/julia#32310 actually landed.
            Sadly, the gradient rule for `eachslice` is unable to handle this case right now,
            please make an issue at https://github.com/JuliaDiff/ChainRules.jl"""))
    end
    allslices(dy) = (NoTangent(), ∇eachslice(unthunk(dy), x, Val(dim)))
    return y, allslices
end

# Using Val(dim) here is worth a factor of 2 in this, on Julia 1.8-
# @btime rrule(eachcol, $([1 2; 3 4]))[2]($([[10, 20], [30, 40]]))
function ∇eachslice(dys, x::AbstractArray, vd::Val{dim}) where {dim}
    i1 = findfirst(dy -> dy isa AbstractArray, dys)
    if i1 === nothing  # all slices are Zero!
        return (zero(x),)
    end
    T = promote_type(eltype(dys[i1]), eltype(x))
    # The whole point of this gradient is that we can allocate one `dx` array:
    dx = similar(x, T)
    for i in axes(x, dim)
        slice = selectdim(dx, dim, i)
        if dys[i] isa AbstractArray
            copyto!(slice, dys[i])
        else
            slice .= zero.(slice)
        end
    end
    return dx
end
