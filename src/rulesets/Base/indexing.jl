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



