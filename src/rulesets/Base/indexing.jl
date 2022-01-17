#####
##### getindex
#####

function rrule(::typeof(getindex), x::Tuple, ind)
    y = x[ind]
    z = map(Returns(NoTangent()), x)
    project = ProjectTo(x)
    function getindex_pullback(ȳ, ind::Integer)
        x̄ = Base.setindex(z, ȳ, ind)
        return (NoTangent(), project(x̄), NoTangent())
    end
    function getindex_pullback(ȳ, ind)
        x̄ = z
        for (i, yi) in zip(ind, unthunk(ȳ))
            x̄ = ntuple(k -> k==i ? x̄[i] + yi : x̄[k], length(z))
        end
        return (NoTangent(), project(x̄), NoTangent())
    end
    getindex_pullback(ȳ) = getindex_pullback(ȳ, ind)
    return y, getindex_pullback
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
##### setindex
#####

function rrule(::typeof(Base.setindex), x::Tuple, val, i::Integer)
    y = Base.setindex(x, val, i)
    valN = Val(length(x))
    project = ProjectTo(x)
    function setindex_pullback(ȳ)
        v̄āl̄ = ȳ[i]
        x̄ = ntuple(k -> k==i ? ZeroTangent() : ȳ[k], valN)
        return (NoTangent(), project(x̄), v̄āl̄, NoTangent())
    end
    return y, setindex_pullback
end

