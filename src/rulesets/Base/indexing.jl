#####
##### getindex
#####

function rrule(::typeof(getindex), x::Array, inds...)
    # removes any logical indexing, CartesianIndex etc
    # leaving us just with a tuple of Int, Arrays of Int and Ranges of Int
    plain_inds = Base.to_indices(x, inds)
    y = getindex(x, plain_inds...)
    x_shape = size(x)
    x_eltype = eltype(x)
    function getindex_pullback(ȳ)
        function getindex_add!(Δ)
            # this a optimizes away for simple cases
            for (ȳ_ii, ii) in zip(ȳ, Iterators.product(plain_inds...))
                Δ[ii...] += ȳ_ii
            end
            return Δ
        end

        x̄ = InplaceableThunk(
            Thunk() do
                z = if x_eltype <: Number && isconcretetype(x_eltype)
                    zeros(x_eltype, x_shape)
                else
                    # TODO this can probably be optimized to something more concrete
                    Array{Any}(undef, x_shape) .= Zero()
                end
                getindex_add!(z)
            end,
            getindex_add!
        )
        return (NO_FIELDS, x̄, (DoesNotExist() for _ in inds)...)
    end

    return y, getindex_pullback
end
