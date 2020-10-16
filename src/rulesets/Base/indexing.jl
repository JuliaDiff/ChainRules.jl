#####
##### getindex
#####

function rrule(::typeof(getindex), x::Array{<:Number}, inds::Vararg{Int})
    y = getindex(x, inds...)
    function getindex_pullback(ȳ)
        function getindex_add!(Δ)
            Δ[inds...] = Δ[inds...] .+ ȳ
            return Δ
        end

        x̄ = InplaceableThunk(
            @thunk(getindex_add!(zero(x))),
            getindex_add!
        )
        return (NO_FIELDS, x̄, (DoesNotExist() for _ in inds)...)
    end

    return y, getindex_pullback
end
