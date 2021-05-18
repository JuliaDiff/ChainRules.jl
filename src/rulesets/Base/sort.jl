function rrule(::typeof(partialsort), xs::AbstractVector, k::Union{Integer,OrdinalRange}; kwargs...)
    inds = partialsortperm(xs, k; kwargs...)
    ys = xs[inds]

    function partialsort_pullback(Δys)
        function partialsort_add!(Δxs)
            Δxs[inds] += Δys
            return Δxs
        end

        Δxs = InplaceableThunk(@thunk(partialsort_add!(zero(xs))), partialsort_add!)

        return NO_FIELDS, Δxs, NoTangent()
    end

    return ys, partialsort_pullback
end

function rrule(::typeof(sort), xs::AbstractVector; kwargs...)
    inds = sortperm(xs; kwargs...)
    ys = xs[inds]

    function sort_pullback(Δys)
        function sort_add!(Δxs)
            Δxs[inds] += Δys
            return Δxs
        end

        Δxs = InplaceableThunk(@thunk(sort_add!(zero(Δys))), sort_add!)

        return NO_FIELDS, Δxs
    end
    return ys, sort_pullback
end
