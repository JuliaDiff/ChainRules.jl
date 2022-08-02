#####
##### `sort`
#####

function frule((_, ẋs, _), ::typeof(partialsort), xs::AbstractVector, k; kw...)
    inds = partialsortperm(xs, k; kw...)
    return xs[inds], ẋs[inds]
end

function rrule(::typeof(partialsort), xs::AbstractVector, k::Union{Integer,OrdinalRange}; kwargs...)
    inds = partialsortperm(xs, k; kwargs...)
    ys = xs[inds]

    function partialsort_pullback(Δys)
        function partialsort_add!(Δxs)
            Δxs[inds] += Δys
            return Δxs
        end

        Δxs = InplaceableThunk(partialsort_add!, @thunk(partialsort_add!(zero(xs))))

        return NoTangent(), Δxs, NoTangent()
    end

    return ys, partialsort_pullback
end

function frule((_, ẋs), ::typeof(sort), xs::AbstractVector; kw...)
    inds = sortperm(xs; kw...)
    return xs[inds], ẋs[inds]
end

function rrule(::typeof(sort), xs::AbstractVector; kwargs...)
    inds = sortperm(xs; kwargs...)
    ys = xs[inds]

    function sort_pullback(ȳ)
        Δys = unthunk(ȳ)
        function sort_add!(Δxs)
            Δxs[inds] += Δys
            return Δxs
        end

        Δxs = InplaceableThunk(sort_add!, @thunk(sort_add!(zero(Δys))))

        return NoTangent(), Δxs
    end
    return ys, sort_pullback
end

#####
##### `sortslices`
#####

function frule((_, ẋ), ::typeof(sortslices), x::AbstractArray; dims::Integer, kw...)
    p = sortperm(collect(eachslice(x; dims=dims)); kw...)
    inds = ntuple(d -> d == dims ? p : (:), ndims(x))
    return x[inds...], ẋ[inds...]
end

function rrule(::typeof(sortslices), x::AbstractArray; dims::Integer, kw...)
    p = sortperm(collect(eachslice(x; dims=dims)); kw...)
    inds = ntuple(d -> d == dims ? p : (:), ndims(x))
    function sortslices_pullback(dy)
        return (NoTangent(), ∇getindex(x, unthunk(dy), inds...))
    end
    return x[inds...], sortslices_pullback
end

#####
##### `unique`
#####

function rrule(::typeof(unique), x::AbstractArray{<:Number}; dims=:)
    axes_x = axes(x)
    y = unique(x; dims=dims)  # accepts only dims=: or dims::Integer
    function unique_pullback(dy_raw)
        dy = unthunk(dy_raw)
        if length(x) == length(y)
            # Short-circuit for the case of all unique, since `mask` is fairly expensive:
            dx = reshape(dy, axes_x)
            return (NoTangent(), ProjectTo(x)(dx))
        end

        if dims isa Colon
            xs, ys = vec(x), y
        else
            xs, ys = collect(eachslice(x; dims=dims)), collect(eachslice(y; dims=dims))
        end
        mask = isequal.(permutedims(ys), xs)  # unique([0.0, -0.0, NaN, NaN])
        mask .= (mask .== cumsum(mask, dims=1) .== true)  # this implements  findfirst(mask; dims=1)
        keep = map(I -> I[1], findall(mask))
        if dims isa Colon
            # The function `∇getindex` allows second derivatives.
            dx = reshape(∇getindex(vec(x), vec(dy), keep), axes_x) ## TODO understand again why vec!
        else
            inds = ntuple(d -> d==dims ? keep : (:), length(axes_x))
            dx = ∇getindex(x, dy, inds...)
        end
        return (NoTangent(), ProjectTo(x)(dx))
    end
    return y, unique_pullback
end
