#####
##### `sort`
#####

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

function rrule(::typeof(sortslices), x::AbstractArray{<:Number}; dims::Integer, kw...)
    p = sortperm(collect(eachslice(x; dims=dims)); kw...)
    inds = ntuple(d -> d == dims ? p : (:), ndims(x))
    function sortslices_pullback(dy)
        # No actual need to zero this, and if you didn't, then you could widen eltype
        # Also, you could use similar(dy) here not x, same size?
        dx = _zerolike_writeat(x, unthunk(dy), (), inds...)
        return (NoTangent(), ProjectTo(x)(dx))
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
        mask .= (mask .== cumsum(mask, dims=1) .== true)
        keep = map(I -> I[1], findall(mask))
        if dims isa Colon
            # The function `_zerolike_writeat` is defined near `maximum`, allows
            # second derivatives. Should perhaps eventually be shared with `getindex`.
            dx = reshape(_zerolike_writeat(vec(x), vec(dy), (), keep), axes_x)
        else
            inds = ntuple(d -> d==dims ? keep : (:), length(axes_x))
            dx = _zerolike_writeat(x, dy, (), inds...)
        end
        return (NoTangent(), ProjectTo(x)(dx))
    end
    return y, unique_pullback
end

function _zerolike_writeat(x, dy, dims, ind...)
    # It's unfortunate to close over `x`, but `similar(typeof(x), axes(x))` doesn't 
    # allow `eltype(dy)`, nor does it work for many structured matrices.
    dx = fill!(similar(x, eltype(dy), axes(x)), false)
    view(dx, ind...) .= dy  # possibly 0-dim view, allows dy::Number and dy::Array, and dx::CuArray
    dx
end
