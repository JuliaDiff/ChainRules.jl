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
##### `unique`
#####

function rrule(::typeof(unique), x::AbstractArray{<:Number}; dims=:)
    axes_x = axes(x)
    project = ProjectTo(x)
    y = unique(x; dims=dims)  # accepts only dims=: or dims::Integer
    if dims isa Colon
        xs, ys = vec(x), y
    else
        xs, ys = collect(eachslice(x; dims=dims)), collect(eachslice(y; dims=dims))
    end
    mask = isequal.(permutedims(ys), xs)  # unique([0.0, -0.0, NaN, NaN])
    mask .= (mask .== cumsum(mask, dims=1) .== true)
    keep = map(I -> I[1], findall(mask))
    function unique_pullback(dy_raw)
        dy = unthunk(dy_raw)
        if dims isa Colon
            # The function `_zerolike_writeat` is defined near `maximum`, allows
            # second derivatives. Should perhaps eventually be shared with `getindex`.
            dx = reshape(_zerolike_writeat(vec(x), vec(dy), (), keep), axes_x)
        else
            inds = ntuple(d -> d==dims ? keep : (:), length(axes_x))
            dx = _zerolike_writeat(x, dy, (), inds...)
        end            
        return (NoTangent(), project(dx))
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

#=

rrule(unique, [1,1,2,3])[2]([10,20,30]) == (NoTangent(), [10, 0, 20, 30])
rrule(unique, [1 2; 1 4])[2]([10,20,30]) == (NoTangent(), [10 20; 0 30])

rrule(unique, [1 2 1 2; 1 2 1 4], dims=2)[2]([10 20 30; 40 50 60])[2] == [10 20 0 30; 40 50 0 60]

rrule(unique, Diagonal([1,2,3]))[2]([10 20 30 40])[2] == [10.0 0.0 0.0; 0.0 30.0 0.0; 0.0 0.0 40.0]

=#
