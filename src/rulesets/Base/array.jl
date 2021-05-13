#####
##### `reshape`
#####

function rrule(::typeof(reshape), A::AbstractArray, dims::Tuple{Vararg{Union{Colon,Int}}})
    A_dims = size(A)
    function reshape_pullback(Ȳ)
        return (NoTangent(), reshape(Ȳ, A_dims), NoTangent())
    end
    return reshape(A, dims), reshape_pullback
end

function rrule(::typeof(reshape), A::AbstractArray, dims::Union{Colon,Int}...)
    A_dims = size(A)
    function reshape_pullback(Ȳ)
        ∂A = reshape(Ȳ, A_dims)
        ∂dims = broadcast(_ -> NoTangent(), dims)
        return (NoTangent(), ∂A, ∂dims...)
    end
    return reshape(A, dims...), reshape_pullback
end

#####
##### `hcat` (🐈)
#####

# work around https://github.com/JuliaLang/julia/issues/40809
_get(x::Tuple, i::Int, default) = i in 1:length(x) ? x[i] : default

function rrule(::typeof(hcat), Xs...)
    Y = hcat(Xs...)  # note that Y always has 1-based indexing, even if X isa OffsetArray
    ndimsY = Val(ndims(Y))  # this avoids closing over Y, Val() is essential for type-stability
    sizes = map(size, Xs)   # this avoids closing over Xs
    function 🐈_pullback(dY)
        hi = Ref(0)  # Ref avoids hi::Core.Box
        dXs = map(sizes) do sizeX
            ndimsX = length(sizeX)
            lo = hi[] + 1
            hi[] += _get(sizeX, 2, 1)
            ind = ntuple(ndimsY) do d
                if d==2
                    d > ndimsX ? lo : lo:hi[]
                else
                    d > ndimsX ? 1 : (:)
                end
            end
            dY[ind...]  # no thunk as Xs may have 1 arg but 1 thunk is disallowed,
                        # and perhaps better to GC clean up dY.
        end
        return (NO_FIELDS, dXs...)
    end
    return Y, 🐈_pullback
end

function rrule(::typeof(reduce), ::typeof(hcat), As::AbstractVector{<:AbstractVecOrMat})
    function reduce_hcat_pullback(ΔY)
        sizes = size.(As, 2)
        cumsizes = cumsum(sizes)
        ∂As = map(cumsizes, sizes) do post, diff
            pre = post - diff + 1
            return ΔY[:, pre:post]
        end
        return (NoTangent(), NoTangent(), ∂As)
    end
    return reduce(hcat, As), reduce_hcat_pullback
end

#####
##### `vcat`
#####

function rrule(::typeof(vcat), Xs...)
    Y = vcat(Xs...)
    ndimsY = Val(ndims(Y))
    sizes = map(size, Xs)
    function vcat_pullback(dY)
        hi = Ref(0)
        dXs = map(sizes) do sizeX
            ndimsX = length(sizeX)
            lo = hi[] + 1
            hi[] += _get(sizeX, 1, 1)
            ind = ntuple(ndimsY) do d
                if d==1
                    d > ndimsX ? lo : lo:hi[]
                else
                    d > ndimsX ? 1 : (:)
                end
            end
            dY[ind...]
        end
        return (NO_FIELDS, dXs...)
    end
    return Y, vcat_pullback
end

function rrule(::typeof(reduce), ::typeof(vcat), As::AbstractVector{<:AbstractVecOrMat})
    function reduce_vcat_pullback(ΔY)
        sizes = size.(As, 1)
        cumsizes = cumsum(sizes)
        ∂As = map(cumsizes, sizes) do post, diff
            pre = post - diff + 1
            return ΔY[pre:post, :]
        end
        return (NoTangent(), NoTangent(), ∂As)
    end
    return reduce(vcat, As), reduce_vcat_pullback
end

#####
##### `cat`
#####

_val(::Val{x}) where {x} = x

function rrule(::typeof(cat), Xs...; dims)
    Y = cat(Xs...; dims)
    cdims = dims isa Val ? Int(_val(dims)) : dims isa Integer ? Int(dims) : Tuple(dims)
    ndimsY = Val(ndims(Y))
    sizes = map(size, Xs)
    function cat_pullback(dY)
        prev = fill(0, _val(ndimsY))  # note that Y always has 1-based indexing, even if X isa OffsetArray
        dXs = map(sizes) do sizeX
            ndimsX = length(sizeX)
            index = ntuple(ndimsY) do d
                if d in cdims
                    d > ndimsX ? (prev[d]+1) : (prev[d]+1:prev[d]+sizeX[d])
                else
                    d > ndimsX ? 1 : (:)
                end
            end
            for d in cdims
                prev[d] += _get(sizeX, d, 1)
            end
            dY[index...]
        end
        return (NO_FIELDS, dXs...)
    end
    return Y, cat_pullback
end

#####
##### `hvcat`
#####

function rrule(::typeof(hvcat), rows, values...)
    Y = hvcat(rows, values...)
    cols = size(Y,2)
    ndimsY = Val(ndims(Y))
    sizes = map(size, values)
    function hvcat_pullback(dY)
        prev = fill(0, 2)
        dXs = map(sizes) do sizeX
            ndimsX = length(sizeX)
            index = ntuple(ndimsY) do d
                if d in (1, 2)
                    d > ndimsX ? (prev[d]+1) : (prev[d]+1:prev[d]+sizeX[d])
                else
                    d > ndimsX ? 1 : (:)
                end
            end
            prev[2] += _get(sizeX, 2, 1)
            if prev[2] == cols
                prev[2] = 0
                prev[1] += _get(sizeX, 1, 1)
            end
            dY[index...]
        end
        return (NO_FIELDS, DoesNotExist(), dXs...)
    end
    return Y, hvcat_pullback
end

#####
##### `fill`
#####

function rrule(::typeof(fill), value::Any, dims::Tuple{Vararg{Int}})
    function fill_pullback(Ȳ)
        return (NoTangent(), sum(Ȳ), NoTangent())
    end
    return fill(value, dims), fill_pullback
end

function rrule(::typeof(fill), value::Any, dims::Int...)
    function fill_pullback(Ȳ)
        return (NoTangent(), sum(Ȳ), ntuple(_->NoTangent(), length(dims))...)
    end
    return fill(value, dims), fill_pullback
end
