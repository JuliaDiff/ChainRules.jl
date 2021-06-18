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

function rrule(::typeof(hcat), A::AbstractArray, Bs::AbstractArray...)
    function hcat_pullback(Ȳ)
        Xs = (A, Bs...)
        ntuple(length(Bs) + 2) do full_i
            full_i == 1 && return NoTangent()

            i = full_i - 1
            l = mapreduce(j->size(Xs[j], 2), Base.add_sum, 1:i-1; init=0)
            u = l + size(Xs[i], 2)
            dim = u > l + 1 ? (l+1:u) : u
            # NOTE: The copy here is defensive, since `selectdim` returns a view which we can
            # materialize with `copy`
            copy(selectdim(Ȳ, 2, dim))
        end
    end
    return hcat(A, Bs...), hcat_pullback
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

function rrule(::typeof(vcat), A::AbstractArray, Bs::AbstractArray...)
    function vcat_pullback(ȳ)
        Ȳ = unthunk(ȳ)
        n = size(A, 1)
        ∂A = copy(selectdim(Ȳ, 1, 1:n))
        ∂Bs = ntuple(length(Bs)) do i
            l = n + mapreduce(j->size(Bs[j], 1), Base.add_sum, 1:i-1; init=0)
            u = l + size(Bs[i], 1)
            copy(selectdim(Ȳ, 1, l+1:u))
        end
        return (NoTangent(), ∂A, ∂Bs...)
    end
    return vcat(A, Bs...), vcat_pullback
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
