#####
##### `reshape`
#####

function rrule(::typeof(reshape), A::AbstractArray, dims::Tuple{Vararg{Int}})
    function reshape_pullback(Ȳ)
        return (NO_FIELDS, reshape(Ȳ, dims), DoesNotExist())
    end
    return reshape(A, dims), reshape_pullback
end

function rrule(::typeof(reshape), A::AbstractArray, dims::Int...)
    function reshape_pullback(Ȳ)
        ∂A = reshape(Ȳ, dims)
        return (NO_FIELDS, ∂A, fill(DoesNotExist(), length(dims))...)
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
            full_i == 1 && return NO_FIELDS

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
        return (NO_FIELDS, NO_FIELDS, ∂As)
    end
    return reduce(hcat, As), reduce_hcat_pullback
end

#####
##### `vcat`
#####

function rrule(::typeof(vcat), A::AbstractArray, Bs::AbstractArray...)
    function vcat_pullback(Ȳ)
        n = size(A, 1)
        ∂A = copy(selectdim(Ȳ, 1, 1:n))
        ∂Bs = ntuple(length(Bs)) do i
            l = n + mapreduce(j->size(Bs[j], 1), Base.add_sum, 1:i-1; init=0)
            u = l + size(Bs[i], 1)
            copy(selectdim(Ȳ, 1, l+1:u))
        end
        return (NO_FIELDS, ∂A, ∂Bs...)
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
        return (NO_FIELDS, NO_FIELDS, ∂As)
    end
    return reduce(vcat, As), reduce_vcat_pullback
end

#####
##### `fill`
#####

function rrule(::typeof(fill), value::Any, dims::Tuple{Vararg{Int}})
    function fill_pullback(Ȳ)
        return (NO_FIELDS, sum(Ȳ), DoesNotExist())
    end
    return fill(value, dims), fill_pullback
end

function rrule(::typeof(fill), value::Any, dims::Int...)
    function fill_pullback(Ȳ)
        return (NO_FIELDS, sum(Ȳ), ntuple(_->DoesNotExist(), length(dims))...)
    end
    return fill(value, dims), fill_pullback
end
