#####
##### `reshape`
#####

function rrule(::typeof(reshape), A::AbstractArray, dims::Tuple{Vararg{Int}})
    return reshape(A, dims), Ȳ -> (NO_FIELDS, reshape(Ȳ, dims), DNE())
end

function rrule(::typeof(reshape), A::AbstractArray, dims::Int...)
    return (
        reshape(A, dims...),
        Ȳ -> (NO_FIELDS, reshape(Ȳ, dims), fill(DNE(), length(dims))...)
    )
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

#####
##### `fill`
#####

function rrule(::typeof(fill), value::Any, dims::Tuple{Vararg{Int}})
    return fill(value, dims), Ȳ -> (NO_FIELDS, sum(Ȳ), DNE())
end

function rrule(::typeof(fill), value::Any, dims::Int...)
    return fill(value, dims), Ȳ -> (NO_FIELDS, sum(Ȳ), ntuple(_->DNE(), length(dims))...)
end
