#####
##### `reshape`
#####

function rrule(::typeof(reshape), A::AbstractArray, dims::Tuple{Vararg{Int}})
    return reshape(A, dims), (NO_FIELDS_RULE, Rule(Ȳ->reshape(Ȳ, dims)), DNERule())
end

function rrule(::typeof(reshape), A::AbstractArray, dims::Int...)
    Y, (nofields, rule, dne) = rrule(reshape, A, dims)[2]
    @assert no_fields === NO_FIELDS_RULE
    @assert dne === DNERule()
    return Y, (NO_FIELDS_RULE, rule, fill(DNERule(), length(dims))...)
end

#####
##### `hcat` (🐈)
#####

function rrule(::typeof(hcat), A::AbstractArray, Bs::AbstractArray...)
    Y = hcat(A, Bs...)
    Xs = (A, Bs...)
    rules = ntuple(length(Bs) + 1) do i
        l = mapreduce(j->size(Xs[j], 2), Base.add_sum, 1:i-1; init=0)
        u = l + size(Xs[i], 2)
        dim = u > l + 1 ? (l+1:u) : u
        # NOTE: The copy here is defensive, since `selectdim` returns a view which we can
        # materialize with `copy`
        Rule(Ȳ->copy(selectdim(Ȳ, 2, dim)))
    end
    return Y, (NO_FIELDS_RULE, rules...)
end

#####
##### `vcat`
#####

function rrule(::typeof(vcat), A::AbstractArray, Bs::AbstractArray...)
    Y = vcat(A, Bs...)
    n = size(A, 1)
    ∂A = Rule(Ȳ->copy(selectdim(Ȳ, 1, 1:n)))
    ∂Bs = ntuple(length(Bs)) do i
        l = n + mapreduce(j->size(Bs[j], 1), Base.add_sum, 1:i-1; init=0)
        u = l + size(Bs[i], 1)
        Rule(Ȳ->copy(selectdim(Ȳ, 1, l+1:u)))
    end
    return Y, (NO_FIELDS_RULE, ∂A, ∂Bs...)
end

#####
##### `fill`
#####

function rrule(::typeof(fill), value::Any, dims::Tuple{Vararg{Int}})
    return fill(value, dims), (NO_FIELDS_RULE, Rule(sum), DNERule())
end

function rrule(::typeof(fill), value::Any, dims::Int...)
    return fill(value, dims), (NO_FIELDS_RULE, Rule(sum), ntuple(_->DNERule(), length(dims))...)
end
