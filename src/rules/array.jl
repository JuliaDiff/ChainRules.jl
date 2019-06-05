#####
##### `reshape`
#####

function rrule(::typeof(reshape), A::AbstractArray, dims::Tuple{Vararg{Int}})
    return reshape(A, dims), (Rule(Ȳ->reshape(Ȳ, dims)), DNERule())
end

function rrule(::typeof(reshape), A::AbstractArray, dims::Int...)
    Y, (rule, _) = rrule(reshape, A, dims)
    return Y, (rule, fill(DNERule(), length(dims))...)
end
