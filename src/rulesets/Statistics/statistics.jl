#####
##### `mean`
#####

_denom(x, dims::Colon) = length(x)
_denom(x, dims::Integer) = size(x, dims)
_denom(x, dims) = mapreduce(i->size(x, i), Base.mul_prod, unique(dims), init=1)

# TODO: We have `mean(f, x; dims)` as of 1.3.0-DEV.36

function rrule(::typeof(mean), x::AbstractArray{<:Real}; dims=:)
    _, dx = rrule(sum, x; dims=dims)
    n = _denom(x, dims)
    return mean(x; dims=dims), Rule(ȳ -> dx(ȳ) / n)
end

function rrule(::typeof(mean), f, x::AbstractArray{<:Real})
    _, (_, dx) = rrule(sum, f, x)
    n = _denom(x, :)
    return mean(f, x), (DNERule(), Rule(ȳ -> dx(ȳ) / n))
end
