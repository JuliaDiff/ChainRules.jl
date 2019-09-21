#####
##### `mean`
#####

_denom(x, dims::Colon) = length(x)
_denom(x, dims::Integer) = size(x, dims)
_denom(x, dims) = mapreduce(i->size(x, i), Base.mul_prod, unique(dims), init=1)

# TODO: We have `mean(f, x; dims)` as of 1.3.0-DEV.36

function rrule(::typeof(mean), x::AbstractArray{<:Real}; dims=:)
    y_sum, sum_pullback = rrule(sum, x; dims=dims)
    n = _denom(x, dims)
    function mean_pullback(ȳ)
        ∂x = Thunk() do
            _, ∂sum_x = sum_pullback(ȳ)
            extern(∂sum_x) / n
        end
        return (NO_FIELDS, ∂x)
    end
    return y_sum / n, mean_pullback
end

function rrule(::typeof(mean), f, x::AbstractArray{<:Real})
    y_sum, sum_pullback = rrule(sum, f, x)
    n = _denom(x, :)
    function mean_pullback(ȳ)
        ∂x = Thunk() do
            _, _, ∂sum_x = sum_pullback(ȳ)
            extern(∂sum_x) / n
        end
        return (NO_FIELDS, DoesNotExist(), ∂x)
    end
    return y_sum / n, mean_pullback
end
