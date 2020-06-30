#####
##### `sum`
#####

function frule((_, ẋ), ::typeof(sum), x; dims=:)
    return sum(x; dims=dims), sum(ẋ; dims=dims)
end

function rrule(::typeof(sum), x::AbstractArray{T}; dims=:) where {T<:Number}
    y = sum(sum, x; dims=dims)
    function sum_pullback(ȳ)
        # broadcasting the two works out the size no-matter `dims`
        x̄ = @thunk broadcast(x, ȳ) do xi, ȳi
            ȳi
        end
        return (NO_FIELDS, x̄)
    end
    return y, sum_pullback
end

function frule(
    (_, _, ẋ),
    ::typeof(sum),
    ::typeof(abs2),
    x::AbstractArray{T};
    dims=:,
) where {T<:Union{Real,Complex}}
    y = sum(abs2, x; dims=dims)
    ∂y = if dims isa Colon
        2 * real(dot(x, ẋ))
    elseif VERSION ≥ v"1.2" # multi-iterator mapreduce introduced in v1.2
        2 * mapreduce(_realconjtimes, +, x, ẋ; dims=dims)
    else
        2 * sum(_realconjtimes.(x, ẋ); dims=dims)
    end
    return y, ∂y
end

function rrule(
    ::typeof(sum),
    ::typeof(abs2),
    x::AbstractArray{T};
    dims=:,
) where {T<:Union{Real,Complex}}
    y = sum(abs2, x; dims=dims)
    function sum_abs2_pullback(ȳ)
        return (NO_FIELDS, DoesNotExist(), @thunk(2 .* real.(ȳ) .* x))
    end
    return y, sum_abs2_pullback
end
