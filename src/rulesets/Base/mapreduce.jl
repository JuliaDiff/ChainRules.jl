#####
##### `sum`
#####

function frule((_, ẋ), ::typeof(sum), x)
    return sum(x), sum(ẋ)
end

function rrule(::typeof(sum), x::AbstractArray{<:Real}; dims=:)
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

function rrule(::typeof(sum), ::typeof(abs2), x::AbstractArray{<:Real}; dims=:)
    y = sum(abs2, x; dims=dims)
    function sum_abs2_pullback(ȳ)
        return (NO_FIELDS, DoesNotExist(), @thunk(2ȳ .* x))
    end
    return y, sum_abs2_pullback
end
