#####
##### `mean`
#####

_denom(x, dims) = size(x, dims)
_denom(x, dims::Colon) = length(x)
_denom(x, dims::Union{Tuple, AbstractArray}) = mapreduce(i->size(x, i), Base.mul_prod, unique(dims), init=1)

function rrule(::typeof(mean), x::AbstractArray{<:Union{Real,Complex,AbstractArray}}; dims=:)
    y_sum, sum_pullback = rrule(sum, x; dims)
    n = _denom(x, dims)
    function mean_pullback(ȳ)
        _, ∂x = sum_pullback(unthunk(ȳ) / n)
        return (NoTangent(), ∂x)
    end
    return y_sum / n, mean_pullback
end

function rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(mean),
    f::F,
    x::AbstractArray{T};
    dims=:,
) where {F, T<:Union{Real,Complex,AbstractArray}}
    y_sum, sum_pullback = rrule(config, sum, f, x; dims)
    n = _denom(x, dims)
    function mean_pullback_f(ȳ)
        return sum_pullback(unthunk(ȳ) / n)
    end
    return y_sum / n, mean_pullback_f
end

#####
##### variance
#####

function rrule(
    ::typeof(Statistics.var),
    x::AbstractArray{<:Number};
    corrected::Bool=true,
    dims=:,
    mean=mean(x, dims=dims)
)
    y = Statistics.var(x; corrected=corrected, mean=mean, dims=dims)
    function variance_pullback(dy)
        pre = 2 // (_denom(x, dims) - corrected)
        dx = pre .* unthunk(dy) .* (x .- mean)
        return (NoTangent(), ProjectTo(x)(dx))
    end
    y, variance_pullback
end

function rrule(
    ::typeof(Statistics.std),
    x::AbstractArray{<:Number}; 
    corrected::Bool=true,
    dims=:,
    mean=mean(x, dims=dims)
)
    y = Statistics.std(x; corrected=corrected, mean=mean, dims=dims)
    function std_pullback(dy)
        pre = 1 // (_denom(x, dims) - corrected)
        dx = pre .* unthunk(dy) .* (x .- mean) ./ y
        return (NoTangent(), ProjectTo(x)(dx))
    end
    y, std_pullback
end
