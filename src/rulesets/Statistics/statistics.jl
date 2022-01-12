#####
##### `mean`
#####

_denom(x, dims::Colon) = length(x)
_denom(x, dims::Integer) = size(x, dims)
_denom(x, dims) = mapreduce(i->size(x, i), Base.mul_prod, unique(dims), init=1)

# TODO: We have `mean(f, x; dims)` as of 1.3.0-DEV.36
# https://github.com/JuliaDiff/ChainRules.jl/issues/85
function rrule(::typeof(mean), x::AbstractArray{<:Real}; dims=:)
    y_sum, sum_pullback = rrule(sum, x; dims=dims)
    n = _denom(x, dims)
    function mean_pullback(ȳ)
        _, ∂sum_x = sum_pullback(ȳ)
        ∂x = unthunk(∂sum_x) / n
        return (NoTangent(), ∂x)
    end
    return y_sum / n, mean_pullback
end

#####
##### variance
#####

function rrule(::typeof(Statistics.var), x::AbstractArray{<:Number}; 
        corrected::Bool=true, dims=:, mean=mean(x, dims=dims))
    y = Statistics.var(x; corrected=corrected, mean=mean, dims=dims)
    function variance_pullback(dy)
        pre = 2 // (_denom(x, dims) - corrected)
        dx = pre .* unthunk(dy) .* (x .- mean)
        return (NoTangent(), ProjectTo(x)(dx))
    end
    y, variance_pullback
end

function rrule(::typeof(Statistics.std), x::AbstractArray{<:Number}; 
        corrected::Bool=true, dims=:, mean=mean(x, dims=dims))
    y = Statistics.std(x; corrected=corrected, mean=mean, dims=dims)
    function std_pullback(dy)
        pre = 1 // (_denom(x, dims) - corrected)
        dx = pre .* unthunk(dy) .* (x .- mean) ./ y
        return (NoTangent(), ProjectTo(x)(dx))
    end
    y, std_pullback
end
