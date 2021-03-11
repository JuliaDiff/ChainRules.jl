#####
##### `sum`
#####

function frule((_, ẋ), ::typeof(sum), x; dims=:)
    return sum(x; dims=dims), sum(ẋ; dims=dims)
end

function rrule(::typeof(sum), x::AbstractArray{T}; dims=:) where {T<:Number}
    y = sum(x; dims=dims)
    function sum_pullback(ȳ)
        # broadcasting the two works out the size no-matter `dims`
        x̄ = InplaceableThunk(
            @thunk(broadcast(last∘tuple, x, ȳ)),
            x -> x .+= ȳ
        )
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
        mapreduce(+, x, ẋ; dims=dims) do xi, dxi
            2 * _realconjtimes(xi, dxi)
        end
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
        x_thunk = InplaceableThunk(
            @thunk(2 .* real.(ȳ) .* x),
            dx -> dx .+= 2 .* real.(ȳ) .* x
        )
        return (NO_FIELDS, DoesNotExist(), x_thunk)
    end
    return y, sum_abs2_pullback
end

#####
##### `prod`
#####

function rrule(::typeof(prod), x::AbstractArray{T}; dims=:) where {T<:CommutativeMulNumber}
    y = prod(x; dims=dims)
    function prod_pullback(dy)
        x_thunk = if dims == (:)
            InplaceableThunk(
                @thunk(∇prod(x, dy, y)),  # This is usually y ./ x .* dy
                dx -> ∇prod!(dx, x, dy, y)
            )
        elseif any(iszero, x)  # Only cases where ./x would give NaN
            InplaceableThunk(
                @thunk(∇prod_dims(dims, x, dy, y)),
                dx -> ∇prod_dims!(dx, dims, x, dy, y)
            )
        else
            InplaceableThunk(
                @thunk(y ./ conj.(x) .* conj.(dy)),
                dx -> dx .+= y ./ conj.(x) .* conj.(dy)
            )
        end
        return (NO_FIELDS, x_thunk)
    end
    return y, prod_pullback
end

function ∇prod_dims(dims, x, dy, y=prod(x; dims=dims))
    T = promote_type(eltype(x), eltype(dy))
    dx = fill!(similar(x, T), zero(T))
    ∇prod_dims!(dx, dims, x, dy, y)
    return dx
end

function ∇prod_dims!(dx, dims, x, dy, y)
    iters = ntuple(d -> d in dims ? tuple(:) : axes(x,d), ndims(x))
    @inbounds for ind in Iterators.product(iters...)
        jay = map(i -> i isa Colon ? 1 : i, ind)
        @views ∇prod!(dx[ind...], x[ind...], dy[jay...], y[jay...])
    end
    return dx
end

# To opt out of this mapslices thing, and accept NaN instead, you could define:
# ∇prod_dims!(dx, dims, x::CuArray, dy, y) = dx .+= y ./ x .* dy
# and similarly ∇prod!(dx, x::CuArray, dy, y)

function ∇prod(x, dy::Number=1, y::Number=prod(x))
    T = promote_type(eltype(x), eltype(dy))
    dx = fill!(similar(x, T), zero(T))
    ∇prod!(dx, x, y, dy)
    return dx
end

function ∇prod!(dx, x, dy::Number=1, y::Number=prod(x))
    numzero = iszero(y) ? count(iszero, x) : 0
    if numzero == 0  # This can happen while y==0, if there are several small xs
        dx .+= y ./ conj.(x) .* conj.(dy)
    elseif numzero > 1
        dx
    else
        ∇prod_one_zero!(dx, x, dy)
    end
    return dx
end

function ∇prod_one_zero!(dx, x, dy::Number=1)  # Assumes exactly one x is zero
    i_zero = 0
    p_rest = one(promote_type(eltype(x), typeof(dy)))
    for i in eachindex(x)
        xi = @inbounds x[i]
        p_rest *= ifelse(iszero(xi), one(xi), conj(xi))
        i_zero = ifelse(iszero(xi), i, i_zero)
    end
    dx[i_zero] += p_rest * dy
    return
end
