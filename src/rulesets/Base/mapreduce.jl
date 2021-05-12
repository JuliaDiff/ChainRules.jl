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
    vald = dims isa Colon ? nothing : dims isa Integer ? Val(Int(dims)) : Val(Tuple(dims))
    function prod_pullback(dy)
        x_thunk = InplaceableThunk(
            # Out-of-place versions
            @thunk if dims === (:)
                ∇prod(x, dy, y)
            elseif any(iszero, x)  # Then, and only then, will ./x lead to NaN
                ∇prod_dims(vald, x, dy, y)
            else
                conj.(y ./ x) .* dy
            end
            ,
            # In-place versions -- same branching
            dx -> if dims === (:)
                ∇prod!(dx, x, dy, y)
            elseif any(iszero, x) 
                ∇prod_dims!(dx, vald, x, dy, y)
            else
                dx .+= conj.(y ./ x) .* dy
            end
            )
        return (NO_FIELDS, x_thunk)
    end
    return y, prod_pullback
end

function ∇prod_dims(vald::Val{dims}, x, dy, y=prod(x; dims=dims)) where {dims}
    T = promote_type(eltype(x), eltype(dy))
    dx = fill!(similar(x, T, axes(x)), zero(T))
    ∇prod_dims!(dx, vald, x, dy, y)
    return dx
end

function ∇prod_dims!(dx, ::Val{dims}, x, dy, y) where {dims}
    iters = ntuple(d -> d in dims ? tuple(:) : axes(x,d), ndims(x))  # Without Val(dims) this is a serious type instability
    @inbounds for ind in Iterators.product(iters...)
        jay = map(i -> i isa Colon ? 1 : i, ind)
        @views ∇prod!(dx[ind...], x[ind...], dy[jay...], y[jay...])
    end
    return dx
end

# To opt out of the mapslices path, and accept NaN instead, you could define for instance:
# ∇prod_dims!(dx, dims, x::CuArray, dy, y) = dx .+= y ./ conj.(x) .* conj.(dy)
#            ∇prod!(dx, x::CuArray, dy, y) = dx .+= y ./ conj.(x) .* conj.(dy)

function ∇prod(x, dy::Number=1, y::Number=prod(x))
    T = promote_type(eltype(x), eltype(dy))
    dx = fill!(similar(x, T, axes(x)), zero(T)) # axes(x) makes MArray on StaticArrays, Array for structured matrices
    ∇prod!(dx, x, dy, y)
    return dx
end

function ∇prod!(dx, x, dy::Number=1, y::Number=prod(x))
    numzero = iszero(y) ? count(iszero, x) : 0
    if numzero == 0  # This can happen while y==0, if there are several small xs
        dx .+= conj.(y ./ x) .* dy
    elseif numzero == 1
        ∇prod_one_zero!(dx, x, dy)
    else
        # numzero > 1, then all first derivatives are zero
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

#=

julia> @btime gradient(x -> sum(prod(x, dims=1)), x)[1]  setup=(x=rand(10,100); x[1:21:end].=0)  # Zygote
  1.292 μs (3 allocations: 8.83 KiB)
10×100 Matrix{Float64}:
 NaN    1.64248e-6    0.0  2.85863e-5     0.0  …    0.0  0.00412328    0.0  0.000262674

julia> @btime gradient(x -> sum(prod(x, dims=1)), x)[1]  setup=(x=rand(10,100); x[1:21:end].=0)  # this PR
  56.000 μs (1706 allocations: 51.06 KiB)
10×100 Matrix{Float64}:
 1.499e-5  2.54822e-5  0.0         0.000129398  …  0.000634891  0.0         0.00343482

julia> @btime gradient(x -> sum(prod(x, dims=1)), x)[1]  setup=(x=rand(10,100); x[1:21:end].=0)  # dims=1 hard coded
  1.842 μs (5 allocations: 8.86 KiB)
10×100 Matrix{Float64}:
 0.00285835  0.000569394  0.0         0.000422376  …  0.000110362  0.0         0.000481694

julia> @btime gradient(x -> sum(prod(x, dims=1)), x)[1]  setup=(x=rand(10,100); x[1:21:end].=0)  # with Val(Tuple(dims))
  3.359 μs (40 allocations: 10.45 KiB)
10×100 Matrix{Float64}:
 3.15652e-6  6.80856e-5   0.0          7.42845e-5   …  4.69112e-6   0.0         0.000449198

julia> @btime gradient(x -> sum(prod(x, dims=1)), x)[1]  setup=(x=rand(10,100); x[1:21:end].=0)  # with Val(Int(dims))
  1.850 μs (5 allocations: 8.86 KiB)
10×100 Matrix{Float64}:
 1.12526e-6  0.000483517  0.0         5.28479e-6  …  0.000555819  0.0         0.00126222

=#
