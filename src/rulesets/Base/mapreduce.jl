#####
##### `sum`
#####

function frule((_, ẋ), ::typeof(sum), x; dims=:)
    return sum(x; dims=dims), sum(ẋ; dims=dims)
end

# Internal helper for filling while maintaining array type.
# TODO: Ideallty we'd only need typeof(x) here, but Base doesn't have the
# interfaces for that.
function _typed_fill(x, ȳ, axes)
    fill!(similar(x, typeof(ȳ), axes), ȳ)
end

function rrule(::typeof(_typed_fill), x, ȳ, axes)
    function _typed_fill_pullback(Ȳ)
        return (NoTangent(), NoTangent(), sum(Ȳ), NoTangent())
    end
    return _typed_fill(x, ȳ, axes), _typed_fill_pullback
end

function sum_rrule(x, dims::Colon)
    y = sum(x; dims=dims)
    let xdims=size(x)
        function sum_pullback(ȳ)
            x̄ = InplaceableThunk(
                x -> x .+= ȳ,
                @thunk(_typed_fill(x, ȳ, xdims...)),
            )
            return (NoTangent(), x̄)
        end
        y, sum_pullback
    end
end

function sum_rrule(x, dims)
    y = sum(x; dims=dims)
    function sum_pullback(ȳ)
        # broadcasting the two works out the size no-matter `dims`
        x̄ = InplaceableThunk(
            x -> x .+= ȳ,
            @thunk(broadcast(last∘tuple, x, ȳ)),
        )
        return (NoTangent(), x̄)
    end
    return y, sum_pullback
end

function rrule(::typeof(sum), x::AbstractArray{T}; dims=:) where {T<:Number}
    return sum_rrule(x, dims)
end

# Can't map over Adjoint/Transpose Vector
function rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(sum),
    f,
    xs::Union{Adjoint{<:Number,<:AbstractVector},Transpose{<:Number,<:AbstractVector}};
    kwargs...
)
    op = xs isa Adjoint ? adjoint : transpose
    # since summing a vector we don't need to worry about dims which simplifies adjointing
    vector = parent(xs)
    y, vector_sum_pb = rrule(config, sum, f, vector; kwargs...)
    function covector_sum_pb(ȳ)
        s̄um, f̄, v̄ = vector_sum_pb(ȳ)
        return s̄um, f̄, op(v̄)
    end

    return y, covector_sum_pb
end

function rrule(
    config::RuleConfig{>:HasReverseMode}, ::typeof(sum), f, xs::AbstractArray; dims=:
)
    fx_and_pullbacks = map(x->rrule_via_ad(config, f, x), xs)
    y = sum(first, fx_and_pullbacks; dims=dims)

    pullbacks = last.(fx_and_pullbacks)

    project = ProjectTo(xs)

    function sum_pullback(ȳ)
        call(f, x) = f(x)  # we need to broadcast this to handle dims kwarg
        f̄_and_x̄s = call.(pullbacks, ȳ)
        # no point thunking as most of work is in f̄_and_x̄s which we need to compute for both
        f̄ = if fieldcount(typeof(f)) === 0 # Then don't need to worry about derivative wrt f
            NoTangent()
        else
            sum(first, f̄_and_x̄s)
        end
        x̄s = map(unthunk ∘ last, f̄_and_x̄s) # project does not support receiving InplaceableThunks
        return NoTangent(), f̄, project(x̄s)
    end
    return y, sum_pullback
end

function frule(
    (_, _, Δx),
    ::typeof(sum),
    ::typeof(abs2),
    x::AbstractArray{T};
    dims=:,
) where {T<:Union{Real,Complex}}
    ẋ = unthunk(Δx)
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
            dx -> dx .+= 2 .* real.(ȳ) .* x,
            @thunk(2 .* real.(ȳ) .* x),
        )
        return (NoTangent(), NoTangent(), x_thunk)
    end
    return y, sum_abs2_pullback
end

# Fix dispatch for this pidgeon-hole optimization,
# Rules with RuleConfig dispatch with priority over without (regardless of other args).
# and if we don't specify what do do for one that HasReverseMode then it is ambigious
for Config in (RuleConfig, RuleConfig{>:HasReverseMode})
    @eval function rrule(
        ::$Config, ::typeof(sum), ::typeof(abs2), x::AbstractArray{T}; dims=:,
    ) where {T<:Union{Real,Complex}}
        return rrule(sum, abs2, x; dims=dims)
    end
end

#####
##### `prod`
#####

function rrule(::typeof(prod), x::AbstractArray{T}; dims=:) where {T<:CommutativeMulNumber}
    y = prod(x; dims=dims)
    project_x = ProjectTo(x)
    # vald = dims isa Colon ? nothing : dims isa Integer ? Val(Int(dims)) : Val(Tuple(dims))
    function prod_pullback(ȳ)
        dy = unthunk(ȳ)
        x_thunk = InplaceableThunk(
            # In-place versions -- same branching
            dx -> if dims === (:)
                ∇prod!(dx, x, dy, y)
            elseif any(iszero, x)
                vald = dims isa Colon ? nothing : dims isa Integer ? Val(Int(dims)) : Val(Tuple(dims))
                ∇prod_dims!(dx, vald, x, dy, y)
            else
                dx .+= conj.(y ./ x) .* dy
            end,
            # Out-of-place versions
            @thunk project_x(if dims === (:)
                ∇prod(x, dy, y)
            elseif any(iszero, x)  # Then, and only then, will ./x lead to NaN
                vald = dims isa Colon ? nothing : dims isa Integer ? Val(Int(dims)) : Val(Tuple(dims))
                ∇prod_dims(vald, x, dy, y)  # val(Int(dims)) is about 2x faster than Val(Tuple(dims))
            else
                conj.(y ./ x) .* dy
            end)
        )
        return (NoTangent(), x_thunk)
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
