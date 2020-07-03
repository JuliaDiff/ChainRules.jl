using LinearAlgebra: AbstractTriangular

# Matrix wrapper types that we know are square and are thus potentially invertible. For
# these we can use simpler definitions for `/` and `\`.
const SquareMatrix{T} = Union{Diagonal{T},AbstractTriangular{T}}

#####
##### `dot`
#####

function frule((_, Δx, Δy), ::typeof(dot), x, y)
    return dot(x, y), dot(Δx, y) + dot(x, Δy)
end

function rrule(::typeof(dot), x, y)
    function dot_pullback(ΔΩ)
        return (NO_FIELDS, @thunk(y .* ΔΩ'), @thunk(x .* ΔΩ))
    end
    return dot(x, y), dot_pullback
end

#####
##### `cross`
#####

function frule((_, Δa, Δb), ::typeof(cross), a::AbstractVector, b::AbstractVector)
    return cross(a, b), cross(Δa, b) .+ cross(a, Δb)
end

# TODO: support complex vectors
function rrule(::typeof(cross), a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    Ω = cross(a, b)
    function cross_pullback(ΔΩ)
        return (NO_FIELDS, @thunk(cross(b, ΔΩ)), @thunk(cross(ΔΩ, a)))
    end
    return Ω, cross_pullback
end

#####
##### `inv`
#####

function frule((_, Δx), ::typeof(inv), x::AbstractArray)
    Ω = inv(x)
    return Ω, -Ω * Δx * Ω
end

function rrule(::typeof(inv), x::AbstractArray)
    Ω = inv(x)
    function inv_pullback(ΔΩ)
        return NO_FIELDS, -Ω' * ΔΩ * Ω'
    end
    return Ω, inv_pullback
end

#####
##### `det`
#####

function frule((_, ẋ), ::typeof(det), x::Union{Number, AbstractMatrix})
    Ω = det(x)
    # TODO Performance optimization: probably there is an efficent
    # way to compute this trace without during the full compution within
    return Ω, Ω * tr(inv(x) * ẋ)
end

function rrule(::typeof(det), x::Union{Number, AbstractMatrix})
    Ω = det(x)
    function det_pullback(ΔΩ)
        return NO_FIELDS, Ω * ΔΩ * inv(x)'
    end
    return Ω, det_pullback
end

#####
##### `logdet`
#####

function frule((_, Δx), ::typeof(logdet), x::Union{Number, AbstractMatrix})
    Ω = logdet(x)
    return Ω, tr(inv(x) * Δx)
end

function rrule(::typeof(logdet), x::Union{Number, AbstractMatrix})
    Ω = logdet(x)
    function logdet_pullback(ΔΩ)
        return (NO_FIELDS, ΔΩ * inv(x)')
    end
    return Ω, logdet_pullback
end

#####
##### `trace`
#####

function frule((_, Δx), ::typeof(tr), x)
    return tr(x), tr(Δx)
end

function rrule(::typeof(tr), x)
    # This should really be a FillArray
    # see https://github.com/JuliaDiff/ChainRules.jl/issues/46
    function tr_pullback(ΔΩ)
        return (NO_FIELDS, @thunk Diagonal(fill(ΔΩ, size(x, 1))))
    end
    return tr(x), tr_pullback
end


#####
##### `*`
#####

function rrule(::typeof(*), A::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real})
    function times_pullback(Ȳ)
        return (NO_FIELDS, @thunk(Ȳ * B'), @thunk(A' * Ȳ))
    end
    return A * B, times_pullback
end

#####
##### `pinv`
#####

@scalar_rule pinv(x::Number) -(Ω ^ 2)

function frule(
    (_, Δx),
    ::typeof(pinv),
    x::AbstractVector{T};
    kwargs...,
) where {T<:Union{Real,Complex}}
    y = pinv(x; kwargs...)
    # make sure ∂y is the same type as y
    fdual = y isa Transpose ? transpose : adjoint
    ∂y = fdual(sum(abs2, y') .* Δx .- 2real(y * Δx) .* y')
    return y, ∂y
end

function frule((_, ΔA), ::typeof(pinv), A::AbstractMatrix{T}; kwargs...) where {T}
    Y = pinv(A; kwargs...)
    ∂Y = -Y * ΔA * Y
    m, n = size(A)
    if m < n # right inverse (A * Y = I)
        ∂Y = _add!(∂Y, (I - Y * A) * ΔA' * Y' * Y)
    elseif m > n # left inverse (Y * A = I)
        ∂Y = _add!(∂Y, Y * Y' * ΔA' * (I - A * Y))
    end
    return Y, ∂Y
end

function rrule(
    ::typeof(pinv),
    x::AbstractVector{T};
    kwargs...,
) where {T<:Union{Real,Complex}}
    y = pinv(x; kwargs...)
    function pinv_pullback(Δy)
        ∂x = sum(abs2, y') .* vec(Δy') .- 2real(y * Δy') .* y'
        return (NO_FIELDS, ∂x)
    end
    return y, pinv_pullback
end

function rrule(::typeof(pinv), A::AbstractMatrix{T}; kwargs...) where {T}
    Y = pinv(A; kwargs...)
    function pinv_pullback(ΔY)
        ∂A = -Y' * ΔY * Y'
        m, n = size(A)
        if m < n # right inverse (A * Y = I)
            ∂A = _add!(∂A, Y' * Y * ΔY' * (I - Y * A))
        elseif m > n # left inverse (Y * A = I)
            ∂A = _add!(∂A, (I - A * Y) * ΔY' * Y * Y')
        end
        return (NO_FIELDS, ∂A)
    end
    return Y, pinv_pullback
end

#####
##### `/`
#####

function rrule(::typeof(/), A::AbstractMatrix{<:Real}, B::T) where T<:SquareMatrix{<:Real}
    Y = A / B
    function slash_pullback(Ȳ)
        S = T.name.wrapper
        ∂A = @thunk Ȳ / B'
        ∂B = @thunk S(-Y' * (Ȳ / B'))
        return (NO_FIELDS, ∂A, ∂B)
    end
    return Y, slash_pullback
end

function rrule(::typeof(/), A::AbstractVecOrMat{<:Real}, B::AbstractVecOrMat{<:Real})
    Aᵀ, dA_pb = rrule(adjoint, A)
    Bᵀ, dB_pb = rrule(adjoint, B)
    Cᵀ, dS_pb = rrule(\, Bᵀ, Aᵀ)
    C, dC_pb = rrule(adjoint, Cᵀ)
    function slash_pullback(Ȳ)
        # Optimization note: dAᵀ, dBᵀ, dC are calculated no matter which partial you want
        _, dC = dC_pb(Ȳ)
        _, dBᵀ, dAᵀ = dS_pb(unthunk(dC))

        ∂A = last(dA_pb(unthunk(dAᵀ)))
        ∂B = last(dA_pb(unthunk(dBᵀ)))

        (NO_FIELDS, ∂A, ∂B)
    end
    return C, slash_pullback
end

#####
##### `\`
#####

function rrule(::typeof(\), A::T, B::AbstractVecOrMat{<:Real}) where T<:SquareMatrix{<:Real}
    Y = A \ B
    function backslash_pullback(Ȳ)
        S = T.name.wrapper
        ∂A = @thunk S(-(A' \ Ȳ) * Y')
        ∂B = @thunk A' \ Ȳ
        return NO_FIELDS, ∂A, ∂B
    end
    return Y, backslash_pullback
end

function rrule(::typeof(\), A::AbstractVecOrMat{<:Real}, B::AbstractVecOrMat{<:Real})
    Y = A \ B
    function backslash_pullback(Ȳ)
        ∂A = @thunk begin
            B̄ = A' \ Ȳ
            Ā = -B̄ * Y'
            _add!(Ā, (B - A * Y) * B̄' / A')
            _add!(Ā, A' \ Y * (Ȳ' - B̄'A))
            Ā
        end
        ∂B = @thunk A' \ Ȳ
        return NO_FIELDS, ∂A, ∂B
    end
    return Y, backslash_pullback

end

#####
##### `norm`
#####

function rrule(::typeof(norm), A::AbstractArray{<:Real}, p::Real=2)
    y = norm(A, p)
    function norm_pullback(ȳ)
        u = y^(1-p)
        ∂A = @thunk ȳ .* u .* abs.(A).^p ./ A
        ∂p = @thunk ȳ * (u * sum(a->abs(a)^p * log(abs(a)), A) - y * log(y)) / p
        (NO_FIELDS, ∂A, ∂p)
    end
    return y, norm_pullback
end

function rrule(::typeof(norm), x::Real, p::Real=2)
    function norm_pullback(ȳ)
        ∂x = @thunk ȳ * sign(x)
        ∂p = @thunk zero(x)  # TODO: should this be Zero()?
        (NO_FIELDS, ∂x, ∂p)
    end
    return norm(x, p), norm_pullback
end
