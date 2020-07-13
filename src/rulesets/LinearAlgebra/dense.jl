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

function frule((_, Δx), ::typeof(det), x::AbstractMatrix)
    Ω = det(x)
    # TODO Performance optimization: probably there is an efficent
    # way to compute this trace without during the full compution within
    return Ω, Ω * tr(x \ Δx)
end
frule((_, Δx), ::typeof(det), x::Number) = (det(x), Δx)

function rrule(::typeof(det), x::Union{Number, AbstractMatrix})
    Ω = det(x)
    function det_pullback(ΔΩ)
        ∂x = x isa Number ? ΔΩ : Ω * ΔΩ * inv(x)'
        return (NO_FIELDS, ∂x)
    end
    return Ω, det_pullback
end

#####
##### `logdet`
#####

function frule((_, Δx), ::typeof(logdet), x::Union{Number, AbstractMatrix})
    Ω = logdet(x)
    return Ω, tr(x \ Δx)
end

function rrule(::typeof(logdet), x::Union{Number, AbstractMatrix})
    Ω = logdet(x)
    function logdet_pullback(ΔΩ)
        ∂x = x isa Number ? ΔΩ / x' : ΔΩ * inv(x)'
        return (NO_FIELDS, ∂x)
    end
    return Ω, logdet_pullback
end

#####
##### `logabsdet`
#####

function frule((_, Δx), ::typeof(logabsdet), x::AbstractMatrix)
    Ω = logabsdet(x)
    (y, signy) = Ω
    ∂detx = tr(x \ Δx)
    (∂y, b) = reim(∂detx)
    signy_r, signy_i = reim(signy)
    ∂signy = complex(-signy_i * b, signy_r * b) # signy * b * im
    ∂Ω = Composite{typeof(Ω)}(∂y, ∂signy)
    return Ω, ∂Ω
end

function rrule(::typeof(logabsdet), x::AbstractMatrix)
    Ω = logabsdet(x)
    function logabsdet_pullback(ΔΩ)
        (Δy, Δsigny) = ΔΩ
        (_, signy) = Ω
        ∂logdetx = real(Δy) + im * _imagconjtimes(signy, Δsigny)
        ∂x = ∂logdetx * inv(x)'
        return (NO_FIELDS, ∂x)
    end
    return Ω, logabsdet_pullback
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
        return (NO_FIELDS, Diagonal(fill(ΔΩ, size(x, 1))))
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

@scalar_rule pinv(x) -(Ω ^ 2)

function frule(
    (_, Δx),
    ::typeof(pinv),
    x::AbstractVector{T},
    tol::Real = 0,
) where {T<:Union{Real,Complex}}
    y = pinv(x, tol)
    ∂y′ = sum(abs2, parent(y)) .* Δx .- 2real(y * Δx) .* parent(y)
    ∂y = y isa Transpose ? transpose(∂y′) : adjoint(∂y′)
    return y, ∂y
end

function frule(
    (_, Δx),
    ::typeof(pinv),
    x::LinearAlgebra.AdjOrTransAbsVec{T},
    tol::Real = 0,
) where {T<:Union{Real,Complex}}
    y = pinv(x, tol)
    ∂y = sum(abs2, y) .* vec(Δx') .- 2real(Δx * y) .* y
    return y, ∂y
end

# Formula for derivative adapted from Eq 4.12 of
# Golub, Gene H., and Victor Pereyra. "The Differentiation of Pseudo-Inverses and Nonlinear
# Least Squares Problems Whose Variables Separate."
# SIAM Journal on Numerical Analysis 10(2). (1973). 413-432. doi: 10.1137/0710036
function frule((_, ΔA), ::typeof(pinv), A::AbstractMatrix{T}; kwargs...) where {T}
    Y = pinv(A; kwargs...)
    m, n = size(A)
    # contract over the largest dimension
    if m ≤ n
        ∂Y = -Y * (ΔA * Y)
        _add!(∂Y, (ΔA' - Y * (A * ΔA')) * (Y' * Y)) # (I - Y A) ΔA' Y' Y
        _add!(∂Y, Y * (Y' * ΔA') * (I - A * Y)) # Y Y' ΔA' (I - A Y)
    else
        ∂Y = -(Y * ΔA) * Y
        _add!(∂Y, (I - Y * A) * (ΔA' * Y') * Y) # (I - Y A) ΔA' Y' Y
        _add!(∂Y, (Y * Y') * (ΔA' - (ΔA' * A) * Y)) # Y Y' ΔA' (I - A Y)
    end
    return Y, ∂Y
end

function rrule(
    ::typeof(pinv),
    x::AbstractVector{T},
    tol::Real = 0,
) where {T<:Union{Real,Complex}}
    y = pinv(x, tol)
    function pinv_pullback(Δy)
        ∂x = sum(abs2, parent(y)) .* vec(Δy') .- 2real(y * Δy') .* parent(y)
        return (NO_FIELDS, ∂x, Zero())
    end
    return y, pinv_pullback
end

function rrule(
    ::typeof(pinv),
    x::LinearAlgebra.AdjOrTransAbsVec{T},
    tol::Real = 0,
) where {T<:Union{Real,Complex}}
    y = pinv(x, tol)
    function pinv_pullback(Δy)
        ∂x′ = sum(abs2, y) .* Δy .- 2real(y' * Δy) .* y
        ∂x = x isa Transpose ? transpose(conj(∂x′)) : adjoint(∂x′)
        return (NO_FIELDS, ∂x, Zero())
    end
    return y, pinv_pullback
end

function rrule(::typeof(pinv), A::AbstractMatrix{T}; kwargs...) where {T}
    Y = pinv(A; kwargs...)
    function pinv_pullback(ΔY)
        m, n = size(A)
        # contract over the largest dimension
        if m ≤ n
            ∂A = (Y' * -ΔY) * Y'
            _add!(∂A, (Y' * Y) * (ΔY' - (ΔY' * Y) * A)) # Y' Y ΔY' (I - Y A)
            _add!(∂A, (I - A * Y) * (ΔY' * Y) * Y') # (I - A Y) ΔY' Y Y'
        elseif m > n
            ∂A = Y' * (-ΔY * Y')
            _add!(∂A, Y' * (Y * ΔY') * (I - Y * A)) # Y' Y ΔY' (I - Y A)
            _add!(∂A, (ΔY' - A * (Y * ΔY')) * (Y * Y')) # (I - A Y) ΔY' Y Y'
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
