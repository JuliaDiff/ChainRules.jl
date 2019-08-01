using LinearAlgebra: AbstractTriangular

# Matrix wrapper types that we know are square and are thus potentially invertible. For
# these we can use simpler definitions for `/` and `\`.
const SquareMatrix{T} = Union{Diagonal{T},AbstractTriangular{T}}

#####
##### `dot`
#####

function frule(::typeof(dot), x, y)
    return dot(x, y), Rule((Δx, Δy) -> sum(Δx * cast(y)) + sum(cast(x) * Δy))
end

function rrule(::typeof(dot), x, y)
    return dot(x, y), (Rule(ΔΩ -> ΔΩ * cast(y)), Rule(ΔΩ -> cast(x) * ΔΩ))
end

#####
##### `inv`
#####

function frule(::typeof(inv), x::AbstractArray)
    Ω = inv(x)
    m = @thunk(-Ω)
    return Ω, Rule(Δx -> m * Δx * Ω)
end

function rrule(::typeof(inv), x::AbstractArray)
    Ω = inv(x)
    m = @thunk(-Ω')
    return Ω, Rule(ΔΩ -> m * ΔΩ * Ω')
end

#####
##### `det`
#####

function frule(::typeof(det), x)
    Ω, m = det(x), @thunk(inv(x))
    return Ω, Rule(Δx -> Ω * tr(extern(m * Δx)))
end

function rrule(::typeof(det), x)
    Ω, m = det(x), @thunk(inv(x)')
    return Ω, Rule(ΔΩ -> Ω * ΔΩ * m)
end

#####
##### `logdet`
#####

function frule(::typeof(logdet), x)
    Ω, m = logdet(x), @thunk(inv(x))
    return Ω, Rule(Δx -> tr(extern(m * Δx)))
end

function rrule(::typeof(logdet), x)
    Ω, m = logdet(x), @thunk(inv(x)')
    return Ω, Rule(ΔΩ -> ΔΩ * m)
end

#####
##### `trace`
#####

frule(::typeof(tr), x) = (tr(x), Rule(Δx -> tr(extern(Δx))))

rrule(::typeof(tr), x) = (tr(x), Rule(ΔΩ -> Diagonal(fill(ΔΩ, size(x, 1)))))

#####
##### `*`
#####

function rrule(::typeof(*), A::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real})
    return A * B, (Rule(Ȳ -> Ȳ * B'), Rule(Ȳ -> A' * Ȳ))
end

#####
##### `/`
#####

function rrule(::typeof(/), A::AbstractMatrix{<:Real}, B::T) where T<:SquareMatrix{<:Real}
    Y = A / B
    S = T.name.wrapper
    ∂A = Rule(Ȳ -> Ȳ / B')
    ∂B = Rule(Ȳ -> S(-Y' * (Ȳ / B')))
    return Y, (∂A, ∂B)
end

function rrule(::typeof(/), A::AbstractVecOrMat{<:Real}, B::AbstractVecOrMat{<:Real})
    Aᵀ, dA = rrule(adjoint, A)
    Bᵀ, dB = rrule(adjoint, B)
    Cᵀ, (dBᵀ, dAᵀ) = rrule(\, Bᵀ, Aᵀ)
    C, dC = rrule(adjoint, Cᵀ)
    ∂A = Rule(dA∘dAᵀ∘dC)
    ∂B = Rule(dA∘dBᵀ∘dC)
    return C, (∂A, ∂B)
end

#####
##### `\`
#####

function rrule(::typeof(\), A::T, B::AbstractVecOrMat{<:Real}) where T<:SquareMatrix{<:Real}
    Y = A \ B
    S = T.name.wrapper
    ∂A = Rule(Ȳ -> S(-(A' \ Ȳ) * Y'))
    ∂B = Rule(Ȳ -> A' \ Ȳ)
    return Y, (∂A, ∂B)
end

function rrule(::typeof(\), A::AbstractVecOrMat{<:Real}, B::AbstractVecOrMat{<:Real})
    Y = A \ B
    ∂A = Rule() do Ȳ
        B̄ = A' \ Ȳ
        Ā = -B̄ * Y'
        _add!(Ā, (B - A * Y) * B̄' / A')
        _add!(Ā, A' \ Y * (Ȳ' - B̄'A))
        Ā
    end
    ∂B = Rule(Ȳ -> A' \ Ȳ)
    return Y, (∂A, ∂B)
end

#####
##### `norm`
#####

function rrule(::typeof(norm), A::AbstractArray{<:Real}, p::Real=2)
    y = norm(A, p)
    u = y^(1-p)
    ∂A = Rule(ȳ -> ȳ .* u .* abs.(A).^p ./ A)
    ∂p = Rule(ȳ -> ȳ * (u * sum(a->abs(a)^p * log(abs(a)), A) - y * log(y)) / p)
    return y, (∂A, ∂p)
end

function rrule(::typeof(norm), x::Real, p::Real=2)
    return norm(x, p), (Rule(ȳ -> ȳ * sign(x)), Rule(_ -> zero(x)))
end
