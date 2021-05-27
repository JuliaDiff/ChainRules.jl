# Structured matrices
using LinearAlgebra: AbstractTriangular

# Matrix wrapper types that we know are square and are thus potentially invertible. For
# these we can use simpler definitions for `/` and `\`.
const SquareMatrix{T} = Union{Diagonal{T}, AbstractTriangular{T}}

function rrule(::typeof(/), A::AbstractMatrix{<:Real}, B::T) where T<:SquareMatrix{<:Real}
    Y = A / B
    function slash_pullback(Ȳ)
        ∂A = @thunk Ȳ / B'
        ∂B = @thunk _unionall_wrapper(T)(-Y' * (Ȳ / B'))
        return (NO_FIELDS, ∂A, ∂B)
    end
    return Y, slash_pullback
end

function rrule(::typeof(\), A::T, B::AbstractVecOrMat{<:Real}) where T<:SquareMatrix{<:Real}
    Y = A \ B
    function backslash_pullback(Ȳ)
        ∂A = @thunk _unionall_wrapper(T)(-(A' \ Ȳ) * Y')
        ∂B = @thunk A' \ Ȳ
        return NO_FIELDS, ∂A, ∂B
    end
    return Y, backslash_pullback
end

#####
##### `Diagonal`
#####

function rrule(::Type{<:Diagonal}, d::AbstractVector)
    function Diagonal_pullback(ȳ::AbstractMatrix)
        return (NO_FIELDS, diag(ȳ))
    end
    function Diagonal_pullback(ȳ::Tangent)
        # TODO: Assert about the primal type in the Tangent, It should be Diagonal
        # infact it should be exactly the type of `Diagonal(d)`
        # but right now Zygote loses primal type information so we can't use it.
        # See https://github.com/FluxML/Zygote.jl/issues/603
        return (NO_FIELDS, ȳ.diag)
    end
    return Diagonal(d), Diagonal_pullback
end

function rrule(::typeof(diag), A::AbstractMatrix)
    function diag_pullback(ȳ)
        return (NO_FIELDS, Diagonal(ȳ))
    end
    return diag(A), diag_pullback
end
if VERSION ≥ v"1.3"
    function rrule(::typeof(diag), A::AbstractMatrix, k::Integer)
        function diag_pullback(ȳ)
            return (NO_FIELDS, diagm(size(A)..., k => ȳ), NoTangent())
        end
        return diag(A, k), diag_pullback
    end

    function rrule(::typeof(diagm), m::Integer, n::Integer, kv::Pair{<:Integer,<:AbstractVector}...)
        function diagm_pullback(ȳ)
            return (NO_FIELDS, NoTangent(), NoTangent(), _diagm_back.(kv, Ref(ȳ))...)
        end
        return diagm(m, n, kv...), diagm_pullback
    end
end
function rrule(::typeof(diagm), kv::Pair{<:Integer,<:AbstractVector}...)
    function diagm_pullback(ȳ)
        return (NO_FIELDS, _diagm_back.(kv, Ref(ȳ))...)
    end
    return diagm(kv...), diagm_pullback
end

function _diagm_back(p, ȳ)
    k, v = p
    d = diag(ȳ, k)[1:length(v)] # handle if diagonal was smaller than matrix
    return Tangent{typeof(p)}(second = d)
end

function rrule(::typeof(*), D::Diagonal{<:Real}, V::AbstractVector{<:Real})
    function times_pullback(Ȳ)
        return (NO_FIELDS, @thunk(Diagonal(Ȳ .* V)), @thunk(D * Ȳ))
    end
    return D * V, times_pullback
end

#####
##### `Adjoint`
#####

function rrule(::Type{<:Adjoint}, A::AbstractMatrix{<:Number})
    Adjoint_pullback(ȳ::Tangent) = (NO_FIELDS, ȳ.parent)
    Adjoint_pullback(ȳ::AbstractVecOrMat) = (NO_FIELDS, adjoint(ȳ))
    return Adjoint(A), Adjoint_pullback
end

function rrule(::Type{<:Adjoint}, A::AbstractVector{<:Number})
    Adjoint_pullback(ȳ::Tangent) = (NO_FIELDS, vec(ȳ.parent))
    Adjoint_pullback(ȳ::AbstractMatrix) = (NO_FIELDS, vec(adjoint(ȳ)))
    return Adjoint(A), Adjoint_pullback
end

function rrule(::typeof(adjoint), A::AbstractMatrix{<:Number})
    adjoint_pullback(ȳ::Tangent) = (NO_FIELDS, ȳ.parent)
    adjoint_pullback(ȳ::AbstractVecOrMat) = (NO_FIELDS, adjoint(ȳ))
    return adjoint(A), adjoint_pullback
end

function rrule(::typeof(adjoint), A::AbstractVector{<:Number})
    adjoint_pullback(ȳ::Tangent) = (NO_FIELDS, vec(ȳ.parent))
    adjoint_pullback(ȳ::AbstractMatrix) = (NO_FIELDS, vec(adjoint(ȳ)))
    return adjoint(A), adjoint_pullback
end

#####
##### `Transpose`
#####

function rrule(::Type{<:Transpose}, A::AbstractMatrix{<:Number})
    Transpose_pullback(ȳ::Tangent) = (NO_FIELDS, ȳ.parent)
    Transpose_pullback(ȳ::AbstractVecOrMat) = (NO_FIELDS, Transpose(ȳ))
    return Transpose(A), Transpose_pullback
end

function rrule(::Type{<:Transpose}, A::AbstractVector{<:Number})
    Transpose_pullback(ȳ::Tangent) = (NO_FIELDS, vec(ȳ.parent))
    Transpose_pullback(ȳ::AbstractMatrix) = (NO_FIELDS, vec(Transpose(ȳ)))
    return Transpose(A), Transpose_pullback
end

function rrule(::typeof(transpose), A::AbstractMatrix{<:Number})
    transpose_pullback(ȳ::Tangent) = (NO_FIELDS, ȳ.parent)
    transpose_pullback(ȳ::AbstractVecOrMat) = (NO_FIELDS, transpose(ȳ))
    return transpose(A), transpose_pullback
end

function rrule(::typeof(transpose), A::AbstractVector{<:Number})
    transpose_pullback(ȳ::Tangent) = (NO_FIELDS, vec(ȳ.parent))
    transpose_pullback(ȳ::AbstractMatrix) = (NO_FIELDS, vec(transpose(ȳ)))
    return transpose(A), transpose_pullback
end

#####
##### Triangular matrices
#####

function rrule(::Type{<:UpperTriangular}, A::AbstractMatrix)
    function UpperTriangular_pullback(ȳ)
        return (NO_FIELDS, Matrix(ȳ))
    end
    return UpperTriangular(A), UpperTriangular_pullback
end

function rrule(::Type{<:LowerTriangular}, A::AbstractMatrix)
    function LowerTriangular_pullback(ȳ)
        return (NO_FIELDS, Matrix(ȳ))
    end
    return LowerTriangular(A), LowerTriangular_pullback
end

function rrule(::typeof(triu), A::AbstractMatrix, k::Integer)
    function triu_pullback(ȳ)
        return (NO_FIELDS, triu(ȳ, k), NoTangent())
    end
    return triu(A, k), triu_pullback
end
function rrule(::typeof(triu), A::AbstractMatrix)
    function triu_pullback(ȳ)
        return (NO_FIELDS, triu(ȳ))
    end
    return triu(A), triu_pullback
end

function rrule(::typeof(tril), A::AbstractMatrix, k::Integer)
    function tril_pullback(ȳ)
        return (NO_FIELDS, tril(ȳ, k), NoTangent())
    end
    return tril(A, k), tril_pullback
end
function rrule(::typeof(tril), A::AbstractMatrix)
    function tril_pullback(ȳ)
        return (NO_FIELDS, tril(ȳ))
    end
    return tril(A), tril_pullback
end

_diag_view(X) = view(X, diagind(X))
_diag_view(X::Diagonal) = parent(X)  #Diagonal wraps a Vector of just Diagonal elements

function rrule(::typeof(det), X::Union{Diagonal, AbstractTriangular})
    y = det(X)
    s = conj!(y ./ _diag_view(X))
    function det_pullback(ȳ)
        return (NO_FIELDS, Diagonal(ȳ .* s))
    end
    return y, det_pullback
end

function rrule(::typeof(logdet), X::Union{Diagonal, AbstractTriangular})
    y = logdet(X)
    s = conj!(one(eltype(X)) ./ _diag_view(X))
    function logdet_pullback(ȳ)
        return (NO_FIELDS, Diagonal(ȳ .* s))
    end
    return y, logdet_pullback
end
