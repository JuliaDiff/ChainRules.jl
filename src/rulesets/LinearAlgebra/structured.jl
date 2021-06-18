# Structured matrices
using LinearAlgebra: AbstractTriangular

# Matrix wrapper types that we know are square and are thus potentially invertible. For
# these we can use simpler definitions for `/` and `\`.
const SquareMatrix{T} = Union{Diagonal{T}, AbstractTriangular{T}}

function rrule(::typeof(/), A::AbstractMatrix{<:Real}, B::T) where T<:SquareMatrix{<:Real}
    Y = A / B
    function slash_pullback(Ȳ) # DECIDE
        ∂A = @thunk Ȳ / B'
        ∂B = @thunk _unionall_wrapper(T)(-Y' * (Ȳ / B'))
        return (NoTangent(), ∂A, ∂B)
    end
    return Y, slash_pullback
end

function rrule(::typeof(\), A::T, B::AbstractVecOrMat{<:Real}) where T<:SquareMatrix{<:Real}
    Y = A \ B
    function backslash_pullback(Ȳ) # DECIDE
        ∂A = @thunk _unionall_wrapper(T)(-(A' \ Ȳ) * Y')
        ∂B = @thunk A' \ Ȳ
        return NoTangent(), ∂A, ∂B
    end
    return Y, backslash_pullback
end

#####
##### `Diagonal`
#####
_Diagonal_pullback(ȳ::AbstractMatrix) = return (NoTangent(), diag(ȳ))
function _Diagonal_pullback(ȳ::Tangent)
    # TODO: Assert about the primal type in the Tangent, It should be Diagonal
    # infact it should be exactly the type of `Diagonal(d)`
    # but right now Zygote loses primal type information so we can't use it.
    # See https://github.com/FluxML/Zygote.jl/issues/603
    return (NoTangent(), ȳ.diag)
end
_Diagonal_pullback(ȳ::AbstractThunk) = return _Diagonal_pullback(unthunk(ȳ))

function rrule(::Type{<:Diagonal}, d::AbstractVector)
    return Diagonal(d), _Diagonal_pullback
end

function rrule(::typeof(diag), A::AbstractMatrix)
    function diag_pullback(ȳ)
        return (NoTangent(), Diagonal(ȳ))
    end
    return diag(A), diag_pullback
end
if VERSION ≥ v"1.3"
    function rrule(::typeof(diag), A::AbstractMatrix, k::Integer)
        function diag_pullback(ȳ)
            return (NoTangent(), diagm(size(A)..., k => ȳ), NoTangent())
        end
        return diag(A, k), diag_pullback
    end

    function rrule(::typeof(diagm), m::Integer, n::Integer, kv::Pair{<:Integer,<:AbstractVector}...)
        function diagm_pullback(ȳ)
            return (NoTangent(), NoTangent(), NoTangent(), _diagm_back.(kv, Ref(ȳ))...)
        end
        return diagm(m, n, kv...), diagm_pullback
    end
end
function rrule(::typeof(diagm), kv::Pair{<:Integer,<:AbstractVector}...)
    function diagm_pullback(ȳ)
        return (NoTangent(), _diagm_back.(kv, Ref(ȳ))...)
    end
    return diagm(kv...), diagm_pullback
end

function _diagm_back(p, ȳ)
    k, v = p
    d = diag(ȳ, k)[1:length(v)] # handle if diagonal was smaller than matrix
    return Tangent{typeof(p)}(second = d)
end

function rrule(::typeof(*), D::Diagonal{<:Real}, V::AbstractVector{<:Real})
    function times_pullback(Ȳ) # DECIDE
        return (NoTangent(), @thunk(Diagonal(Ȳ .* V)), @thunk(D * Ȳ))
    end
    return D * V, times_pullback
end

#####
##### `Adjoint`
#####

Adjoint_mat_pullback(ȳ::Tangent) = (NoTangent(), ȳ.parent)
Adjoint_mat_pullback(ȳ::AbstractVecOrMat) = (NoTangent(), adjoint(ȳ))
Adjoint_mat_pullback(ȳ::AbstractThunk) = return Adjoint_mat_pullback(unthunk(ȳ))
function rrule(::Type{<:Adjoint}, A::AbstractMatrix{<:Number})
    return Adjoint(A), Adjoint_mat_pullback
end

_Adjoint_vec_pullback(ȳ::Tangent) = (NoTangent(), vec(ȳ.parent))
_Adjoint_vec_pullback(ȳ::AbstractMatrix) = (NoTangent(), vec(adjoint(ȳ)))
_Adjoint_vec_pullback(ȳ::AbstractThunk) = return _Adjoint_vec_pullback(unthunk(ȳ))
function rrule(::Type{<:Adjoint}, A::AbstractVector{<:Number})
    return Adjoint(A), _Adjoint_vec_pullback
end

_adjoint_mat_pullback(ȳ::Tangent) = (NoTangent(), ȳ.parent)
_adjoint_mat_pullback(ȳ::AbstractVecOrMat) = (NoTangent(), adjoint(ȳ))
_adjoint_mat_pullback(ȳ::AbstractThunk) = return _adjoint_mat_pullback(unthunk(ȳ))
function rrule(::typeof(adjoint), A::AbstractMatrix{<:Number})
    return adjoint(A), _adjoint_mat_pullback
end

_adjoint_vec_pullback(ȳ::Tangent) = (NoTangent(), vec(ȳ.parent))
_adjoint_vec_pullback(ȳ::AbstractMatrix) = (NoTangent(), vec(adjoint(ȳ)))
_adjoint_vec_pullback(ȳ::AbstractThunk) = return _adjoint_vec_pullback(unthunk(ȳ))
function rrule(::typeof(adjoint), A::AbstractVector{<:Number})
    return adjoint(A), _adjoint_vec_pullback
end

#####
##### `Transpose`
#####

_Transpose_mat_pullback(ȳ::Tangent) = (NoTangent(), ȳ.parent)
_Transpose_mat_pullback(ȳ::AbstractVecOrMat) = (NoTangent(), Transpose(ȳ))
_Transpose_mat_pullback(ȳ::AbstractThunk) = return _Transpose_mat_pullback(unthunk(ȳ))
function rrule(::Type{<:Transpose}, A::AbstractMatrix{<:Number})
    return Transpose(A), _Transpose_mat_pullback
end

_Transpose_vec_pullback(ȳ::Tangent) = (NoTangent(), vec(ȳ.parent))
_Transpose_vec_pullback(ȳ::AbstractMatrix) = (NoTangent(), vec(Transpose(ȳ)))
_Transpose_vec_pullback(ȳ::AbstractThunk) = return _Transpose_vec_pullback(unthunk(ȳ))
function rrule(::Type{<:Transpose}, A::AbstractVector{<:Number})
    return Transpose(A), _Transpose_vec_pullback
end

_transpose_mat_pullback(ȳ::Tangent) = (NoTangent(), ȳ.parent)
_transpose_mat_pullback(ȳ::AbstractVecOrMat) = (NoTangent(), transpose(ȳ))
_transpose_mat_pullback(ȳ::AbstractThunk) = return _transpose_mat_pullback(unthunk(ȳ))
function rrule(::typeof(transpose), A::AbstractMatrix{<:Number})
    return transpose(A), _transpose_mat_pullback
end

_transpose_vec_pullback(ȳ::Tangent) = (NoTangent(), vec(ȳ.parent))
_transpose_vec_pullback(ȳ::AbstractMatrix) = (NoTangent(), vec(transpose(ȳ)))
_transpose_vec_pullback(ȳ::AbstractThunk) = return _transpose_vec_pullback(unthunk(ȳ))
function rrule(::typeof(transpose), A::AbstractVector{<:Number})
    return transpose(A), _transpose_vec_pullback
end

#####
##### Triangular matrices
#####

function rrule(::Type{<:UpperTriangular}, A::AbstractMatrix)
    function UpperTriangular_pullback(ȳ)
        return (NoTangent(), Matrix(ȳ))
    end
    return UpperTriangular(A), UpperTriangular_pullback
end

function rrule(::Type{<:LowerTriangular}, A::AbstractMatrix)
    function LowerTriangular_pullback(ȳ)
        return (NoTangent(), Matrix(ȳ))
    end
    return LowerTriangular(A), LowerTriangular_pullback
end

function rrule(::typeof(triu), A::AbstractMatrix, k::Integer)
    function triu_pullback(ȳ)
        return (NoTangent(), triu(ȳ, k), NoTangent())
    end
    return triu(A, k), triu_pullback
end
function rrule(::typeof(triu), A::AbstractMatrix)
    function triu_pullback(ȳ)
        return (NoTangent(), triu(ȳ))
    end
    return triu(A), triu_pullback
end

function rrule(::typeof(tril), A::AbstractMatrix, k::Integer)
    function tril_pullback(ȳ)
        return (NoTangent(), tril(ȳ, k), NoTangent())
    end
    return tril(A, k), tril_pullback
end
function rrule(::typeof(tril), A::AbstractMatrix)
    function tril_pullback(ȳ)
        return (NoTangent(), tril(ȳ))
    end
    return tril(A), tril_pullback
end

_diag_view(X) = view(X, diagind(X))
_diag_view(X::Diagonal) = parent(X)  #Diagonal wraps a Vector of just Diagonal elements

function rrule(::typeof(det), X::Union{Diagonal, AbstractTriangular})
    y = det(X)
    s = conj!(y ./ _diag_view(X))
    function det_pullback(ȳ)
        return (NoTangent(), Diagonal(ȳ .* s))
    end
    return y, det_pullback
end

function rrule(::typeof(logdet), X::Union{Diagonal, AbstractTriangular})
    y = logdet(X)
    s = conj!(one(eltype(X)) ./ _diag_view(X))
    function logdet_pullback(ȳ)
        return (NoTangent(), Diagonal(ȳ .* s))
    end
    return y, logdet_pullback
end
