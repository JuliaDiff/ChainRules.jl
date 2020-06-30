# Structured matrices


#####
##### `Diagonal`
#####

function rrule(::Type{<:Diagonal}, d::AbstractVector)
    function Diagonal_pullback(ȳ::AbstractMatrix)
        return (NO_FIELDS, diag(ȳ))
    end
    function Diagonal_pullback(ȳ::Composite)
        # TODO: Assert about the primal type in the Composite, It should be Diagonal
        # infact it should be exactly the type of `Diagonal(d)`
        # but right now Zygote loses primal type information so we can't use it.
        # See https://github.com/FluxML/Zygote.jl/issues/603
        return (NO_FIELDS, ȳ.diag)
    end
    return Diagonal(d), Diagonal_pullback
end

function rrule(::typeof(diag), A::AbstractMatrix)
    function diag_pullback(ȳ)
        return (NO_FIELDS, @thunk(Diagonal(ȳ)))
    end
    return diag(A), diag_pullback
end
if VERSION ≥ v"1.3"
    function rrule(::typeof(diag), A::AbstractMatrix, k::Integer)
        function diag_pullback(ȳ)
            return (NO_FIELDS, @thunk(diagm(size(A)..., k => ȳ)), DoesNotExist())
        end
        return diag(A, k), diag_pullback
    end

    function rrule(::typeof(diagm), m::Integer, n::Integer, kv::Pair{<:Integer,<:AbstractVector}...)
        function diagm_pullback(ȳ)
            return (NO_FIELDS, DoesNotExist(), DoesNotExist(), _diagm_back.(kv, Ref(ȳ))...)
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
    return Thunk() do
        k, v = p
        d = diag(ȳ, k)[1:length(v)] # handle if diagonal was smaller than matrix
        return Composite{typeof(p)}(second = d)
    end
end

function rrule(::typeof(*), D::Diagonal{<:Real}, V::AbstractVector{<:Real})
    function times_pullback(Ȳ)
        return (NO_FIELDS, @thunk(Diagonal(Ȳ .* V)), @thunk(D * Ȳ))
    end
    return D * V, times_pullback
end

#####
##### `Symmetric`/`Hermitian`
#####

function frule(
    (_, ΔA, _),
    ::Type{T},
    A::AbstractMatrix,
    uplo,
) where {T<:LinearAlgebra.HermOrSym}
    return T(A, uplo), T(ΔA, uplo)
end

function rrule(::Type{T}, A::AbstractMatrix, uplo) where {T<:LinearAlgebra.HermOrSym}
    Ω = T(A, uplo)
    function HermOrSym_pullback(ΔΩ)
        return (NO_FIELDS, @thunk(_symherm_back(T, ΔΩ, Ω.uplo)), DoesNotExist())
    end
    return Ω, HermOrSym_pullback
end

_symherm_back(::Type{<:Symmetric}, ΔΩ, uplo) = _symmetric_back(ΔΩ, uplo)
function _symherm_back(::Type{<:Hermitian}, ΔΩ::AbstractMatrix{<:Real}, uplo)
    return _symmetric_back(ΔΩ, uplo)
end
_symherm_back(::Type{<:Hermitian}, ΔΩ, uplo) = _hermitian_back(ΔΩ, uplo)

function _symmetric_back(ΔΩ, uplo)
    L, U, D = LowerTriangular(ΔΩ), UpperTriangular(ΔΩ), Diagonal(ΔΩ)
    return uplo == 'U' ? U .+ transpose(L) - D : L .+ transpose(U) - D
end
_symmetric_back(ΔΩ::Diagonal, uplo) = ΔΩ
_symmetric_back(ΔΩ::UpperTriangular, uplo) = collect(uplo == 'U' ? ΔΩ : transpose(ΔΩ))
_symmetric_back(ΔΩ::LowerTriangular, uplo) = collect(uplo == 'U' ? transpose(ΔΩ) : ΔΩ)

function _hermitian_back(ΔΩ, uplo)
    isreal(ΔΩ) && return _symmetric_back(ΔΩ, uplo)
    L, U, rD = LowerTriangular(ΔΩ), UpperTriangular(ΔΩ), real.(Diagonal(ΔΩ))
    return uplo == 'U' ? U .+ L' - rD : L .+ U' - rD
end
_hermitian_back(ΔΩ::Diagonal, uplo) = real.(ΔΩ)
function _hermitian_back(ΔΩ::LinearAlgebra.AbstractTriangular, uplo)
    isreal(ΔΩ) && return _symmetric_back(ΔΩ, uplo)
    ∂UL = ΔΩ .- Diagonal(_extract_imag(diag(ΔΩ)))
    return if istriu(ΔΩ)
        return collect(uplo == 'U' ? ∂UL : ∂UL')
    else
        return collect(uplo == 'U' ? ∂UL' : ∂UL)
    end
end

#####
##### `Adjoint`
#####

# ✖️✖️✖️TODO: Deal with complex-valued arrays as well
function rrule(::Type{<:Adjoint}, A::AbstractMatrix{<:Real})
    function Adjoint_pullback(ȳ)
        return (NO_FIELDS, @thunk(adjoint(ȳ)))
    end
    return Adjoint(A), Adjoint_pullback
end

function rrule(::Type{<:Adjoint}, A::AbstractVector{<:Real})
    function Adjoint_pullback(ȳ)
        return (NO_FIELDS, @thunk(vec(adjoint(ȳ))))
    end
    return Adjoint(A), Adjoint_pullback
end

function rrule(::typeof(adjoint), A::AbstractMatrix{<:Real})
    function adjoint_pullback(ȳ)
        return (NO_FIELDS, @thunk(adjoint(ȳ)))
    end
    return adjoint(A), adjoint_pullback
end

function rrule(::typeof(adjoint), A::AbstractVector{<:Real})
    function adjoint_pullback(ȳ)
        return (NO_FIELDS, @thunk(vec(adjoint(ȳ))))
    end
    return adjoint(A), adjoint_pullback
end

#####
##### `Transpose`
#####

function rrule(::Type{<:Transpose}, A::AbstractMatrix)
    function Transpose_pullback(ȳ)
        return (NO_FIELDS, @thunk transpose(ȳ))
    end
    return Transpose(A), Transpose_pullback
end

function rrule(::Type{<:Transpose}, A::AbstractVector)
    function Transpose_pullback(ȳ)
        return (NO_FIELDS, @thunk vec(transpose(ȳ)))
    end
    return Transpose(A), Transpose_pullback
end

function rrule(::typeof(transpose), A::AbstractMatrix)
    function transpose_pullback(ȳ)
        return (NO_FIELDS, @thunk transpose(ȳ))
    end
    return transpose(A), transpose_pullback
end

function rrule(::typeof(transpose), A::AbstractVector)
    function transpose_pullback(ȳ)
        return (NO_FIELDS, @thunk vec(transpose(ȳ)))
    end
    return transpose(A), transpose_pullback
end

#####
##### Triangular matrices
#####

function rrule(::Type{<:UpperTriangular}, A::AbstractMatrix)
    function UpperTriangular_pullback(ȳ)
        return (NO_FIELDS, @thunk Matrix(ȳ))
    end
    return UpperTriangular(A), UpperTriangular_pullback
end

function rrule(::Type{<:LowerTriangular}, A::AbstractMatrix)
    function LowerTriangular_pullback(ȳ)
        return (NO_FIELDS, @thunk Matrix(ȳ))
    end
    return LowerTriangular(A), LowerTriangular_pullback
end

function rrule(::typeof(triu), A::AbstractMatrix, k::Integer)
    function triu_pullback(ȳ)
        return (NO_FIELDS, @thunk(triu(ȳ, k)), DoesNotExist())
    end
    return triu(A, k), triu_pullback
end
function rrule(::typeof(triu), A::AbstractMatrix)
    function triu_pullback(ȳ)
        return (NO_FIELDS, @thunk triu(ȳ))
    end
    return triu(A), triu_pullback
end

function rrule(::typeof(tril), A::AbstractMatrix, k::Integer)
    function tril_pullback(ȳ)
        return (NO_FIELDS, @thunk(tril(ȳ, k)), DoesNotExist())
    end
    return tril(A, k), tril_pullback
end
function rrule(::typeof(tril), A::AbstractMatrix)
    function tril_pullback(ȳ)
        return (NO_FIELDS, @thunk tril(ȳ))
    end
    return tril(A), tril_pullback
end
