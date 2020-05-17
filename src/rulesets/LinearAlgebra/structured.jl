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

function rrule(::typeof(*), D::Diagonal{<:Real}, V::AbstractVector{<:Real})
    function times_pullback(Ȳ)
        return (NO_FIELDS, @thunk(Diagonal(Ȳ .* V)), @thunk(D * Ȳ))
    end
    return D * V, times_pullback
end

#####
##### `Symmetric`
#####

function rrule(::Type{<:Symmetric}, A::AbstractMatrix)
    function Symmetric_pullback(ȳ)
        return (NO_FIELDS, @thunk(_symmetric_back(ȳ)))
    end
    return Symmetric(A), Symmetric_pullback
end

_symmetric_back(ΔΩ) = UpperTriangular(ΔΩ) + LowerTriangular(ΔΩ)' - Diagonal(ΔΩ)
_symmetric_back(ΔΩ::Union{Diagonal,UpperTriangular}) = ΔΩ

#####
##### `Symmetric{<:Real}`/`Hermitian` eigendecomposition
#####

function frule((_, ΔA), ::typeof(eigen), A::LinearAlgebra.RealHermSymComplexHerm)
    F = eigen(A)
    ∂F = Thunk() do
        λ, U = F
        M = U' * ΔA * U
        ∂λ = real(diag(M)) # if ΔA is Hermitian, so is U′ΔAU, so its diagonal is real
        # K is skew-hermitian with zero diag
        K = M ./ _nonzero.(λ' .- λ)
        _setdiag!(K, Zero())
        ∂U = U * K
        return Composite{typeof(F)}(values = ∂λ, vectors = ∂U)
    end
    return F, ∂F
end

function rrule(::typeof(eigen), A::LinearAlgebra.RealHermSymComplexHerm)
    F = eigen(A)
    function eigen_pullback(ΔF)
        ∂A = Thunk() do
            λ, U = F
            ∂λ, ∂U = ΔF.values, ΔF.vectors
            # K is skew-hermitian
            K = U' * ∂U
            # unstable for degenerate matrices
            U′∂AU = K ./ (_nonzero.(λ' .- λ))
            setdiag!(U'∂AU, ∂λ)
            return _symherm(A, U * U′∂AU * U')
        end
        return NO_FIELDS, ∂A
    end
    return F, eigen_pullback
end

function frule((_, ΔA), ::typeof(eigvals), A::LinearAlgebra.RealHermSymComplexHerm)
    λ, U = eigen(A)
    return λ, @thunk real(diag(U' * ΔA * U))
end

function rrule(::typeof(eigvals), A::LinearAlgebra.RealHermSymComplexHerm)
    F, back = rrule(eigen, A)
    λ, U = F
    function eigvals_pullback(Δλ)
        ∂A = Thunk() do
            ∂F = Composite{typeof(F)}(values = Δλ)
            return unthunk(back(∂F)[2])
        end
        return NO_FIELDS, ∂A
    end
    return λ, eigvals_pullback
end

_nonzero(x) = abs(x) > eps(eltype(x)) ? x : eps(eltype(x)) * sign(x)

_setdiag!(A, d) = (A[diagind(A)] = d)
_setdiag!(A, d::AbstractZero) = (A[diagind(A)] .= 0)
_setdiag!(A::AbstractZero, d) = Diagonal(d)

# constrain B to have same Symmetric/Hermitian type as A
_symherm(A::Symmetric, B) = Symmetric(B, Symbol(A.uplo))
_symherm(A::Hermitian, B) = Hermitian(B, Symbol(A.uplo))
_symherm(f, B::AbstractMatrix{<:Real}, uplo) = Symmetric(B, uplo)
_symherm(f, B::AbstractMatrix{<:Complex}, uplo) = Hermitian(B, uplo)

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
