#####
##### `Symmetric`/`Hermitian`
#####

function frule((_, ΔA, _), T::Type{<:LinearAlgebra.HermOrSym}, A::AbstractMatrix, uplo)
    return T(A, uplo), T(ΔA, uplo)
end

function rrule(T::Type{<:LinearAlgebra.HermOrSym}, A::AbstractMatrix, uplo)
    Ω = T(A, uplo)
    function HermOrSym_pullback(ΔΩ)
        return (NO_FIELDS, _symherm_back(T, ΔΩ, Ω.uplo), DoesNotExist())
    end
    return Ω, HermOrSym_pullback
end

function frule((_, ΔA), TM::Type{<:Matrix}, A::LinearAlgebra.HermOrSym)
    return TM(A), TM(_symherm_forward(A, ΔA))
end
function frule((_, ΔA), ::Type{Array}, A::LinearAlgebra.HermOrSym)
    return Array(A), Array(_symherm_forward(A, ΔA))
end

function rrule(TM::Type{<:Matrix}, A::LinearAlgebra.HermOrSym)
    function Matrix_pullback(ΔΩ)
        TA = _symhermtype(A)
        T∂A = TA{eltype(ΔΩ),typeof(ΔΩ)}
        uplo = A.uplo
        ∂A = T∂A(_symherm_back(A, ΔΩ, uplo), uplo)
        return NO_FIELDS, ∂A
    end
    return TM(A), Matrix_pullback
end
rrule(::Type{Array}, A::LinearAlgebra.HermOrSym) = rrule(Matrix, A)

# Get type (Symmetric or Hermitian) from type or matrix
_symhermtype(::Type{<:Symmetric}) = Symmetric
_symhermtype(::Type{<:Hermitian}) = Hermitian
_symhermtype(A) = _symhermtype(typeof(A))

# for Ω = Matrix(A::HermOrSym), push forward ΔA to get ∂Ω
function _symherm_forward(A, ΔA)
    TA = _symhermtype(A)
    return if ΔA isa TA
        ΔA
    else
        TA{eltype(ΔA),typeof(ΔA)}(ΔA, A.uplo)
    end
end

# for Ω = HermOrSym(A, uplo), pull back ΔΩ to get ∂A
_symherm_back(::Type{<:Symmetric}, ΔΩ, uplo) = _symmetric_back(ΔΩ, uplo)
function _symherm_back(::Type{<:Hermitian}, ΔΩ::AbstractMatrix{<:Real}, uplo)
    return _symmetric_back(ΔΩ, uplo)
end
_symherm_back(::Type{<:Hermitian}, ΔΩ, uplo) = _hermitian_back(ΔΩ, uplo)
_symherm_back(Ω, ΔΩ, uplo) = _symherm_back(typeof(Ω), ΔΩ, uplo)

function _symmetric_back(ΔΩ, uplo)
    L, U, D = LowerTriangular(ΔΩ), UpperTriangular(ΔΩ), Diagonal(ΔΩ)
    return uplo == 'U' ? U .+ transpose(L) - D : L .+ transpose(U) - D
end
_symmetric_back(ΔΩ::Diagonal, uplo) = ΔΩ
_symmetric_back(ΔΩ::UpperTriangular, uplo) = Matrix(uplo == 'U' ? ΔΩ : transpose(ΔΩ))
_symmetric_back(ΔΩ::LowerTriangular, uplo) = Matrix(uplo == 'U' ? transpose(ΔΩ) : ΔΩ)

function _hermitian_back(ΔΩ, uplo)
    L, U, rD = LowerTriangular(ΔΩ), UpperTriangular(ΔΩ), real.(Diagonal(ΔΩ))
    return uplo == 'U' ? U .+ L' - rD : L .+ U' - rD
end
_hermitian_back(ΔΩ::Diagonal, uplo) = real.(ΔΩ)
function _hermitian_back(ΔΩ::LinearAlgebra.AbstractTriangular, uplo)
    ∂UL = ΔΩ .- Diagonal(_extract_imag.(diag(ΔΩ)))
    return if istriu(ΔΩ)
        return Matrix(uplo == 'U' ? ∂UL : ∂UL')
    else
        return Matrix(uplo == 'U' ? ∂UL' : ∂UL)
    end
end

#####
##### `eigen!`/`eigen`
#####

function frule(
    (_, ΔA),
    ::typeof(eigen!),
    A::LinearAlgebra.RealHermSymComplexHerm{<:BlasReal,<:StridedMatrix};
    sortby::Union{Function,Nothing}=nothing,
)
    F = eigen!(A; sortby=sortby)
    ΔA isa AbstractZero && return F, ΔA
    λ, U = F.values, F.vectors
    tmp = U' * ΔA
    ∂K = mul!(ΔA.data, tmp, U)
    ∂Kdiag = @view ∂K[diagind(∂K)]
    ∂λ = real.(∂Kdiag)
    ∂K ./= λ' .- λ
    fill!(∂Kdiag, 0)
    ∂U = mul!(tmp, U, ∂K)
    _eigen_norm_phase_fwd!(∂U, A, U)
    ∂F = Composite{typeof(F)}(values = ∂λ, vectors = ∂U)
    return F, ∂F
end

function rrule(
    ::typeof(eigen),
    A::LinearAlgebra.RealHermSymComplexHerm;
    sortby::Union{Function,Nothing}=nothing,
)
    F = eigen(A; sortby=sortby)
    function eigen_pullback(ΔF::Composite{<:Eigen})
        λ, U = F.values, F.vectors
        Δλ, ΔU = ΔF.values, ΔF.vectors
        if ΔU isa AbstractZero
            Δλ isa AbstractZero && return (NO_FIELDS, Δλ + ΔU)
            ∂K = Diagonal(Δλ)
            ∂A = U * ∂K * U'
        else
            ∂U = copyto!(similar(ΔU), ΔU)
            _eigen_norm_phase_rev!(∂U, A, U)
            ∂K = U' * ∂U
            ∂K ./= λ' .- λ
            ∂K[diagind(∂K)] = Δλ
            ∂A = mul!(∂K, U * ∂K, U')
        end
        return NO_FIELDS, ∂A
    end
    eigen_pullback(ΔF::AbstractZero) = (NO_FIELDS, ΔF)
    return F, eigen_pullback
end

_eigen_norm_phase_fwd!(∂V, ::LinearAlgebra.RealHermSym, V) = ∂V
function _eigen_norm_phase_fwd!(∂V, A::Hermitian, V)
    k = A.uplo === 'U' ? size(A, 1) : 1
    @inbounds for i in axes(V, 2)
        vᵢ = @view V[:, i]
        vₖᵢ, ∂vₖᵢ = real(vᵢ[k]), ∂V[k, i]
        ∂vᵢ .-= vᵢ .* (imag(∂vₖᵢ) / ifelse(iszero(vₖᵢ), one(vₖᵢ), vₖᵢ))
    end
    return ∂V
end

_eigen_norm_phase_rev!(∂V, ::LinearAlgebra.RealHermSym, V) = ∂V
function _eigen_norm_phase_rev!(∂V, A::Hermitian, V)
    k = A.uplo === 'U' ? size(A, 1) : 1
    @inbounds for i in axes(V, 2)
        vᵢ, ∂vᵢ = @views V[:, i], ∂V[:, i]
        vₖᵢ = real(vᵢ[k])
        ∂cᵢ = dot(vᵢ, ∂vᵢ)
        ∂vᵢ[k] -= im * (imag(∂cᵢ) / ifelse(iszero(vₖᵢ), one(vₖᵢ), vₖᵢ))
    end
    return ∂V
end

#####
##### `eigvals!`/`eigvals`
#####

function frule(
    (_, ΔA),
    ::typeof(eigvals!),
    A::LinearAlgebra.RealHermSymComplexHerm{<:BlasReal,<:StridedMatrix};
    sortby::Union{Function,Nothing}=nothing,
)
    ΔA isa AbstractZero && return eigvals!(A; sortby=sortby), ΔA
    F = eigen!(A; sortby=sortby)
    λ, U = F.values, F.vectors
    tmp = ΔA * U
    # diag(U' * tmp) without computing matrix product
    ∂λ = similar(λ)
    @inbounds for i in eachindex(λ)
        ∂λ[i] = @views real(dot(U[:, i], tmp[:, i]))
    end
    return λ, ∂λ
end

function rrule(
    ::typeof(eigvals),
    A::LinearAlgebra.RealHermSymComplexHerm;
    sortby::Union{Function,Nothing}=nothing,
)
    F = eigen(A; sortby=sortby)
    λ = F.values
    function eigvals_pullback(Δλ)
        U = F.vectors
        ∂A = U * Diagonal(Δλ) * U'
        return NO_FIELDS, ∂A
    end
    eigvals_pullback(Δλ::AbstractZero) = (NO_FIELDS, Δλ)
    return λ, eigvals_pullback
end
