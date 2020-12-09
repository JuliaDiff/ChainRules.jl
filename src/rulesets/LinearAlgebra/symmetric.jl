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
    A::LinearAlgebra.RealHermSymComplexHerm{<:BLAS.BlasReal,<:StridedMatrix};
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
    A::LinearAlgebra.RealHermSymComplexHerm{<:BLAS.BlasReal,<:StridedMatrix};
    sortby::Union{Function,Nothing}=nothing,
)
    F = eigen(A; sortby=sortby)
    function eigen_pullback(ΔF::Composite{<:Eigen})
        λ, U = F.values, F.vectors
        Δλ, ΔU = ΔF.values, ΔF.vectors
        ΔU = ΔU isa AbstractZero ? ΔU : copy(ΔU)
        ∂A = eigen_rev!(A, λ, U, Δλ, ΔU)
        return NO_FIELDS, ∂A
    end
    eigen_pullback(ΔF::AbstractZero) = (NO_FIELDS, ΔF)
    return F, eigen_pullback
end

# ∂U is overwritten if not an `AbstractZero`
function eigen_rev!(A::LinearAlgebra.RealHermSymComplexHerm, λ, U, ∂λ, ∂U)
    ∂λ isa AbstractZero && ∂U isa AbstractZero && return ∂λ + ∂U
    ∂A = similar(A, eltype(U))
    tmp = ∂U
    if ∂U isa AbstractZero
        mul!(∂A.data, U, real.(∂λ) .* U')
    else
        _eigen_norm_phase_rev!(∂U, A, U)
        ∂K = mul!(∂A.data, U', ∂U)
        ∂K ./= λ' .- λ
        ∂K[diagind(∂K)] .= real.(∂λ)
        mul!(tmp, ∂K, U')
        mul!(∂A.data, U, tmp)
        @inbounds _hermitrize!(∂A.data)
    end
    return ∂A
end

_eigen_norm_phase_fwd!(∂V, ::Union{Symmetric{T,S},Hermitian{T,S}}, V) where {T<:Real,S} = ∂V
function _eigen_norm_phase_fwd!(∂V, A::Hermitian{<:Complex}, V)
    k = A.uplo === 'U' ? size(A, 1) : 1
    @inbounds for i in axes(V, 2)
        v, ∂v = @views V[:, i], ∂V[:, i]
        vₖ, ∂vₖ = real(v[k]), ∂v[k]
        ∂v .-= v .* (im * (imag(∂vₖ) / ifelse(iszero(vₖ), one(vₖ), vₖ)))
    end
    return ∂V
end

_eigen_norm_phase_rev!(∂V, ::Union{Symmetric{T,S},Hermitian{T,S}}, V) where {T<:Real,S} = ∂V
function _eigen_norm_phase_rev!(∂V, A::Hermitian{<:Complex}, V)
    k = A.uplo === 'U' ? size(A, 1) : 1
    @inbounds for i in axes(V, 2)
        v, ∂v = @views V[:, i], ∂V[:, i]
        vₖ = real(v[k])
        ∂c = dot(v, ∂v)
        ∂v[k] -= im * (imag(∂c) / ifelse(iszero(vₖ), one(vₖ), vₖ))
    end
    return ∂V
end

#####
##### `eigvals!`/`eigvals`
#####

function frule(
    (_, ΔA),
    ::typeof(eigvals!),
    A::LinearAlgebra.RealHermSymComplexHerm{<:BLAS.BlasReal,<:StridedMatrix};
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
    A::LinearAlgebra.RealHermSymComplexHerm{<:BLAS.BlasReal,<:StridedMatrix};
    sortby::Union{Function,Nothing}=nothing,
)
    F = eigen(A; sortby=sortby)
    λ = F.values
    function eigvals_pullback(Δλ)
        U = F.vectors
        ∂A = similar(A)
        mul!(∂A.data, U, real.(Δλ) .* U')
        return NO_FIELDS, ∂A
    end
    eigvals_pullback(Δλ::AbstractZero) = (NO_FIELDS, Δλ)
    return λ, eigvals_pullback
end

#####
##### `svd`
#####

function rrule(::typeof(svd), A::LinearAlgebra.RealHermSymComplexHerm{<:BLAS.BlasReal,<:StridedMatrix})
    F = svd(A)
    function svd_pullback(ΔF::Composite{<:SVD})
        U, V = F.U, F.V
        c = _svd_eigvals_sign!(similar(F.S), U, V)
        λ = F.S .* c
        ∂λ = ΔF.S isa AbstractZero ? ΔF.S : ΔF.S .* c
        if all(x -> x isa AbstractZero, (ΔF.U, ΔF.V, ΔF.Vt))
            ∂U = ΔF.U + ΔF.V + ΔF.Vt
        else
            ∂U = ΔF.U .+ (ΔF.V .+ ΔF.Vt') .* c'
        end
        ∂A = eigen_rev!(A, λ, U, ∂λ, ∂U)
        return NO_FIELDS, ∂A
    end
    svd_pullback(ΔF::AbstractZero) = (NO_FIELDS, ΔF)
    return F, svd_pullback
end

# given singular vectors, compute sign of eigenvalues corresponding to singular values
function _svd_eigvals_sign!(c, U, V)
    n = size(U, 1)
    @inbounds broadcast!(c, eachindex(c)) do i
        u = @views U[:, i]
        # find element not close to zero
        # at least one element has abs2 ≥ 1/n > 1/(n + 1)
        k = findfirst(x -> (n + 1) * abs2(x) ≥ 1, u)
        return sign(real(u[k]) * real(V[k, i]))
    end
    return c
end

#####
##### `svdvals`
#####

function rrule(::typeof(svdvals), A::LinearAlgebra.RealHermSymComplexHerm{<:BLAS.BlasReal,<:StridedMatrix})
    # sorting doesn't affect the eigvals pullback, and it simplifies this rrule
    λ, back = rrule(eigvals, A; sortby = x -> -abs2(x))
    S = abs.(λ)
    function svdvals_pullback(ΔS)
        ∂λ = ΔS .* S ./ ifelse.(iszero.(λ), one.(λ), λ)
        ∂A = back(∂λ)
        return NO_FIELDS, ∂A
    end
    svdvals_pullback(ΔS::AbstractZero) = (NO_FIELDS, ΔS)
    return S, svdvals_pullback
end

#####
##### utilities
#####

# Get type (Symmetric or Hermitian) from type or matrix
_symhermtype(::Type{<:Symmetric}) = Symmetric
_symhermtype(::Type{<:Hermitian}) = Hermitian
_symhermtype(A) = _symhermtype(typeof(A))

# in-place hermitrize matrix
function _hermitrize!(A)
    n = size(A, 1)
    for i in 1:n
        for j in (i + 1):n
            A[i, j] = (A[i, j] + conj(A[j, i])) / 2
            A[j, i] = conj(A[i, j])
        end
        A[i, i] = real(A[i, i])
    end
    return A
end
