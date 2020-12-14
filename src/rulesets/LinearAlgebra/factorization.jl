using LinearAlgebra: checksquare
using LinearAlgebra.BLAS: gemv, gemv!, gemm!, trsm!, axpy!, ger!

#####
##### `svd`
#####

function rrule(::typeof(svd), X::AbstractMatrix{<:Real})
    F = svd(X)
    function svd_pullback(Ȳ::Composite)
        # `getproperty` on `Composite`s ensures we have no thunks.
        ∂X = svd_rev(F, Ȳ.U, Ȳ.S, Ȳ.V)
        return (NO_FIELDS, ∂X)
    end
    return F, svd_pullback
end

function rrule(::typeof(getproperty), F::T, x::Symbol) where T <: SVD
    function getproperty_svd_pullback(Ȳ)
        C = Composite{T}
        ∂F = if x === :U
            C(U=Ȳ,)
        elseif x === :S
            C(S=Ȳ,)
        elseif x === :V
            C(V=Ȳ,)
        elseif x === :Vt
            # TODO: https://github.com/JuliaDiff/ChainRules.jl/issues/106
            throw(ArgumentError("Vt is unsupported; use V and transpose the result"))
        end
        return NO_FIELDS, ∂F, DoesNotExist()
    end
    return getproperty(F, x), getproperty_svd_pullback
end

# When not `Zero`s expect `Ū::AbstractMatrix, s̄::AbstractVector, V̄::AbstractMatrix`
function svd_rev(USV::SVD, Ū, s̄, V̄)
    # Note: assuming a thin factorization, i.e. svd(A, full=false), which is the default
    U = USV.U
    s = USV.S
    V = USV.V
    Vt = USV.Vt

    k = length(s)
    T = eltype(s)
    F = T[i == j ? 1 : inv(@inbounds s[j]^2 - s[i]^2) for i = 1:k, j = 1:k]

    # We do a lot of matrix operations here, so we'll try to be memory-friendly and do
    # as many of the computations in-place as possible. Benchmarking shows that the in-
    # place functions here are significantly faster than their out-of-place, naively
    # implemented counterparts, and allocate no additional memory.
    Ut = U'
    FUᵀŪ = _mulsubtrans!!(Ut*Ū, F)  # F .* (UᵀŪ - ŪᵀU)
    FVᵀV̄ = _mulsubtrans!!(Vt*V̄, F)  # F .* (VᵀV̄ - V̄ᵀV)
    ImUUᵀ = _eyesubx!(U*Ut)  # I - UUᵀ
    ImVVᵀ = _eyesubx!(V*Vt)  # I - VVᵀ

    S = Diagonal(s)
    S̄ = s̄ isa AbstractZero ? s̄ : Diagonal(s̄)

    # TODO: consider using MuladdMacro here
    Ā = add!!(U * FUᵀŪ * S, ImUUᵀ * (Ū / S)) * Vt
    Ā = add!!(Ā, U * S̄ * Vt)
    Ā = add!!(Ā, U * add!!(S * FVᵀV̄ * Vt, (S \ V̄') * ImVVᵀ))

    return Ā
end

#####
##### `eigen`
#####

# TODO:
# - support correct differential of phase convention when A is hermitian
# - simplify when A is diagonal
# - support degenerate matrices (see #144)

function frule((_, ΔA), ::typeof(eigen!), A::StridedMatrix{T}; kwargs...) where {T<:BlasFloat}
    if ΔA isa AbstractZero
        F = eigen!(A; kwargs...)
        return F, ΔA
    end
    ishermitian(A) && return frule((Zero(), Hermitian(ΔA)), eigen!, Hermitian(A); kwargs...)
    F = eigen!(A; kwargs...)
    λ, V = F.values, F.vectors
    tmp = V \ ΔA
    ∂K = tmp * V
    ∂Kdiag = @view ∂K[diagind(∂K)]
    ∂λ = eltype(λ) <: Real ? real.(∂Kdiag) : copy(∂Kdiag)
    ∂K ./= transpose(λ) .- λ
    fill!(∂Kdiag, 0)
    ∂V = mul!(tmp, V, ∂K)
    _eigen_norm_phase_fwd!(∂V, A, V)
    ∂F = Composite{typeof(F)}(values = ∂λ, vectors = ∂V)
    return F, ∂F
end

function rrule(::typeof(eigen), A::StridedMatrix{T}; kwargs...) where {T<:Union{Real,Complex}}
    F = eigen(A; kwargs...)
    function eigen_pullback(ΔF::Composite{<:Eigen})
        λ, V = F.values, F.vectors
        Δλ, ΔV = ΔF.values, ΔF.vectors
        ΔV isa AbstractZero && Δλ isa AbstractZero && return (NO_FIELDS, Δλ + ΔV)
        if ishermitian(A)
            hermA = Hermitian(A)
            ∂V = copyto!(similar(ΔV), ΔV)
            ∂hermA = eigen_rev!(hermA, λ, V, Δλ, ∂V)
            ∂Atriu = _symherm_back(typeof(hermA), ∂hermA, hermA.uplo)
            ∂A = triu!(∂Atriu.data)
        elseif ΔV isa AbstractZero
            ∂K = Diagonal(Δλ)
            ∂A = V' \ ∂K * V'
        else
            ∂V = copyto!(similar(ΔV), ΔV)
            _eigen_norm_phase_rev!(∂V, A, V)
            ∂K = V' * ∂V
            ∂K ./= λ' .- conj.(λ)
            ∂K[diagind(∂K)] .= Δλ
            ∂A = mul!(∂K, V' \ ∂K, V')
        end
        return NO_FIELDS, T <: Real ? real(∂A) : ∂A
    end
    eigen_pullback(ΔF::AbstractZero) = (NO_FIELDS, ΔF)
    return F, eigen_pullback
end

# mutate ∂V to account for the (arbitrary but consistent) normalization and phase condition
# applied to the eigenvectors.
# these implementations assume the convention used by eigen in LinearAlgebra (i.e. that of
# LAPACK.geevx!; eigenvectors have unit norm, and the element with the largest absolute
# value is real), but they can be specialized for `A`

function _eigen_norm_phase_fwd!(∂V, A, V)
    @inbounds for i in axes(V, 2)
        v, ∂v = @views V[:, i], ∂V[:, i]
        # account for unit normalization
        ∂c_norm = -real(dot(v, ∂v))
        if eltype(V) <: Real
            ∂c = ∂c_norm
        else
            # account for rotation of largest element to real
            k = _findrealmaxabs2(v)
            ∂c_phase = -imag(∂v[k]) / real(v[k])
            ∂c = complex(∂c_norm, ∂c_phase)
        end
        ∂v .+= v .* ∂c
    end
    return ∂V
end

function _eigen_norm_phase_rev!(∂V, A, V)
    @inbounds for i in axes(V, 2)
        v, ∂v = @views V[:, i], ∂V[:, i]
        ∂c = dot(v, ∂v)
        # account for unit normalization
        ∂v .-= real(∂c) .* v
        if !(eltype(V) <: Real)
            # account for rotation of largest element to real
            k = _findrealmaxabs2(v)
            @inbounds ∂v[k] -= im * (imag(∂c) / real(v[k]))
        end
    end
    return ∂V
end

# workaround for findmax not taking a mapped function
function _findrealmaxabs2(x)
    amax = abs2(first(x))
    imax = 1
    @inbounds for i in 2:length(x)
        xi = x[i]
        !isreal(xi) && continue
        a = abs2(xi)
        a < amax && continue
        amax, imax = a, i
    end
    return imax
end

#####
##### `eigvals`
#####

function frule((_, ΔA), ::typeof(eigvals!), A::StridedMatrix{T}; kwargs...) where {T<:BlasFloat}
    ΔA isa AbstractZero && return eigvals!(A; kwargs...), ΔA
    F = eigen!(A; kwargs...)
    λ, V = F.values, F.vectors
    tmp = V \ ΔA
    ∂λ = similar(λ)
    # diag(tmp * V) without computing full matrix product
    if eltype(∂λ) <: Real
        broadcast!((a, b) -> sum(real ∘ prod, zip(a, b)), ∂λ, eachrow(tmp), eachcol(V))
    else
        broadcast!((a, b) -> sum(prod, zip(a, b)), ∂λ, eachrow(tmp), eachcol(V))
    end
    return λ, ∂λ
end

function rrule(::typeof(eigvals), A::StridedMatrix{T}; kwargs...) where {T<:Union{Real,Complex}}
    F = eigen(A; kwargs...)
    λ = F.values
    function eigvals_pullback(Δλ)
        V = F.vectors
        ∂A = V' \ Diagonal(Δλ) * V'
        return NO_FIELDS, T <: Real ? real(∂A) : ∂A
    end
    eigvals_pullback(Δλ::AbstractZero) = (NO_FIELDS, Δλ)
    return λ, eigvals_pullback
end

#####
##### `cholesky`
#####

function rrule(::typeof(cholesky), A::Real, uplo::Symbol=:U)
    C = cholesky(A, uplo)
    function cholesky_pullback(ΔC::Composite)
        return NO_FIELDS, ΔC.factors[1, 1] / (2 * C.U[1, 1]), DoesNotExist()
    end
    return C, cholesky_pullback
end

function rrule(::typeof(cholesky), A::Diagonal{<:Real}, ::Val{false}; check::Bool=true)
    C = cholesky(A, Val(false); check=check)
    function cholesky_pullback(ΔC::Composite)
        Ā = Diagonal(diag(ΔC.factors) .* inv.(2 .* C.factors.diag))
        return NO_FIELDS, Ā, DoesNotExist()
    end
    return C, cholesky_pullback
end

# The appropriate cotangent is different depending upon whether A is Symmetric / Hermitian,
# or just a StridedMatrix.
# Implementation due to Seeger, Matthias, et al. "Auto-differentiating linear algebra."
function rrule(
    ::typeof(cholesky),
    A::LinearAlgebra.HermOrSym{<:LinearAlgebra.BlasReal, <:StridedMatrix},
    ::Val{false};
    check::Bool=true,
)
    C = cholesky(A, Val(false); check=check)
    function cholesky_pullback(ΔC::Composite)
        Ā, U = _cholesky_pullback_shared_code(C, ΔC)
        Ā = BLAS.trsm!('R', 'U', 'C', 'N', one(eltype(Ā)) / 2, U.data, Ā)
        return NO_FIELDS, _symhermtype(A)(Ā), DoesNotExist()
    end
    return C, cholesky_pullback
end

function rrule(
    ::typeof(cholesky),
    A::StridedMatrix{<:LinearAlgebra.BlasReal},
    ::Val{false};
    check::Bool=true,
)
    C = cholesky(A, Val(false); check=check)
    function cholesky_pullback(ΔC::Composite)
        Ā, U = _cholesky_pullback_shared_code(C, ΔC)
        Ā = BLAS.trsm!('R', 'U', 'C', 'N', one(eltype(Ā)), U.data, Ā)
        idx = diagind(Ā)
        @views Ā[idx] .= real.(Ā[idx]) ./ 2
        return (NO_FIELDS, UpperTriangular(Ā), DoesNotExist())
    end
    return C, cholesky_pullback
end

function _cholesky_pullback_shared_code(C, ΔC)
    U = C.U
    Ū = ΔC.U
    Ā = similar(U.data)
    Ā = mul!(Ā, Ū, U')
    Ā = LinearAlgebra.copytri!(Ā, 'U', true)
    Ā = ldiv!(U, Ā)
    return Ā, U
end

function rrule(::typeof(getproperty), F::T, x::Symbol) where {T <: Cholesky}
    function getproperty_cholesky_pullback(Ȳ)
        C = Composite{T}
        ∂F = if x === :U
            if F.uplo === 'U'
                C(U=UpperTriangular(Ȳ),)
            else
                C(L=LowerTriangular(Ȳ'),)
            end
        elseif x === :L
            if F.uplo === 'L'
                C(L=LowerTriangular(Ȳ),)
            else
                C(U=UpperTriangular(Ȳ'),)
            end
        end
        return NO_FIELDS, ∂F, DoesNotExist()
    end
    return getproperty(F, x), getproperty_cholesky_pullback
end
