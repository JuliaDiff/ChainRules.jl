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
