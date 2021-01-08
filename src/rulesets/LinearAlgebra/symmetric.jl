#####
##### `Symmetric`/`Hermitian`
#####

function frule((_, ΔA, _), T::Type{<:LinearAlgebra.HermOrSym}, A::AbstractMatrix, uplo)
    return T(A, uplo), T(ΔA, uplo)
end

function rrule(T::Type{<:LinearAlgebra.HermOrSym}, A::AbstractMatrix, uplo)
    Ω = T(A, uplo)
    function HermOrSym_pullback(ΔΩ)
        return (NO_FIELDS, _symherm_back(typeof(Ω), ΔΩ, Ω.uplo), DoesNotExist())
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

# rule is old but the usual references are
# real rules:
# Giles M. B., An extended collection of matrix derivative results for forward and reverse
# mode algorithmic differentiation.
# https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf.
# complex rules:
# Boeddeker C., Hanebrink P., et al, On the Computation of Complex-valued Gradients with
# Application to Statistically Optimum Beamforming. arXiv:1701.00392v2 [cs.NA]
#
# accounting for normalization convention appears in Boeddeker && Hanebrink.
# account for phase convention is unpublished.
function frule(
    (_, ΔA),
    ::typeof(eigen!),
    A::LinearAlgebra.RealHermSymComplexHerm{<:BLAS.BlasReal,<:StridedMatrix};
    kwargs...,
)
    F = eigen!(A; kwargs...)
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
    kwargs...,
)
    F = eigen(A; kwargs...)
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

# NOTE: for small vₖ, the derivative of sign(vₖ) explodes, causing the tangents to become
# unstable even for phase-invariant programs. So for small vₖ we don't account for the phase
# in the gradient. Then derivatives are accurate for phase-invariant programs but inaccurate
# for phase-dependent programs that have low vₖ.

_eigen_norm_phase_fwd!(∂V, ::Union{Symmetric{T,S},Hermitian{T,S}}, V) where {T<:Real,S} = ∂V
function _eigen_norm_phase_fwd!(∂V, A::Hermitian{<:Complex}, V)
    k = A.uplo === 'U' ? size(A, 1) : 1
    ϵ = sqrt(eps(real(eltype(V))))
    @inbounds for i in axes(V, 2)
        v = @view V[:, i]
        vₖ = real(v[k])
        if abs(vₖ) > ϵ
            ∂v = @view ∂V[:, i]
            ∂v .-= v .* (im * (imag(∂v[k]) / vₖ))
        end
    end
    return ∂V
end

_eigen_norm_phase_rev!(∂V, ::Union{Symmetric{T,S},Hermitian{T,S}}, V) where {T<:Real,S} = ∂V
function _eigen_norm_phase_rev!(∂V, A::Hermitian{<:Complex}, V)
    k = A.uplo === 'U' ? size(A, 1) : 1
    ϵ = sqrt(eps(real(eltype(V))))
    @inbounds for i in axes(V, 2)
        v = @view V[:, i]
        vₖ = real(v[k])
        if abs(vₖ) > ϵ
            ∂v = @view ∂V[:, i]
            ∂c = dot(v, ∂v)
            ∂v[k] -= im * (imag(∂c) / vₖ)
        end
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
    kwargs...,
)
    ΔA isa AbstractZero && return eigvals!(A; kwargs...), ΔA
    F = eigen!(A; kwargs...)
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
    kwargs...,
)
    F, eigen_back = rrule(eigen, A; kwargs...)
    λ = F.values
    function eigvals_pullback(Δλ)
        ∂F = Composite{typeof(F)}(values = Δλ)
        _, ∂A = eigen_back(∂F)
        return NO_FIELDS, ∂A
    end
    return λ, eigvals_pullback
end

#####
##### `svd`
#####

# NOTE: rrule defined because the `svd` primal mutates after calling `eigen`.
# otherwise, this rule just applies the chain rule and can be removed when mutation
# is supported by reverse-mode AD packages
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
        # at least one element satisfies abs2(x) ≥ 1/n > 1/(n + 1)
        k = findfirst(x -> (n + 1) * abs2(x) ≥ 1, u)
        return sign(real(u[k]) * real(V[k, i]))
    end
    return c
end

#####
##### `svdvals`
#####

# NOTE: rrule defined because `svdvals` calls mutating `svdvals!` internally.
# can be removed when mutation is supported by reverse-mode AD packages
function rrule(::typeof(svdvals), A::LinearAlgebra.RealHermSymComplexHerm{<:BLAS.BlasReal,<:StridedMatrix})
    λ, back = rrule(eigvals, A)
    S = abs.(λ)
    p = sortperm(S; rev=true)
    permute!(S, p)
    function svdvals_pullback(ΔS)
        ∂λ = real.(ΔS)
        invpermute!(∂λ, p)
        ∂λ .*= sign.(λ)
        _, ∂A = back(∂λ)
        return NO_FIELDS, unthunk(∂A)
    end
    svdvals_pullback(ΔS::AbstractZero) = (NO_FIELDS, ΔS)
    return S, svdvals_pullback
end

#####
##### `Symmetric{<:Real}`/`Hermitian` matrix functions
#####

# Formula comes from so-called Daleckiĭ-Kreĭn theorem originally due to
# Ju. L. Daleckiĭ and S. G. Kreĭn. Integration and differentiation of functions of Hermitian
# operators and applications to the theory of perturbations.
# Amer. Math. Soc. Transl., Series 2, 47:1–30, 1965.
# Stabilization for almost-degenerate matrices due to
# S. D. Axen, 2020. Representing Ensembles of Molecules.
# Appendix D: Automatic differentation rules for power series functions of diagonalizable matrices
# https://escholarship.org/uc/item/6s62d8pw
# These rules are more stable for degenerate matrices than applying the chain rule to the
# rules for `eigen`.

for func in (:exp, :log, :sqrt, :cos, :sin, :tan, :cosh, :sinh, :tanh, :acos, :asin, :atan, :acosh, :asinh, :atanh)
    @eval begin
        function frule((_, ΔA), ::typeof($func), A::LinearAlgebra.RealHermSymComplexHerm)
            ΔA isa AbstractZero && return $func(A), ΔA
            Y, cache = _matfun($func, A)
            Ȳ = _matfun_frechet($func, A, Y, ΔA, cache)
            # If ΔA was hermitian, then ∂Y has the same structure as Y
            ∂Y = if ishermitian(ΔA) && (isa(Y, Symmetric) || isa(Y, Hermitian))
                _symhermlike!(Ȳ, Y)
            else
                Ȳ
            end
            return Y, ∂Y
        end

        function rrule(::typeof($func), A::LinearAlgebra.RealHermSymComplexHerm)
            Y, cache = _matfun($func, A)
            $(Symbol("$(func)_pullback"))(ΔY::AbstractZero) = (NO_FIELDS, ΔY)
            function $(Symbol("$(func)_pullback"))(ΔY)
                # for Hermitian Y, we don't need to realify the diagonal of ΔY, since the
                # effect is the same as applying _hermitrize! at the end
                ∂Y = eltype(Y) <: Real ? real(ΔY) : ΔY
                # for matrix functions, the pullback is related to the pushforward by an adjoint
                Ā = _matfun_frechet($func, A, Y, ∂Y', cache)
                # the cotangent of Hermitian A should be Hermitian
                ∂A = typeof(A)(eltype(A) <: Real ? real(Ā) : Ā, A.uplo)
                _hermitrize!(∂A.data)
                return NO_FIELDS, ∂A
            end
            return Y, $(Symbol("$(func)_pullback"))
        end
    end
end

function frule((_, ΔA), ::typeof(sincos), A::LinearAlgebra.RealHermSymComplexHerm)
    ΔA isa AbstractZero && return sincos(A), ΔA
    sinA, (λ, U, sinλ, cosλ) = _matfun(sin, A)
    cosA = _symhermtype(sinA)((U * Diagonal(cosλ)) * U')
    tmp = ΔA * U
    ∂Λ = U' * tmp
    ∂sinΛ = _muldiffquotmat!(similar(∂Λ), sin, λ, sinλ, cosλ, ∂Λ)
    ∂cosΛ = _muldiffquotmat!(∂Λ, cos, λ, cosλ, -sinλ, ∂Λ)
    ∂sinA = _symhermlike!(mul!(∂sinΛ, U, mul!(tmp, ∂sinΛ, U')), sinA)
    ∂cosA = _symhermlike!(mul!(∂cosΛ, U, mul!(tmp, ∂cosΛ, U')), cosA)
    Y = (sinA, cosA)
    ∂Y = Composite{typeof(Y)}(∂sinA, ∂cosA)
    return Y, ∂Y
end

function rrule(::typeof(sincos), A::LinearAlgebra.RealHermSymComplexHerm)
    sinA, (λ, U, sinλ, cosλ) = _matfun(sin, A)
    cosA = _symhermtype(sinA)((U * Diagonal(cosλ)) * U')
    Y = (sinA, cosA)
    sincos_pullback(ΔY::AbstractZero) = (NO_FIELDS, ΔY)
    function sincos_pullback(ΔY::Composite)
        ΔsinA, ΔcosA = ΔY
        if eltype(A) <: Real
            ∂sinA, ∂cosA = real(ΔsinA), real(ΔcosA)
        else
            ∂sinA, ∂cosA = ΔsinA, ΔcosA
        end
        tmp = ∂sinA * U
        ∂sinΛ = U' * tmp
        mul!(tmp, ∂cosA, U)
        ∂cosΛ = U' * tmp
        ∂Λ = _muldiffquotmat!(∂sinΛ, sin, λ, sinλ, cosλ, ∂sinΛ)
        ∂Λ = _muldiffquotmat!(∂Λ, cos, λ, cosλ, -sinλ, ∂cosΛ, true)
        Ā = mul!(∂Λ, U, mul!(tmp, ∂Λ, U'))
        _hermitrize!(Ā)
        ∂A = typeof(A)(Ā, A.uplo)
        return NO_FIELDS, ∂A
    end
    return Y, sincos_pullback
end

# compute the matrix function f(A), returning also a cache of intermediates for computing
# the pushforward or pullback.
function _matfun(f, A::LinearAlgebra.RealHermSymComplexHerm)
    λ, U = eigen(A)
    if all(λi -> _isindomain(f, λi), λ)
        fλ_df_dλ = map(λi -> frule((Zero(), One()), f, λi), λ)
    else # promote to complex if necessary
        fλ_df_dλ = map(λi -> frule((Zero(), One()), f, complex(λi)), λ)
    end
    fλ = first.(fλ_df_dλ)
    df_dλ = last.(unthunk.(fλ_df_dλ))
    fA = (U * Diagonal(fλ)) * U'
    Y = if eltype(A) <: Real
        Symmetric(fA)
    elseif eltype(fλ) <: Complex
        fA
    else
        Hermitian(fA)
    end
    cache = (λ, U, fλ, df_dλ)
    return Y, cache
end

# Fréchet derivative of matrix function f
# Computes ∂Y = U * (P .* (U' * ΔA * U)) * U' with fewer allocations
function _matfun_frechet(f, A::LinearAlgebra.RealHermSymComplexHerm, Y, ΔA, (λ, U, fλ, df_dλ))
    tmp = ΔA * U
    ∂Λ = U' * tmp
    ∂fΛ = _muldiffquotmat!(∂Λ, f, λ, fλ, df_dλ, ∂Λ)
    # reuse intermediate if possible
    if eltype(tmp) <: Real && eltype(∂fΛ) <: Complex
        tmp2 = ∂fΛ * U'
    else
        tmp2 = mul!(tmp, ∂fΛ, U')
    end
    ∂Y = mul!(∂fΛ, U, tmp2)
    return ∂Y
end

# difference quotient, i.e. Pᵢⱼ = (f(λⱼ) - f(λᵢ)) / (λⱼ - λᵢ), with f'(λᵢ) when λᵢ=λⱼ
function _diffquot(f, λi, λj, fλi, fλj, ∂fλi, ∂fλj)
    T = Base.promote_typeof(λi, λj, fλi, fλj, ∂fλi, ∂fλj)
    Δλ = λj - λi
    iszero(Δλ) && return T(∂fλi)
    # handle round-off error using Maclaurin series of (f(λᵢ + Δλ) - f(λᵢ)) / Δλ wrt Δλ
    # and approximating f''(λᵢ) with forward difference (f'(λᵢ + Δλ) - f'(λᵢ)) / Δλ
    # so (f(λᵢ + Δλ) - f(λᵢ)) / Δλ = (f'(λᵢ + Δλ) + f'(λᵢ)) / 2 + O(Δλ^2)
    # total error on the order of f(λᵢ) * eps()^(2/3)
    abs(Δλ) < cbrt(eps(real(T))) && return T((∂fλj + ∂fλi) / 2)
    Δfλ = fλj - fλi
    return T(Δfλ / Δλ)
end

# broadcast multiply Δ by the matrix of difference quotients P, storing the result in PΔ.
# If β is is nonzero, then @. PΔ = β*PΔ + P*Δ
# if type of PΔ is incompatible with result, new matrix is allocated
function _muldiffquotmat!(PΔ, f, λ, fλ, ∂fλ, Δ, β = false)
    if eltype(PΔ) <: Real && eltype(fλ) <: Complex
        return β .* PΔ .+ _diffquot.(f, λ, λ', fλ, transpose(fλ), ∂fλ, transpose(∂fλ)) .* Δ
    else
        PΔ .= β .* PΔ .+ _diffquot.(f, λ, λ', fλ, transpose(fλ), ∂fλ, transpose(∂fλ)) .* Δ
        return PΔ
    end
end

_isindomain(f, x) = true
_isindomain(::Union{typeof(acos),typeof(asin)}, x::Real) = -1 ≤ x ≤ 1
_isindomain(::typeof(acosh), x::Real) = x ≥ 1
_isindomain(::Union{typeof(log),typeof(sqrt)}, x::Real) = x ≥ 0

#####
##### utilities
#####

# Get type (Symmetric or Hermitian) from type or matrix
_symhermtype(::Type{<:Symmetric}) = Symmetric
_symhermtype(::Type{<:Hermitian}) = Hermitian
_symhermtype(A) = _symhermtype(typeof(A))

function _realifydiag!(A)
    for i in axes(A, 1)
        @inbounds A[i, i] = real(A[i, i])
    end
    return A
end

function _symhermlike!(A, S::Union{Symmetric,Hermitian})
    A isa Hermitian{<:Complex} && _realifydiag!(A)
    return typeof(S)(A, S.uplo)
end

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
