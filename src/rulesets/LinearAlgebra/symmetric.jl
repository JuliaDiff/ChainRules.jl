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

# Currently only defined for series functions whose codomain is ℝ
# These are type-stable and closed under `func`


# TODO: support log, sqrt, acos, asin, and non-int pow, which are type-unstable
for func in (:exp, :cos, :sin, :tan, :cosh, :sinh, :tanh, :atan, :asinh, :atanh)
    @eval begin
        function frule((_, ΔA), ::typeof($func), A::LinearAlgebra.RealHermSymComplexHerm)
            Y, λ, U, fλ, df_dλ = _matfun_shared($func, A)
            ∂Λ = U' * ΔA * U
            U′∂YU = _muldiffquotmat(λ, fλ, df_dλ, ∂Λ)
            ∂Y = _symhermlike!(U * U′∂YU * U', Y)
            return Y, ∂Y
        end

        function rrule(::typeof($func), A::LinearAlgebra.RealHermSymComplexHerm)
            Y, λ, U, fλ, df_dλ = _matfun_shared($func, A)
            function $(Symbol("$(func)_pullback"))(ΔY)
                ∂fΛ = U' * _realifydiag(ΔY) * U
                U′∂AU = _muldiffquotmat(λ, fλ, df_dλ, ∂fΛ)
                ∂A = _hermitrizeback!(U * U′∂AU * U', A)
                return NO_FIELDS, ∂A
            end
            return Y, $(Symbol("$(func)_pullback"))
        end
    end
end

# code shared by `frule`s and `rrule`s of matrix functions of Hermitian matrices
function _matfun_shared(f, A::LinearAlgebra.RealHermSymComplexHerm)
    λ, U = eigen(A)
    fλ_df_dλ = map(λi -> frule((Zero(), One()), f, λi), λ)
    T = Base.promote_eltype(λ, eltype(fλ_df_dλ))
    fλ = T.(first.(fλ_df_dλ))
    df_dλ = T.(last.(unthunk.(fλ_df_dλ)))
    Y = _symhermfwd!(U * Diagonal(fλ) * U')
    return Y, λ, U, fλ, df_dλ
end

function frule((_, ΔA), ::typeof(sincos), A::LinearAlgebra.RealHermSymComplexHerm)
    λ, U = eigen(A)
    sinλ, cosλ = sin.(λ), cos.(λ)
    sinA = _symhermfwd!(U * Diagonal(sinλ) * U')
    cosA = _symhermfwd!(U * Diagonal(cosλ) * U')
    sincosA = (sinA, cosA)
    ∂Λ = U' * ΔA * U
    U′∂sinAU = _muldiffquotmat(λ, sinλ, cosλ, ∂Λ)
    ∂sinA = _symhermlike!(U * U′∂sinAU * U', sinA)
    U′∂cosAU = _muldiffquotmat(λ, cosλ, -sinλ, ∂Λ)
    ∂cosA = _symhermlike!(U * U′∂cosAU * U', cosA)
    ∂sincosA = Composite{typeof(sincosA)}(∂sinA, ∂cosA)
    return sincosA, ∂sincosA
end

function rrule(::typeof(sincos), A::LinearAlgebra.RealHermSymComplexHerm)
    λ, U = eigen(A)
    sinλ, cosλ = sin.(λ), cos.(λ)
    sinA = _symhermfwd!(U * Diagonal(sinλ) * U')
    cosA = _symhermfwd!(U * Diagonal(cosλ) * U')
    sincosA = (sinA, cosA)
    function sincos_pullback(ΔsincosA)
        ΔsinA, ΔcosA = ΔsincosA
        ∂sinΛ, ∂cosΛ = U' * _realifydiag(ΔsinA) * U, U' * _realifydiag(ΔcosA) * U
        inds = eachindex(λ)
        U′∂AU = @inbounds begin
            _diffquot.(inds, inds', Ref(λ), Ref(sinλ), Ref(cosλ)) .* ∂sinΛ .+
            _diffquot.(inds, inds', Ref(λ), Ref(cosλ), Ref(-sinλ)) .* ∂cosΛ
        end
        ∂A = _hermitrizeback!(U * U′∂AU * U', A)
        return NO_FIELDS, ∂A
    end
    return sincosA, sincos_pullback
end

# difference quotient, i.e. Pᵢⱼ = (f(λᵢ) - f(λⱼ)) / (λᵢ - λⱼ), with f'(λᵢ) when i==j
Base.@propagate_inbounds function _diffquot(i, j, λ, fλ, df_dλ)
    T = typeof(zero(eltype(fλ)) / one(eltype(λ)) + zero(eltype(df_dλ)))
    i == j && return T(df_dλ[i])
    Δλ = λ[i] - λ[j]
    # handle round-off error by taylor expanding Δfλ / Δλ as Δλ → 0
    # total error on the order of eps()^(2/3)
    abs(Δλ) < cbrt(eps(real(T))) && return T((df_dλ[i] + df_dλ[j]) / 2)
    return T((fλ[i] - fλ[j]) / Δλ)
end

# multiply Δ by the matrix of difference quotients P
function _muldiffquotmat(λ, fλ, df_dλ, Δ)
    inds = eachindex(λ)
    return @inbounds _diffquot.(inds, inds', Ref(λ), Ref(fλ), Ref(df_dλ)) .* Δ
end

#####
##### utilities
#####

# Get type (Symmetric or Hermitian) from type or matrix
_symhermtype(::Type{<:Symmetric}) = Symmetric
_symhermtype(::Type{<:Hermitian}) = Hermitian
_symhermtype(A) = _symhermtype(typeof(A))

_pureimag(x) = x - real(x)

function _realifydiag!(A)
    for i in axes(A, 1)
        @inbounds A[i, i] = real(A[i, i])
    end
    return A
end
_realifydiag!(A::AbstractMatrix{<:Real}) = A

_realifydiag(A) = A .- Diagonal(_pureimag.(diag(A)))
_realifydiag(A::AbstractMatrix{<:Real}) = A

_symherm(A::AbstractMatrix{<:Real}, uplo = :U) = Symmetric(A, uplo)
_symherm(A::AbstractMatrix{<:Complex}, uplo = :U) = Hermitian(A, uplo)

function _symhermlike!(A, S::LinearAlgebra.RealHermSymComplexHerm)
    _realifydiag!(A)
    return typeof(S)(A, S.uplo)
end

function _symhermfwd!(A, uplo = :U)
    _realifydiag!(A)
    return _symherm(A, uplo)
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

# pullback of hermitrization
function _hermitrizeback!(∂A, A)
    @inbounds for i in axes(∂A, 1)
        for j in 1:(i - 1)
            if A.uplo === 'U'
                ∂A[j, i] = (∂A[j, i] + ∂A[i, j]') / 2
            else
                ∂A[i, j] = (∂A[i, j] + ∂A[j, i]') / 2
            end
        end
        if eltype(∂A) <: Complex
            ∂A[i, i] = real(∂A[i, i])
        end
    end
    return typeof(A)(∂A, A.uplo)
end
