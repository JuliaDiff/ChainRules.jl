using LinearAlgebra: checksquare
using LinearAlgebra.BLAS: gemv, gemv!, gemm!, trsm!, axpy!, ger!

#####
##### `lu`
#####

# These rules are necessary because the primals call LAPACK functions

# frule for square matrix was introduced in Eq. 3.6 of
# de Hoog, F.R., Anderssen, R.S. and Lukas, M.A. (2011)
# Differentiation of matrix functionals using triangular factorization.
# Mathematics of Computation, 80 (275). p. 1585.
# doi: http://doi.org/10.1090/S0025-5718-2011-02451-8
# for derivations for wide and tall matrices, see
# https://sethaxen.com/blog/2021/02/differentiating-the-lu-decomposition/

const LU_RowMaximum = VERSION >= v"1.7.0-DEV.1188" ? RowMaximum : Val{true}
const LU_NoPivot = VERSION >= v"1.7.0-DEV.1188" ? NoPivot : Val{false}

const CHOLESKY_NoPivot = VERSION >= v"1.8.0-rc1" ? Union{NoPivot, Val{false}} : Val{false}

function frule(
    (_, Ȧ), ::typeof(lu!), A::StridedMatrix, pivot::Union{LU_RowMaximum,LU_NoPivot}; kwargs...
)
    ΔA = unthunk(Ȧ)
    F = lu!(A, pivot; kwargs...)
    ∂factors = pivot isa LU_RowMaximum ? ΔA[F.p, :] : ΔA
    m, n = size(∂factors)
    q = min(m, n)
    if m == n  # square A
        # minimal allocation computation of
        # ∂L = L * tril(L \ (P * ΔA) / U, -1)
        # ∂U = triu(L \ (P * ΔA) / U) * U
        # ∂factors = ∂L + ∂U
        L = UnitLowerTriangular(F.factors)
        U = UpperTriangular(F.factors)
        rdiv!(∂factors, U)
        ldiv!(L, ∂factors)
        ∂L = lmul!(L, tril(∂factors, -1))
        ∂U = rmul!(triu(∂factors), U)
        ∂factors .= ∂L .+ ∂U
    elseif m < n  # wide A, system is [P*A1 P*A2] = [L*U1 L*U2]
        L = UnitLowerTriangular(F.L)
        U = F.U
        ldiv!(L, ∂factors)
        @views begin
            ∂factors1 = ∂factors[:, 1:q]
            ∂factors2 = ∂factors[:, (q + 1):end]
            U1 = UpperTriangular(U[:, 1:q])
            U2 = U[:, (q + 1):end]
        end
        rdiv!(∂factors1, U1)
        ∂L = tril(∂factors1, -1)
        mul!(∂factors2, ∂L, U2, -1, 1)
        lmul!(L, ∂L)
        rmul!(triu!(∂factors1), U1)
        ∂factors1 .+= ∂L
    else  # tall A, system is [P1*A; P2*A] = [L1*U; L2*U]
        L = F.L
        U = UpperTriangular(F.U)
        rdiv!(∂factors, U)
        @views begin
            ∂factors1 = ∂factors[1:q, :]
            ∂factors2 = ∂factors[(q + 1):end, :]
            L1 = UnitLowerTriangular(L[1:q, :])
            L2 = L[(q + 1):end, :]
        end
        ldiv!(L1, ∂factors1)
        ∂U = triu(∂factors1)
        mul!(∂factors2, L2, ∂U, -1, 1)
        rmul!(∂U, U)
        lmul!(L1, tril!(∂factors1, -1))
        ∂factors1 .+= ∂U
    end
    ∂F = Tangent{typeof(F)}(; factors=∂factors)
    return F, ∂F
end

# these functions are defined outside the rrule because otherwise type inference breaks
# see https://github.com/JuliaLang/julia/issues/40990
function _lu_pullback(ΔF::Tangent, m, n, eltypeA, pivot, F)
    Δfactors = ΔF.factors
    Δfactors isa AbstractZero && return (NoTangent(), Δfactors, NoTangent())
    factors = F.factors
    ∂factors = eltypeA <: Real ? real(Δfactors) : Δfactors
    ∂A = similar(factors)
    q = min(m, n)
    if m == n  # square A
        # ∂A = P' * (L' \ (tril(L' * ∂L, -1) + triu(∂U * U')) / U')
        L = UnitLowerTriangular(factors)
        U = UpperTriangular(factors)
        ∂U = UpperTriangular(∂factors)
        tril!(copyto!(∂A, ∂factors), -1)
        lmul!(L', ∂A)
        copyto!(UpperTriangular(∂A), UpperTriangular(∂U * U'))
        rdiv!(∂A, U')
        ldiv!(L', ∂A)
    elseif m < n  # wide A, system is [P*A1 P*A2] = [L*U1 L*U2]
        triu!(copyto!(∂A, ∂factors))
        @views begin
            factors1 = factors[:, 1:q]
            U2 = factors[:, (q + 1):end]
            ∂A1 = ∂A[:, 1:q]
            ∂A2 = ∂A[:, (q + 1):end]
            ∂L = tril(∂factors[:, 1:q], -1)
        end
        L = UnitLowerTriangular(factors1)
        U1 = UpperTriangular(factors1)
        triu!(rmul!(∂A1, U1'))
        ∂A1 .+= tril!(mul!(lmul!(L', ∂L), ∂A2, U2', -1, 1), -1)
        rdiv!(∂A1, U1')
        ldiv!(L', ∂A)
    else  # tall A, system is [P1*A; P2*A] = [L1*U; L2*U]
        tril!(copyto!(∂A, ∂factors), -1)
        @views begin
            factors1 = factors[1:q, :]
            L2 = factors[(q + 1):end, :]
            ∂A1 = ∂A[1:q, :]
            ∂A2 = ∂A[(q + 1):end, :]
            ∂U = triu(∂factors[1:q, :])
        end
        U = UpperTriangular(factors1)
        L1 = UnitLowerTriangular(factors1)
        tril!(lmul!(L1', ∂A1), -1)
        ∂A1 .+= triu!(mul!(rmul!(∂U, U'), L2', ∂A2, -1, 1))
        ldiv!(L1', ∂A1)
        rdiv!(∂A, U')
    end
    if pivot isa LU_RowMaximum
        ∂A = ∂A[invperm(F.p), :]
    end
    return NoTangent(), ∂A, NoTangent()
end
_lu_pullback(ΔF::AbstractThunk, m, n, eltypeA, pivot, F) = _lu_pullback(unthunk(ΔF), m, n, eltypeA, pivot, F)
function rrule(
    ::typeof(lu), A::StridedMatrix, pivot::Union{LU_RowMaximum,LU_NoPivot}; kwargs...
)
    m, n = size(A)
    F = lu(A, pivot; kwargs...)
    lu_pullback(ȳ) = _lu_pullback(ȳ, m, n, eltype(A), pivot, F)
    return F, lu_pullback
end

#####
##### functions of `LU`
#####

# this rrule is necessary because the primal mutates

function rrule(::typeof(getproperty), F::TF, x::Symbol) where {T,TF<:LU{T,<:StridedMatrix{T}}}
    function getproperty_LU_pullback(ΔY)
        ∂factors = if x === :L
            m, n = size(F.factors)
            S = eltype(ΔY)
            tril!([ΔY zeros(S, m, max(0, n - m))], -1)
        elseif x === :U
            m, n = size(F.factors)
            S = eltype(ΔY)
            triu!([ΔY; zeros(S, max(0, m - n), n)])
        elseif x === :factors
            Matrix(ΔY)
        else
            return (NoTangent(), NoTangent(), NoTangent())
        end
        ∂F = Tangent{TF}(; factors=∂factors)
        return NoTangent(), ∂F, NoTangent()
    end
    getproperty_LU_pullback(ΔY::AbstractThunk) = getproperty_LU_pullback(unthunk(ΔY))
    return getproperty(F, x), getproperty_LU_pullback
end

# these rules are needed because the primal calls a LAPACK function

function frule((_, Ḟ), ::typeof(LinearAlgebra.inv!), F::LU{<:Any,<:StridedMatrix})
    ΔF = unthunk(Ḟ)
    # factors must be square if the primal did not error
    L = UnitLowerTriangular(F.factors)
    U = UpperTriangular(F.factors)
    # compute ∂Y = -(U \ (L \ ∂L + ∂U / U) / L) * P while minimizing allocations
    m, n = size(F.factors)
    q = min(m, n)
    ∂L = tril(m ≥ n ? ΔF.factors : view(ΔF.factors, :, 1:q), -1)
    ∂U = triu(m ≤ n ? ΔF.factors : view(ΔF.factors, 1:q, :))
    ∂Y = ldiv!(L, ∂L)
    ∂Y .+= rdiv!(∂U, U)
    ldiv!(U, ∂Y)
    rdiv!(∂Y, L)
    rmul!(∂Y, -1)
    return LinearAlgebra.inv!(F), ∂Y[:, invperm(F.p)]
end

function rrule(::typeof(inv), F::LU{<:Any,<:StridedMatrix})
    function inv_LU_pullback(ΔY)
        # factors must be square if the primal did not error
        L = UnitLowerTriangular(F.factors)
        U = UpperTriangular(F.factors)
        # compute the following while minimizing allocations
        # ∂U = - triu((U' \ ∂Y * P' / L') / U')
        # ∂L = - tril(L' \ (U' \ ∂Y * P' / L'), -1)
        ∂factors = ΔY[:, F.p]
        ldiv!(U', ∂factors)
        rdiv!(∂factors, L')
        rmul!(∂factors, -1)
        ∂L = tril!(L' \ ∂factors, -1)
        triu!(rdiv!(∂factors, U'))
        ∂factors .+= ∂L
        ∂F = Tangent{typeof(F)}(; factors=∂factors)
        return NoTangent(), ∂F
    end
    return inv(F), inv_LU_pullback
end

#####
##### `svd`
#####

function _svd_pullback(Ȳ::Tangent, F)
    ∂X = svd_rev(F, Ȳ.U, Ȳ.S, Ȳ.Vt')
    return (NoTangent(), ∂X)
end
_svd_pullback(Ȳ::AbstractThunk, F) = _svd_pullback(unthunk(Ȳ), F)
function rrule(::typeof(svd), X::AbstractMatrix{<:Real})
    F = svd(X)
    svd_pullback(ȳ) = _svd_pullback(ȳ, F)
    return F, svd_pullback
end

function rrule(::typeof(getproperty), F::T, x::Symbol) where T <: SVD
    function getproperty_svd_pullback(Ȳ)
        C = Tangent{T}
        ∂F = if x === :U
            C(U=Ȳ,)
        elseif x === :S
            C(S=Ȳ,)
        elseif x === :V
            C(Vt=Ȳ',)
        elseif x === :Vt
            C(Vt=Ȳ,)
        end
        return NoTangent(), ∂F, NoTangent()
    end
    return getproperty(F, x), getproperty_svd_pullback
end

# When not `ZeroTangent`s expect `Ū::AbstractMatrix, s̄::AbstractVector, V̄::AbstractMatrix`
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
##### `svdvals`
#####

function rrule(::typeof(svdvals), A::AbstractMatrix{<:Number})
    F = svd(A)
    U = F.U
    Vt = F.Vt
    project_A = ProjectTo(A)
    function svdvals_pullback(s̄)
        S̄ = s̄ isa AbstractZero ? s̄ : Diagonal(unthunk(s̄))
        return (NoTangent(), project_A(U * S̄ * Vt))
    end
    return F.S, svdvals_pullback
end

#####
##### `eigen`
#####

# TODO:
# - support correct differential of phase convention when A is hermitian
# - simplify when A is diagonal
# - support degenerate matrices (see #144)

function frule((_, ΔA), ::typeof(eigen!), A::StridedMatrix{T}; kwargs...) where {T<:BlasFloat}
    ΔA isa AbstractZero && return (eigen!(A; kwargs...), ΔA)
    if ishermitian(A)
        sortby = get(kwargs, :sortby, LinearAlgebra.eigsortby)
        return if sortby === nothing
            frule((ZeroTangent(), Hermitian(ΔA)), eigen!, Hermitian(A))
        else
            frule((ZeroTangent(), Hermitian(ΔA)), eigen!, Hermitian(A); sortby=sortby)
        end
    end
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
    ∂F = Tangent{typeof(F)}(values = ∂λ, vectors = ∂V)
    return F, ∂F
end

function rrule(::typeof(eigen), A::StridedMatrix{T}; kwargs...) where {T<:Union{Real,Complex}}
    F = eigen(A; kwargs...)
    function eigen_pullback(ΔF::Tangent)
        λ, V = F.values, F.vectors
        Δλ, ΔV = ΔF.values, ΔF.vectors
        ΔV isa AbstractZero && Δλ isa AbstractZero && return (NoTangent(), Δλ + ΔV)
        if eltype(λ) <: Real && ishermitian(A)
            hermA = Hermitian(A)
            ∂V = ΔV isa AbstractZero ? ΔV : copyto!(similar(ΔV), ΔV)
            ∂hermA = eigen_rev!(hermA, λ, V, Δλ, ∂V)
            ∂Atriu = _symherm_back(typeof(hermA), ∂hermA, Symbol(hermA.uplo))
            ∂A = ∂Atriu isa AbstractTriangular ? triu!(∂Atriu.data) : ∂Atriu
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
        return NoTangent(), T <: Real ? real(∂A) : ∂A
    end
    eigen_pullback(Ȳ::AbstractThunk) = eigen_pullback(unthunk(Ȳ))
    eigen_pullback(ΔF::AbstractZero) = (NoTangent(), ΔF)
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
        ∂c_norm = -realdot(v, ∂v)
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
    if ishermitian(A)
        λ, ∂λ = frule((ZeroTangent(), Hermitian(ΔA)), eigvals!, Hermitian(A))
        sortby = get(kwargs, :sortby, LinearAlgebra.eigsortby)
        _sorteig!_fwd(∂λ, λ, sortby)
    else
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
    end
    return λ, ∂λ
end

function rrule(::typeof(eigvals), A::StridedMatrix{T}; kwargs...) where {T<:Union{Real,Complex}}
    F, eigen_back = rrule(eigen, A; kwargs...)
    λ = F.values
    function eigvals_pullback(Δλ)
        ∂F = Tangent{typeof(F)}(values = Δλ)
        _, ∂A = eigen_back(∂F)
        return NoTangent(), ∂A
    end
    return λ, eigvals_pullback
end

# adapted from LinearAlgebra.sorteig!
function _sorteig!_fwd(Δλ, λ, sortby)
    Δλ isa AbstractZero && return (sort!(λ; by=sortby), Δλ)
    if sortby !== nothing
        p = sortperm(λ; alg=QuickSort, by=sortby)
        permute!(λ, p)
        permute!(Δλ, p)
    end
    return (λ, Δλ)
end

#####
##### `cholesky`
#####

function rrule(::typeof(cholesky), x::Number, uplo::Symbol)
    C = cholesky(x, uplo)
    function cholesky_pullback(ΔC)
        Ā = real(only(unthunk(ΔC).factors)) / (2 * sign(real(x)) * only(C.factors))
        return NoTangent(), Ā, NoTangent()
    end
    return C, cholesky_pullback
end

function _cholesky_Diagonal_pullback(ΔC, C)
    Udiag = C.factors.diag
    ΔUdiag = diag(ΔC.factors)
    Ādiag = real.(ΔUdiag) ./ (2 .* Udiag)
    if !issuccess(C)
        # cholesky computes the factor diagonal from the beginning until it encounters the
        # first failure. The remainder of the diagonal is then copied from the input.
        i = findfirst(x -> !isreal(x) || !(real(x) > 0), Udiag)
        Ādiag[i:end] .= ΔUdiag[i:end]
    end
    return NoTangent(), Diagonal(Ādiag), NoTangent()
end
function rrule(::typeof(cholesky), A::Diagonal{<:Number}, pivot::CHOLESKY_NoPivot; check::Bool=true)
    C = cholesky(A, pivot; check=check)
    cholesky_pullback(ȳ) = _cholesky_Diagonal_pullback(unthunk(ȳ), C)
    return C, cholesky_pullback
end

# The appropriate cotangent is different depending upon whether A is Symmetric / Hermitian,
# or just a StridedMatrix.
# Implementation due to Seeger, Matthias, et al. "Auto-differentiating linear algebra."
function rrule(
    ::typeof(cholesky),
    A::LinearAlgebra.RealHermSymComplexHerm{<:Real, <:StridedMatrix},
    pivot::CHOLESKY_NoPivot;
    check::Bool=true,
)
    C = cholesky(A, pivot; check=check)
    function cholesky_HermOrSym_pullback(ΔC)
        Ā = _cholesky_pullback_shared_code(C, unthunk(ΔC))
        rmul!(Ā, one(eltype(Ā)) / 2)
        return NoTangent(), _symhermtype(A)(Ā), NoTangent()
    end
    return C, cholesky_HermOrSym_pullback
end

function rrule(
    ::typeof(cholesky),
    A::StridedMatrix{<:Union{Real,Complex}},
    pivot::CHOLESKY_NoPivot;
    check::Bool=true,
)
    C = cholesky(A, pivot; check=check)
    function cholesky_Strided_pullback(ΔC)
        Ā = _cholesky_pullback_shared_code(C, unthunk(ΔC))
        idx = diagind(Ā)
        @views Ā[idx] .= real.(Ā[idx]) ./ 2
        return (NoTangent(), UpperTriangular(Ā), NoTangent())
    end
    return C, cholesky_Strided_pullback
end

function _cholesky_pullback_shared_code(C, ΔC)
    Δfactors = ΔC.factors
    Ā = similar(C.factors)
    if C.uplo === 'U'
        U = C.U
        Ū = eltype(U) <: Real ? real(_maybeUpperTri(Δfactors)) : _maybeUpperTri(Δfactors)
        mul!(Ā, Ū, U')
        LinearAlgebra.copytri!(Ā, 'U', true)
        eltype(Ā) <: Real || _realifydiag!(Ā)
        ldiv!(U, Ā)
        rdiv!(Ā, U')
    else  # C.uplo === 'L'
        L = C.L
        L̄ = eltype(L) <: Real ? real(_maybeLowerTri(Δfactors)) : _maybeLowerTri(Δfactors)
        mul!(Ā, L', L̄)
        LinearAlgebra.copytri!(Ā, 'L', true)
        eltype(Ā) <: Real || _realifydiag!(Ā)
        rdiv!(Ā, L)
        ldiv!(L', Ā)
    end
    return Ā
end

function rrule(::typeof(getproperty), F::T, x::Symbol) where {T <: Cholesky}
    function getproperty_cholesky_pullback(Ȳ)
        C = Tangent{T}
        ∂F = if x === :U
            if F.uplo === 'U'
                C(factors=_maybeUpperTri(Ȳ),)
            else
                C(factors=_maybeLowerTri(Ȳ'),)
            end
        elseif x === :L
            if F.uplo === 'L'
                C(factors=_maybeLowerTri(Ȳ),)
            else
                C(factors=_maybeUpperTri(Ȳ'),)
            end
        end
        return NoTangent(), ∂F, NoTangent()
    end
    return getproperty(F, x), getproperty_cholesky_pullback
end

_maybeUpperTri(A) = UpperTriangular(A)
_maybeUpperTri(A::Diagonal) = A
_maybeLowerTri(A) = LowerTriangular(A)
_maybeLowerTri(A::Diagonal) = A

# `det` and `logdet` for `Cholesky`
function rrule(::typeof(det), C::Cholesky)
    y = det(C)
    diagF = _diag_view(C.factors)
    function det_Cholesky_pullback(ȳ)
        ΔF = Diagonal(_x_divide_conj_y.(2 * ȳ * conj(y), diagF))
        ΔC = Tangent{typeof(C)}(; factors=ΔF)
        return NoTangent(), ΔC
    end
    return y, det_Cholesky_pullback
end

function rrule(::typeof(logdet), C::Cholesky)
    y = logdet(C)
    diagF = _diag_view(C.factors)
    function logdet_Cholesky_pullback(ȳ)
        ΔC = Tangent{typeof(C)}(; factors=Diagonal(_x_divide_conj_y.(2 * ȳ, diagF)))
        return NoTangent(), ΔC
    end
    return y, logdet_Cholesky_pullback
end

# Return `x / conj(y)`, or a type-stable 0 if `iszero(x)`
function _x_divide_conj_y(x, y)
    z = x / conj(y)
    return iszero(x) ? zero(z) : z
end

# these rules exists because the primals mutates using `ldiv!` and `rdiv!`
function rrule(::typeof(\), A::Cholesky, B::AbstractVecOrMat{<:Union{Real,Complex}})
    U, getproperty_back = rrule(getproperty, A, :U)
    Z = U' \ B
    Y = U \ Z
    project_B = ProjectTo(B)
    function ldiv_Cholesky_AbsVecOrMat_pullback(ΔY)
        ∂Z = U' \ ΔY
        ∂B = U \ ∂Z
        ∂A = Thunk() do
            _, Ā = getproperty_back(-add!!(∂Z * Y', Z * ∂B'))
            return Ā
        end
        return NoTangent(), ∂A, project_B(∂B)
    end
    return Y, ldiv_Cholesky_AbsVecOrMat_pullback
end

function rrule(::typeof(/), B::AbstractMatrix{<:Union{Real,Complex}}, A::Cholesky)
    U, getproperty_back = rrule(getproperty, A, :U)
    Z = B / U
    Y = Z / U'
    project_B = ProjectTo(B)
    function rdiv_AbstractMatrix_Cholesky_pullback(ΔY)
        ∂Z = ΔY / U
        ∂B = ∂Z / U'
        ∂A = Thunk() do
            _, Ā = getproperty_back(-add!!(∂Z' * Y, Z' * ∂B))
            return Ā
        end
        return NoTangent(), project_B(∂B), ∂A
    end
    return Y, rdiv_AbstractMatrix_Cholesky_pullback
end
