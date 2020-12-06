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

function frule((_, ΔA), ::typeof(eigen), A::StridedMatrix{T}; kwargs...) where {T<:Union{Real,Complex}}
    F = eigen(A; kwargs...)
    ΔA isa AbstractZero && return F, ΔA
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
        if ΔV isa AbstractZero
            Δλ isa AbstractZero && return (NO_FIELDS, Δλ + ΔV)
            ∂K = Diagonal(Δλ)
            tmp = Matrix(∂K)
        else
            ∂V = copyto!(similar(ΔV), ΔV)
            _eigen_norm_phase_rev!(∂V, A, V)
            ∂K = V' * ∂V
            ∂K ./= λ' .- conj.(λ)
            ∂K[diagind(∂K)] .= Δλ
            tmp = ∂K
        end
        ∂A = mul!(tmp, V' \ ∂K, V')
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
        vᵢ, ∂vᵢ = @views V[:, i], ∂V[:, i]
        # account for unit normalization
        ∂cᵢnorm = -real(dot(vᵢ, ∂vᵢ))
        if eltype(V) <: Real
            ∂cᵢ = ∂cᵢnorm
        else
            # account for rotation of largest element to real
            k = _findrealmaxabs2(vᵢ)
            ∂cᵢphase = -imag(∂vᵢ[k]) / real(vᵢ[k])
            ∂cᵢ = complex(∂cᵢnorm, ∂cᵢphase)
        end
        ∂vᵢ .+= vᵢ .* ∂cᵢ
    end
    return ∂V
end

function _eigen_norm_phase_rev!(∂V, A, V)
    @inbounds for i in axes(V, 2)
        vᵢ, ∂vᵢ = @views V[:, i], ∂V[:, i]
        ∂cᵢ = dot(vᵢ, ∂vᵢ)
        # account for unit normalization
        ∂vᵢ .-= real(∂cᵢ) .* vᵢ
        if !(eltype(V) <: Real)
            # account for rotation of largest element to real
            k = _findrealmaxabs2(vᵢ)
            ∂vᵢ[k] -= im * (imag(∂cᵢ) / real(vᵢ[k]))
        end
    end
    return ∂V
end

# workaround for findmax not taking a mapped function
function _findrealmaxabs2(x)
    amax = abs2(first(x))
    imax = 1
    for i in 2:length(x)
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

function frule((_, ΔA), ::typeof(eigvals), A::StridedMatrix{T}; kwargs...) where {T<:Union{Real,Complex}}
    ΔA isa AbstractZero && return eigvals(A; kwargs...), ΔA
    F = eigen(A; kwargs...)
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

function rrule(::typeof(cholesky), X::AbstractMatrix{<:Real})
    F = cholesky(X)
    function cholesky_pullback(Ȳ::Composite)
        ∂X = if F.uplo === 'U'
            chol_blocked_rev(Ȳ.U, F.U, 25, true)
        else
            chol_blocked_rev(Ȳ.L, F.L, 25, false)
        end
        return (NO_FIELDS, ∂X)
    end
    return F, cholesky_pullback
end

function rrule(::typeof(getproperty), F::T, x::Symbol) where T <: Cholesky
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

# See "Differentiation of the Cholesky decomposition" (Murray 2016), pages 5-9 in particular,
# for derivations. Here we're implementing the algorithms and their transposes.

"""
    level2partition(A::AbstractMatrix, j::Integer, upper::Bool)

Returns views to various bits of the lower triangle of `A` according to the
`level2partition` procedure defined in [1] if `upper` is `false`. If `upper` is `true` then
the transposed views are returned from the upper triangle of `A`.

[1]: "Differentiation of the Cholesky decomposition", Murray 2016
"""
function level2partition(A::AbstractMatrix, j::Integer, upper::Bool)
    n = checksquare(A)
    @boundscheck checkbounds(1:n, j)
    if upper
        r = view(A, 1:j-1, j)
        d = view(A, j, j)
        B = view(A, 1:j-1, j+1:n)
        c = view(A, j, j+1:n)
    else
        r = view(A, j, 1:j-1)
        d = view(A, j, j)
        B = view(A, j+1:n, 1:j-1)
        c = view(A, j+1:n, j)
    end
    return r, d, B, c
end

"""
    level3partition(A::AbstractMatrix, j::Integer, k::Integer, upper::Bool)

Returns views to various bits of the lower triangle of `A` according to the
`level3partition` procedure defined in [1] if `upper` is `false`. If `upper` is `true` then
the transposed views are returned from the upper triangle of `A`.

[1]: "Differentiation of the Cholesky decomposition", Murray 2016
"""
function level3partition(A::AbstractMatrix, j::Integer, k::Integer, upper::Bool)
    n = checksquare(A)
    @boundscheck checkbounds(1:n, j)
    if upper
        R = view(A, 1:j-1, j:k)
        D = view(A, j:k, j:k)
        B = view(A, 1:j-1, k+1:n)
        C = view(A, j:k, k+1:n)
    else
        R = view(A, j:k, 1:j-1)
        D = view(A, j:k, j:k)
        B = view(A, k+1:n, 1:j-1)
        C = view(A, k+1:n, j:k)
    end
    return R, D, B, C
end

"""
    chol_unblocked_rev!(Ā::AbstractMatrix, L::AbstractMatrix, upper::Bool)

Compute the reverse-mode sensitivities of the Cholesky factorization in an unblocked manner.
If `upper` is `false`, then the sensitivites are computed from and stored in the lower triangle
of `Ā` and `L` respectively. If `upper` is `true` then they are computed and stored in the
upper triangles. If at input `upper` is `false` and `tril(Ā) = L̄`, at output
`tril(Ā) = tril(Σ̄)`, where `Σ = LLᵀ`. Analogously, if at input `upper` is `true` and
`triu(Ā) = triu(Ū)`, at output `triu(Ā) = triu(Σ̄)` where `Σ = UᵀU`.
"""
function chol_unblocked_rev!(Σ̄::AbstractMatrix{T}, L::AbstractMatrix{T}, upper::Bool) where T<:Real
    n = checksquare(Σ̄)
    j = n
    @inbounds for _ in 1:n
        r, d, B, c = level2partition(L, j, upper)
        r̄, d̄, B̄, c̄ = level2partition(Σ̄, j, upper)

        # d̄ <- d̄ - c'c̄ / d.
        d̄[1] -= dot(c, c̄) / d[1]

        # [d̄ c̄'] <- [d̄ c̄'] / d.
        d̄ ./= d
        c̄ ./= d

        # r̄ <- r̄ - [d̄ c̄'] [r' B']'.
        r̄ = axpy!(-Σ̄[j,j], r, r̄)
        r̄ = gemv!(upper ? 'n' : 'T', -one(T), B, c̄, one(T), r̄)

        # B̄ <- B̄ - c̄ r.
        B̄ = upper ? ger!(-one(T), r, c̄, B̄) : ger!(-one(T), c̄, r, B̄)
        d̄ ./= 2
        j -= 1
    end
    return (upper ? triu! : tril!)(Σ̄)
end

function chol_unblocked_rev(Σ̄::AbstractMatrix, L::AbstractMatrix, upper::Bool)
    return chol_unblocked_rev!(copy(Σ̄), L, upper)
end

"""
    chol_blocked_rev!(Σ̄::StridedMatrix, L::StridedMatrix, nb::Integer, upper::Bool)

Compute the sensitivities of the Cholesky factorization using a blocked, cache-friendly
procedure. `Σ̄` are the sensitivities of `L`, and will be transformed into the sensitivities
of `Σ`, where `Σ = LLᵀ`. `nb` is the block size to use. If the upper triangle has been used
to represent the factorization, that is `Σ = UᵀU` where `U := Lᵀ`, then this should be
indicated by passing `upper = true`.
"""
function chol_blocked_rev!(Σ̄::StridedMatrix{T}, L::StridedMatrix{T}, nb::Integer, upper::Bool) where T<:Real
    n = checksquare(Σ̄)
    tmp = Matrix{T}(undef, nb, nb)
    k = n
    if upper
        @inbounds for _ in 1:nb:n
            j = max(1, k - nb + 1)
            R, D, B, C = level3partition(L, j, k, true)
            R̄, D̄, B̄, C̄ = level3partition(Σ̄, j, k, true)

            C̄ = trsm!('L', 'U', 'N', 'N', one(T), D, C̄)
            gemm!('N', 'N', -one(T), R, C̄, one(T), B̄)
            gemm!('N', 'T', -one(T), C, C̄, one(T), D̄)
            chol_unblocked_rev!(D̄, D, true)
            gemm!('N', 'T', -one(T), B, C̄, one(T), R̄)
            if size(D̄, 1) == nb
                tmp = axpy!(one(T), D̄, transpose!(tmp, D̄))
                gemm!('N', 'N', -one(T), R, tmp, one(T), R̄)
            else
                gemm!('N', 'N', -one(T), R, D̄ + D̄', one(T), R̄)
            end

            k -= nb
        end
        return triu!(Σ̄)
    else
        @inbounds for _ in 1:nb:n
            j = max(1, k - nb + 1)
            R, D, B, C = level3partition(L, j, k, false)
            R̄, D̄, B̄, C̄ = level3partition(Σ̄, j, k, false)

            C̄ = trsm!('R', 'L', 'N', 'N', one(T), D, C̄)
            gemm!('N', 'N', -one(T), C̄, R, one(T), B̄)
            gemm!('T', 'N', -one(T), C̄, C, one(T), D̄)
            chol_unblocked_rev!(D̄, D, false)
            gemm!('T', 'N', -one(T), C̄, B, one(T), R̄)
            if size(D̄, 1) == nb
                tmp = axpy!(one(T), D̄, transpose!(tmp, D̄))
                gemm!('N', 'N', -one(T), tmp, R, one(T), R̄)
            else
                gemm!('N', 'N', -one(T), D̄ + D̄', R, one(T), R̄)
            end

            k -= nb
        end
        return tril!(Σ̄)
    end
end

function chol_blocked_rev(Σ̄::AbstractMatrix, L::AbstractMatrix, nb::Integer, upper::Bool)
    # Convert to `Matrix`s because blas functions require StridedMatrix input.
    return chol_blocked_rev!(Matrix(Σ̄), Matrix(L), nb, upper)
end
