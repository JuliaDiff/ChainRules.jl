using LinearAlgebra: checksquare
using LinearAlgebra.BLAS: gemv, gemv!, gemm!, trsm!, axpy!, ger!

#####
##### `svd`
#####

function rrule(::typeof(svd), X::AbstractMatrix{<:Real})
    F = svd(X)
    ∂X = Rule() do Ȳ::NamedTuple{(:U,:S,:V)}
        svd_rev(F, Ȳ.U, Ȳ.S, Ȳ.V)
    end
    return F, ∂X
end

function rrule(::typeof(getproperty), F::SVD, x::Symbol)
    if x === :U
        rule = Ȳ->(U=Ȳ, S=zero(F.S), V=zero(F.V))
    elseif x === :S
        rule = Ȳ->(U=zero(F.U), S=Ȳ, V=zero(F.V))
    elseif x === :V
        rule = Ȳ->(U=zero(F.U), S=zero(F.S), V=Ȳ)
    elseif x === :Vt
        # TODO: This could be made to work, but it'd be a pain
        throw(ArgumentError("Vt is unsupported; use V and transpose the result"))
    end
    update = (X̄::NamedTuple{(:U,:S,:V)}, Ȳ)->_update!(X̄, rule(Ȳ), x)
    return getproperty(F, x), (Rule(rule, update), DNERule())
end

function svd_rev(USV::SVD, Ū::AbstractMatrix, s̄::AbstractVector, V̄::AbstractMatrix)
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
    FUᵀŪ = _mulsubtrans!(Ut*Ū, F)  # F .* (UᵀŪ - ŪᵀU)
    FVᵀV̄ = _mulsubtrans!(Vt*V̄, F)  # F .* (VᵀV̄ - V̄ᵀV)
    ImUUᵀ = _eyesubx!(U*Ut)        # I - UUᵀ
    ImVVᵀ = _eyesubx!(V*Vt)        # I - VVᵀ

    S = Diagonal(s)
    S̄ = Diagonal(s̄)

    Ā = _add!(U * FUᵀŪ * S, ImUUᵀ * (Ū / S)) * Vt
    _add!(Ā, U * S̄ * Vt)
    _add!(Ā, U * _add!(S * FVᵀV̄ * Vt, (S \ V̄') * ImVVᵀ))

    return Ā
end

#####
##### `cholesky`
#####

function rrule(::typeof(cholesky), X::AbstractMatrix{<:Real})
    F = cholesky(X)
    ∂X = Rule(Ȳ->chol_blocked_rev(Matrix(Ȳ), Matrix(F.U), 25, true))
    return F, ∂X
end

function rrule(::typeof(getproperty), F::Cholesky, x::Symbol)
    if x === :U
        if F.uplo === 'U'
            ∂F = Ȳ->UpperTriangular(Ȳ)
        else
            ∂F = Ȳ->LowerTriangular(Ȳ')
        end
    elseif x === :L
        if F.uplo === 'L'
            ∂F = Ȳ->LowerTriangular(Ȳ)
        else
            ∂F = Ȳ->UpperTriangular(Ȳ')
        end
    end
    return getproperty(F, x), (Rule(∂F), DNERule())
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
    chol_unblocked_rev!(Ā::AbstractMatrix, L::AbstractMatrix, upper::Bool)

Compute the reverse-mode sensitivities of the Cholesky factorization in an unblocked manner.
If `upper` is `false`, then the sensitivites are computed from and stored in the lower triangle
of `Ā` and `L` respectively. If `upper` is `true` then they are computed and stored in the
upper triangles. If at input `upper` is `false` and `tril(Ā) = L̄`, at output
`tril(Ā) = tril(Σ̄)`, where `Σ = LLᵀ`. Analogously, if at input `upper` is `true` and
`triu(Ā) = triu(Ū)`, at output `triu(Ā) = triu(Σ̄)` where `Σ = UᵀU`.
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
    chol_blocked_rev!(Σ̄::AbstractMatrix, L::AbstractMatrix, nb::Integer, upper::Bool)

Compute the sensitivities of the Cholesky factorization using a blocked, cache-friendly 
procedure. `Σ̄` are the sensitivities of `L`, and will be transformed into the sensitivities
of `Σ`, where `Σ = LLᵀ`. `nb` is the block size to use. If the upper triangle has been used
to represent the factorization, that is `Σ = UᵀU` where `U := Lᵀ`, then this should be
indicated by passing `upper = true`.
"""
function chol_blocked_rev!(Σ̄::AbstractMatrix{T}, L::AbstractMatrix{T}, nb::Integer, upper::Bool) where T<:Real
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
    return chol_blocked_rev!(copy(Σ̄), L, nb, upper)
end
