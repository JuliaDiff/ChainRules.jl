module SparseArraysExt

using ChainRulesCore
using LinearAlgebra
using SparseArrays

using ChainRules: _diagm_back

import ChainRulesCore: rrule

function rrule(::typeof(sparse), I::AbstractVector, J::AbstractVector, V::AbstractVector, m, n, combine::typeof(+))
    project_V = ProjectTo(V)
    
    function sparse_pullback(Ω̄)
        ΔΩ = unthunk(Ω̄)
        ΔV = project_V(ΔΩ[I .+ m .* (J .- 1)])
        return NoTangent(), NoTangent(), NoTangent(), ΔV, NoTangent(), NoTangent(), NoTangent()
    end

    return sparse(I, J, V, m, n, combine), sparse_pullback
end

function rrule(::Type{T}, A::AbstractMatrix) where T <: AbstractSparseMatrix
    function sparse_pullback(Ω̄)
        return NoTangent(), Ω̄
    end
    return T(A), sparse_pullback
end

function rrule(::Type{T}, v::AbstractVector) where T <: AbstractSparseVector
    function sparse_pullback(Ω̄)
        return NoTangent(), Ω̄
    end
    return T(v), sparse_pullback
end

function rrule(::typeof(findnz), A::AbstractSparseMatrix)
    I, J, V = findnz(A)
    m, n = size(A)

    function findnz_pullback(Δ)
        V̄ = unthunk(Δ)[3]
        V̄ isa AbstractZero && return (NoTangent(), V̄)
        return NoTangent(), sparse(I, J, V̄, m, n)
    end

    return (I, J, V), findnz_pullback
end

function rrule(::typeof(findnz), v::AbstractSparseVector)
    I, V = findnz(v)
    n = length(v)

    function findnz_pullback(Δ)
        V̄ = unthunk(Δ)[2]
        V̄ isa AbstractZero && return (NoTangent(), V̄)
        return NoTangent(), sparsevec(I, V̄, n)
    end

    return (I, V), findnz_pullback
end

if Base.USE_GPL_LIBS # Don't define rrules for sparse determinants if we don't have CHOLMOD from SuiteSparse.jl
    using SparseArrays: CHOLMOD

    # Cholesky factorization of `x`, and whether it succeeded, i.e., whether `x` is Hermitian
    # positive definite. If `x` is not Hermitian, an empty matrix is factorized instead: this
    # avoids a pointless factorization of `x` but keeps the type of the factorization, and
    # hence of the pullbacks capturing it, inferrable.
    # The `Hermitian` wrapper makes CHOLMOD accept `x` even if explicitly stored zeros make
    # its sparsity pattern asymmetric.
    function _cholesky(x::SparseMatrixCSC)
        x_ishermitian = ishermitian(x)
        F = cholesky(Hermitian(x_ishermitian ? x : similar(x, 0, 0), :L); check=false)
        return F, x_ishermitian && issuccess(F)
    end

    function _check_posdef(success::Bool)
        success || throw(ArgumentError(
            "rrules for `det`, `logdet` and `logabsdet` of a `SparseMatrixCSC` only support " *
            "Hermitian positive definite matrices"
        ))
        return nothing
    end

    # Selected inverse: the lower triangle of `Z = inv(A[p, p])` on the sparsity pattern of
    # the Cholesky factor `L` of `F = cholesky(A)`, where `p = F.p`.
    # Since `A[p, p] = L * L'`, `Z * L = inv(L')` is upper triangular with diagonal
    # `1 ./ diag(L)`. Hence, with `S` the rows of the entries of `L[:, i]` below the diagonal,
    #     Z[S, i] = -Z[S, S] * L[S, i] / L[i, i],
    #     Z[i, i] = (1 / L[i, i] - Z[S, i]' * L[S, i]) / L[i, i].
    # Computed backwards from the last column, this only requires entries of `Z` within the
    # pattern of `L + L'` (Takahashi, Fagan & Chin, 1973; Erisman & Tinney, 1975).
    function _selinv(F::CHOLMOD.Factor)
        L = sparse(F.L)
        n = size(L, 1)
        colptr, rowval, lval = SparseArrays.getcolptr(L), rowvals(L), nonzeros(L)
        zval = similar(lval)
        # `pos[r]` is the index in `lval` of the entry in row `r` of the current column of `L`,
        # or 0 if there is none; `acc[r]` accumulates `(Z[S, S] * L[S, i])[r]`
        pos = zeros(Int, n)
        acc = zeros(eltype(zval), n)
        for i in n:-1:1
            # The diagonal is the first stored entry of each column of `L`
            diag_idx = colptr[i]
            d = real(lval[diag_idx])
            S = (diag_idx + 1):(colptr[i + 1] - 1)
            for k in S
                pos[rowval[k]] = k
                acc[rowval[k]] = zero(eltype(acc))
            end

            # Accumulate `acc[r] = Σ_{l ∈ S} Z[r, l] * L[l, i]` for all `r ∈ S`. Columns `l > i`
            # of `Z` are complete, and only their lower triangle is stored: entry `Z[r, l]`
            # with `r > l` also contributes `Z[l, r] = conj(Z[r, l])` to row `l`.
            for kl in S
                l = rowval[kl]
                for t in colptr[l]:(colptr[l + 1] - 1)
                    r = rowval[t]
                    kr = pos[r]
                    kr == 0 && continue
                    z = zval[t]
                    acc[r] += z * lval[kl]
                    r == l || (acc[l] += conj(z) * lval[kr])
                end
            end

            zii = inv(d)
            for k in S
                r = rowval[k]
                zri = -acc[r] / d
                zval[k] = zri
                zii -= real(conj(zri) * lval[k])
                pos[r] = 0
            end
            zval[diag_idx] = zii / d
        end
        return SparseMatrixCSC(n, n, colptr, rowval, zval)
    end

    # `inv(x)` on the sparsity pattern of `x`, for `F = cholesky(Hermitian(x, :L))`.
    # Since `x` is Hermitian, this is also `inv(x)'` on that pattern.
    function _inv_on_pattern(F::CHOLMOD.Factor, x::SparseMatrixCSC)
        Zl = _selinv(F)
        zrows, zval = rowvals(Zl), nonzeros(Zl)
        ip = invperm(F.p)
        xrows = rowvals(x)
        vals = similar(zval, nnz(x))
        # Columns of `x` with entries outside of the pattern of `L + L'`, e.g., explicitly stored
        # zeros in the strict upper triangle, which CHOLMOD ignores
        uncovered = Int[]
        for c in axes(x, 2), k in nzrange(x, c)
            i, j = ip[xrows[k]], ip[c]
            range = nzrange(Zl, min(i, j))
            t = first(range) - 1 + searchsortedfirst(view(zrows, range), max(i, j))
            if t <= last(range) && zrows[t] == max(i, j)
                vals[k] = i >= j ? zval[t] : conj(zval[t])
            elseif isempty(uncovered) || last(uncovered) != c
                push!(uncovered, c)
            end
        end
        for c in uncovered
            e = zeros(eltype(vals), size(x, 1))
            e[c] = 1
            invx_c = F \ e
            for k in nzrange(x, c)
                vals[k] = invx_c[xrows[k]]
            end
        end
        return SparseMatrixCSC(size(x)..., copy(SparseArrays.getcolptr(x)), copy(xrows), vals)
    end

    function rrule(::typeof(logabsdet), x::SparseMatrixCSC)
        F, success = _cholesky(x)
        Ω = success ? (real(logdet(F)), one(float(eltype(x)))) : logabsdet(x)
        function logabsdet_pullback(ΔΩ)
            _check_posdef(success)
            (Δy, Δsigny) = unthunk(ΔΩ)
            (_, signy) = Ω
            f = signy' * Δsigny
            imagf = f - real(f)
            g = real(Δy) + imagf
            ∂x = g * _inv_on_pattern(F, x)
            return (NoTangent(), ∂x)
        end
        return Ω, logabsdet_pullback
    end

    function rrule(::typeof(logdet), x::SparseMatrixCSC)
        F, success = _cholesky(x)
        Ω = success ? logdet(F) : logdet(x)
        function logdet_pullback(ΔΩ)
            _check_posdef(success)
            ∂x = unthunk(ΔΩ) * _inv_on_pattern(F, x)
            return (NoTangent(), ∂x)
        end
        return Ω, logdet_pullback
    end

    function rrule(::typeof(det), x::SparseMatrixCSC)
        F, success = _cholesky(x)
        Ω = success ? det(F) : det(x)
        function det_pullback(ΔΩ)
            _check_posdef(success)
            ∂x = _inv_on_pattern(F, x) * dot(Ω, unthunk(ΔΩ))
            return (NoTangent(), ∂x)
        end
        return Ω, det_pullback
    end

end # rrules that depend on CHOLMOD

function rrule(::typeof(spdiagm), m::Integer, n::Integer, kv::Pair{<:Integer,<:AbstractVector}...)

    function spdiagm_pullback(ȳ)
        return (NoTangent(), NoTangent(), NoTangent(), _diagm_back.(kv, Ref(ȳ))...)
    end
    return spdiagm(m, n, kv...), spdiagm_pullback
end

function rrule(::typeof(spdiagm), kv::Pair{<:Integer,<:AbstractVector}...)
    function spdiagm_pullback(ȳ)
        return (NoTangent(), _diagm_back.(kv, Ref(ȳ))...)
    end
    return spdiagm(kv...), spdiagm_pullback
end

function rrule(::typeof(spdiagm), v::AbstractVector)
    function spdiagm_pullback(ȳ)
        return (NoTangent(), diag(unthunk(ȳ)))
    end
    return spdiagm(v), spdiagm_pullback
end

end # module
