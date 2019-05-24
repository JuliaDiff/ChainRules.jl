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
        return F.U, (Rule(Ȳ->(U=Ȳ, S=zero(F.S), V=zero(F.V))), DNERule())
    elseif x === :S
        return F.S, (Rule(Ȳ->(U=zero(F.U), S=Ȳ, V=zero(F.V))), DNERule())
    elseif x === :V
        return F.V, (Rule(Ȳ->(U=zero(F.U), S=zero(F.S), V=Ȳ)), DNERule())
    elseif x === :Vt
        return F.Vt, (Rule(Ȳ->(U=zero(F.U), S=zero(F.S), V=Ȳ')), DNERule())
    end
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

function _mulsubtrans!(X::AbstractMatrix{T}, F::AbstractMatrix{T}) where T<:Real
    k = size(X, 1)
    @inbounds for j = 1:k, i = 1:j  # Iterate the upper triangle
        if i == j
            X[i,i] = zero(T)
        else
            X[i,j], X[j,i] = F[i,j] * (X[i,j] - X[j,i]), F[j,i] * (X[j,i] - X[i,j])
        end
    end
    X
end

function _eyesubx!(X::AbstractMatrix{T}) where T<:Real
    n, m = size(X)
    @inbounds for j = 1:m, i = 1:n
        X[i,j] = (i == j) - X[i,j]
    end
    X
end

function _add!(X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where T<:Real
    @inbounds for i = eachindex(X, Y)
        X[i] += Y[i]
    end
    X
end
