#=
These implementations were ported from the wonderful DiffLinearAlgebra
package (https://github.com/invenia/DiffLinearAlgebra.jl).
=#

using LinearAlgebra: BlasFloat

_zeros(x) = fill!(similar(x), zero(eltype(x)))

#####
##### `BLAS.dot`
#####

frule((Δself, Δx, Δy), ::typeof(BLAS.dot), x, y) = frule((Δself, Δx, Δy), dot, x, y)

rrule(::typeof(BLAS.dot), x, y) = rrule(dot, x, y)

function rrule(::typeof(BLAS.dot), n, X, incx, Y, incy)
    Ω = BLAS.dot(n, X, incx, Y, incy)
    function blas_dot_pullback(ΔΩ)
        if ΔΩ isa Zero
            ∂X = Zero()
            ∂Y = Zero()
        else
            ΔΩ = extern(ΔΩ)
            ∂X = @thunk scal!(n, ΔΩ, blascopy!(n, Y, incy, _zeros(X), incx), incx)
            ∂Y = @thunk scal!(n, ΔΩ, blascopy!(n, X, incx, _zeros(Y), incy), incy)
        end
        return (NO_FIELDS, DoesNotExist(), ∂X, DoesNotExist(), ∂Y, DoesNotExist())
    end
    return Ω, blas_dot_pullback
end

#####
##### `BLAS.nrm2`
#####

function frule((_, Δx), ::typeof(BLAS.nrm2), x)
    Ω = BLAS.nrm2(x)
    s = ifelse(iszero(Ω), one(Ω), Ω)
    ∂Ω = if x isa Real
        BLAS.dot(x, Δx) / s
    else
        sum(y -> _realconjtimes(y...), zip(x, Δx)) / s
    end
    return Ω, ∂Ω
end

function rrule(::typeof(BLAS.nrm2), x)
    Ω = BLAS.nrm2(x)
    function nrm2_pullback(ΔΩ)
        return NO_FIELDS, x .* (real(ΔΩ) / ifelse(iszero(Ω), one(Ω), Ω))
    end
    return Ω, nrm2_pullback
end

function rrule(::typeof(BLAS.nrm2), n, X, incx)
    Ω = BLAS.nrm2(n, X, incx)
    nrm2_pullback(::Zero) = (NO_FIELDS, DoesNotExist(), Zero(), DoesNotExist())
    function nrm2_pullback(ΔΩ)
        # BLAS.scal! requires s has the same eltype as X
        s = eltype(X)(real(ΔΩ) / ifelse(iszero(Ω), one(Ω), Ω))
        ∂X = scal!(n, s, blascopy!(n, X, incx, _zeros(X), incx), incx)
        return (NO_FIELDS, DoesNotExist(), ∂X, DoesNotExist())
    end
    return Ω, nrm2_pullback
end

#####
##### `BLAS.asum`
#####

function frule((_, Δx), ::typeof(BLAS.asum), x)
    ∂Ω = sum(zip(x, Δx)) do (xi, Δxi)
        return _realconjtimes(_signcomp(xi), Δxi)
    end
    return BLAS.asum(x), ∂Ω
end

function rrule(::typeof(BLAS.asum), x)
    function asum_pullback(ΔΩ)
        return (NO_FIELDS, _signcomp.(x) .* real(ΔΩ))
    end
    return BLAS.asum(x), asum_pullback
end

function rrule(::typeof(BLAS.asum), n, X, incx)
    Ω = BLAS.asum(n, X, incx)
    asum_pullback(::Zero) = (NO_FIELDS, DoesNotExist(), Zero(), DoesNotExist())
    function asum_pullback(ΔΩ)
        # BLAS.scal! requires s has the same eltype as X
        s = eltype(X)(real(ΔΩ))
        ∂X = scal!(n, s, blascopy!(n, _signcomp.(X), incx, _zeros(X), incx), incx)
        return (NO_FIELDS, DoesNotExist(), ∂X, DoesNotExist())
    end
    return Ω, asum_pullback
end

# component-wise sign, e.g. sign(x) + i sign(y)
@inline _signcomp(x::Real) = sign(x)
@inline _signcomp(x::Complex) = Complex(sign(real(x)), sign(imag(x)))

#####
##### `BLAS.gemv`
#####

function rrule(::typeof(gemv), tA::Char, α::T, A::AbstractMatrix{T},
               x::AbstractVector{T}) where T<:BlasFloat
    y = gemv(tA, α, A, x)
    function gemv_pullback(ȳ)
        if uppercase(tA) === 'N'
            ∂A = InplaceableThunk(
                @thunk(α' * ȳ * x'),
                Ā -> ger!(α', ȳ, x, Ā)
            )
            ∂x = InplaceableThunk(
                @thunk(gemv('C', α', A, ȳ)),
                x̄ -> gemv!('C', α', A, ȳ, one(T), x̄)
            )
        elseif uppercase(tA) === 'C'
            ∂A = InplaceableThunk(
                @thunk(α * x * ȳ'),
                Ā -> ger!(α, x, ȳ, Ā)
            )
            ∂x = InplaceableThunk(
                @thunk(gemv('N', α', A, ȳ)),
                x̄ -> gemv!('N', α', A, ȳ, one(T), x̄)
            )
        else  # uppercase(tA) === 'T'
            ∂A = InplaceableThunk(
                @thunk(conj(α * x * ȳ')),
                Ā -> conj!(ger!(α, x, ȳ, Ā))
            )
            ∂x = InplaceableThunk(
                @thunk(gemv('N', α', conj(A), ȳ)),
                x̄ -> gemv!('N', α', conj(A), ȳ, one(T), x̄)
            )
        end
        return (NO_FIELDS, DoesNotExist(), @thunk(dot(y, ȳ) / α'), ∂A, ∂x)
    end
    return y, gemv_pullback
end

function rrule(
    ::typeof(gemv), tA::Char, A::AbstractMatrix{T}, x::AbstractVector{T}
) where T<:BlasFloat
    y, inner_pullback = rrule(gemv, tA, one(T), A, x)
    function gemv_pullback(Ȳ)
        (_, dtA, _, dA, dx) = inner_pullback(Ȳ)
        return (NO_FIELDS, dtA, dA, dx)
    end
    return y, gemv_pullback
end

#####
##### `BLAS.gemm`
#####

function rrule(
    ::typeof(gemm), tA::Char, tB::Char, α::T, A::AbstractMatrix{T}, B::AbstractMatrix{T}
) where T<:BlasFloat
    C = gemm(tA, tB, α, A, B)
    function gemv_pullback(C̄)
        β = one(T)
        if uppercase(tA) === 'N'
            if uppercase(tB) === 'N'
                ∂A = InplaceableThunk(
                    @thunk(gemm('N', 'C', α', C̄, B)),
                    Ā -> gemm!('N', 'C', α', C̄, B, β, Ā)
                )
                ∂B = InplaceableThunk(
                    @thunk(gemm('C', 'N', α', A, C̄)),
                    B̄ -> gemm!('C', 'N', α', A, C̄, β, B̄)
                )
            elseif uppercase(tB) === 'C'
                ∂A = InplaceableThunk(
                    @thunk(gemm('N', 'N', α', C̄, B)),
                    Ā -> gemm!('N', 'N', α', C̄, B, β, Ā)
                )
                ∂B = InplaceableThunk(
                    @thunk(gemm('C', 'N', α, C̄, A)),
                    B̄ -> gemm!('C', 'N', α, C̄, A, β, B̄)
                )
            else  # uppercase(tB) === 'T'
                ∂A = InplaceableThunk(
                    @thunk(gemm('N', 'N', α', C̄, conj(B))),
                    Ā -> gemm!('N', 'N', α', C̄, conj(B), β, Ā)
                )
                ∂B = InplaceableThunk(
                    @thunk(conj(gemm('C', 'N', α, C̄, A))),
                    B̄ -> conj!(gemm!('C', 'N', α, C̄, A, β, B̄))
                )
            end
        elseif uppercase(tA) === 'C'
            if uppercase(tB) === 'N'
                ∂A = InplaceableThunk(
                    @thunk(gemm('N', 'C', α, B, C̄)),
                    Ā -> gemm!('N', 'C', α, B, C̄, β, Ā)
                )
                ∂B = InplaceableThunk(
                    @thunk(gemm('N', 'N', α', A, C̄)),
                    B̄ -> gemm!('N', 'N', α', A, C̄, β, B̄)
                )
            elseif uppercase(tB) === 'C'
                ∂A = InplaceableThunk(
                    @thunk(gemm('C', 'C', α, B, C̄)),
                    Ā -> gemm!('C', 'C', α, B, C̄, β, Ā)
                )
                ∂B = InplaceableThunk(
                    @thunk(gemm('C', 'C', α, C̄, A)),
                    B̄ -> gemm!('C', 'C', α, C̄, A, β, B̄)
                )
            else  # uppercase(tB) === 'T'
                ∂A = InplaceableThunk(
                    @thunk(gemm('T', 'C', α, B, C̄)),
                    Ā -> gemm!('T', 'C', α, B, C̄, β, Ā)
                )
                ∂B = InplaceableThunk(
                    @thunk(gemm('T', 'T', α', C̄, A)),
                    B̄ -> gemm!('T', 'T', α', C̄, A, β, B̄)
                )
            end
        else  # uppercase(tA) === 'T'
            if uppercase(tB) === 'N'
                ∂A = InplaceableThunk(
                    @thunk(conj(gemm('N', 'C', α, B, C̄))),
                    Ā -> conj!(gemm!('N', 'C', α, B, C̄, β, Ā))
                )
                ∂B = InplaceableThunk(
                    @thunk(gemm('N', 'N', α', conj(A), C̄)),
                    B̄ -> gemm!('N', 'N', α', conj(A), C̄, β, B̄)
                )
            elseif uppercase(tB) === 'C'
                ∂A = InplaceableThunk(
                    @thunk(gemm('T', 'T', α', B, C̄)),
                    Ā -> gemm!('T', 'T', α', B, C̄, β, Ā)
                )
                ∂B = InplaceableThunk(
                    @thunk(gemm('C', 'T', α, C̄, A)),
                    B̄ -> gemm!('C', 'T', α, C̄, A, β, B̄)
                )
            else  # uppercase(tB) === 'T'
                ∂A = InplaceableThunk(
                    @thunk(gemm('C', 'T', α', B, C̄)),
                    Ā -> gemm!('C', 'T', α', B, C̄, β, Ā)
                )
                ∂B = InplaceableThunk(
                    @thunk(gemm('T', 'C', α', C̄, A)),
                    B̄ -> gemm!('T', 'C', α', C̄, A, β, B̄)
                )
            end
        end
        return (NO_FIELDS, DoesNotExist(), DoesNotExist(), @thunk(dot(C, C̄) / α'), ∂A, ∂B)
    end
    return C, gemv_pullback
end

function rrule(
    ::typeof(gemm), tA::Char, tB::Char, A::AbstractMatrix{T}, B::AbstractMatrix{T}
) where T<:BlasFloat
    C, inner_pullback = rrule(gemm, tA, tB, one(T), A, B)
    function gemv_pullback(Ȳ)
        (_, dtA, dtB, _, dA, dB) = inner_pullback(Ȳ)
        return (NO_FIELDS, dtA, dtB, dA, dB)
    end
    return C, gemm_pullback
end
