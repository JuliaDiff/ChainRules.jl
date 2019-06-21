#=
These implementations were ported from the wonderful DiffLinearAlgebra
package (https://github.com/invenia/DiffLinearAlgebra.jl).
=#

using LinearAlgebra: BlasFloat

_zeros(x) = fill!(similar(x), zero(eltype(x)))

_rule_via(∂) = Rule(ΔΩ -> isa(ΔΩ, Zero) ? ΔΩ : ∂(extern(ΔΩ)))

#####
##### `BLAS.dot`
#####

frule(::typeof(BLAS.dot), x, y) = frule(dot, x, y)

rrule(::typeof(BLAS.dot), x, y) = rrule(dot, x, y)

function rrule(::typeof(BLAS.dot), n, X, incx, Y, incy)
    Ω = BLAS.dot(n, X, incx, Y, incy)
    ∂X = ΔΩ -> scal!(n, ΔΩ, blascopy!(n, Y, incy, _zeros(X), incx), incx)
    ∂Y = ΔΩ -> scal!(n, ΔΩ, blascopy!(n, X, incx, _zeros(Y), incy), incy)
    return Ω, (DNERule(), _rule_via(∂X), DNERule(), _rule_via(∂Y), DNERule())
end

#####
##### `BLAS.nrm2`
#####

function frule(::typeof(BLAS.nrm2), x)
    Ω = BLAS.nrm2(x)
    return Ω, Rule(Δx -> sum(Δx * cast(@thunk(x * inv(Ω)))))
end

function rrule(::typeof(BLAS.nrm2), x)
    Ω = BLAS.nrm2(x)
    return Ω, Rule(ΔΩ -> ΔΩ * @thunk(x * inv(Ω)))
end

function rrule(::typeof(BLAS.nrm2), n, X, incx)
    Ω = BLAS.nrm2(n, X, incx)
    ∂X = ΔΩ -> scal!(n, ΔΩ / Ω, blascopy!(n, X, incx, _zeros(X), incx), incx)
    return Ω, (DNERule(), _rule_via(∂X), DNERule())
end

#####
##### `BLAS.asum`
#####

frule(::typeof(BLAS.asum), x) = (BLAS.asum(x), Rule(Δx -> sum(cast(sign, x) * Δx)))

rrule(::typeof(BLAS.asum), x) = (BLAS.asum(x), Rule(ΔΩ -> ΔΩ * cast(sign, x)))

function rrule(::typeof(BLAS.asum), n, X, incx)
    Ω = BLAS.asum(n, X, incx)
    ∂X = ΔΩ -> scal!(n, ΔΩ, blascopy!(n, sign.(X), incx, _zeros(X), incx), incx)
    return Ω, (DNERule(), _rule_via(∂X), DNERule())
end

#####
##### `BLAS.gemv`
#####

function rrule(::typeof(gemv), tA::Char, α::T, A::AbstractMatrix{T},
               x::AbstractVector{T}) where T<:BlasFloat
    y = gemv(tA, α, A, x)
    if uppercase(tA) === 'N'
        ∂A = Rule(ȳ -> α * ȳ * x', (Ā, ȳ) -> ger!(α, ȳ, x, Ā))
        ∂x = Rule(ȳ -> gemv('T', α, A, ȳ), (x̄, ȳ) -> gemv!('T', α, A, ȳ, one(T), x̄))
    else
        ∂A = Rule(ȳ -> α * x * ȳ', (Ā, ȳ) -> ger!(α, x, ȳ, Ā))
        ∂x = Rule(ȳ -> gemv('N', α, A, ȳ), (x̄, ȳ) -> gemv!('N', α, A, ȳ, one(T), x̄))
    end
    return y, (DNERule(), Rule(ȳ -> dot(ȳ, y) / α), ∂A, ∂x)
end

function rrule(::typeof(gemv), tA::Char, A::AbstractMatrix{T},
               x::AbstractVector{T}) where T<:BlasFloat
    y, (dtA, _, dA, dx) = rrule(gemv, tA, one(T), A, x)
    return y, (dtA, dA, dx)
end

#####
##### `BLAS.gemm`
#####

function rrule(::typeof(gemm), tA::Char, tB::Char, α::T,
               A::AbstractMatrix{T}, B::AbstractMatrix{T}) where T<:BlasFloat
    C = gemm(tA, tB, α, A, B)
    β = one(T)
    if uppercase(tA) === 'N'
        if uppercase(tB) === 'N'
            ∂A = Rule(C̄ -> gemm('N', 'T', α, C̄, B),
                      (Ā, C̄) -> gemm!('N', 'T', α, C̄, B, β, Ā))
            ∂B = Rule(C̄ -> gemm('T', 'N', α, A, C̄),
                      (B̄, C̄) -> gemm!('T', 'N', α, A, C̄, β, B̄))
        else
            ∂A = Rule(C̄ -> gemm('N', 'N', α, C̄, B),
                      (Ā, C̄) -> gemm!('N', 'N', α, C̄, B, β, Ā))
            ∂B = Rule(C̄ -> gemm('T', 'N', α, C̄, A),
                      (B̄, C̄) -> gemm!('T', 'N', α, C̄, A, β, B̄))
        end
    else
        if uppercase(tB) === 'N'
            ∂A = Rule(C̄ -> gemm('N', 'T', α, B, C̄),
                      (Ā, C̄) -> gemm!('N', 'T', α, B, C̄, β, Ā))
            ∂B = Rule(C̄ -> gemm('N', 'N', α, A, C̄),
                      (B̄, C̄) -> gemm!('N', 'N', α, A, C̄, β, B̄))
        else
            ∂A = Rule(C̄ -> gemm('T', 'T', α, B, C̄),
                      (Ā, C̄) -> gemm!('T', 'T', α, B, C̄, β, Ā))
            ∂B = Rule(C̄ -> gemm('T', 'T', α, C̄, A),
                      (B̄, C̄) -> gemm!('T', 'T', α, C̄, A, β, B̄))
        end
    end
    return C, (DNERule(), DNERule(), Rule(C̄ -> dot(C̄, C) / α), ∂A, ∂B)
end

function rrule(::typeof(gemm), tA::Char, tB::Char,
               A::AbstractMatrix{T}, B::AbstractMatrix{T}) where T<:BlasFloat
    C, (dtA, dtB, _, dA, dB) = rrule(gemm, tA, tB, one(T), A, B)
    return C, (dtA, dtB, dA, dB)
end
