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

function rrule(::typeof(BLAS.gemv), tA, α, A, x)
    Ω = BLAS.gemv(tA, α, A, x)
    ∂α = ΔΩ -> dot(ΔΩ, Ω) / α
    ∂A = ΔΩ -> uppercase(tA) == 'N' ? α * ΔΩ * x' : α * x * ΔΩ'
    ∂x = ΔΩ -> gemv(uppercase(tA) == 'N' ? 'T' : 'N', α, A, ΔΩ)
    return Ω, (DNERule(), _rule_via(∂α), _rule_via(∂A), _rule_via(∂x))
end

function rrule(f::typeof(BLAS.gemv), tA, A, x)
    Ω, (dtA, dα, dA, dx) = rrule(f, tA, one(eltype(A)), A, x)
    return Ω, (dtA, dA, dx)
end

#####
##### `BLAS.gemm`
#####

function rrule(::typeof(gemm), tA::Char, tB::Char, α::T,
               A::AbstractMatrix{T}, B::AbstractMatrix{T}) where T<:BlasFloat
    C = gemm(tA, tB, α, A, B)
    ∂α = C̄ -> sum(C̄ .* C) / α
    if uppercase(tA) === 'N'
        if uppercase(tB) === 'N'
            ∂A = C̄ -> gemm('N', 'T', α, C̄, B)
            ∂B = C̄ -> gemm('T', 'N', α, A, C̄)
        else
            ∂A = C̄ -> gemm('N', 'N', α, C̄, B)
            ∂B = C̄ -> gemm('T', 'N', α, C̄, A)
        end
    else
        if uppercase(tB) === 'N'
            ∂A = C̄ -> gemm('N', 'T', α, B, C̄)
            ∂B = C̄ -> gemm('N', 'N', α, A, C̄)
        else
            ∂A = C̄ -> gemm('T', 'T', α, B, C̄)
            ∂B = C̄ -> gemm('T', 'T', α, C̄, A)
        end
    end
    return C, (DNERule(), DNERule(), _rule_via(∂α), _rule_via(∂A), _rule_via(∂B))
end

function rrule(::typeof(gemm), tA::Char, tB::Char,
               A::AbstractMatrix{T}, B::AbstractMatrix{T}) where T<:BlasFloat
    C, (dtA, dtB, _, dA, dB) = rrule(gemm, tA, tB, one(T), A, B)
    return C, (dtA, dtB, dA, dB)
end
