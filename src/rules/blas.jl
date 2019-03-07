#=
These implementations were ported from the wonderful DiffLinearAlgebra
package (https://github.com/invenia/DiffLinearAlgebra.jl).
=#

_zeros(x) = fill!(similar(x), zero(eltype(x)))

_chain_via(∂) = Chain(ΔΩ -> isa(ΔΩ, Zero) ? ΔΩ : ∂(extern(ΔΩ)))

#####
##### `BLAS.dot`
#####

frule(::typeof(BLAS.dot), x, y) = frule(dot, x, y)

rrule(::typeof(BLAS.dot), x, y) = rrule(dot, x, y)

function rrule(::typeof(BLAS.dot), n, X, incx, Y, incy)
    Ω = BLAS.dot(n, X, incx, Y, incy)
    ∂X = ΔΩ -> scal!(n, ΔΩ, blascopy!(n, Y, incy, _zeros(X), incx), incx)
    ∂Y = ΔΩ -> scal!(n, ΔΩ, blascopy!(n, X, incx, _zeros(Y), incy), incy)
    return Ω, (DNEChain(), _chain_via(∂X), DNEChain(), _chain_via(∂Y), DNEChain())
end

#####
##### `BLAS.nrm2`
#####

function frule(::typeof(BLAS.nrm2), x)
    Ω = BLAS.nrm2(x)
    return Ω, Chain(Δx -> sum(Δx * cast(@thunk(x * inv(Ω)))))
end

function rrule(::typeof(BLAS.nrm2), x)
    Ω = BLAS.nrm2(x)
    return Ω, Chain(ΔΩ -> ΔΩ * @thunk(x * inv(Ω)))
end

function rrule(::typeof(BLAS.nrm2), n, X, incx)
    Ω = BLAS.nrm2(n, X, incx)
    ∂X = ΔΩ -> scal!(n, ΔΩ / Ω, blascopy!(n, X, incx, _zeros(X), incx), incx)
    return Ω, (DNEChain(), _chain_via(∂X), DNEChain())
end

#####
##### `BLAS.asum`
#####

frule(::typeof(BLAS.asum), x) = (BLAS.asum(x), Chain(Δx -> sum(cast(sign, x) * Δx)))

rrule(::typeof(BLAS.asum), x) = (BLAS.asum(x), Chain(ΔΩ -> ΔΩ * cast(sign, x)))

function rrule(::typeof(BLAS.asum), n, X, incx)
    Ω = BLAS.asum(n, X, incx)
    ∂X = ΔΩ -> scal!(n, ΔΩ, blascopy!(n, sign.(X), incx, _zeros(X), incx), incx)
    return Ω, (DNEChain(), _chain_via(∂X), DNEChain())
end

#####
##### `BLAS.gemv`
#####

function rrule(::typeof(BLAS.gemv), tA, α, A, x)
    Ω = BLAS.gemv(tA, α, A, x)
    ∂α = ΔΩ -> dot(ΔΩ, Ω) / α
    ∂A = ΔΩ -> uppercase(tA) == 'N' ? α * ΔΩ * x' : α * x * ΔΩ'
    ∂x = ΔΩ -> gemv(uppercase(tA) == 'N' ? 'T' : 'N', α, A, ΔΩ)
    return Ω, (DNEChain(), _chain_via(∂α), _chain_via(∂A), _chain_via(∂x))
end

function rrule(f::typeof(BLAS.gemv), tA, A, x)
    Ω, (dtA, dα, dA, dx) = rrule(f, tA, one(eltype(A)), A, x)
    return Ω, (dtA, dA, dx)
end
