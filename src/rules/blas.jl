#####
##### `@rule`s
#####

@rule(BLAS.dot(x, y), (cast(y), cast(x)))
@rule(BLAS.nrm2(x), mul(x, inv(Ω)))
@rule(BLAS.asum(x), cast(sign, x))

#####
##### custom rules
#####
#=
These implementations were ported from the wonderful DiffLinearAlgebra
package (https://github.com/invenia/DiffLinearAlgebra.jl).
=#

#=
TODO: Various chain implementations below could check if their input adjoint
`isa` `MaterializeInto` and subsequently perform the relevant in-place
optimizations.
=#

_zeros(x) = fill!(similar(x), zero(eltype(x)))

_chain_via(∂) = (δ̄, Ω̄) -> add(δ̄, isa(Ω̄, Zero) ? Ω̄ : ∂(materialize(Ω̄)))

function rrule(::typeof(BLAS.dot), n, X, incx, Y, incy)
    Ω = BLAS.dot(n, X, incx, Y, incy)
    ∂X = Ω̄ -> scal!(n, Ω̄, blascopy!(n, Y, incy, _zeros(X), incx), incx)
    ∂Y = Ω̄ -> scal!(n, Ω̄, blascopy!(n, X, incx, _zeros(Y), incy), incy)
    return Ω, (@chain(DNE()), _chain_via(∂X), @chain(DNE()), _chain_via(∂Y), @chain(DNE()))
end

function rrule(::typeof(BLAS.nrm2), n, X, incx)
    Ω = BLAS.nrm2(n, X, incx)
    ∂X = Ω̄ -> scal!(n, Ω̄ / Ω, blascopy!(n, X, incx, _zeros(X), incx), incx)
    return Ω, (@chain(DNE()), _chain_via(∂X), @chain(DNE()))
end

function rrule(::typeof(BLAS.asum), n, X, incx)
    Ω = BLAS.asum(n, X, incx)
    ∂X = Ω̄ -> scal!(n, Ω̄, blascopy!(n, sign.(X), incx, _zeros(X), incx), incx)
    return Ω, (@chain(DNE()), _chain_via(∂X), @chain(DNE()))
end

function rrule(::typeof(BLAS.gemv), tA, α, A, x)
    Ω = BLAS.gemv(tA, α, A, x)
    ∂α = Ω̄ -> dot(Ω̄, Ω) / α
    ∂A = Ω̄ -> uppercase(tA) == 'N' ? α * Ω̄ * x' : α * x * Ω̄'
    ∂x = Ω̄ -> gemv(uppercase(tA) == 'N' ? 'T' : 'N', α, A, Ω̄)
    return Ω, (@chain(DNE()), _chain_via(∂α), _chain_via(∂A), _chain_via(∂x))
end

function rrule(f::typeof(BLAS.gemv), tA, A, x)
    Ω, (dtA, dα, dA, dx) = rrule(f, tA, one(eltype(A)), A, x)
    return Ω, (dtA, dA, dx)
end
