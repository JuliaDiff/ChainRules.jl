#####
##### unit-stride rules
#####

@rule(BLAS.dot(x, y), (Cast(y), Cast(x)))
@rule(BLAS.nrm2(x), x * inv(Ω))
@rule(BLAS.asum(x), Cast(broadcasted(sign, x)))

#####
##### arbitrary-stride rules
#####

_zeros(x) = fill!(similar(x), zero(eltype(x)))

function rrule(::typeof(BLAS.dot), n, X, incx, Y, incy)
    Ω = BLAS.dot(n, X, incx, Y, incy)
    X_partial = Ω̄ -> scal!(n, Ω̄, blascopy!(n, Y, incy, _zeros(X), incx), incx)
    X_chain = (X̄, Ω̄) -> add(X̄, isa(Ω̄, Zero) ? Ω̄ : X_partial(materialize(Ω̄)))
    Y_partial = Ω̄ -> scal!(n, Ω̄, blascopy!(n, X, incx, _zeros(Y), incy), incy)
    Y_chain = (X̄, Ω̄) -> add(X̄, isa(Ω̄, Zero) ? Ω̄ : Y_partial(materialize(Ω̄)))
    return Ω, (@chain(DNE()), X_chain, @chain(DNE()), Y_chain, @chain(DNE()))
end

function rrule(::typeof(BLAS.nrm2), n, X, incx)
    Ω = BLAS.nrm2(n, X, incx)
    X_partial = Ω̄ -> scal!(n, Ω̄ / Ω, blascopy!(n, X, incx, _zeros(X), incx), incx)
    X_chain = (X̄, Ω̄) -> add(X̄, isa(Ω̄, Zero) ? Ω̄ : X_partial(materialize(Ω̄)))
    return Ω, (@chain(DNE()), X_chain, @chain(DNE()))
end

function rrule(::typeof(BLAS.asum), n, X, incx)
    Ω = BLAS.asum(n, X, incx)
    X_partial = Ω̄ -> scal!(n, Ω̄, blascopy!(n, sign.(X), incx, _zeros(X), incx), incx)
    X_chain = (X̄, Ω̄) -> add(X̄, isa(Ω̄, Zero) ? Ω̄ : X_partial(materialize(Ω̄)))
    return Ω, (@chain(DNE()), X_chain, @chain(DNE()))
end
