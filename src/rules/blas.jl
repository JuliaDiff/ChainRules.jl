#####
##### unit-stride `@rule`s
#####

@rule(BLAS.dot(x, y), (Bundle(y), Bundle(x)))
@rule(BLAS.nrm2(x), x * inv(Ω))
@rule(BLAS.asum(x), Bundle(broadcasted(sign, x)))

#####
##### arbitrary-stride rules
#####

function nrm2_X_adjoint(Ω̄, Ω, n, X, incx)
    X0 = fill!(similar(X), zero(eltype(X)))
    blascopy!(n, X, incx, X0, incx)
    a = materialize(mul(Ω̄, inv(Ω)))
    return BLAS.scal!(n, a, X0, incx)
end

function rrule(::typeof(BLAS.nrm2), n, X, incx)
    Ω = BLAS.nrm2(n, X, incx)
    return Ω, (@chain(DNE()),
               (X̄, Ω̄) -> add(X̄, nrm2_X_adjoint(Ω̄, Ω, n, X, incx)),
               @chain(DNE()))
end

function asum_X_adjoint(Ω̄, Ω, n, X, incx)
    X0 = fill!(similar(X), zero(eltype(X)))
    blascopy!(n, sign.(X), incx, X0, incx)
    a = materialize(Ω̄)
    return BLAS.scal!(n, a, X0, incx)
end

function rrule(::typeof(BLAS.asum), n, X, incx)
    Ω = BLAS.asum(n, X, incx)
    return Ω, (@chain(DNE()),
               (X̄, Ω̄) -> add(X̄, asum_X_adjoint(Ω̄, Ω, n, X, incx)),
               @chain(DNE()))
end

#####
##### custom rules
#####
