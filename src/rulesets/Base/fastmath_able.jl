let
    # Include inside this quote any rules that should have FastMath versions
    fastable_ast = quote
        #  Trig-Basics
        @scalar_rule cos(x) -(sin(x))
        @scalar_rule sin(x) cos(x)
        @scalar_rule tan(x) 1 + Ω ^ 2


        # Trig-Hyperbolic
        @scalar_rule cosh(x) sinh(x)
        @scalar_rule sinh(x) cosh(x)
        @scalar_rule tanh(x) 1 - Ω ^ 2

        # Trig- Inverses
        @scalar_rule acos(x) -(inv(sqrt(1 - x ^ 2)))
        @scalar_rule asin(x) inv(sqrt(1 - x ^ 2))
        @scalar_rule atan(x) inv(1 + x ^ 2)

        # Trig-Multivariate
        @scalar_rule atan(y, x) @setup(u = x ^ 2 + y ^ 2) (x / u, -y / u)
        @scalar_rule sincos(x) @setup((sinx, cosx) = Ω) cosx -sinx

        # exponents
        @scalar_rule cbrt(x) inv(3 * Ω ^ 2)
        @scalar_rule inv(x) -(Ω ^ 2)
        @scalar_rule sqrt(x) inv(2Ω)
        @scalar_rule exp(x::Real) Ω
        @scalar_rule exp10(x) Ω * log(oftype(x, 10))
        @scalar_rule exp2(x) Ω * log(oftype(x, 2))
        @scalar_rule expm1(x) exp(x)
        @scalar_rule log(x) inv(x)
        @scalar_rule log10(x) inv(x) / log(oftype(x, 10))
        @scalar_rule log1p(x) inv(x + 1)
        @scalar_rule log2(x) inv(x) / log(oftype(x, 2))


        # Unary complex functions
        @scalar_rule abs(x::Real) sign(x)
        @scalar_rule abs2(x::Real) 2x
        @scalar_rule angle(x::Real) Zero()
        @scalar_rule conj(x::Real) One()
        function frule((Δx,), abs, x::ComplexF64)
            Ω = abs(x)
            return Ω, (real(x) * real(Δx) + imag(x) * imag(Δx)) / Ω
        end
        function frule((Δx,), abs2, x::ComplexF64)
            return abs2(x), 2 * (real(x) * real(Δx) + imag(x) * imag(Δx))
        end        

        # Binary functions
        @scalar_rule hypot(x::Real, y::Real) (x / Ω, y / Ω)
        @scalar_rule x + y (One(), One())
        @scalar_rule x - y (One(), -1)
        @scalar_rule x / y (inv(y), -((x / y) / y))
        #log(complex(x)) is require so it give correct complex answer for x<0
        @scalar_rule(x ^ y,
            (ifelse(iszero(y), zero(Ω), y * x ^ (y - 1)), Ω * log(complex(x))),
        )
        @scalar_rule(
            rem(x, y),
            @setup((u, nan) = promote(x / y, NaN16), isint = isinteger(x / y)),
            (ifelse(isint, nan, one(u)), ifelse(isint, nan, -trunc(u))),
        )
        @scalar_rule max(x, y) @setup(gt = x > y) (gt, !gt)
        @scalar_rule min(x, y) @setup(gt = x > y) (!gt, gt)

        # Unary functions
        @scalar_rule +x One()
        @scalar_rule -x -1


        @scalar_rule sign(x) Zero()


        # product rule requires special care for arguments where `mul` is non-commutative
        function frule((_, Δx, Δy), ::typeof(*), x::Number, y::Number)
            # Optimized version of `Δx .* y .+ x .* Δy`. Also, it is potentially more
            # accurate on machines with FMA instructions, since there are only two
            # rounding operations, one in `muladd/fma` and the other in `*`.
            ∂xy = muladd.(Δx, y, x .* Δy)
            return x * y, ∂xy
        end

        function rrule(::typeof(*), x::Number, y::Number)
            function times_pullback(ΔΩ)
                return (NO_FIELDS,  @thunk(ΔΩ * y), @thunk(x * ΔΩ))
            end
            return x * y, times_pullback
        end
    end

    # Rewrite everything to use fast_math functions, including the type-constraints
    eval(Base.FastMath.make_fastmath(fastable_ast))
    eval(fastable_ast)  # Get original definitions
    # we do this second so it overwrites anything we included by mistake in the fastable
end
