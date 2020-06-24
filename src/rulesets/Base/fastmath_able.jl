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
        ## abs
        function frule((_, Δx), ::typeof(abs), x::Real)
            return abs(x), sign(x) * real(Δx)
        end
        function frule((_, Δz), ::typeof(abs), z::Complex)
            Ω = abs(z)
            return Ω, (real(z) * real(Δz) + imag(z) * imag(Δz)) / Ω
        end
        
        function rrule(::typeof(abs), x::Real)
            function abs_pullback(Δf)
                return (NO_FIELDS, real(Δf)*sign(x))
            end
            return abs(x), abs_pullback
        end
        function rrule(::typeof(abs), z::Complex)
            Ω = abs(z)
            function abs_pullback(Δf)
                Δu = real(Δf) 
                return (NO_FIELDS, Δu*z/Ω)
            end
            return Ω, abs_pullback
        end

        ## abs2
        function frule((_, Δx), ::typeof(abs2), x::Real)
            return abs2(x), 2x * real(Δx)
        end
        function frule((_, Δz), ::typeof(abs2), z::Complex)
            return abs2(z), 2 * (real(z) * real(Δz) + imag(z) * imag(Δz))
        end
        
        function rrule(::typeof(abs2), x::Real)
            function abs2_pullback(Δx)
                return (NO_FIELDS, 2real(Δx)*x)
            end
            return abs2(x), abs2_pullback
        end
        function rrule(::typeof(abs2), z::Complex)
            function abs2_pullback(Δf)
                Δu = real(Δf)
                return (NO_FIELDS, 2real(Δu)*z)
            end
            return abs2(z), abs2_pullback
        end

        ## conj
        function frule((_, Δz), ::typeof(conj), z::Union{Real, Complex})
            return conj(z), conj(Δz) 
        end
        function rrule(::typeof(conj), z::Union{Real, Complex})
            function conj_pullback(Δf)
                return (NO_FIELDS, conj(Δf))
            end
            return conj(z), conj_pullback
        end

        ## angle
        function frule((_, Δz), ::typeof(angle), x::Real)
            Δx, Δy = reim(Δz)
            return angle(x), Δy/x  
        end
        function frule((_, Δz), ::typeof(angle), z::Complex)
            x,  y  = reim(z)
            Δx, Δy = reim(Δz)
            return angle(z), (-y*Δx + x*Δy)/abs2(z)  
        end
        function rrule(::typeof(angle), x::Real)
            function angle_pullback(Δf)
                Δu, Δv = reim(Δf)
                return (NO_FIELDS, im*Δu/x)
            end
            return angle(x), angle_pullback 
        end
        function rrule(::typeof(angle), z::Complex)
            function angle_pullback(Δf)
                x,  y  = reim(z)
                Δu, Δv = reim(Δf)
                return (NO_FIELDS, (-y + im*x)*Δu/abs2(z))
            end
            return angle(z), angle_pullback 
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
