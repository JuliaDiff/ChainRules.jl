let
    # Include inside this quote any rules that should have FastMath versions
    # IMPORTANT:
    # Do not add any rules here for functions that do not have varients in Base.FastMath
    # e.g. do not add `foo` unless `Base.FastMath.foo_fast` exists.
    fastable_ast = quote
        #  Trig-Basics
        ## use `sincos` to compute `sin` and `cos` at the same time
        ## for the rules for `sin` and `cos`
        ## See issue: https://github.com/JuliaDiff/ChainRules.jl/issues/291
        ## sin
        function rrule(::typeof(sin), x::Number)
            sinx, cosx = sincos(x)
            sin_pullback(Δy) = (NO_FIELDS, cosx' * Δy)
            return (sinx, sin_pullback)
        end

        function frule((_, Δx), ::typeof(sin), x::Number)
            sinx, cosx = sincos(x)
            return (sinx, cosx * Δx)
        end

        ## cos
        function rrule(::typeof(cos), x::Number)
            sinx, cosx = sincos(x)
            cos_pullback(Δy) = (NO_FIELDS, -sinx' * Δy)
            return (cosx, cos_pullback)
        end
        
        function frule((_, Δx), ::typeof(cos), x::Number)
            sinx, cosx = sincos(x)
            return (cosx, -sinx * Δx)
        end
        
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
        @scalar_rule exp(x) Ω
        @scalar_rule exp10(x) Ω * log(oftype(x, 10))
        @scalar_rule exp2(x) Ω * log(oftype(x, 2))
        @scalar_rule expm1(x) exp(x)
        @scalar_rule log(x) inv(x)
        @scalar_rule log10(x) inv(x) / log(oftype(x, 10))
        @scalar_rule log1p(x) inv(x + 1)
        @scalar_rule log2(x) inv(x) / log(oftype(x, 2))

        # Unary complex functions
        ## abs
        function frule((_, Δx), ::typeof(abs), x::Union{Real, Complex})
            Ω = abs(x)
            # `ifelse` is applied only to denominator to ensure type-stability.
            signx = x isa Real ? sign(x) : x / ifelse(iszero(x), one(Ω), Ω)
            return Ω, _realconjtimes(signx, Δx)
        end

        function rrule(::typeof(abs), x::Union{Real, Complex})
            Ω = abs(x)
            function abs_pullback(ΔΩ)
                signx = x isa Real ? sign(x) : x / ifelse(iszero(x), one(Ω), Ω)
                return (NO_FIELDS, signx * real(ΔΩ))
            end
            return Ω, abs_pullback
        end

        ## abs2
        function frule((_, Δz), ::typeof(abs2), z::Union{Real, Complex})
            return abs2(z), 2 * _realconjtimes(z, Δz)
        end

        function rrule(::typeof(abs2), z::Union{Real, Complex})
            function abs2_pullback(ΔΩ)
                Δu = real(ΔΩ)
                return (NO_FIELDS, 2real(ΔΩ)*z)
            end
            return abs2(z), abs2_pullback
        end

        ## conj
        function frule((_, Δz), ::typeof(conj), z::Union{Real, Complex})
            return conj(z), conj(Δz)
        end
        function rrule(::typeof(conj), z::Union{Real, Complex})
            function conj_pullback(ΔΩ)
                return (NO_FIELDS, conj(ΔΩ))
            end
            return conj(z), conj_pullback
        end

        ## angle
        function frule((_, Δx), ::typeof(angle), x)
            Ω = angle(x)
            # `ifelse` is applied only to denominator to ensure type-stability.
            n = ifelse(iszero(x), one(real(x)), abs2(x))
            ∂Ω = _imagconjtimes(x, Δx) / n
            return Ω, ∂Ω
        end

        function rrule(::typeof(angle), x::Real)
            function angle_pullback(ΔΩ::Real)
                return (NO_FIELDS, Zero())
            end
            function angle_pullback(ΔΩ)
                Δu, Δv = reim(ΔΩ)
                return (NO_FIELDS, im*Δu/ifelse(iszero(x), one(x), x))
                # `ifelse` is applied only to denominator to ensure type-stability.
            end
            return angle(x), angle_pullback
        end
        function rrule(::typeof(angle), z::Complex)
            function angle_pullback(ΔΩ)
                x,  y  = reim(z)
                Δu, Δv = reim(ΔΩ)
                # `ifelse` is applied only to denominator to ensure type-stability.
                n = ifelse(iszero(z), one(real(z)), abs2(z))
                return (NO_FIELDS, (-y + im*x)*Δu/n)
            end
            return angle(z), angle_pullback
        end

        # Binary functions

        # `hypot`

        function frule(
            (_, Δx, Δy),
            ::typeof(hypot),
            x::T,
            y::T,
        ) where {T<:Union{Real,Complex}}
            Ω = hypot(x, y)
            n = ifelse(iszero(Ω), one(Ω), Ω)
            ∂Ω = (_realconjtimes(x, Δx) + _realconjtimes(y, Δy)) / n
            return Ω, ∂Ω
        end

        function rrule(::typeof(hypot), x::T, y::T) where {T<:Union{Real,Complex}}
            Ω = hypot(x, y)
            function hypot_pullback(ΔΩ)
                c = real(ΔΩ) / ifelse(iszero(Ω), one(Ω), Ω)
                return (NO_FIELDS, c * x, c * y)
            end
            return (Ω, hypot_pullback)
        end

        @scalar_rule x + y (One(), One())
        @scalar_rule x - y (One(), -1)
        @scalar_rule x / y (one(x) / y, -(Ω / y))
        #log(complex(x)) is required so it gives correct complex answer for x<0
        @scalar_rule(x ^ y,
            (ifelse(iszero(x), zero(Ω), y * Ω / x), Ω * log(complex(x))),
        )
        # x^y for x < 0 errors when y is not an integer, but then derivative wrt y
        # is undefined, so we adopt subgradient convention and set derivative to 0.
        @scalar_rule(x::Real ^ y::Real,
            (ifelse(iszero(x), zero(Ω), y * Ω / x), Ω * log(oftype(Ω, ifelse(x ≤ 0, one(x), x)))),
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

        # `sign`

        function frule((_, Δx), ::typeof(sign), x)
            n = ifelse(iszero(x), one(real(x)), abs(x))
            Ω = x isa Real ? sign(x) : x / n
            ∂Ω = Ω * (_imagconjtimes(Ω, Δx) / n) * im
            return Ω, ∂Ω
        end

        function rrule(::typeof(sign), x)
            n = ifelse(iszero(x), one(real(x)), abs(x))
            Ω = x isa Real ? sign(x) : x / n
            function sign_pullback(ΔΩ)
                ∂x = Ω * (_imagconjtimes(Ω, ΔΩ) / n) * im
                return (NO_FIELDS, ∂x)
            end
            return Ω, sign_pullback
        end

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
                return (NO_FIELDS,  ΔΩ * y', x' * ΔΩ)
            end
            return x * y, times_pullback
        end
    end  # fastable_ast

    # Rewrite everything to use fast_math functions, including the type-constraints
    fast_ast = Base.FastMath.make_fastmath(fastable_ast)

    # Guard against mistakenly defining something as fast-able when it isn't.
    non_transformed_definitions = intersect(fastable_ast.args, fast_ast.args)
    filter!(expr->!(expr isa LineNumberNode), non_transformed_definitions)
    if !isempty(non_transformed_definitions)
        error(
            "Non-FastMath compatible rules defined in fastmath_able.jl. \n Definitions:\n" *
            join(non_transformed_definitions, "\n")
        )
    end

    eval(fast_ast)
    eval(fastable_ast)  # Get original definitions
    # we do this second so it overwrites anything we included by mistake in the fastable
end
