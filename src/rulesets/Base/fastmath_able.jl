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
        function rrule(::typeof(sin), x::CommutativeMulNumber)
            sinx, cosx = sincos(x)
            sin_pullback(Δy) = (NoTangent(), cosx' * Δy)
            return (sinx, sin_pullback)
        end

        function frule((_, Δx), ::typeof(sin), x::CommutativeMulNumber)
            sinx, cosx = sincos(x)
            return (sinx, cosx * Δx)
        end

        ## cos
        function rrule(::typeof(cos), x::CommutativeMulNumber)
            sinx, cosx = sincos(x)
            cos_pullback(Δy) = (NoTangent(), -sinx' * Δy)
            return (cosx, cos_pullback)
        end
        
        function frule((_, Δx), ::typeof(cos), x::CommutativeMulNumber)
            sinx, cosx = sincos(x)
            return (cosx, -sinx * Δx)
        end
        
        @scalar_rule tan(x::CommutativeMulNumber) 1 + Ω ^ 2


        # Trig-Hyperbolic
        @scalar_rule cosh(x::CommutativeMulNumber) sinh(x)
        @scalar_rule sinh(x::CommutativeMulNumber) cosh(x)
        @scalar_rule tanh(x::CommutativeMulNumber) 1 - Ω ^ 2

        # Trig- Inverses
        @scalar_rule acos(x::CommutativeMulNumber) -(inv(sqrt(1 - x ^ 2)))
        @scalar_rule asin(x::CommutativeMulNumber) inv(sqrt(1 - x ^ 2))
        @scalar_rule atan(x::CommutativeMulNumber) inv(1 + x ^ 2)

        # Trig-Multivariate
        @scalar_rule atan(y::Real, x::Real) @setup(u = x ^ 2 + y ^ 2) (x / u, -y / u)
        @scalar_rule sincos(x::CommutativeMulNumber) @setup((sinx, cosx) = Ω) cosx -sinx

        # exponents
        @scalar_rule cbrt(x::CommutativeMulNumber) inv(3 * Ω ^ 2)
        @scalar_rule inv(x::CommutativeMulNumber) -(Ω ^ 2)
        function frule((_, Δx), ::typeof(inv), x::Number)
            Ω = inv(x)
            return Ω, Ω * -Δx * Ω
        end
        function rrule(::typeof(inv), x::Number)
            Ω = inv(x)
            project_x = ProjectTo(x)
            function inv_pullback(ΔΩ)
                Ω′ = conj(Ω)
                return NoTangent(), project_x(Ω′ * -ΔΩ * Ω′)
            end
            return Ω, inv_pullback
        end
        @scalar_rule sqrt(x::CommutativeMulNumber) inv(2Ω)  # gradient +Inf at x==0
        @scalar_rule exp(x::CommutativeMulNumber) Ω
        @scalar_rule exp10(x::CommutativeMulNumber) Ω * log(oftype(x, 10))
        @scalar_rule exp2(x::CommutativeMulNumber) Ω * log(oftype(x, 2))
        @scalar_rule expm1(x::CommutativeMulNumber) exp(x)
        @scalar_rule log(x::CommutativeMulNumber) inv(x)
        @scalar_rule log10(x::CommutativeMulNumber) inv(x) / log(oftype(x, 10))
        @scalar_rule log1p(x::CommutativeMulNumber) inv(x + 1)
        @scalar_rule log2(x::CommutativeMulNumber) inv(x) / log(oftype(x, 2))

        # Unary complex functions
        ## abs
        function frule((_, Δx), ::typeof(abs), x::Number)
            Ω = abs(x)
            # `ifelse` is applied only to denominator to ensure type-stability.
            signx = x isa Real ? sign(x) : x / ifelse(iszero(x), one(Ω), Ω)
            return Ω, _realconjtimes(signx, Δx)
        end

        function rrule(::typeof(abs), x::Number)
            Ω = abs(x)
            project_x = ProjectTo(x)
            function abs_pullback(ΔΩ)
                signx = x isa Real ? sign(x) : x / ifelse(iszero(x), one(Ω), Ω)
                return (NoTangent(), project_x(signx * real(ΔΩ)))
            end
            return Ω, abs_pullback
        end

        ## abs2
        function frule((_, Δz), ::typeof(abs2), z::Number)
            return abs2(z), 2 * _realconjtimes(z, Δz)
        end

        function rrule(::typeof(abs2), z::Number)
            project_z = ProjectTo(z)
            function abs2_pullback(ΔΩ)
                Δu = real(ΔΩ)
                return (NoTangent(), project_z(2Δu * z))
            end
            return abs2(z), abs2_pullback
        end

        ## conj
        function frule((_, Δz), ::typeof(conj), z::Number)
            return conj(z), conj(Δz)
        end
        function rrule(::typeof(conj), z::Number)
            project_z = ProjectTo(z)
            function conj_pullback(ΔΩ)
                return (NoTangent(), project_z(conj(ΔΩ)))
            end
            return conj(z), conj_pullback
        end

        ## angle
        function frule((_, Δx), ::typeof(angle), x::Union{Real,Complex})
            Ω = angle(x)
            # `ifelse` is applied only to denominator to ensure type-stability.
            n = ifelse(iszero(x), one(real(x)), abs2(x))
            ∂Ω = _imagconjtimes(x, Δx) / n
            return Ω, ∂Ω
        end

        function rrule(::typeof(angle), x::Real)
            function angle_pullback(ΔΩ::Real)
                return (NoTangent(), ZeroTangent())
            end
            function angle_pullback(ΔΩ)
                Δu, Δv = reim(ΔΩ)
                return (NoTangent(), im*Δu/ifelse(iszero(x), one(x), x))
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
                return (NoTangent(), (-y + im*x)*Δu/n)
            end
            return angle(z), angle_pullback
        end

        # Binary functions

        ## `hypot`
        function frule(
            (_, Δx, Δy),
            ::typeof(hypot),
            x::T,
            y::T,
        ) where {T<:Number}
            Ω = hypot(x, y)
            n = ifelse(iszero(Ω), one(Ω), Ω)
            ∂Ω = (_realconjtimes(x, Δx) + _realconjtimes(y, Δy)) / n
            return Ω, ∂Ω
        end

        function rrule(::typeof(hypot), x::T, y::T) where {T<:Number}
            Ω = hypot(x, y)
            project_x = ProjectTo(x)
            project_y = ProjectTo(y)
            function hypot_pullback(ΔΩ)
                c = real(ΔΩ) / ifelse(iszero(Ω), one(Ω), Ω)
                return (NoTangent(), project_x(c * x), project_y(c * y))
            end
            return (Ω, hypot_pullback)
        end

        @scalar_rule x + y (true, true)
        @scalar_rule x - y (true, -1)
        @scalar_rule x / y::CommutativeMulNumber (one(x) / y, -(Ω / y))
        function frule((_, Δx, Δy), ::typeof(/), x::Number, y::Number)
            Ω = x / y
            return Ω, muladd(Δx, Ω, -Δy) / y
        end
        function rrule(::typeof(/), x::Number, y::Number)
            Ω = x / y
            project_x = ProjectTo(x)
            project_y = ProjectTo(y)
            function slash_pullback(ΔΩ)
                ∂x = ΔΩ / y'
                return NoTangent(), project_x(∂x), project_y(Ω' * -∂x)
            end
            return Ω, slash_pullback
        end
        ## power
        # literal_pow is in base.jl
        function frule((_, Δx, Δp), ::typeof(^), x::CommutativeMulNumber, p::CommutativeMulNumber)
            y = x ^ p
            _dx = _pow_grad_x(x, p, float(y))
            if iszero(Δp)
                # Treat this as a strong zero, to avoid NaN, and save the cost of log
                return y, _dx * Δx
            else
                # This may do real(log(complex(...))) which matches ProjectTo in rrule
                _dp = _pow_grad_p(x, p, float(y))
                return y, muladd(_dp, Δp, _dx * Δx)
            end
        end

        function rrule(::typeof(^), x::CommutativeMulNumber, p::CommutativeMulNumber)
            y = x^p
            project_x = ProjectTo(x)
            project_p = ProjectTo(p)
            function power_pullback(dy)
                _dx = _pow_grad_x(x, p, float(y))
                return (
                    NoTangent(), 
                    project_x(conj(_dx) * dy),
                    # _pow_grad_p contains log, perhaps worth thunking:
                    @thunk project_p(conj(_pow_grad_p(x, p, float(y))) * dy)
                )
            end
            return y, power_pullback
        end

        ## `rem`
        @scalar_rule(
            rem(x, y),
            @setup((u, nan) = promote(x / y, NaN16), isint = isinteger(x / y)),
            (ifelse(isint, nan, one(u)), ifelse(isint, nan, -trunc(u))),
        )
        ## `min`, `max`
        @scalar_rule max(x, y) @setup(gt = x > y) (gt, !gt)
        @scalar_rule min(x, y) @setup(gt = x > y) (!gt, gt)

        # Unary functions
        @scalar_rule +x true
        @scalar_rule -x -1

        ## `sign`
        function frule((_, Δx), ::typeof(sign), x::Number)
            n = ifelse(iszero(x), one(real(x)), abs(x))
            Ω = x isa Real ? sign(x) : x / n
            ∂Ω = Ω * (_imagconjtimes(Ω, Δx) / n) * im
            return Ω, ∂Ω
        end

        function rrule(::typeof(sign), x::Number)
            n = ifelse(iszero(x), one(real(x)), abs(x))
            Ω = x isa Real ? sign(x) : x / n
            project_x = ProjectTo(x)
            function sign_pullback(ΔΩ)
                ∂x = Ω * (_imagconjtimes(Ω, ΔΩ) / n) * im
                return (NoTangent(), project_x(∂x))
            end
            return Ω, sign_pullback
        end

        function frule((_, Δx, Δy), ::typeof(*), x::Number, y::Number)
            # Optimized version of `Δx * y + x * Δy`. Also, it is potentially more
            # accurate on machines with FMA instructions, since there are only two
            # rounding operations, one in `muladd/fma` and the other in `*`.
            ∂xy = muladd(Δx, y, x * Δy)
            return x * y, ∂xy
        end

        function rrule(::typeof(*), x::Number, y::Number)
            project_x = ProjectTo(x)
            project_y = ProjectTo(y)
            function times_pullback(Ω̇)
                ΔΩ = unthunk(Ω̇)
                return (NoTangent(),  project_x(ΔΩ * y'), project_y(x' * ΔΩ))
            end
            return x * y, times_pullback
        end
    end  # fastable_ast

    # Rewrite everything to use fast_math functions, including the type-constraints
    fast_ast = Base.FastMath.make_fastmath(fastable_ast)

    # Guard against mistakenly defining something as fast-able when it isn't.
    # NOTE: this check is not infallible, it will be tricked if a function itself is not
    # fastmath_able but it's pullback uses something that is. So manual check should also be
    # done.
    non_transformed_definitions = intersect(fastable_ast.args, fast_ast.args)
    filter!(expr->!(expr isa LineNumberNode), non_transformed_definitions)
    if !isempty(non_transformed_definitions)
        error(
            "Non-FastMath compatible rules defined in fastmath_able.jl. \n Definitions:\n" *
            join(non_transformed_definitions, "\n")
        )
        # This error() may not play well with Revise. But a wanring @error does:
        # @error "Non-FastMath compatible rules defined in fastmath_able.jl." non_transformed_definitions
    end

    eval(fast_ast)
    eval(fastable_ast)  # Get original definitions
    # we do this second so it overwrites anything we included by mistake in the fastable
end

## power
# Thes functions need to be defined outside the eval() block.
# The special cases they aim to hit are in POWERGRADS in tests.
_pow_grad_x(x, p, y) = (p * y / x)
function _pow_grad_x(x::Real, p::Real, y)
    return if !iszero(x) || p < 0
        p * y / x
    elseif isone(p)
        one(y)
    elseif iszero(p) || p > 1
        zero(y)
    else
        oftype(y, Inf)
    end
end

_pow_grad_p(x, p, y) = y * log(complex(x))
function _pow_grad_p(x::Real, p::Real, y)
    return if !iszero(x)
        y * real(log(complex(x)))
    elseif p > 0
        zero(y)
    else
        oftype(y, NaN)
    end
end
