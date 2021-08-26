# Add tests to the quote for functions with  FastMath varients.
function jacobian_via_frule(f,z)
    du_dx, dv_dx = reim(frule((ZeroTangent(), 1),f,z)[2])
    du_dy, dv_dy = reim(frule((ZeroTangent(),im),f,z)[2])
    return [
        du_dx  du_dy
        dv_dx  dv_dy
    ]
end
function jacobian_via_rrule(f,z)
    _, pullback = rrule(f,z)
    du_dx, du_dy = reim(pullback( 1)[2])
    dv_dx, dv_dy = reim(pullback(im)[2])
    return [
        du_dx  du_dy
        dv_dx  dv_dy
    ]
end

function jacobian_via_fdm(f, z::Union{Real, Complex})
    fR2((x, y)) = (collect ∘ reim ∘ f)(x + im*y)
    v = float([real(z)
               imag(z)])
    j = jacobian(central_fdm(5,1), fR2, v)[1]
    if size(j) == (2,2)
        j
    elseif size(j) == (1, 2)
        [j
         false false]
    else
        error("Invalid Jacobian size $(size(j))")
    end
end

function complex_jacobian_test(f, z)
    @test jacobian_via_fdm(f, z) ≈ jacobian_via_frule(f, z)
    @test jacobian_via_fdm(f, z) ≈ jacobian_via_rrule(f, z)
end

# IMPORTANT:
# Do not add any tests here for functions that do not have varients in Base.FastMath
# e.g. do not add `foo` unless `Base.FastMath.foo_fast` exists.
const FASTABLE_AST = quote
    @testset "Trig" begin
        @testset "Basics" for x = (Float64(π)-0.01, Complex(π, π/2))
            test_scalar(sin, x)
            test_scalar(cos, x)
            test_scalar(tan, x)
        end
        @testset "Hyperbolic" for x = (Float64(π)-0.01, Complex(π-0.01, π/2))
            test_scalar(sinh, x)
            test_scalar(cosh, x)
            test_scalar(tanh, x)
        end
        @testset "Inverses" for x = (0.5, Complex(0.5, 0.25))
            test_scalar(asin, x)
            test_scalar(acos, x)
            test_scalar(atan, x)
        end
        @testset "Multivariate" begin
            @testset "sincos(x::$T)" for T in (Float64, ComplexF64)
                Δz = Tangent{Tuple{T,T}}(randn(T), randn(T))

                test_frule(sincos, randn(T))
                test_rrule(sincos, randn(T); output_tangent=Δz)
            end
        end
    end

    @testset "exponents" begin
        for x in (-0.1, 7.9, 0.5 + 0.25im)
            test_scalar(inv, x)

            test_scalar(exp, x)
            test_scalar(exp2, x)
            test_scalar(exp10, x)
            test_scalar(expm1, x)

            if x isa Real
                test_scalar(cbrt, x)
            end

            if x isa Complex || x >= 0
                test_scalar(sqrt, x)
                test_scalar(log, x)
                test_scalar(log2, x)
                test_scalar(log10, x)
                test_scalar(log1p, x)
            end
        end
    end

    @testset "Unary complex functions" begin
        for f ∈ (abs, abs2, conj), z ∈ (-4.1-0.02im, 6.4, 3 + im)
            @testset "Unary complex functions f = $f, z = $z" begin
                complex_jacobian_test(f, z)
            end
        end
        # As per PR #196, angle gives a ZeroTangent() pullback for Real z and ΔΩ, rather than
        # the one you'd get from considering the reals as embedded in the complex plane
        # so we need to special case it's tests
        for z ∈ (-4.1-0.02im, 6.4 + 0im, 3 + im)
            complex_jacobian_test(angle, z)
        end
        @test frule((ZeroTangent(), randn()), angle, randn())[2] === ZeroTangent()
        @test rrule(angle, randn())[2](randn())[2] === ZeroTangent()

        # test that real primal with complex tangent gives complex tangent
        ΔΩ = randn(ComplexF64)
        for x in (-0.5, 2.0)
            @test isapprox(
                frule((ZeroTangent(), ΔΩ), angle, x)[2],
                frule((ZeroTangent(), ΔΩ), angle, complex(x))[2],
            )
        end
    end

    @testset "Unary functions" begin
        for x in (-4.1, 6.4, 0.0, 0.0 + 0.0im, 0.5 + 0.25im)
            test_scalar(+, x)
            test_scalar(-, x)
            test_scalar(atan, x)
        end
    end

    @testset "binary functions" begin
        @testset "$f(x, y)" for f in (atan, rem, max, min)
            # be careful not to sample near singularities for `rem`
            base = rand() + 1
            test_frule(f, (rand(0:10) + .6rand() + .2) * base, base)
            base = rand() + 1
            test_rrule(f, (rand(0:10) + .6rand() + .2) * base, base)
        end

        @testset "$f(x::$T, y::$T)" for f in (/, +, -, hypot), T in (Float64, ComplexF64)
            test_frule(f, 10rand(T), rand(T))
            test_rrule(f, 10rand(T), rand(T))
        end

        @testset "$f(x::$T, y::$T) type check" for f in (/, +, -,\, hypot), T in (Float32, Float64)
            x, Δx, x̄ = 10rand(T, 3)
            y, Δy, ȳ = rand(T, 3)
            @assert T == typeof(f(x, y))
            Δz = randn(typeof(f(x, y)))

            @test frule((ZeroTangent(), Δx, Δy), f, x, y) isa Tuple{T, T}
            _, ∂x, ∂y = rrule(f, x, y)[2](Δz)
            @test (∂x, ∂y) isa Tuple{T, T}

            if f != hypot
                # Issue #233
                @test frule((ZeroTangent(), Δx, Δy), f, x, 2) isa Tuple{T, T}
                _, ∂x, ∂y = rrule(f, x, 2)[2](Δz)
                @test (∂x, ∂y) isa Tuple{T, Float64}

                @test frule((ZeroTangent(), Δx, Δy), f, 2, y) isa Tuple{T, T}
                _, ∂x, ∂y = rrule(f, 2, y)[2](Δz)
                @test (∂x, ∂y) isa Tuple{Float64, T}
            end
        end

        @testset "^(x::$T, p::$S)" for T in (Float64, ComplexF64), S in (Float64, ComplexF64)
            test_frule(^, rand(T) + 3, rand(T) + 3)
            test_rrule(^, rand(T) + 3, rand(T) + 3)

            # When both x & p are Real, and !(isinteger(p)), 
            # then x must be positive to avoid a DomainError
            T <: Real && S <: Real && continue
            # In other cases, we can test values near zero:

            test_frule(^, randn(T), rand(T))
            test_rrule(^, rand(T), rand(T))
        end

        # @testset "^(x::$T, $p::Int)" for T in (Float64, ComplexF64), p in -2:2
        #     test_frule(^, randn(T) + 3, p ⊢ NoTangent())  # this doesn't just skip p's tangent
        #     test_rrule(^, randn(T) + 3, p ⊢ NoTangent())
        # end

        # Tests for power functions, at values near to zero.

        POWERGRADS = [ # (x,p) => (dx,dp)
        # Some regular points, as sanity checks:
          (1.0, 2)   => (2.0, 0.0),
          (2.0, 2)   => (4.0, 2.772588722239781),
        # At x=0, gradients for x seem clear, 
        # for p less certain but I think 0 or NaN right?
          (0.0, 2)   => (0.0, 0.0),
          (-0.0, 2)  => (-0.0, 0.0),
          (0.0, 1)   => (1.0, 0.0),
          (-0.0, 1)  => (1.0, 0.0),
          (0.0, 0)   => (0.0, NaN),
          (-0.0, 0)  => (0.0, NaN),
          (0.0, -1)  => (-Inf, NaN),
          (-0.0, -1) => (-Inf, NaN),
          (0.0, -2)  => (-Inf, NaN),
          (-0.0, -2) => (Inf, NaN),
        # Non-integer powers:
          (0.0, 0.5)   => (Inf, 0.0),
          (0.0, 3.5)   => (0.0, 0.0),
          (0.0, -1.5)  => (-Inf, NaN),
        ]

        @testset "$x ^ $p" for ((x,p), (∂x, ∂p)) in POWERGRADS
            y = x^p

            # Forward
            y_f = frule((1,1,1), ^, x, p)[1]
            @test isequal(y, y_f) # || println("^ forward value for $x^$p: got $y_f, expected $y")

            ∂x_fwd = frule((0,1,0), ^, x, p)[1]
            ∂p_fwd = frule((0,0,1), ^, x, p)[2]
            # isequal(∂x, ∂x_fwd) || println("^ forward `x` gradient for $y = $x^$p: got $∂x_fwd, expected $∂x, maybe!")
            # isequal(∂p, ∂p_fwd) || println("^ forward `p` gradient for $x^$p: got $∂p_fwd, expected $∂p, maybe")

            # Reverse
            y_r = rrule(^, x, p)[1]
            @test isequal(y, y_r) # || println("^ reverse value for $x^$p: got $y_r, expected $y")
            
            ∂x_rev, ∂p_rev = unthunk.(rrule(^, x, p)[2](1))[2:3]
            if ∂x === -0.0 # happens at at x === -0.0 && p === 2, ignore the sign
                @test 0.0 == ∂x_rev
            else
                @test isequal(∂x, ∂x_rev) # || println("^ reverse `x` gradient for $x^$p: got $∂x_rev, expected $∂x")
            end
            @test isequal(∂p, ∂p_rev) # || println("^ reverse `p` gradient for $x^$p: got $∂p_rev, expected $∂p")
        end

        @testset "literal_pow $x ^ $p" for ((x,p), (∂x, ∂p)) in POWERGRADS
            # p isa Int || continue
            # x isa Real || continue

            y = x^p

            # Forward
            y_f = frule((1,1,1,1), Base.literal_pow, ^, x, Val(p))[1]
            @test isequal(y, y_f) # || println("literal_pow forward value for $x^$p: got $y_f, expected $y")

            ∂x_fwd = frule((0,0,1,0), Base.literal_pow, ^, x, Val(p))[1]
            # isequal(∂x, ∂x_fwd) || println("literal_pow forward `x` gradient for $x^$p: got $∂x_fwd, expected $∂x, maybe, y=$y")

            # Reverse
            y_r = rrule(Base.literal_pow, ^, x, Val(p))[1]
            @test isequal(y, y_r) # || println("literal_pow reverse value for $x^$p: got $y_r, expected $y")

            ∂x_rev = unthunk(rrule(Base.literal_pow, ^, x, Val(p))[2](1))[3]
            @test isequal(∂x, ∂x_rev) # || println("literal_pow `x` gradient for $x^$p: got $∂x_rev, expected $∂x")
        end
    end

    @testset "sign" begin
        @testset "real" begin
            @testset "at $x" for x in (-1.1, -1.1, 0.5, 100.0)
                test_scalar(sign, x)
            end

            @testset "ZeroTangent over the point discontinuity" begin
                # Can't do finite differencing because we are lying
                # following the subgradient convention.

                _, pb = rrule(sign, 0.0)
                _, x̄ = pb(10.5)
                test_approx(x̄, 0)

                _, ẏ = frule((ZeroTangent(), 10.5), sign, 0.0)
                test_approx(ẏ, 0)
            end
        end
        @testset "complex" begin
            @testset "at $z" for z in (-1.1 + randn() * im, 0.5 + randn() * im)
                test_scalar(sign, z)

                # test that complex (co)tangents with real primal gives same result as
                # complex primal with zero imaginary part

                ż, ΔΩ = randn(ComplexF64, 2)
                Ω, ∂Ω = frule((ZeroTangent(), ż), sign, real(z))
                @test Ω == sign(real(z))
                @test ∂Ω ≈ frule((ZeroTangent(), ż), sign, real(z) + 0im)[2]

                Ω, pb = rrule(sign, real(z))
                @test Ω == sign(real(z))
                @test pb(ΔΩ)[2] ≈ rrule(sign, real(z) + 0im)[2](ΔΩ)[2]
            end

            @testset "zero over the point discontinuity" begin
                # Can't do finite differencing because we are lying
                # following the subgradient convention.

                _, pb = rrule(sign, 0.0 + 0.0im)
                _, z̄ = pb(randn(ComplexF64))
                @test z̄ == 0.0 + 0.0im

                _, Ω̇ = frule((ZeroTangent(), randn(ComplexF64)), sign, 0.0 + 0.0im)
                @test Ω̇ == 0.0 + 0.0im
            end
        end
    end
end

# Now we generate tests for fast and nonfast versions
@eval @interpret (function()
    @testset "fastmath_able Base functions" begin
        $FASTABLE_AST
    end
    
    
    @testset "fastmath_able FastMath functions" begin
        $(Base.FastMath.make_fastmath(FASTABLE_AST))
    end
end)()
