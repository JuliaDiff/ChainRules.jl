# Add tests to the quote for functions with  FastMath varients.
function jacobian_via_frule(f,z)
    du_dx, dv_dx = reim(frule((Zero(), 1),f,z)[2])
    du_dy, dv_dy = reim(frule((Zero(),im),f,z)[2])
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

const FASTABLE_AST = quote
    is_fastmath_mode = sin === Base.FastMath.sin_fast

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
                x, Δx, x̄ = randn(T, 3)
                Δz = (randn(T), randn(T))

                frule_test(sincos, (x, Δx))
                rrule_test(sincos, Δz, (x, x̄))
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

    @testset "rounding" begin
        for x in (-0.6, -0.2, -0.1, 0.1, 0.2, 0.6, 6.4)
            test_scalar(floor, x, fdm = backward_fdm(5, 1))
            test_scalar(ceil, x, fdm = forward_fdm(5, 1))
        end
    end

    @testset "Unary complex functions" begin
        for f ∈ (abs, abs2, conj), z ∈ (-4.1-0.02im, 6.4, 3 + im)
            @testset "Unary complex functions f = $f, z = $z" begin
                complex_jacobian_test(f, z)
            end
        end
        # As per PR #196, angle gives a Zero() pullback for Real z and ΔΩ, rather than
        # the one you'd get from considering the reals as embedded in the complex plane
        # so we need to special case it's tests
        for z ∈ (-4.1-0.02im, 6.4 + 0im, 3 + im)
            complex_jacobian_test(angle, z)
        end
        @test frule((Zero(), randn()), angle, randn())[2] === Zero()
        @test rrule(angle, randn())[2](randn())[2]        === Zero()

        # test that real primal with complex tangent gives complex tangent
        ΔΩ = randn(ComplexF64)
        for x in (-0.5, 2.0)
            @test isapprox(
                frule((Zero(), ΔΩ), angle, x)[2],
                frule((Zero(), ΔΩ), angle, complex(x))[2],
            )
        end
    end

    @testset "Unary functions" begin
        for x in (-4.1, 6.4, 0.0, 0.0 + 0.0im, 0.5 + 0.25im)
            test_scalar(+, x)
            test_scalar(-, x)
            test_scalar(atan, x; rtol=(is_fastmath_mode ? 1e-7 : 1e-9))
        end
    end

    @testset "binary functions" begin
        @testset "$f(x, y)" for f in (atan, rem, max, min)
            x, Δx, x̄ = 100rand(3)
            y, Δy, ȳ = 10rand(3)
            Δz = rand()

            frule_test(f, (x, Δx), (y, Δy))
            rrule_test(f, Δz, (x, x̄), (y, ȳ))
        end

        @testset "$f(x::$T, y::$T)" for f in (/, +, -, hypot), T in (Float64, ComplexF64)
            x, Δx, x̄ = 10rand(T, 3)
            y, Δy, ȳ = rand(T, 3)
            Δz = randn(typeof(f(x, y)))

            # some tests struggle in fast_math mode to get accurasy so we relax it some.
            frule_test(f, (x, Δx), (y, Δy); rtol=(is_fastmath_mode ? 1e-5 : 1e-7))
            rrule_test(f, Δz, (x, x̄), (y, ȳ); rtol=(is_fastmath_mode ? 1e-5 : 1e-7))
        end

        @testset "$f(x::$T, y::$T) type check" for f in (/, +, -,\, hypot, ^), T in (Float32, Float64)
            x, Δx, x̄ = 10rand(T, 3)
            y, Δy, ȳ = rand(T, 3)
            @assert T == typeof(f(x, y))
            Δz = randn(typeof(f(x, y)))

            @test frule((Zero(), Δx, Δy), f, x, y) isa Tuple{T, T}
            _, ∂x, ∂y = rrule(f, x, y)[2](Δz)
            @test extern.((∂x, ∂y)) isa Tuple{T, T}

            if f != hypot
                # Issue #233
                @test frule((Zero(), Δx, Δy), f, x, 2) isa Tuple{T, T}
                _, ∂x, ∂y = rrule(f, x, 2)[2](Δz)
                @test extern.((∂x, ∂y)) isa Tuple{T, T}

                @test frule((Zero(), Δx, Δy), f, 2, y) isa Tuple{T, T}
                _, ∂x, ∂y = rrule(f, 2, y)[2](Δz)
                @test extern.((∂x, ∂y)) isa Tuple{T, T}
            end
        end

        @testset "^(x::$T, n::$T)" for T in (Float64, ComplexF64)
            # for real x and n, x must be >0
            x, Δx, x̄ = rand(T, 3) .+ 3
            y, Δy, ȳ = rand(T, 3) .+ 3
            Δz = rand(T)

            frule_test(^, (x, Δx), (y, Δy))
            rrule_test(^, Δz, (x, x̄), (y, ȳ))

            T <: Real && @testset "discontinuity for ^(x::Real, n::Int) when x ≤ 0" begin
                # finite differences doesn't work for x < 0, so we check manually
                x = -rand(T) .- 3
                y = 3
                Δx = randn(T)
                Δy = randn(T)
                Δz = randn(T)

                @test frule((Zero(), Δx, Δy), ^, x, y)[2] ≈ Δx * y * x^(y - 1)
                @test frule((Zero(), Δx, Δy), ^, zero(x), y)[2] ≈ 0
                _, ∂x, ∂y = rrule(^, x, y)[2](Δz)
                @test ∂x ≈ Δz * y * x^(y - 1)
                @test ∂y ≈ 0
                _, ∂x, ∂y = rrule(^, zero(x), y)[2](Δz)
                @test ∂x ≈ 0
                @test ∂y ≈ 0
            end
        end
    end

    @testset "sign" begin
        @testset "real" begin
            @testset "at $x" for x in (-1.1, -1.1, 0.5, 100.0)
                test_scalar(sign, x)
            end

            @testset "Zero over the point discontinuity" begin
                # Can't do finite differencing because we are lying
                # following the subgradient convention.

                _, pb = rrule(sign, 0.0)
                _, x̄ = pb(10.5)
                @test extern(x̄) == 0

                _, ẏ = frule((Zero(), 10.5), sign, 0.0)
                @test extern(ẏ) == 0
            end
        end
        @testset "complex" begin
            @testset "at $z" for z in (-1.1 + randn() * im, 0.5 + randn() * im)
                test_scalar(sign, z)

                # test that complex (co)tangents with real primal gives same result as
                # complex primal with zero imaginary part

                ż, ΔΩ = randn(ComplexF64, 2)
                Ω, ∂Ω = frule((Zero(), ż), sign, real(z))
                @test Ω == sign(real(z))
                @test ∂Ω ≈ frule((Zero(), ż), sign, real(z) + 0im)[2]

                Ω, pb = rrule(sign, real(z))
                @test Ω == sign(real(z))
                @test pb(ΔΩ)[2] ≈ rrule(sign, real(z) + 0im)[2](ΔΩ)[2]
            end

            @testset "zero over the point discontinuity" begin
                # Can't do finite differencing because we are lying
                # following the subgradient convention.

                _, pb = rrule(sign, 0.0 + 0.0im)
                _, z̄ = pb(randn(ComplexF64))
                @test extern(z̄) == 0.0 + 0.0im

                _, Ω̇ = frule((Zero(), randn(ComplexF64)), sign, 0.0 + 0.0im)
                @test extern(Ω̇) == 0.0 + 0.0im
            end
        end
    end
end

# Now we generate tests for fast and nonfast versions
@eval @testset "fastmath_able Base functions" begin
    $FASTABLE_AST
end


@eval @testset "fastmath_able FastMath functions" begin
    $(Base.FastMath.make_fastmath(FASTABLE_AST))
end
