@testset "base" begin
    @testset "Trig" begin
        @testset "Basics" for x = (Float64(π)-0.01, Complex(π, π/2))
            test_scalar(sec, x)
            test_scalar(csc, x)
            test_scalar(cot, x)
            test_scalar(sinpi, x)
            test_scalar(cospi, x)
        end
        @testset "Hyperbolic" for x = (Float64(π)-0.01, Complex(π-0.01, π/2))
            test_scalar(sech, x)
            test_scalar(csch, x)
            test_scalar(coth, x)
        end
        @testset "Degrees" begin
            x = 45.0
            test_scalar(sind, x)
            test_scalar(cosd, x)
            test_scalar(tand, x)
            test_scalar(secd, x)
            test_scalar(cscd, x)
            test_scalar(cotd, x)
        end
        @testset "Inverses" for x = (0.5, Complex(0.5, 0.25))
            test_scalar(asec, 1/x)
            test_scalar(acsc, 1/x)
            test_scalar(acot, 1/x)
        end
        @testset "Inverse hyperbolic" for x = (0.5, Complex(0.5, 0.25),  Complex(-2.1 -3.1im))
            test_scalar(asinh, x)
            test_scalar(acosh, x + 1)  # +1 accounts for domain for real
            test_scalar(atanh, x)
            test_scalar(asech, x)
            test_scalar(acsch, x)
            test_scalar(acoth, x + 1)
        end

        @testset "Inverse degrees" for x = (0.5, Complex(0.5, 0.25))
            test_scalar(asind, x)
            test_scalar(acosd, x)
            test_scalar(atand, x)
            test_scalar(asecd, 1/x)
            test_scalar(acscd, 1/x)
            test_scalar(acotd, 1/x)
        end
    end  # Trig

    @testset "Angles" begin
        for x in (-0.1, 6.4, 0.5 + 0.25im)
            test_scalar(deg2rad, x)
            test_scalar(rad2deg, x)
        end
    end

    @testset "Unary complex functions" begin
        for x in (-4.1, 6.4, 0.0, 0.0 + 0.0im, 0.5 + 0.25im)
            test_scalar(real, x)
            test_scalar(imag, x)
            test_scalar(hypot, x)
            test_scalar(adjoint, x)
        end
    end

    @testset "Complex" begin
        test_scalar(Complex, randn())
        test_scalar(Complex, randn(ComplexF64))
        x, ẋ, x̄ = randn(3)
        y, ẏ, ȳ = randn(3)
        Δz = randn(ComplexF64)
        frule_test(Complex, (x, ẋ), (y, ẏ))
        rrule_test(Complex, Δz, (x, x̄), (y, ȳ))
    end

    @testset "*(x, y) (scalar)" begin
        # This is pretty important so testing it fairly heavily
        test_points = (0.0, -2.1, 3.2, 3.7+2.12im, 14.2-7.1im)
        @testset "($x) * ($y); (perturbed by: $perturb)" for
            x in test_points, y in test_points, perturb in test_points

            # ensure all complex if any complex for FiniteDifferences
            x, y, perturb = Base.promote(x, y, perturb)

            # give small off-set so as can't slip in symmetry
            x̄ = ẋ = 0.5 + perturb
            ȳ = ẏ = 0.6 + perturb
            Δz = perturb

            frule_test(*, (x, ẋ), (y, ẏ))
            rrule_test(*, Δz, (x, x̄), (y, ȳ))
        end
    end

    @testset "ldexp" begin
        x, Δx, x̄ = 10rand(3)
        Δz = rand()

        for n in (0,1,20)
            # TODO: Forward test does not work when parameter is Integer
            # See: https://github.com/JuliaDiff/ChainRulesTestUtils.jl/issues/22
            #frule_test(ldexp, (x, Δx), (n, nothing))
            rrule_test(ldexp, Δz, (x, x̄), (n, nothing))
        end
    end

    @testset "\\(x::$T, y::$T) (scalar)" for T in (Float64, ComplexF64)
        x, ẋ, x̄, y, ẏ, ȳ, Δz = randn(T, 7)
        frule_test(*, (x, ẋ), (y, ẏ))
        rrule_test(*, Δz, (x, x̄), (y, ȳ))
    end

    @testset "mod" begin
        x, Δx, x̄ = 10rand(3)
        y, Δy, ȳ = rand(3)
        Δz = rand()

        frule_test(mod, (x, Δx), (y, Δy))
        rrule_test(mod, Δz, (x, x̄), (y, ȳ))
    end

    @testset "identity" for T in (Float64, ComplexF64)
        frule_test(identity, (randn(T), randn(T)))
        frule_test(identity, (randn(T, 4), randn(T, 4)))
        frule_test(
            identity,
            (Composite{Tuple}(randn(T, 3)...), Composite{Tuple}(randn(T, 3)...))
        )

        rrule_test(identity, randn(T), (randn(T), randn(T)))
        rrule_test(identity, randn(T, 4), (randn(T, 4), randn(T, 4)))
        rrule_test(
            identity, Tuple(randn(T, 3)),
            (Composite{Tuple}(randn(T, 3)...), Composite{Tuple}(randn(T, 3)...))
        )
    end

    VERSION ≥ v"1.4" && @testset "evalpoly" begin
        # test fallbacks for when code generation fails
        @testset "fallbacks for $T" for T in (Float64, ComplexF64)
            x, p = randn(T), Tuple(randn(T, 10))
            y_fb, ys_fb = ChainRules._evalpoly_intermediates_fallback(x, p)
            y, ys = ChainRules._evalpoly_intermediates(x, p)
            @test y_fb ≈ y
            @test collect(ys_fb) ≈ collect(ys)

            Δy, ys = randn(T), Tuple(randn(T, 9))
            ∂x_fb, ∂p_fb = ChainRules._evalpoly_back_fallback(x, p, ys, Δy)
            ∂x, ∂p = ChainRules._evalpoly_back(x, p, ys, Δy)
            @test ∂x_fb ≈ ∂x
            @test collect(∂p_fb) ≈ collect(∂p)
        end

        @testset "x dim: $(nx), pi dim: $(np), type: $T" for T in (Float64, ComplexF64), nx in (tuple(), 3), np in (tuple(), 3)
            # skip x::Matrix, pi::Number case, which is not supported by evalpoly
            isempty(np) && !isempty(nx) && continue
            m = 5
            sx = (nx..., nx...)
            sp = (np..., np...)
            x, ẋ, x̄ = randn(T, sx...), randn(T, sx...), randn(T, sx...)
            p = [randn(T, sp...) for _ in 1:m]
            ṗ = [randn(T, sp...) for _ in 1:m]
            p̄ = [randn(T, sp...) for _ in 1:m]
            Ω = evalpoly(x, p)
            Ω̄ = randn(T, size(Ω)...)
            frule_test(evalpoly, (x, ẋ), (p, ṗ))
            frule_test(evalpoly, (x, ẋ), (Tuple(p), Tuple(ṗ)))
            rrule_test(evalpoly, Ω̄, (x, x̄), (p, p̄))
            rrule_test(evalpoly, Ω̄, (x, x̄), (Tuple(p), Tuple(p̄)))
        end
    end

    @testset "Constants" for x in (-0.1, 6.4, 1.0+0.5im, -10.0+0im, 0.0+200im)
        test_scalar(one, x)
        test_scalar(zero, x)
    end

    @testset "muladd(x::$T, y::$T, z::$T)" for T in (Float64, ComplexF64)
        x, Δx, x̄ = 10randn(T, 3)
        y, Δy, ȳ = randn(T, 3)
        z, Δz, z̄ = randn(T, 3)
        Δk = randn(T)

        frule_test(muladd, (x, Δx), (y, Δy), (z, Δz))
        rrule_test(muladd, Δk, (x, x̄), (y, ȳ), (z, z̄))
    end

    @testset "fma" begin
        x, Δx, x̄ = 10randn(3)
        y, Δy, ȳ = randn(3)
        z, Δz, z̄ = randn(3)
        Δk = randn()

        frule_test(fma, (x, Δx), (y, Δy), (z, Δz))
        rrule_test(fma, Δk, (x, x̄), (y, ȳ), (z, z̄))
    end

    @testset "clamp"  begin
        x̄, ȳ, z̄    = randn(3)
        Δx, Δy, Δz = randn(3)
        Δk = randn()

        x, y, z = 1., 2., 3.  # to left
        frule_test(clamp, (x, Δx), (y, Δy), (z, Δz))
        rrule_test(clamp, Δk, (x, x̄), (y, ȳ), (z, z̄))

        x, y, z = 2.5, 2., 3.  # in the middle
        frule_test(clamp, (x, Δx), (y, Δy), (z, Δz))
        rrule_test(clamp, Δk, (x, x̄), (y, ȳ), (z, z̄))

        x, y, z = 4., 2., 3.  # to right
        frule_test(clamp, (x, Δx), (y, Δy), (z, Δz))
        rrule_test(clamp, Δk, (x, x̄), (y, ȳ), (z, z̄))
    end
end
