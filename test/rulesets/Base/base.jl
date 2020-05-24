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
        for x in (-0.1, 6.4)
            test_scalar(deg2rad, x)
            test_scalar(rad2deg, x)
        end
    end

    @testset "Unary complex functions" begin
        for x in (-4.1, 6.4)
            test_scalar(real, x)
            test_scalar(imag, x)
            test_scalar(hypot, x)
            test_scalar(adjoint, x)
        end
    end

    @testset "*(x, y) (scalar)" begin
        # This is pretty important so testing it fairly heavily
        test_points = (0.0, -2.1, 3.2, 3.7+2.12im, 14.2-7.1im)
        @testset "$x * $y; (perturbed by: $perturb)" for
            x in test_points, y in test_points, perturb in test_points

            # give small off-set so as can't slip in symmetry
            x̄ = ẋ = 0.5 + perturb
            ȳ = ẏ = 0.6 + perturb
            Δz = perturb

            frule_test(*, (x, ẋ), (y, ẏ))
            rrule_test(*, Δz, (x, x̄), (y, ȳ))
        end
    end

    @testset "matmul *(x, y)" begin
        x, y = rand(3, 2), rand(2, 5)
        z, pullback = rrule(*, x, y)

        @test z == x * y

        z̄ = rand(3, 5)
        (ds, dx, dy) = pullback(z̄)

        @test ds === NO_FIELDS

        @test extern(dx) == extern(zeros(3, 2) .+ dx)
        @test extern(dy) == extern(zeros(2, 5) .+ dy)
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

    @testset "binary function ($f)" for f in (mod, \)
        x, Δx, x̄ = 10rand(3)
        y, Δy, ȳ = rand(3)
        Δz = rand()

        frule_test(f, (x, Δx), (y, Δy))
        rrule_test(f, Δz, (x, x̄), (y, ȳ))
    end

    @testset "x^n for x<0" begin
        x = -15*rand()
        Δx, x̄ = 10rand(2)
        y, Δy, ȳ = rand(3)
        Δz = rand()

        frule_test(^, (-x, Δx), (y, Δy))
        rrule_test(^, Δz, (-x, x̄), (y, ȳ))
    end

    @testset "identity" begin
        rrule_test(identity, randn(), (randn(), randn()))
        rrule_test(identity, randn(4), (randn(4), randn(4)))

        rrule_test(
            identity, Tuple(randn(3)),
            (Composite{Tuple}(randn(3)...), Composite{Tuple}(randn(3)...))
        )
    end

    @testset "Constants" for x in (-0.1, 6.4, 1.0+0.5im, -10.0+0im, 0+200im)
        test_scalar(one, x)
        test_scalar(zero, x)
    end

    @testset "trinary ($f)" for f in (muladd, fma)
        x, Δx, x̄ = 10randn(3)
        y, Δy, ȳ = randn(3)
        z, Δz, z̄ = randn(3)
        Δk = randn()

        frule_test(f, (x, Δx), (y, Δy), (z, Δz))
        rrule_test(f, Δk, (x, x̄), (y, ȳ), (z, z̄))
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
