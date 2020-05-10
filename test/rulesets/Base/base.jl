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

    VERSION ≥ v"1.4" && @testset "evalpoly" begin
        # test fallbacks for when code generation fails
        @testset "fallbacks" begin
            x, p = randn(), Tuple(randn(10))
            @test ChainRules._evalpoly_intermediates_fallback(x, p) == ChainRules._evalpoly_intermediates(x, p)
            Δy, ys = randn(), Tuple(randn(10))
            @test ChainRules._evalpoly_back_fallback(x, p, ys, Δy) == ChainRules._evalpoly_back(x, p, ys, Δy)
        end
        @testset "frule" begin
            @testset "scalar" begin
                frule_test(evalpoly, (randn(), randn()), (randn(5), randn(5)))
                frule_test(evalpoly, (randn(), randn()), (Tuple(randn(5)), Tuple(randn(5))))
            end

            @testset "matrix" begin
                frule_test(
                    evalpoly, (randn(3, 3), randn(3, 3)),
                    ([randn(3, 3) for _ in 1:5], [randn(3, 3) for _ in 1:5]),
                )
                frule_test(
                    evalpoly, (randn(3, 3), randn(3, 3)),
                    (Tuple([randn(3, 3) for _ in 1:5]), Tuple([randn(3, 3) for _ in 1:5])),
                )
            end
        end

        @testset "rrule" begin
            @testset "$T" for T in (Float64,) # TODO: test ComplexF64
                @testset "(x::Number, pi::Number)" begin
                    rrule_test(
                        evalpoly, randn(T), (randn(T), randn(T)),
                        (randn(T, 5), randn(T, 5)),
                    )
                    rrule_test(
                        evalpoly, randn(T), (randn(T), randn(T)),
                        (Tuple(randn(T, 5)), Tuple(randn(T, 5))),
                    )
                end

                @testset "(x::Matrix, pi::Matrix)" begin
                    rrule_test(
                        evalpoly, randn(T, 3, 3), (randn(T, 3, 3), randn(T, 3, 3)),
                        ([randn(T, 3, 3) for i in 1:5], [randn(T, 3, 3) for i in 1:5]),
                    )
                    rrule_test(
                        evalpoly, randn(T, 3, 3), (randn(T, 3, 3), randn(T, 3, 3)),
                        (Tuple([randn(T, 3, 3) for i in 1:5]),
                         Tuple([randn(T, 3, 3) for i in 1:5])),
                    )
                end

                @testset "(x::Number, pi::Matrix)" begin
                    rrule_test(
                        evalpoly, randn(T, 3, 3), (randn(T), randn(T)),
                        ([randn(T, 3, 3) for i in 1:5], [randn(T, 3, 3) for i in 1:5]),
                    )
                    rrule_test(
                        evalpoly, randn(T, 3, 3), (randn(T), randn(T)),
                        (Tuple([randn(T, 3, 3) for i in 1:5]),
                         Tuple([randn(T, 3, 3) for i in 1:5])),
                    )
                end
            end
        end
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
end
