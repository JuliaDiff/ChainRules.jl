@testset "base" begin
    @testset "Trig" begin
        @testset "Basics" for x = (Float64(π)-0.01, Complex(π, π/2))
            test_scalar(sin, x)
            test_scalar(cos, x)
            test_scalar(tan, x)
            test_scalar(sec, x)
            test_scalar(csc, x)
            test_scalar(cot, x)
            test_scalar(sinpi, x)
            test_scalar(cospi, x)
        end
        @testset "Hyperbolic" for x = (Float64(π)-0.01, Complex(π-0.01, π/2))
            test_scalar(sinh, x)
            test_scalar(cosh, x)
            test_scalar(tanh, x)
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
            test_scalar(asin, x)
            test_scalar(acos, x)
            test_scalar(atan, x)
            test_scalar(asec, 1/x)
            test_scalar(acsc, 1/x)
            test_scalar(acot, 1/x)
        end
        @testset "Inverse hyperbolic" for x = (0.5, Complex(0.5, 0.25))
            test_scalar(asinh, x)
            test_scalar(acosh, x + 1)  # +1 accounts for domain
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
        @testset "Multivariate" begin
            @testset "atan2" begin
                # https://en.wikipedia.org/wiki/Atan2
                x, y = rand(2)
                ratan = atan(x, y)
                u = x^2 + y^2
                datan = y/u - 2x/u

                r, pushforward = frule(atan, x, y)
                @test r === ratan
                @test pushforward(NamedTuple(), 1, 2) === datan

                r, pullback = rrule(atan, x, y)
                @test r === ratan
                dself, df1, df2 = pullback(1)
                @test dself == NO_FIELDS
                @test df1 + 2df2 === datan
            end

            @testset "sincos" begin
                x, Δx, x̄ = randn(3)
                Δz = (randn(), randn())

                frule_test(sincos, (x, Δx))
                rrule_test(sincos, Δz, (x, x̄))
            end
        end
    end  # Trig

    @testset "math" begin
        for x in (-0.1, 6.4, 1.0+0.5im, -10.0+0im)
            test_scalar(deg2rad, x)
            test_scalar(rad2deg, x)

            test_scalar(inv, x)

            test_scalar(exp, x)
            test_scalar(exp2, x)
            test_scalar(exp10, x)

            x isa Real && test_scalar(cbrt, x)
            if (x isa Real && x >= 0) || x isa Complex
                # this check is needed because these have discontinuities between
                # `-10 + im*eps()` and `-10 - im*eps()`
                should_test_wirtinger = imag(x) != 0 && real(x) < 0
                test_scalar(sqrt, x; test_wirtinger=should_test_wirtinger)
                test_scalar(log, x; test_wirtinger=should_test_wirtinger)
                test_scalar(log2, x; test_wirtinger=should_test_wirtinger)
                test_scalar(log10, x; test_wirtinger=should_test_wirtinger)
                test_scalar(log1p, x; test_wirtinger=should_test_wirtinger)
            end
        end
    end

    @testset "Unary complex functions" begin
        for x in (-4.1, 6.4, 1.0+0.5im, -10.0+1.5im)
            test_scalar(real, x)
            test_scalar(imag, x)

            test_scalar(abs, x)
            test_scalar(hypot, x)

            test_scalar(angle, x)
            test_scalar(abs2, x)
            test_scalar(conj, x)
        end
    end

    @testset "*(x, y)" begin
        x, y = rand(3, 2), rand(2, 5)
        z, pullback = rrule(*, x, y)

        @test z == x * y

        z̄ = rand(3, 5)
        (ds, dx, dy) = pullback(z̄)

        @test ds === NO_FIELDS

        @test extern(dx) == extern(accumulate(zeros(3, 2), dx))
        @test extern(dy) == extern(accumulate(zeros(2, 5), dy))

        test_accumulation(rand(3, 2), dx)
        test_accumulation(rand(2, 5), dy)
    end

    @testset "binary trig ($f)" for f in (hypot, atan)
        rng = MersenneTwister(123456)
        x, Δx, x̄ = 10randn(rng, 3)
        y, Δy, ȳ = randn(rng, 3)
        Δz = randn(rng)

        frule_test(f, (x, Δx), (y, Δy))
        rrule_test(f, Δz, (x, x̄), (y, ȳ))
    end

    @testset "identity" begin
        rng = MersenneTwister(1)
        rrule_test(identity, randn(rng), (randn(rng), randn(rng)))
        rrule_test(identity, randn(rng, 4), (randn(rng, 4), randn(rng, 4)))
    end

    @testset "Constants" for x in (-0.1, 6.4, 1.0+0.5im, -10.0+0im, 0+200im)
        test_scalar(one, x)
        test_scalar(zero, x)
    end

    @testset "sign" begin
        @testset "at points" for x in (-1.1, -1.1, 0.5, 100)
            test_scalar(sign, x)
        end

        @testset "Zero over the point discontinuity" begin
            # Can't do finite differencing because we are lying
            # following the subgradient convention.

            _, pb = rrule(sign, 0.0)
            _, x̄ = pb(10.5)
            @test extern(x̄) == 0

            _, pf = frule(sign, 0.0)
            ẏ = pf(NamedTuple(), 10.5)
            @test extern(ẏ) == 0
        end
    end
end
