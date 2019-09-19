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
            x, y = rand(2)
            @testset "atan2" begin
                # https://en.wikipedia.org/wiki/Atan2
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
                rsincos = sincos(x)
                dsincos = cos(x) - 2sin(x)

                r, pushforward = frule(sincos, x)
                @test r === rsincos
                df1, df2 = pushforward(NamedTuple(), 1)
                @test df1 + 2df2 === dsincos

                r, pullback = rrule(sincos, x)
                @test r === rsincos
                ds, df = pullback(1, 2)
                @test df === dsincos
                @test ds === NO_FIELDS
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
        for x in (-6, rand.((Float32, Float64, Complex{Float32}, Complex{Float64}))...)
            rtol = x isa Complex{Float32} ? 1e-6 : 1e-9
            test_scalar(real, x; rtol=rtol)
            test_scalar(imag, x; rtol=rtol)

            test_scalar(abs, x; rtol=rtol)
            test_scalar(hypot, x; rtol=rtol)

            test_scalar(angle, x; rtol=rtol)
            test_scalar(abs2, x; rtol=rtol)
            test_scalar(conj, x; rtol=rtol)
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

    @testset "hypot(x, y)" begin
        x, y = rand(2)
        h, pushforward = frule(hypot, x, y)
        dxy(x, y) = pushforward(NamedTuple(), x, y)

        @test extern(dxy(One(), Zero())) === x / h
        @test extern(dxy(Zero(), One())) === y / h

        cx, cy = cast((One(), Zero())), cast((Zero(), One()))
        dx, dy = extern(dxy(cx, cy))
        @test dx === x / h
        @test dy === y / h

        cx, cy = cast((rand(), Zero())), cast((Zero(), rand()))
        dx, dy = extern(dxy(cx, cy))
        @test dx === x / h * cx.value[1]
        @test dy === y / h * cy.value[2]
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
end
