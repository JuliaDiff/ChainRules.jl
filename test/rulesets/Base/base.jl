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
        @testset "Inverse degrees" begin
            x = 0.5
            test_scalar(asind, x)
            test_scalar(acosd, x)
            test_scalar(atand, x)
            test_scalar(asecd, 1/x)
            test_scalar(acscd, 1/x)
            test_scalar(acotd, 1/x)
        end
        @testset "Multivariate" begin
            x, y = rand(2)
            ratan = atan(x, y) # https://en.wikipedia.org/wiki/Atan2
            u = x^2 + y^2
            datan = y/u - 2x/u
            r, df = frule(atan, x, y)
            @test r === ratan
            @test df(1, 2) === datan
            r, (df1, df2) = rrule(atan, x, y)
            @test r === ratan
            @test df1(1) + df2(2) === datan

            rsincos = sincos(x)
            dsincos = cos(x) - 2sin(x)
            r, (df1, df2) = frule(sincos, x)
            @test r === rsincos
            @test df1(1) + df2(2) === dsincos
            r, df = rrule(sincos, x)
            @test r === rsincos
            @test df(1, 2) === dsincos
        end
    end
    @testset "Misc. Tests" begin
        @testset "*(x, y)" begin
            x, y = rand(3, 2), rand(2, 5)
            z, (dx, dy) = rrule(*, x, y)

            @test z == x * y

            z̄ = rand(3, 5)

            @test dx(z̄) == extern(accumulate(zeros(3, 2), dx, z̄))
            @test dy(z̄) == extern(accumulate(zeros(2, 5), dy, z̄))

            test_accumulation(rand(3, 2), dx, z̄, z̄ * y')
            test_accumulation(rand(2, 5), dy, z̄, x' * z̄)
        end
        @testset "hypot(x, y)" begin
            x, y = rand(2)
            h, dxy = frule(hypot, x, y)

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
    end
    @testset "identity" begin
        rng = MersenneTwister(1)
        n = 4
        rrule_test(identity, randn(rng), (randn(rng), randn(rng)))
        rrule_test(identity, randn(rng, 4), (randn(rng, 4), randn(rng, 4)))
    end
end
# TODO: Non-trig stuff
