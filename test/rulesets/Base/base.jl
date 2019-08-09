function test_scalar(f, f′, xs...)
    for r = (rrule, frule)
        rr = r(f, xs...)
        @test rr !== nothing
        fx, ∂x = rr
        @test fx == f(xs...)
        @test ∂x(1) ≈ f′(xs...) atol=1e-5
    end
end

@testset "base" begin
    @testset "Trig" begin
        @testset "Basics" for x = (Float64(π), Complex(π, π/2))
            test_scalar(sin, cos, x)
            test_scalar(cos, x -> -sin(x), x)
            test_scalar(tan, x -> 1 + tan(x)^2, x)
            test_scalar(sec, x -> sec(x) * tan(x), x)
            test_scalar(csc, x -> -csc(x) * cot(x), x)
            test_scalar(cot, x -> -1 - cot(x)^2, x)
            test_scalar(sinpi, x -> π * cospi(x), x)
            test_scalar(cospi, x -> -π * sinpi(x), x)
        end
        @testset "Hyperbolic" for x = (Float64(π), Complex(π, π/2))
            test_scalar(sinh, cosh, x)
            test_scalar(cosh, sinh, x)
            test_scalar(tanh, x -> sech(x)^2, x)
            test_scalar(sech, x -> -tanh(x) * sech(x), x)
            test_scalar(csch, x -> -coth(x) * csch(x), x)
            test_scalar(coth, x -> -csch(x)^2, x)
        end
        @testset "Degrees" begin
            x = 45.0
            test_scalar(sind, x -> (π / 180) * cosd(x), x)
            test_scalar(cosd, x -> (-π / 180) * sind(x), x)
            test_scalar(tand, x -> (π / 180) * (1 + tand(x)^2), x)
            test_scalar(secd, x -> (π / 180) * secd(x) * tand(x), x)
            test_scalar(cscd, x -> (-π / 180) * cscd(x) * cotd(x), x)
            test_scalar(cotd, x -> (-π / 180) * (1 + cotd(x)^2), x)
        end
        @testset "Inverses" for x = (1.0, Complex(1.0, 0.25))
            test_scalar(asin, x -> 1 / sqrt(1 - x^2), x)
            test_scalar(acos, x -> -1 / sqrt(1 - x^2), x)
            test_scalar(atan, x -> 1 / (1 + x^2), x)
            test_scalar(asec, x -> 1 / (abs(x) * sqrt(x^2 - 1)), x)
            test_scalar(acsc, x -> -1 / (abs(x) * sqrt(x^2 - 1)), x)
            test_scalar(acot, x -> -1 / (1 + x^2), x)
        end
        @testset "Inverse hyperbolic" for x = (0.0, Complex(0.0, 0.25))
            test_scalar(asinh, x -> 1 / sqrt(x^2 + 1), x)
            test_scalar(acosh, x -> 1 / sqrt(x^2 - 1), x + 1)  # +1 accounts for domain
            test_scalar(atanh, x -> 1 / (1 - x^2), x)
            test_scalar(asech, x -> -1 / x / sqrt(1 - x^2), x)
            test_scalar(acsch, x -> -1 / abs(x) / sqrt(1 + x^2), x)
            test_scalar(acoth, x -> 1 / (1 - x^2), x + 1)
        end
        @testset "Inverse degrees" begin
            x = 1.0
            test_scalar(asind, x -> 180 / π / sqrt(1 - x^2), x)
            test_scalar(acosd, x -> -180 / π / sqrt(1 - x^2), x)
            test_scalar(atand, x -> 180 / π / (1 + x^2), x)
            test_scalar(asecd, x -> 180 / π / abs(x) / sqrt(x^2 - 1), x)
            test_scalar(acscd, x -> -180 / π / abs(x) / sqrt(x^2 - 1), x)
            test_scalar(acotd, x -> -180 / π / (1 + x^2), x)
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
