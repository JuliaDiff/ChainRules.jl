function test_scalar(f, f′, xs...)
    for r = (rrule, frule)
        rr = r(f, xs...)
        @test rr !== nothing
        fx, ∂x = rr
        @test fx == f(xs...)
        @test ∂x(1) ≈ f′(xs...) atol=1e-5
    end
end

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
    # TODO: atan2 sincos
end

# TODO: Non-trig stuff
