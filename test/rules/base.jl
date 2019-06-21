_isapprox(x, y; kwargs...) = isapprox(x, y; kwargs...)

const AbstractDifferential = ChainRules.AbstractDifferential

_isapprox(x::AbstractDifferential, y; kwargs...) = _isapprox(extern(x), y; kwargs...)
_isapprox(x, y::AbstractDifferential; kwargs...) = _isapprox(x, extern(y); kwargs...)
_isapprox(x::AbstractDifferential, y::AbstractDifferential; kwargs...) = _isapprox(extern(x), extern(y); kwargs...)

_isapprox(x::Wirtinger, y; kwargs...) = _isapprox(x.primal, y; kwargs...) &&
    _isapprox(x.conjugate, Zero(); kwargs...)
_isapprox(x, y::Wirtinger; kwargs...) = _isapprox(x, y.primal; kwargs...) &&
    _isapprox(Zero(), y.conjugate; kwargs...)
_isapprox(x::Wirtinger, y::Wirtinger; kwargs...) = _isapprox(x.primal, y.primal; kwargs...) &&
    _isapprox(x.conjugate, y.conjugate; kwargs...)

const ≈₁ = _isapprox

function test_scalar(f, f′, xs...)
    for r = (rrule, frule)
        rr = @inferred r(f, xs...)
        @test rr !== nothing
        fx, ∂x = rr
        @test fx == f(xs...)
        @test ∂x(1) ≈₁ f′(xs...) atol=1e-5
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
        # TODO: atan2 sincos
    end
    @testset "Misc. Tests" begin
        @testset "*(x, y)" for T in (Float32, Float64, Complex{Float64})
            x, y = rand(T, 3, 2), rand(T, 2, 5)
            z, (dx, dy) = rrule(*, x, y)

            @test z == x * y

            z̄ = rand(T, 3, 5)

            @test dx(z̄) == extern(accumulate(zeros(T, 3, 2), dx, z̄))
            @test dy(z̄) == extern(accumulate(zeros(T, 2, 5), dy, z̄))

            test_accumulation(rand(T, 3, 2), dx, z̄, z̄ * transpose(y))
            test_accumulation(rand(T, 2, 5), dy, z̄, transpose(x) * z̄)
        end
        @testset "hypot(x, y)" begin
            x, y = rand(2)
            h, dxy = frule(hypot, x, y)

            @test extern(dxy(One(), Zero())) ≈ x / h
            @test extern(dxy(Zero(), One())) ≈ y / h

            cx, cy = cast((One(), Zero())), cast((Zero(), One()))
            dx, dy = extern(dxy(cx, cy))
            @test dx ≈ x / h
            @test dy ≈ y / h

            cx, cy = cast((rand(), Zero())), cast((Zero(), rand()))
            dx, dy = extern(dxy(cx, cy))
            @test dx ≈ x / h * cx.value[1]
            @test dy ≈ y / h * cy.value[2]
        end
    end

    @testset "Unary complex functions" begin
        @testset "real input type" for x in rand.((Int, Float32, Float64))
            test_scalar(abs, x -> 1, abs(x))
            test_scalar(abs, x -> -1, -abs(x))
        end
        @testset "complex input type" for x in rand.(complex.((Float32, Float64)))
            test_scalar(abs, z -> Wirtinger(z' / 2abs(z), z / 2abs(z)), x)
        end
    end
end
# TODO: Non-trig stuff
