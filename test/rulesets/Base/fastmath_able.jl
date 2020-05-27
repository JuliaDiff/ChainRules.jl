# Add tests to the quote for functions with  FastMath varients.
const FASTABLE_AST = quote
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
            @testset "sincos" begin
                x, Δx, x̄ = randn(3)
                Δz = (randn(), randn())

                frule_test(sincos, (x, Δx))
                rrule_test(sincos, Δz, (x, x̄))
            end
        end
    end

    @testset "exponents" begin
        for x in (-0.1, 6.4)
            test_scalar(inv, x)

            test_scalar(exp, x)
            test_scalar(exp2, x)
            test_scalar(exp10, x)
            test_scalar(expm1, x)

            test_scalar(cbrt, x)

            if x >= 0
                test_scalar(sqrt, x)
                test_scalar(log, x)
                test_scalar(log2, x)
                test_scalar(log10, x)
                test_scalar(log1p, x)
            end
        end
    end

    @testset "Unary complex functions" begin
        for x in (-4.1, 6.4, 3 + im)
            test_scalar(abs, x)
            test_scalar(angle, x)
            test_scalar(abs2, x)
            test_scalar(conj, x)
        end
    end

    @testset "Unary functions" begin
        for x in (-4.1, 6.4)
            test_scalar(+, x)
            test_scalar(-, x)
        end
    end

    @testset "binary function ($f)" for f in (/, +, -, hypot, atan, rem, ^, max, min)
        x, Δx, x̄ = 10rand(3)
        y, Δy, ȳ = rand(3)
        Δz = rand()

        frule_test(f, (x, Δx), (y, Δy))
        rrule_test(f, Δz, (x, x̄), (y, ȳ))
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

            _, ẏ = frule((Zero(), 10.5), sign, 0.0)
            @test extern(ẏ) == 0
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
