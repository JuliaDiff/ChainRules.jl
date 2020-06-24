# Add tests to the quote for functions with  FastMath varients.
function jacobian_via_frule(f,z)
    du_dx, dv_dx = reim(frule((Zero(), 1),f,z)[2])
    du_dy, dv_dy = reim(frule((Zero(),im),f,z)[2])
    return [
        du_dx  du_dy
        dv_dx  dv_dy
    ]
end
function jacobian_via_rrule(f,z)
    _, pullback = rrule(f,z)
    du_dx, du_dy = reim(pullback( 1)[2])
    dv_dx, dv_dy = reim(pullback(im)[2])
    return [
        du_dx  du_dy
        dv_dx  dv_dy
    ]
end

function jacobian_via_fdm(f, z::Union{Real, Complex})
    fR2((x, y)) = (collect ∘ reim ∘ f)(x + im*y)
    v = float([real(z)
               imag(z)])
    j = jacobian(central_fdm(5,1), fR2, v)[1]
    if size(j) == (2,2)
        j
    elseif size(j) == (1, 2)
        [j 
         false false]
    else
        error("Invalid Jacobian size $(size(j))")
    end
end

function complex_jacobian_test(f, z)
    @test jacobian_via_fdm(f, z) ≈ jacobian_via_frule(f, z) 
    @test jacobian_via_fdm(f, z) ≈ jacobian_via_rrule(f, z)
end

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
        for f ∈ (abs, abs2, angle, conj), z ∈ (-4.1-0.02im, 6.4, 3 + im)
            @testset "Unary complex functions f = $f, z = $z" begin
                complex_jacobian_test(f, z)
            end
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
