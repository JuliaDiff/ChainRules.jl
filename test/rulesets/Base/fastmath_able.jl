# Add tests to the quote for functions with  FastMath varients.
function jacobian_via_frule(f,z)
    du_dx, dv_dx = reim(frule((ZeroTangent(), 1),f,z)[2])
    du_dy, dv_dy = reim(frule((ZeroTangent(),im),f,z)[2])
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

# IMPORTANT:
# Do not add any tests here for functions that do not have varients in Base.FastMath
# e.g. do not add `foo` unless `Base.FastMath.foo_fast` exists.
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
            @testset "sincos(x::$T)" for T in (Float64, ComplexF64)
                Δz = Tangent{Tuple{T,T}}(randn(T), randn(T))

                test_frule(sincos, randn(T))
                test_rrule(sincos, randn(T); output_tangent=Δz)
            end
            if VERSION ≥ v"1.6"
                @testset "sincospi(x::$T)" for T in (Float64, ComplexF64)
                    Δz = Tangent{Tuple{T,T}}(randn(T), randn(T))

                    test_frule(sincospi, randn(T))
                    test_rrule(sincospi, randn(T); output_tangent=Δz)
                end
            end
        end
    end

    @testset "exponents" begin
        for x in (-0.1, 7.9, 0.5 + 0.25im)
            test_scalar(inv, x)

            test_scalar(exp, x)
            test_scalar(exp2, x)
            test_scalar(exp10, x)
            test_scalar(expm1, x)

            if x isa Real
                test_scalar(cbrt, x)
            end

            if x isa Complex || x >= 0
                test_scalar(sqrt, x)
                test_scalar(log, x)
                test_scalar(log2, x)
                test_scalar(log10, x)
                test_scalar(log1p, x)
            end
        end
    end

    @testset "Unary complex functions" begin
        for f ∈ (abs, abs2, conj), z ∈ (-4.1-0.02im, 6.4, 3 + im)
            @testset "Unary complex functions f = $f, z = $z" begin
                complex_jacobian_test(f, z)
            end
        end
        # As per PR #196, angle gives a ZeroTangent() pullback for Real z and ΔΩ, rather than
        # the one you'd get from considering the reals as embedded in the complex plane
        # so we need to special case it's tests
        for z ∈ (-4.1-0.02im, 6.4 + 0im, 3 + im)
            complex_jacobian_test(angle, z)
        end
        @test frule((ZeroTangent(), randn()), angle, randn())[2] === ZeroTangent()
        @test rrule(angle, randn())[2](randn())[2] === ZeroTangent()

        # test that real primal with complex tangent gives complex tangent
        ΔΩ = randn(ComplexF64)
        for x in (-0.5, 2.0)
            @test isapprox(
                frule((ZeroTangent(), ΔΩ), angle, x)[2],
                frule((ZeroTangent(), ΔΩ), angle, complex(x))[2],
            )
        end
    end

    @testset "Unary functions" begin
        for x in (-4.1, 6.4, 0.0, 0.0 + 0.0im, 0.5 + 0.25im)
            test_scalar(+, x)
            test_scalar(-, x)
            test_scalar(atan, x)
        end
    end

    @testset "binary functions" begin
        @testset "$f(x, y)" for f in (atan, rem, max, min)
            # be careful not to sample near singularities for `rem`
            base = rand() + 1
            test_frule(f, (rand(0:10) + .6rand() + .2) * base, base)
            base = rand() + 1
            test_rrule(f, (rand(0:10) + .6rand() + .2) * base, base)
        end

        @testset "$f(x::$T, y::$T)" for f in (/, +, -, hypot), T in (Float64, ComplexF64)
            test_frule(f, 10rand(T), rand(T))
            test_rrule(f, 10rand(T), rand(T))
        end

        @testset "$f(x::$T, y::$T) type check" for f in (/, +, -,\, hypot, ^), T in (Float32, Float64)
            x, Δx, x̄ = 10rand(T, 3)
            y, Δy, ȳ = rand(T, 3)
            @assert T == typeof(f(x, y))
            Δz = randn(typeof(f(x, y)))

            @test frule((ZeroTangent(), Δx, Δy), f, x, y) isa Tuple{T, T}
            _, ∂x, ∂y = rrule(f, x, y)[2](Δz)
            @test (∂x, ∂y) isa Tuple{T, T}

            if f != hypot
                # Issue #233
                @test frule((ZeroTangent(), Δx, Δy), f, x, 2) isa Tuple{T, T}
                _, ∂x, ∂y = rrule(f, x, 2)[2](Δz)
                @test (∂x, ∂y) isa Tuple{T, Float64}

                @test frule((ZeroTangent(), Δx, Δy), f, 2, y) isa Tuple{T, T}
                _, ∂x, ∂y = rrule(f, 2, y)[2](Δz)
                @test (∂x, ∂y) isa Tuple{Float64, T}
            end
        end

        @testset "^(x::$T, n::$T)" for T in (Float64, ComplexF64)
            # for real x and n, x must be >0
            test_frule(^, rand(T) + 3, rand(T) + 3)
            test_rrule(^, rand(T) + 3, rand(T) + 3)

            T <: Real && @testset "discontinuity for ^(x::Real, n::Int) when x ≤ 0" begin
                # finite differences doesn't work for x < 0, so we check manually
                x = -rand(T) .- 3
                y = 3
                Δx = randn(T)
                Δy = randn(T)
                Δz = randn(T)

                @test frule((ZeroTangent(), Δx, Δy), ^, x, y)[2] ≈ Δx * y * x^(y - 1)
                @test frule((ZeroTangent(), Δx, Δy), ^, zero(x), y)[2] ≈ 0
                _, ∂x, ∂y = rrule(^, x, y)[2](Δz)
                @test ∂x ≈ Δz * y * x^(y - 1)
                @test ∂y ≈ 0
                _, ∂x, ∂y = rrule(^, zero(x), y)[2](Δz)
                @test ∂x ≈ 0
                @test ∂y ≈ 0
            end
        end
    end

    @testset "sign" begin
        @testset "real" begin
            @testset "at $x" for x in (-1.1, -1.1, 0.5, 100.0)
                test_scalar(sign, x)
            end

            @testset "ZeroTangent over the point discontinuity" begin
                # Can't do finite differencing because we are lying
                # following the subgradient convention.

                _, pb = rrule(sign, 0.0)
                _, x̄ = pb(10.5)
                test_approx(x̄, 0)

                _, ẏ = frule((ZeroTangent(), 10.5), sign, 0.0)
                test_approx(ẏ, 0)
            end
        end
        @testset "complex" begin
            @testset "at $z" for z in (-1.1 + randn() * im, 0.5 + randn() * im)
                test_scalar(sign, z)

                # test that complex (co)tangents with real primal gives same result as
                # complex primal with zero imaginary part

                ż, ΔΩ = randn(ComplexF64, 2)
                Ω, ∂Ω = frule((ZeroTangent(), ż), sign, real(z))
                @test Ω == sign(real(z))
                @test ∂Ω ≈ frule((ZeroTangent(), ż), sign, real(z) + 0im)[2]

                Ω, pb = rrule(sign, real(z))
                @test Ω == sign(real(z))
                @test pb(ΔΩ)[2] ≈ rrule(sign, real(z) + 0im)[2](ΔΩ)[2]
            end

            @testset "zero over the point discontinuity" begin
                # Can't do finite differencing because we are lying
                # following the subgradient convention.

                _, pb = rrule(sign, 0.0 + 0.0im)
                _, z̄ = pb(randn(ComplexF64))
                @test z̄ == 0.0 + 0.0im

                _, Ω̇ = frule((ZeroTangent(), randn(ComplexF64)), sign, 0.0 + 0.0im)
                @test Ω̇ == 0.0 + 0.0im
            end
        end
    end
end

# Now we generate tests for fast and nonfast versions
@eval @interpret (function()
    @testset "fastmath_able Base functions" begin
        $FASTABLE_AST
    end
    
    
    @testset "fastmath_able FastMath functions" begin
        $(Base.FastMath.make_fastmath(FASTABLE_AST))
    end
end)()
