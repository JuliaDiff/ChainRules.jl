    VERSION ≥ v"1.4" && @testset "evalpoly" begin
        # test fallbacks for when code generation fails
        @testset "fallbacks for $T" for T in (Float64, ComplexF64)
            x, p = randn(T), Tuple(randn(T, 10))
            y_fb, ys_fb = ChainRules._evalpoly_intermediates_fallback(x, p)
            y, ys = ChainRules._evalpoly_intermediates(x, p)
            @test y_fb ≈ y
            @test collect(ys_fb) ≈ collect(ys)

            Δy, ys = randn(T), Tuple(randn(T, 9))
            ∂x_fb, ∂p_fb = ChainRules._evalpoly_back_fallback(x, p, ys, Δy)
            ∂x, ∂p = ChainRules._evalpoly_back(x, p, ys, Δy)
            @test ∂x_fb ≈ ∂x
            @test collect(∂p_fb) ≈ collect(∂p)
        end

        @testset "x dim: $(nx), pi dim: $(np), type: $T" for T in (Float64, ComplexF64), nx in (tuple(), 3), np in (tuple(), 3)
            # skip x::Matrix, pi::Number case, which is not supported by evalpoly
            isempty(np) && !isempty(nx) && continue
            m = 5
            sx = (nx..., nx...)
            sp = (np..., np...)
            x, ẋ, x̄ = randn(T, sx...), randn(T, sx...), randn(T, sx...)
            p = [randn(T, sp...) for _ in 1:m]
            ṗ = [randn(T, sp...) for _ in 1:m]
            p̄ = [randn(T, sp...) for _ in 1:m]
            Ω = evalpoly(x, p)
            Ω̄ = randn(T, size(Ω)...)
            frule_test(evalpoly, (x, ẋ), (p, ṗ))
            frule_test(evalpoly, (x, ẋ), (Tuple(p), Tuple(ṗ)))
            rrule_test(evalpoly, Ω̄, (x, x̄), (p, p̄))
            rrule_test(evalpoly, Ω̄, (x, x̄), (Tuple(p), Tuple(p̄)))
        end
    end

