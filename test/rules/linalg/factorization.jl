@testset "Factorizations" begin
    @testset "svd" begin
        rng = MersenneTwister(2)
        for n in [4, 6, 10], m in [3, 5, 10]
            X = randn(rng, n, m)
            F, dX = rrule(svd, X)
            for p in [:U, :S, :V, :Vt]
                Y, (dF, dp) = rrule(getproperty, F, p)
                @test dp isa ChainRules.DNERule
                Ȳ = randn(rng, size(Y)...)
                X̄_ad = dX(dF(Ȳ))
                X̄_fd = j′vp(central_fdm(5, 1), X->getproperty(svd(X), p), Ȳ, X)
                @test X̄_ad ≈ X̄_fd rtol=1e-6 atol=1e-6
            end
        end
        @testset "Helper functions" begin
            X = randn(rng, 10, 10)
            Y = randn(rng, 10, 10)
            @test ChainRules._mulsubtrans!(copy(X), Y) ≈ Y .* (X - X')
            @test ChainRules._eyesubx!(copy(X)) ≈ I - X
            @test ChainRules._add!(copy(X), Y) ≈ X + Y
        end
    end
end
