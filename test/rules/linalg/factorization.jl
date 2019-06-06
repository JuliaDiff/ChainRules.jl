@testset "Factorizations" begin
    @testset "svd" begin
        rng = MersenneTwister(2)
        for n in [4, 6, 10], m in [3, 5, 10]
            X = randn(rng, n, m)
            F, dX = rrule(svd, X)
            for p in [:U, :S, :V]
                Y, (dF, dp) = rrule(getproperty, F, p)
                @test dp isa ChainRules.DNERule
                Ȳ = randn(rng, size(Y)...)
                X̄_ad = dX(dF(Ȳ))
                X̄_fd = j′vp(central_fdm(5, 1), X->getproperty(svd(X), p), Ȳ, X)
                @test X̄_ad ≈ X̄_fd rtol=1e-6 atol=1e-6
            end
            @test_throws ArgumentError rrule(getproperty, F, :Vt)
        end
        @testset "accumulate!" begin
            X = [1.0 2.0; 3.0 4.0; 5.0 6.0]
            F, dX = rrule(svd, X)
            X̄ = (U=zeros(3, 2), S=zeros(2), V=zeros(2, 2))
            for p in [:U, :S, :V]
                Y, (dF, _) = rrule(getproperty, F, p)
                Ȳ = ones(size(Y)...)
                ChainRules.accumulate!(X̄, dF, Ȳ)
            end
            @test X̄.U ≈ ones(3, 2) atol=1e-6
            @test X̄.S ≈ ones(2) atol=1e-6
            @test X̄.V ≈ ones(2, 2) atol=1e-6
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
