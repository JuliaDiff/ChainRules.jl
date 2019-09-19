@testset "mean" begin
    rng = MersenneTwister(999)
    n = 9

    @testset "Basic" begin
        rrule_test(
            mean,
            randn(rng),
            (randn(rng, n),
            randn(rng, n))
        )
    end

    @testset "with function arg" begin
        rrule_test(
            mean,
            randn(rng),
            (abs2, nothing),
            (randn(rng, n),
            randn(rng, n))
        )
    end

    @testset "with dims kwargs" begin
        X = randn(rng, n, n+1)
        y, mean_pullback = rrule(mean, X; dims=1)
        ȳ = randn(rng, size(y))
        _, dX = mean_pullback(ȳ)
        X̄_ad = extern(dX)
        X̄_fd = j′vp(_fdm, x->mean(x, dims=1), ȳ, X)
        @test X̄_ad ≈ X̄_fd rtol=1e-9 atol=1e-9
    end
end
