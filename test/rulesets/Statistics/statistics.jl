@testset "mean" begin
    rng = MersenneTwister(999)
    n = 9
    rrule_test(mean, randn(rng), (abs2, nothing), (randn(rng, n), randn(rng, n)))
    X = randn(rng, n, n)
    y, dX = rrule(mean, X; dims=1)
    ȳ = randn(rng, size(y))
    X̄_ad = dX(ȳ)
    X̄_fd = j′vp(central_fdm(5, 1), x->mean(x, dims=1), ȳ, X)
    @test X̄_ad ≈ X̄_fd rtol=1e-9 atol=1e-9
end
