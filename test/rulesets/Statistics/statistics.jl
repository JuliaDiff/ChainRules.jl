@testset "mean" begin
    n = 9

    @testset "Basic" begin
        rrule_test(mean, randn(), (randn(n), randn(n)))
    end

    @testset "with dims kwargs" begin
        X = randn(n, n+1)
        y, mean_pullback = rrule(mean, X; dims=1)
        ȳ = randn(size(y))
        _, dX = mean_pullback(ȳ)
        X̄_ad = extern(dX)
        X̄_fd = only(j′vp(_fdm, x->mean(x, dims=1), ȳ, X))
        @test X̄_ad ≈ X̄_fd rtol=1e-9 atol=1e-9
    end
end
