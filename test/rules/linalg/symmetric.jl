@testset "symmetric" begin
    @testset "Symmetric" begin
        rng, N = MersenneTwister(123456), 3
        rrule_test(Symmetric, randn(rng, N, N), (randn(rng, N, N), randn(rng, N, N)))
    end
end
