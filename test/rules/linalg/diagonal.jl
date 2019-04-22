@testset "diagonal" begin
    @testset "Diagonal" begin
        rng, N = MersenneTwister(123456), 3
        rrule_test(Diagonal, randn(rng, N, N), (randn(rng, N), randn(rng, N)))
        rrule_test(Diagonal, Diagonal(randn(rng, N)), (randn(rng, N), randn(rng, N)))
    end
    @testset "diag" begin
        rng, N = MersenneTwister(123456), 7
        rrule_test(diag, randn(rng, N), (randn(rng, N, N), randn(rng, N, N)))
        rrule_test(diag, randn(rng, N), (Diagonal(randn(rng, N)), randn(rng, N, N)))
        rrule_test(diag, randn(rng, N), (randn(rng, N, N), Diagonal(randn(rng, N))))
        rrule_test(diag, randn(rng, N), (Diagonal(randn(rng, N)), Diagonal(randn(rng, N))))
    end
end
