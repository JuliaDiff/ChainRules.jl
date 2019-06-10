@testset "Structured Matrices" begin
    @testset "Diagonal" begin
        rng, N = MersenneTwister(123456), 3
        rrule_test(Diagonal, randn(rng, N, N), (randn(rng, N), randn(rng, N)))
        D = Diagonal(randn(rng, N))
        rrule_test(Diagonal, D, (randn(rng, N), randn(rng, N)))
        # Concrete type instead of UnionAll
        rrule_test(typeof(D), D, (randn(rng, N), randn(rng, N)))
    end
    @testset "diag" begin
        rng, N = MersenneTwister(123456), 7
        rrule_test(diag, randn(rng, N), (randn(rng, N, N), randn(rng, N, N)))
        rrule_test(diag, randn(rng, N), (Diagonal(randn(rng, N)), randn(rng, N, N)))
        rrule_test(diag, randn(rng, N), (randn(rng, N, N), Diagonal(randn(rng, N))))
        rrule_test(diag, randn(rng, N), (Diagonal(randn(rng, N)), Diagonal(randn(rng, N))))
    end
    @testset "Symmetric" begin
        rng, N = MersenneTwister(123456), 3
        rrule_test(Symmetric, randn(rng, N, N), (randn(rng, N, N), randn(rng, N, N)))
    end
end
