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
    @testset "$f" for f in (Adjoint, adjoint, Transpose, transpose)
        rng = MersenneTwister(32)
        n = 5
        m = 3
        rrule_test(f, randn(rng, m, n), (randn(rng, n, m), randn(rng, n, m)))
        rrule_test(f, randn(rng, 1, n), (randn(rng, n), randn(rng, n)))
    end
    @testset "$T" for T in (UpperTriangular, LowerTriangular)
        rng = MersenneTwister(33)
        n = 5
        rrule_test(T, T(randn(rng, n, n)), (randn(rng, n, n), randn(rng, n, n)))
    end
end
