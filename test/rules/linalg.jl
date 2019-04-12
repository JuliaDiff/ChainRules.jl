function construct_well_conditioned_matrix(rng, N)
    A = randn(rng, N, N)
    return A * A' + I
end

@testset "linalg" begin
    @testset "sum" begin
        rng, M, N = MersenneTwister(123456), 3, 4
        frule_test(sum, randn(rng, M, N), randn(rng, M, N))
        rrule_test(sum, randn(rng), randn(rng, M, N), randn(rng, M, N))
    end
    @testset "dot" begin
        rng, M, N = MersenneTwister(123456), 3, 4
        x, y = randn(rng, M, N), randn(rng, M, N)
        ẋ, ẏ = randn(rng, M, N), randn(rng, M, N)
        frule_test(dot, (x, y), (ẋ, ẏ))
    end
    @testset "inv" begin
        rng, N = MersenneTwister(123456), 3
        B = construct_well_conditioned_matrix(rng, N)
        frule_test(inv, B, randn(rng, N, N))
        rrule_test(inv, randn(rng, N, N), B, randn(rng, N, N))
    end
    @testset "det" begin
        rng, N = MersenneTwister(123456), 3
        B = construct_well_conditioned_matrix(rng, N)
        frule_test(det, B, randn(rng, N, N))
        rrule_test(det, randn(rng), B, randn(rng, N, N))
    end
    @testset "logdet" begin
        rng, N = MersenneTwister(123456), 3
        B = construct_well_conditioned_matrix(rng, N)
        frule_test(logdet, B, randn(rng, N, N))
        rrule_test(logdet, randn(rng), B, randn(rng, N, N))
    end
end
