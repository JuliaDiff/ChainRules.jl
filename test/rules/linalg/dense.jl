function generate_well_conditioned_matrix(rng, N)
    A = randn(rng, N, N)
    return A * A' + I
end

@testset "linalg" begin
    @testset "sum" begin
        @testset "Vector" begin
            rng, M = MersenneTwister(123456), 3
            frule_test(sum, (randn(rng, M), randn(rng, M)))
            rrule_test(sum, randn(rng), (randn(rng, M), randn(rng, M)))
        end
        @testset "Matrix" begin
            rng, M, N = MersenneTwister(123456), 3, 4
            frule_test(sum, (randn(rng, M, N), randn(rng, M, N)))
            rrule_test(sum, randn(rng), (randn(rng, M, N), randn(rng, M, N)))
        end
        @testset "Array{T, 3}" begin
            rng, M, N, P = MersenneTwister(123456), 3, 7, 11
            frule_test(sum, (randn(rng, M, N, P), randn(rng, M, N, P)))
            rrule_test(sum, randn(rng), (randn(rng, M, N, P), randn(rng, M, N, P)))
        end
    end
    @testset "dot" begin
        @testset "Vector" begin
            rng, M = MersenneTwister(123456), 3
            x, y = randn(rng, M), randn(rng, M)
            ẋ, ẏ = randn(rng, M), randn(rng, M)
            x̄, ȳ = randn(rng, M), randn(rng, M)
            frule_test(dot, (x, ẋ), (y, ẏ))
            rrule_test(dot, randn(rng), (x, x̄), (y, ȳ))
        end
        @testset "Matrix" begin
            rng, M, N = MersenneTwister(123456), 3, 4
            x, y = randn(rng, M, N), randn(rng, M, N)
            ẋ, ẏ = randn(rng, M, N), randn(rng, M, N)
            x̄, ȳ = randn(rng, M, N), randn(rng, M, N)
            frule_test(dot, (x, ẋ), (y, ẏ))
            rrule_test(dot, randn(rng), (x, x̄), (y, ȳ))
        end
        @testset "Array{T, 3}" begin
            rng, M, N, P = MersenneTwister(123456), 3, 4, 5
            x, y = randn(rng, M, N, P), randn(rng, M, N, P)
            ẋ, ẏ = randn(rng, M, N, P), randn(rng, M, N, P)
            x̄, ȳ = randn(rng, M, N, P), randn(rng, M, N, P)
            frule_test(dot, (x, ẋ), (y, ẏ))
            rrule_test(dot, randn(rng), (x, x̄), (y, ȳ))
        end
    end
    @testset "inv" begin
        rng, N = MersenneTwister(123456), 3
        B = generate_well_conditioned_matrix(rng, N)
        frule_test(inv, (B, randn(rng, N, N)))
        rrule_test(inv, randn(rng, N, N), (B, randn(rng, N, N)))
    end
    @testset "det" begin
        rng, N = MersenneTwister(123456), 3
        B = generate_well_conditioned_matrix(rng, N)
        frule_test(det, (B, randn(rng, N, N)))
        rrule_test(det, randn(rng), (B, randn(rng, N, N)))
    end
    @testset "logdet" begin
        rng, N = MersenneTwister(123456), 3
        B = generate_well_conditioned_matrix(rng, N)
        frule_test(logdet, (B, randn(rng, N, N)))
        rrule_test(logdet, randn(rng), (B, randn(rng, N, N)))
    end
    @testset "tr" begin
        rng, N = MersenneTwister(123456), 4
        frule_test(tr, (randn(rng, N, N), randn(rng, N, N)))
        rrule_test(tr, randn(rng), (randn(rng, N, N), randn(rng, N, N)))
    end
end
