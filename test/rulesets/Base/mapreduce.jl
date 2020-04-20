@testset "Maps and Reductions" begin
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
        @testset "keyword arguments" begin
            rng = MersenneTwister(33)
            n = 4
            X = randn(rng, n, n+1)
            y, pullback = rrule(sum, X; dims=2)
            ȳ = randn(rng, size(y))
            _, x̄_ad = pullback(ȳ)
            x̄_fd = only(j′vp(central_fdm(5, 1), x->sum(x, dims=2), ȳ, X))
            @test x̄_ad ≈ x̄_fd atol=1e-9 rtol=1e-9
        end
    end  # sum
end
