@testset "Maps and Reductions" begin
    @testset "map" begin
        rng = MersenneTwister(42)
        n = 10
        x = randn(rng, n)
        vx = randn(rng, n)
        ȳ = randn(rng, n)
        rrule_test(map, ȳ, (sin, nothing), (x, vx))
        rrule_test(map, ȳ, (+, nothing), (x, vx), (randn(rng, n), randn(rng, n)))
    end
    @testset "mapreduce" begin
        rng = MersenneTwister(6)
        n = 10
        x = randn(rng, n)
        vx = randn(rng, n)
        ȳ = randn(rng)
        rrule_test(mapreduce, ȳ, (sin, nothing), (+, nothing), (x, vx))

        # With keyword arguments (not yet supported in rrule_test)
        X = randn(rng, n, n)
        y, pullback = rrule(mapreduce, abs2, +, X; dims=2)
        ȳ = randn(rng, size(y))
        (_, _, _, x̄_ad) = pullback(ȳ)
        x̄_fd = j′vp(central_fdm(5, 1), x->mapreduce(abs2, +, x; dims=2), ȳ, X)
        @test x̄_ad ≈ x̄_fd atol=1e-9 rtol=1e-9
    end
    @testset "$f" for f in (mapfoldl, mapfoldr)
        rng = MersenneTwister(10)
        n = 7
        x = randn(rng, n)
        vx = randn(rng, n)
        ȳ = randn(rng)
        rrule_test(f, ȳ, (cos, nothing), (+, nothing), (x, vx))
    end
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
        @testset "function argument" begin
            rng = MersenneTwister(1)
            n = 8
            rrule_test(sum, randn(rng), (cos, nothing), (randn(rng, n), randn(rng, n)))
            rrule_test(sum, randn(rng), (abs2, nothing), (randn(rng, n), randn(rng, n)))
        end
        @testset "keyword arguments" begin
            rng = MersenneTwister(33)
            n = 4
            X = randn(rng, n, n+1)
            y, pullback = rrule(sum, X; dims=2)
            ȳ = randn(rng, size(y))
            _, x̄_ad = pullback(ȳ)
            x̄_fd = j′vp(central_fdm(5, 1), x->sum(x, dims=2), ȳ, X)
            @test x̄_ad ≈ x̄_fd atol=1e-9 rtol=1e-9
        end
    end  # sum
end
