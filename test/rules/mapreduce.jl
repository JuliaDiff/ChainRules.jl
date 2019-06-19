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
        y, (_, _, dx) = rrule(mapreduce, abs2, +, X; dims=2)
        ȳ = randn(rng, size(y))
        x̄_ad = dx(ȳ)
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
end
