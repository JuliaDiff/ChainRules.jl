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
end
