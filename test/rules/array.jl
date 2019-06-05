@testset "reshape" begin
    rng = MersenneTwister(1)
    A = randn(rng, 4, 5)
    B, (dA, dd) = rrule(reshape, A, (5, 4))
    @test B == reshape(A, (5, 4))
    @test dd isa ChainRules.DNERule
    Ȳ = randn(rng, 4, 5)
    Ā = dA(Ȳ)
    @test Ā == reshape(Ȳ, (5, 4))

    B, (dA, dd1, dd2) = rrule(reshape, A, 5, 4)
    @test B == reshape(A, 5, 4)
    @test dd1 isa ChainRules.DNERule
    @test dd2 isa ChainRules.DNERule
    Ȳ = randn(rng, 4, 5)
    Ā = dA(Ȳ)
    @test Ā == reshape(Ȳ, 5, 4)
end
