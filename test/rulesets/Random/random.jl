@testset "random" begin
    @testset "MersenneTwister" begin
        @testset "no args" begin
            rng, pb = rrule(MersenneTwister)
            @test rng isa MersenneTwister
            @test first(pb(10)) isa Zero
        end
        @testset "unary" begin
            rng, pb = rrule(MersenneTwister, 123)
            @test rng isa MersenneTwister
            @test all(map(x -> x isa Zero, pb(10)))
        end
    end
end
