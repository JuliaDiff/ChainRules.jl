@testset "random" begin
    @testset "MersenneTwister" begin
        @testset "no args" begin
            rng, dΩ = frule((5.0,), MersenneTwister)
            @test rng isa MersenneTwister
            @test dΩ isa Zero

            rng, pb = rrule(MersenneTwister)
            @test rng isa MersenneTwister
            @test first(pb(10)) isa Zero
        end
        @testset "unary" begin
            rng, dΩ = frule((5.0, 4.0), MersenneTwister, 123)
            @test rng isa MersenneTwister
            @test dΩ isa Zero

            rng, pb = rrule(MersenneTwister, 123)
            @test rng isa MersenneTwister
            @test all(map(x -> x isa Zero, pb(10)))
        end
    end

    @testset "rand" begin
        non_differentiables = [((), Float64),
                               ((MersenneTwister(123),), Float64),
                               ((MersenneTwister(123),2,2), Matrix{<:Float64}),
                               ((Float32,), Float32),
                               ((Float32,2,2), Matrix{<:Float32}),
                               ((Float32,(2,2)), Matrix{<:Float32}),
                               ((2,2), Matrix{<:Float64})]

        for (args,xType) in non_differentiables
            x, dΩ = frule((), rand, args...)
            @test x isa xType
            @test dΩ isa DoesNotExist

            x, pb = rrule(rand, args...)
            @test x isa xType
            @test first(pb(10)) isa Zero
        end

        @test frule((), rand, ones(2,2)) === nothing
        @test rrule(rand, ones(2,2)) === nothing
    end
end
