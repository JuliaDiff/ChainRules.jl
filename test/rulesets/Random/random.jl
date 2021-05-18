# Simple Distributions like object for testing purposes
struct NormalDistribution
    μ
    σ
end
Random.rand(d::NormalDistribution) = d.μ + d.σ*randn()

@testset "random" begin
    @testset "MersenneTwister" begin
        @testset "no args" begin
            rng, dΩ = frule((5.0,), MersenneTwister)
            @test rng isa MersenneTwister
            @test dΩ isa ZeroTangent

            rng, pb = rrule(MersenneTwister)
            @test rng isa MersenneTwister
            @test first(pb(10)) isa ZeroTangent
        end
        @testset "unary" begin
            rng, dΩ = frule((5.0, 4.0), MersenneTwister, 123)
            @test rng isa MersenneTwister
            @test dΩ isa ZeroTangent

            rng, pb = rrule(MersenneTwister, 123)
            @test rng isa MersenneTwister
            @test all(map(x -> x isa ZeroTangent, pb(10)))
        end
    end

    @testset "rand" begin
        non_differentiables = [
            ((), Float64),
            ((MersenneTwister(123),), Float64),
            ((MersenneTwister(123),2,2), Matrix{<:Float64}),
            ((Float32,), Float32),
            ((Float32,2,2), Matrix{<:Float32}),
            ((Float32,(2,2)), Matrix{<:Float32}),
            ((2,2), Matrix{<:Float64}),
        ]

        for (args, xType) in non_differentiables
            x, dΩ = frule((ZeroTangent(), randn(args...)), rand, args...)
            @test x isa xType
            @test dΩ isa NoTangent

            x, pb = rrule(rand, args...)
            @test x isa xType
            dself, dargs = Iterators.peel(pb(10.0))
            @test iszero(dself)
            @test all(darg isa NoTangent for darg in dargs)
        end

        # Make sure that we do *not* have these set as non_differentiable. as they are differentiable
        @test nothing === frule(
            (ZeroTangent(), Tangent{NormalDistribution}(μ=0.5,σ=2.0)),
            rand,
            NormalDistribution(0.1,1.5),
        )
        @test rrule(rand, NormalDistribution(0.1,1.5)) === nothing
    end
end
