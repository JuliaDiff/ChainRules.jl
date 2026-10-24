using ChainRules, ChainRulesCore, Random, Test

# Simple Distributions like object for testing purposes
struct NormalDistribution
    μ
    σ
end
Random.rand(d::NormalDistribution) = d.μ + d.σ*randn()

@testset "Random" begin
    rng_types = [MersenneTwister, Xoshiro]
    
    @testset "$Rng" for Rng in rng_types
        @testset "no args" begin
            rng, dΩ = frule((5.0,), Rng)
            @test rng isa Rng
            @test dΩ isa ZeroTangent

            rng, pb = rrule(Rng)
            @test rng isa Rng
            @test first(pb(10)) isa typeof(NoTangent())
        end
        @testset "unary" begin
            rng, dΩ = frule((5.0, 4.0), Rng, 123)
            @test rng isa Rng
            @test dΩ isa ZeroTangent

            rng, pb = rrule(Rng, 123)
            @test rng isa Rng
            @test all(map(x -> x isa AbstractZero, pb(10)))
        end
    end

    @testset "rand" begin
        # Make sure that we do *not* have these set as non_differentiable. as they are differentiable
        @test nothing === frule(
            (ZeroTangent(), Tangent{NormalDistribution}(μ=0.5,σ=2.0)),
            rand,
            NormalDistribution(0.1,1.5),
        )
        @test rrule(rand, NormalDistribution(0.1,1.5)) === nothing
    end

    @testset "randstring" begin
        for args in ((), (5,), (Xoshiro(123),), (Xoshiro(123), 5), ("ab",), (Xoshiro(123), 'a':'z', 5))
            s, ds = frule((NoTangent(), map(Returns(NoTangent()), args)...), randstring, args...)
            @test s isa String
            @test ds isa NoTangent
            s, pb = rrule(randstring, args...)
            @test s isa String
            @test all(darg isa NoTangent for darg in pb(NoTangent()))
        end
    end

    @testset "rand(rng, ::Sampler)" begin
        rng = Xoshiro(123)

        # Samplers without any differentiable data and samplers for integer ranges are non-differentiable
        for X in (
            Float64, Float32, Int, Bool, ComplexF64, BigFloat, "abc", SubString("abc", 2), 'a':'z', ['a', 'b'],
            1:4, Base.OneTo(4), Int8(1):Int8(4), UInt128(1):UInt128(4), typemin(Int):typemax(Int), big(1):big(4),
        )
            sp = Random.Sampler(rng, X, Val(1))
            x, dx = frule((NoTangent(), NoTangent(), NoTangent()), rand, rng, sp)
            @test x isa eltype(sp)
            @test dx isa NoTangent
            x, pb = rrule(rand, rng, sp)
            @test x isa eltype(sp)
            @test all(darg isa NoTangent for darg in pb(1.0))

            A = Vector{eltype(sp)}(undef, 3)
            spInf = Random.Sampler(rng, X, Val(Inf))
            Ω, dΩ = frule((NoTangent(), NoTangent(), NoTangent(), NoTangent()), rand!, rng, A, spInf)
            @test Ω === A
            @test dΩ isa NoTangent
            Ω, pb! = rrule(rand!, rng, A, spInf)
            @test Ω === A
            @test all(darg isa NoTangent for darg in pb!(ones(3)))
        end

        # Samplers for collections with possibly differentiable elements are not
        for X in ([1.0, 2.0], (1.0,), (1.0, 2.0), LinRange(0.0, 1.0, 3), 0.0:0.5:1.0, UnitRange(1.0, 4.0), Dict(1 => 2.0), Set([1.0]))
            @test rrule(rand, rng, Random.Sampler(rng, X, Val(1))) === nothing
            @test rrule(rand!, rng, Vector{Any}(undef, 3), Random.Sampler(rng, X, Val(Inf))) === nothing
        end
        @test rrule(rand, rng, Random.SamplerRangeNDL(1.5, UInt64(4))) === nothing
    end
end
