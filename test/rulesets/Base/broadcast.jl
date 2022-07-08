using Base.Broadcast: broadcasted

@testset "Broadcasting" begin
    @testset "generic 1: trivial path" begin
        # test_rrule(copy∘broadcasted, >, rand(3), rand(3))  # MethodError: no method matching eps(::UInt64) inside FiniteDifferences
        y1, bk1 = rrule(CFG, copy∘broadcasted, >, rand(3), rand(3))
        @test y1 isa AbstractArray{Bool}
        @test all(d -> d isa AbstractZero, bk1(99))
    
        y2, bk2 = rrule(CFG, copy∘broadcasted, isinteger, Tuple(rand(3)))
        @test y2 isa Tuple{Bool,Bool,Bool}
        @test all(d -> d isa AbstractZero, bk2(99))
    end

    @testset "generic 2: fast path" begin
        test_rrule(copy∘broadcasted, log, rand(3))
        test_rrule(copy∘broadcasted, log, Tuple(rand(3)))

        # Two args uses StructArrays
        test_rrule(copy∘broadcasted, atan, rand(3), rand(3))
        test_rrule(copy∘broadcasted, atan, rand(3), rand(4)')
        test_rrule(copy∘broadcasted, atan, rand(3), rand())
        test_rrule(copy∘broadcasted, atan, rand(3), Tuple(rand(1)))
        test_rrule(copy∘broadcasted, atan, Tuple(rand(3)), Tuple(rand(3)))
        
        # Protected by Ref/Tuple:
        test_rrule(copy∘broadcasted, *, rand(3), Ref(rand()))
        test_rrule(copy∘broadcasted, *, rand(3), Ref(rand(2)))
    end

    @testset "generic 3: slow path" begin
        test_rrule(copy∘broadcasted, sin∘cos, rand(3), check_inferred=false)
        test_rrule(copy∘broadcasted, sin∘atan, rand(3), rand(3)', check_inferred=false)
        test_rrule(copy∘broadcasted, sin∘atan, rand(), rand(3), check_inferred=false)
        test_rrule(copy∘broadcasted, ^, rand(3), 3.0, check_inferred=false)

        # From test_helpers.jl
        test_rrule(copy∘broadcasted, Multiplier(rand()), rand(3), check_inferred=false)
        test_rrule(copy∘broadcasted, Multiplier(rand()), rand(3), rand(4)', check_inferred=false)
        @test_skip test_rrule(copy∘broadcasted, Multiplier(rand()), rand(3), 5.0im, check_inferred=false)  # ProjectTo(f) fails to correct this
        test_rrule(copy∘broadcasted, make_two_vec, rand(3), check_inferred=false)
        
        # Non-diff components
        test_rrule(copy∘broadcasted, first∘tuple, rand(3), :sym, rand(4)', check_inferred=false)
        test_rrule(copy∘broadcasted, last∘tuple, rand(3), nothing, rand(4)', check_inferred=false)
        test_rrule(copy∘broadcasted, |>, rand(3), sin, check_inferred=false)
        _call(f, x...) = f(x...)
        test_rrule(copy∘broadcasted, _call, atan, rand(3), rand(4)', check_inferred=false)

        # Protected by Ref/Tuple:
        test_rrule(copy∘broadcasted, conj∘*, rand(3), Ref(rand() + im), check_inferred=false)
        test_rrule(copy∘broadcasted, conj∘*, rand(3), Ref(rand(2) .+ im), check_inferred=false)
        test_rrule(copy∘broadcasted, /, (rand(2),), rand(3), check_inferred=false)
    end

    @testset "lazy rules" begin
        test_rrule(copy∘broadcasted, +, rand(3), rand(3))
        test_rrule(copy∘broadcasted, +, rand(3), rand(4)')
        test_rrule(copy∘broadcasted, +, rand(3), rand(1), rand())
        test_rrule(copy∘broadcasted, +, rand(3), 1.0*im)
        test_rrule(copy∘broadcasted, +, rand(3), true)
        test_rrule(copy∘broadcasted, +, rand(3), Tuple(rand(3)))
    
        test_rrule(copy∘broadcasted, -, rand(3), rand(3))
        test_rrule(copy∘broadcasted, -, rand(3), rand(4)')
        test_rrule(copy∘broadcasted, -, rand(3))
        # test_rrule(copy∘broadcasted, -, Tuple(rand(3))) # MethodError: (::ChainRulesTestUtils.var"#test_approx##kw")(::NamedTuple{(:rtol, :atol), Tuple{Float64, Float64}}, ::typeof(test_approx), ::Thunk{ChainRules.var"#1614#1616"{Tangent{Tuple{Float64, Float64, Float64}, Tuple{Float64, Float64, Float64}}}}, ::Tangent{Tuple{Float64, Float64, Float64}, Tuple{Float64, Float64, Float64}}) is ambiguous.
    
        test_rrule(copy∘broadcasted, *, rand(3), rand(3))
        test_rrule(copy∘broadcasted, *, rand(3), rand())
        test_rrule(copy∘broadcasted, *, rand(), rand(3))
    
        test_rrule(copy∘broadcasted, Base.literal_pow, ^, rand(3), Val(2))
    
        test_rrule(copy∘broadcasted, /, rand(3), rand())
    
        test_rrule(copy∘broadcasted, identity, rand(3))
    
        test_rrule(copy∘broadcasted, conj, rand(3))
        test_rrule(copy∘broadcasted, conj, rand(3) .+ im)
    end

    @testset "scalar rules" begin    
        test_rrule(copy∘broadcasted, sin, rand())
        test_rrule(copy∘broadcasted, atan, rand(), rand())
        # test_rrule(copy∘broadcasted, >, rand(), rand()) # DimensionMismatch from FiniteDifferences

        # Functions with lazy rules
        test_rrule(copy∘broadcasted, +, rand(), rand(), rand())
        test_rrule(copy∘broadcasted, +, rand())
        test_rrule(copy∘broadcasted, -, rand(), rand())
        test_rrule(copy∘broadcasted, -, rand())
        test_rrule(copy∘broadcasted, *, rand(), rand())
        test_rrule(copy∘broadcasted, Base.literal_pow, ^, rand(), Val(2))
        test_rrule(copy∘broadcasted, /, rand(), rand())
    
        test_rrule(copy∘broadcasted, identity, rand())
        test_rrule(copy∘broadcasted, conj, rand())
        test_rrule(copy∘broadcasted, conj, rand() + im)
    end
end