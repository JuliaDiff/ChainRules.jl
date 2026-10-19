using ChainRules, ChainRulesCore, ChainRulesTestUtils, Test

@testset "test_helpers.jl" begin

    @testset "Multiplier" begin
        # One argument
        test_rrule(Multiplier(4.0), 3.0)
        test_rrule(Multiplier(5.0 + 6im), 7.0 + 8im)
        test_rrule(Multiplier(rand(2,3)), rand(3,4))

        # Two arguments
        test_rrule(Multiplier(1.2), 3.4, 5.6)
        test_rrule(Multiplier(1.0 + 2im), 3.0 + 4im, 5.0 - 6im)
        test_rrule(Multiplier(rand(2,3)), rand(3,4), rand(4,5))
    end
    
    @testset "Divider" begin
        test_rrule(Divider(2.3), 4.5)
        test_rrule(Divider(0.2), -3.4)
    end

    @testset "Counter" begin
        c = Counter()
        @test map(c, ones(5)) == 1:5
        @test c == Counter(5)
        y, back = rrule(c, 666)
        @test back(1) == (NoTangent(), 16)
        @test back(10) == (NoTangent(), 260)
        y2, back2 = rrule(c, 777, 888)
        @test back2(1) == (NoTangent(), 37, 47)
    end

    @testset "make_two_vec" begin
        test_rrule(make_two_vec, 1.5)
    end
    
    @testset "fstar, flog" begin
        test_frule(fstar, 1.2, 3.4 + 5im)
        test_frule(flog, 6.7)
        test_frule(flog, 8.9 + im)    
    end

    @testset "ambiguities" begin
        @test [] == filter(Test.detect_ambiguities(ChainRules, ChainRulesCore)) do t
            (t[1].name in [:rrule, Symbol("rrule##kw")])
        end
    end

    @testset "method tables" begin
        test_method_tables()  # Check the global method tables are consistent
    end
end
