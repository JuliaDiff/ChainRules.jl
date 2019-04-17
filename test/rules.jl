cool(x) = x + 1

@testset "rules" begin
    @testset "frule and rrule" begin
        @test frule(cool, 1) === nothing
        @test rrule(cool, 1) === nothing
        ChainRules.@scalar_rule(Main.cool(x), one(x))
        frx, fr = frule(cool, 1)
        @test frx == 2
        @test fr(1) == 1
        rrx, rr = rrule(cool, 1)
        @test rrx == 2
        @test rr(1) == 1
    end
    @testset "iterating rules" begin
        _, rule = frule(+, 1)
        i = 0
        for r in rule
            @test r === rule
            i += 1
        end
        @test i == 1  # rules only iterate once, yielding themselves
    end
end
