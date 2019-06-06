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
    @testset "helper functions" begin
        # Hits fallback, since we can't update `Diagonal`s in place
        X = Diagonal([1, 1])
        Y = copy(X)
        @test ChainRules._update!(X, [1 2; 3 4]) == [2 2; 3 5]
        @test X == Y  # no change to X

        X = [1 2; 3 4]
        Y = copy(X)
        @test ChainRules._update!(X, Diagonal([1, 1])) == [2 2; 3 5]
        @test X != Y  # X has been updated

        # Reusing above X
        @test ChainRules._update!(X, Zero()) === X
        @test ChainRules._update!(Zero(), X) === X
        @test ChainRules._update!(Zero(), Zero()) === Zero()

        X = (A=[1 0; 0 1], B=[2 2; 2 2])
        Y = deepcopy(X)
        @test ChainRules._update!(X, Y) == (A=[2 0; 0 2], B=[4 4; 4 4])
        @test X.A != Y.A
        @test X.B != Y.B
    end
end
