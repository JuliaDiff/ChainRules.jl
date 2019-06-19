cool(x) = x + 1
cool(x, y) = x + y + 1

_second(t) = Base.tuple_type_head(Base.tuple_type_tail(t))

@testset "rules" begin
    @testset "frule and rrule" begin
        @test frule(cool, 1) === nothing
        @test frule(cool, 1; iscool=true) === nothing
        @test rrule(cool, 1) === nothing
        @test rrule(cool, 1; iscool=true) === nothing

        ChainRules.@scalar_rule(Main.cool(x), one(x))
        @test hasmethod(rrule, Tuple{typeof(cool),Number})
        ChainRules.@scalar_rule(Main.cool(x::String), "wow such dfdx")
        @test hasmethod(rrule, Tuple{typeof(cool),String})
        # Ensure those are the *only* methods that have been defined
        cool_methods = Set(m.sig for m in methods(rrule) if _second(m.sig) == typeof(cool))
        only_methods = Set([Tuple{typeof(rrule),typeof(cool),Number},
                            Tuple{typeof(rrule),typeof(cool),String}])
        @test cool_methods == only_methods

        frx, fr = frule(cool, 1)
        @test frx == 2
        @test fr(1) == 1
        rrx, rr = rrule(cool, 1)
        @test rrx == 2
        @test rr(1) == 1
    end
    @testset "iterating and indexing rules" begin
        _, rule = frule(+, 1)
        i = 0
        for r in rule
            @test r === rule
            i += 1
        end
        @test i == 1  # rules only iterate once, yielding themselves
        @test rule[1] == rule
        @test_throws BoundsError rule[2]
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

        try
            # We defined a 2-arg method for `cool` but no `rrule`
            ChainRules._checked_rrule(cool, 1.0, 2.0)
        catch e
            @test e isa ArgumentError
            @test e.msg == "can't differentiate `cool(::Float64, ::Float64)`; no " *
                           "matching `rrule` is defined"
        end
    end
end
