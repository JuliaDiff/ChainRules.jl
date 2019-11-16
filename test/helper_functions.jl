@testset "helper functions" begin
    @testset "_update! Array" begin
        # Hits fallback, since we can't update `Diagonal`s in place
        X = Diagonal([1, 1])
        Y = copy(X)
        @test ChainRules._update!(X, [1 2; 3 4]) == [2 2; 3 5]
        @test X == Y  # no change to X

        X = [1 2; 3 4]
        Y = copy(X)
        @test ChainRules._update!(X, Diagonal([1, 1])) == [2 2; 3 5]
        @test X != Y  # X has been updated
    end
    @testset "_update! Zero" begin
        X = [1 2; 3 4]
        @test ChainRules._update!(X, Zero()) === X
        @test ChainRules._update!(Zero(), X) === X
        @test ChainRules._update!(Zero(), Zero()) === Zero()
    end
    @testset "_update! NamedTuple" begin
        X = (A=[1 0; 0 1], B=[2 2; 2 2])
        old_X = deepcopy(X)
        Y = deepcopy(X)
        @test ChainRules._update!(X, Y, :A) == (A=[2 0; 0 2], B=[2 2; 2 2])
        @test X.A != old_X.A
        @test X.B == old_X.B
    end
    @testset "_checked_rrule" begin
        try
            @eval cool(x,y) = x + y
            # We defined a 2-arg method for `cool` but no `rrule`
            ChainRules._checked_rrule(cool, 1.0, 2.0)
        catch e
            @test e isa ArgumentError
            @test e.msg == "can't differentiate `cool(::Float64, ::Float64)`; no " *
                "matching `rrule` is defined"
        end
    end
end
