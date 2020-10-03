@testset "ignore" begin
    f(x) = x
    @test ChainRules.frule((NO_FIELDS, NO_FIELDS), ChainRules.ignore, f, (2,)...) == (4, nothing)
    @test ChainRules.rrule(ChainRules.ignore, f, (1,)...)[2](1) == (Zero(), nothing)
end
