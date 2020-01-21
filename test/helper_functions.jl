@testset "helper functions" begin
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
