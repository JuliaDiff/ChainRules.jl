@testset "nondiff.jl" begin
    @testset "Sum over boolean array" begin
        rrule_test(sum, 12, (falses(5, 3), nothing))
        rrule_test(sum, 2, ([true, false, true], nothing))
        rrule_test(sum, 2, ((true, false, true), nothing))
    end
end
