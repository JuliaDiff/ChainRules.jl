@testset "getindex" begin
    x = [1.0 2.0 3.0; 10.0 20.0 30.0]
    x̄ = [1.4 2.5 3.7; 10.5 20.1 30.2]
    rrule_test(getindex, 2.3, (x, x̄), (2, nothing))
    rrule_test(getindex, 2.3, (x, x̄), (2, nothing), (1, nothing))
    rrule_test(getindex, 2.3, (x, x̄), (2, nothing), (2, nothing))
end
