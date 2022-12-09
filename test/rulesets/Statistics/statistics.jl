@testset "mean" begin
    @testset "mean(x)" begin
        @gpu test_rrule(mean, randn(9))
        test_rrule(mean, randn(ComplexF64,2,4))
        test_rrule(mean, transpose(rand(3)))
        test_rrule(mean, [rand(3) for _ in 1:4]; check_inferred=false)
    end
    @testset "with dims kwargs" begin
        @gpu test_rrule(mean, randn(9); fkwargs=(;dims=1))
        @gpu test_rrule(mean, randn(9,4); fkwargs=(;dims=2))
        @gpu test_rrule(mean, [rand(2) for _ in 1:3, _ in 1:4]; fkwargs=(;dims=2), check_inferred=false)
    end
    @testset "mean(f, x)" begin
        # This shares its implementation with sum(f, x). Similar tests should cover all cases:
        test_rrule(mean, abs, [-4.0, 2.0, 2.0])
        test_rrule(mean, log, rand(3, 4) .+ 1)
        test_rrule(mean, cbrt, randn(5))
        test_rrule(mean, Multiplier(2.0), [2.0, 4.0, 8.0])  # defined in test_helpers.jl
        test_rrule(mean, Divider(1 + rand()), randn(5)) 

        test_rrule(mean, sum, [[2.0, 4.0], [4.0,1.9]]; check_inferred=false)

        test_rrule(mean, log, rand(ComplexF64, 5))
        test_rrule(mean, sqrt, rand(ComplexF64, 5))
        test_rrule(mean, abs, rand(ComplexF64, 3, 4))
        
        test_rrule(mean, abs, [-2.0 4.0; 5.0 1.9]; fkwargs=(;dims=1))
        test_rrule(mean, abs, [-2.0 4.0; 5.0 1.9]; fkwargs=(;dims=2))
        test_rrule(mean, sqrt, rand(ComplexF64, 3, 4); fkwargs=(;dims=(1,)))
    end
end

@testset "variation: $var" for var in (std, var)
    @gpu test_rrule(var, randn(3))
    @gpu test_rrule(var, randn(4, 5); fkwargs=(; corrected=false))
    test_rrule(var, randn(ComplexF64, 6))
    test_rrule(var, Diagonal(randn(6)))

    test_rrule(var, randn(4, 5); fkwargs=(; dims=1))
    test_rrule(var, randn(ComplexF64, 4, 5); fkwargs=(; dims=2, corrected=false))
    test_rrule(var, UpperTriangular(randn(5, 5)); fkwargs=(; dims=1))

    x = PermutedDimsArray(randn(3, 4, 5), (3, 2, 1))
    xm = mean(x; dims=(1, 3))
    test_rrule(var, x; fkwargs=(; dims=(1, 3), mean=xm))
end
