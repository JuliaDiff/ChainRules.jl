@testset "mean" begin
    n = 9
    @testset "Basic" begin
        test_rrule(mean, randn(n))
    end
    @testset "with dims kwargs" begin
        test_rrule(mean, randn(n); fkwargs=(;dims=1))
        test_rrule(mean, randn(n,4); fkwargs=(;dims=2))
    end
end

@testset "variation: $var" for var in (std, var)
    test_rrule(var, randn(3))
    test_rrule(var, randn(4, 5); fkwargs=(; corrected=false))
    test_rrule(var, randn(ComplexF64, 6))
    test_rrule(var, Diagonal(randn(6)))

    test_rrule(var, randn(4, 5); fkwargs=(; dims=1))
    test_rrule(var, randn(ComplexF64, 4, 5); fkwargs=(; dims=2, corrected=false))
    test_rrule(var, UpperTriangular(randn(5, 5)); fkwargs=(; dims=1))

    x = PermutedDimsArray(randn(3, 4, 5), (3, 2, 1))
    xm = mean(x; dims=(1, 3))
    test_rrule(var, x; fkwargs=(; dims=(1, 3), mean=xm))
end
