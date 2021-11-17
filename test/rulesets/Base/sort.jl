@testset "sort.jl" begin
    @testset "sort" begin
        a = rand(10)
        test_rrule(sort, a)
        test_rrule(sort, a; fkwargs=(;rev=true))
    end
    @testset "partialsort" begin
        a = rand(10)
        test_rrule(partialsort, a, 4)
        test_rrule(partialsort, a, 3:5)
        test_rrule(partialsort, a, 1:2:6)

        test_rrule(partialsort, a, 4, fkwargs=(;rev=true))
    end

    @testset "sortslices" begin
        test_rrule(sortslices, rand(3,4); fkwargs=(; dims=2))
        test_rrule(sortslices, rand(5,4); fkwargs=(; dims=1, rev=true, by=last))
        test_rrule(sortslices, rand(3,4,5); fkwargs=(; dims=3, by=sum))

        @test_throws Exception sortslices(Diagonal(1:3), dims=1)
    end

    @testset "unique" begin
        # Trivial case, all unique:
        test_rrule(unique, rand(5))
        test_rrule(unique, rand(3,4))
        test_rrule(unique, rand(3,4); fkwargs=(; dims=2))

        # Not all unique:
        @test rrule(unique, [1,1,2,3])[1] == [1,2,3]
        @test rrule(unique, [1,1,2,3])[2]([10,20,30]) == (NoTangent(), [10, 0, 20, 30])

        @test rrule(unique, [1 2; 1 4])[1] == [1,2,4]
        @test rrule(unique, [1 2; 1 4])[2]([10,20,30]) == (NoTangent(), [10 20; 0 30])

        @test rrule(unique, [1 2 1 2; 1 2 1 4], dims=2)[1] == [1 2 2; 1 2 4]
        @test rrule(unique, [1 2 1 2; 1 2 1 4], dims=2)[2]([10 20 30; 40 50 60])[2] == [10 20 0 30; 40 50 0 60]

        @test rrule(unique, Diagonal([1,2,3]))[1] == [1,0,2,3]
        @test rrule(unique, Diagonal([1,2,3]))[2]([10 20 30 40])[2] == [10.0 0.0 0.0; 0.0 30.0 0.0; 0.0 0.0 40.0]
        @test rrule(unique, Diagonal([1,2,3]))[2]([10 20 30 40])[2] isa Diagonal
    end
end
