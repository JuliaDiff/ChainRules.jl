
using ChainRules: tuplecast, unzip  # tuplemap, 

@testset "tuplecast" begin
    @testset "basics: $(sprint(show, fun))" for fun in [tuplecast, unzip∘broadcast] # [tuplemap, tuplecast, unzip∘map, unzip∘broadcast]
        @test_throws Exception fun(sqrt, 1:3)

        @test fun(tuple, 1:3, 4:6) == ([1, 2, 3], [4, 5, 6])
        @test fun(tuple, [1, 10, 100]) == ([1, 10, 100],)
        @test fun(tuple, 1:3, fill(nothing, 3)) == (1:3, fill(nothing, 3))
        @test fun(tuple, [1, 10, 100], fill(nothing, 3)) == ([1, 10, 100], fill(nothing, 3))
        @test fun(tuple, fill(nothing, 3), fill(nothing, 3)) == (fill(nothing, 3), fill(nothing, 3))

        if contains(string(fun), "map")
            @test fun(tuple, 1:3, 4:999) == ([1, 2, 3], [4, 5, 6])
        else
            @test fun(tuple, [1,2,3], [4 5]) == ([1 1; 2 2; 3 3], [4 5; 4 5; 4 5])
        end
    end

    # tuplemap(tuple, (1,2,3), (4,5,6)) == ([1, 2, 3], [4, 5, 6])

    @testset "unzip" begin
        @test unzip([(1,2), (3,4), (5,6)]) == ([1, 3, 5], [2, 4, 6])
        @test unzip([(nothing,2), (3,4), (5,6)]) == ([nothing, 3, 5], [2, 4, 6])
        @test unzip([(missing,2), (missing,4), (missing,6)])[2] isa Base.ReinterpretArray

        y, bk = rrule(unzip, [(1,2), (3,4), (5,6)])
        @test y == ([1, 3, 5], [2, 4, 6])
        @test bk(Tangent{Tuple}([1,1,1], [10,100,1000]))[2] isa Vector{<:Tangent{<:Tuple}}
    end
    
    @testset "rrules" begin
        # These exist to allow for second derivatives
        
        # test_rrule(collect∘tuplecast, tuple, [1,2,3.], [4,5,6.], check_inferred=false)
        y1, bk1 = rrule(CFG, tuplecast, tuple, [1,2,3.0], [4,5,6.0])
        @test y1 == ([1, 2, 3], [4, 5, 6])
        @test bk1(([1,10,100.0], [7,8,9.0]))[3] ≈ [1,10,100]
        
        y2, bk2 = rrule(CFG, tuplecast, tuple, [1,2,3.0], [4 5.0], 6.0)
        @test y2 == ([1 1; 2 2; 3 3], [4 5; 4 5; 4 5], [6 6; 6 6; 6 6])
        @test bk2(y2)[5] ≈ 36

        test_rrule(unzip, [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)], check_inferred=false)
    end
end