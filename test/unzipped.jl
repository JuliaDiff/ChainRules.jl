
using ChainRules: unzip_broadcast, unzip #, unzip_map

@testset "unzipped.jl" begin
    @testset "basics: $(sprint(show, fun))" for fun in [unzip_broadcast, unzip∘map, unzip∘broadcast] # unzip_map, 
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

        if contains(string(fun), "map")
            @test fun(tuple, (1,2,3), (4,5,6)) == ((1, 2, 3), (4, 5, 6))
        else
            @test fun(tuple, (1,2,3), (4,5,6)) == ((1, 2, 3), (4, 5, 6))
            @test fun(tuple, (1,2,3), (7,)) == ((1, 2, 3), (7, 7, 7))
            @test fun(tuple, (1,2,3), 8) == ((1, 2, 3), (8, 8, 8))
        end
        @test fun(tuple, (1,2,3), [4,5,6]) == ([1, 2, 3], [4, 5, 6])  # mix tuple & vector
    end

    @testset "rrules" begin
        # These exist to allow for second derivatives

        # test_rrule(collect∘unzip_broadcast, tuple, [1,2,3.], [4,5,6.], collectheck_inferred=false) # return type Tuple{NoTangent, NoTangent, Vector{Float64}, Vector{Float64}} does not match inferred return type NTuple{4, Any}

        y1, bk1 = rrule(CFG, unzip_broadcast, tuple, [1,2,3.0], [4,5,6.0])
        @test y1 == ([1, 2, 3], [4, 5, 6])
        @test bk1(([1,10,100.0], [7,8,9.0]))[3] ≈ [1,10,100]
        
        # bk1(([1,10,100.0], NoTangent()))  # DimensionMismatch in FiniteDifferences
        
        y2, bk2 = rrule(CFG, unzip_broadcast, tuple, [1,2,3.0], [4 5.0], 6.0)
        @test y2 == ([1 1; 2 2; 3 3], [4 5; 4 5; 4 5], [6 6; 6 6; 6 6])
        @test bk2(y2)[5] ≈ 36

        # y4, bk4 = rrule(CFG, unzip_map, tuple, [1,2,3.0], [4,5,6.0])
        # @test y4 == ([1, 2, 3], [4, 5, 6])
        # @test bk4(([1,10,100.0], [7,8,9.0]))[3] ≈ [1,10,100]
    end
    
    @testset "unzip" begin
        @test unzip([(1,2), (3,4), (5,6)]) == ([1, 3, 5], [2, 4, 6])
        @test unzip(Any[(1,2), (3,4), (5,6)]) == ([1, 3, 5], [2, 4, 6])
        
        @test unzip([(nothing,2), (3,4), (5,6)]) == ([nothing, 3, 5], [2, 4, 6])
        @test unzip([(missing,2), (missing,4), (missing,6)])[2] isa Base.ReinterpretArray

        @test unzip([(1,), (3,), (5,)]) == ([1, 3, 5],)
        @test unzip([(1,), (3,), (5,)])[1] isa Base.ReinterpretArray
        
        @test unzip(((1,2), (3,4), (5,6))) == ((1, 3, 5), (2, 4, 6))

        # test_rrule(unzip, [(1,2), (3,4), (5.0,6.0)], check_inferred=false)  # DimensionMismatch: second dimension of A, 6, does not match length of x, 2

        y, bk = rrule(unzip, [(1,2), (3,4), (5,6)])
        @test y == ([1, 3, 5], [2, 4, 6])
        @test bk(Tangent{Tuple}([1,1,1], [10,100,1000]))[2] isa Vector{<:Tangent{<:Tuple}}
        
        y3, bk3 = rrule(unzip, [(1,ZeroTangent()), (3,ZeroTangent()), (5,ZeroTangent())])
        @test y3 == ([1, 3, 5], [ZeroTangent(), ZeroTangent(), ZeroTangent()])
        dx3 = bk3(Tangent{Tuple}([1,1,1], [10,100,1000]))[2]
        @test dx3 isa Vector{<:Tangent{<:Tuple}}
        @test Tuple(dx3[1]) == (1.0, NoTangent())
        
        y5, bk5 = rrule(unzip, ((1,2), (3,4), (5,6)))
        @test y5 == ((1, 3, 5), (2, 4, 6))
        @test bk5(y5)[2] isa Tangent{<:Tuple}
        @test Tuple(bk5(y5)[2][2]) == (3, 4)
        dx5 = bk5(((1,10,100), ZeroTangent()))
        @test dx5[2] isa Tangent{<:Tuple}
        @test Tuple(dx5[2][2]) == (10, ZeroTangent())
    end
    
    @testset "JLArray tests" begin  # fake GPU testing
        (y1, y2), bk = rrule(CFG, unzip_broadcast, tuple, [1,2,3.0], [4 5.0])
        (y1jl, y2jl), bk_jl = rrule(CFG, unzip_broadcast, tuple, jl([1,2,3.0]), jl([4 5.0]))
        @test y1 == Array(y1jl)
        # TODO invent some tests of this rrule's pullback function

        @test unzip(jl([(1,2), (3,4), (5,6)])) == (jl([1, 3, 5]), jl([2, 4, 6]))
        @test unzip(jl([(missing,2), (missing,4), (missing,6)]))[2] == jl([2, 4, 6])
        @test unzip(jl([(1,), (3,), (5,)]))[1] == jl([1, 3, 5])

        # depending on Julia version may get ReinterpretArray or may get JLArray
        # Either is acceptable
        @test unzip(jl([(missing,2), (missing,4), (missing,6)]))[2] isa Union{Base.ReinterpretArray, JLArray}
        @test unzip(jl([(1,), (3,), (5,)]))[1] isa Union{Base.ReinterpretArray, JLArray}
    end
end
