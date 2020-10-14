@testset "utils.jl" begin
    @testset "_subtract!!" begin
        _subtract!! = ChainRules._subtract!!

        @testset "Inplace" begin
            x = [1, 2, 3]
            ret = _subtract!!(x, ones(3))
            @test ret === x
            @test ret == [0, 1, 2]
        end

        @testset "Out of place" begin
            x = Diagonal([2, 2])
            ret = _subtract!!(x, ones(2, 2))
            @test ret !== x
            @test ret == [1 -1; -1 1]
        end

        @testset "Currently out of place, but this could change" begin
            x = Diagonal([3, 3])
            ret = _subtract!!(x, Diagonal([1,1]))
            @test ret !== x
            @test ret isa Diagonal
            @test ret == Diagonal([2, 2])
        end
    end
end
