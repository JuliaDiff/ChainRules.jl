@testset "getindex" begin
    @testset "getindex(::Matrix{<:Number}, ...)" begin
        x = [1.0 2.0 3.0; 10.0 20.0 30.0]
        
        @testset "forward mode" begin
            test_frule(getindex, x, 2)
            test_frule(getindex, x, 2, 1)
            test_frule(getindex, x, CartesianIndex(2, 3))

            test_frule(getindex, x, 2:3)
            test_frule(getindex, x, (:), 2:3)
        end

        @testset "single element" begin
            test_rrule(getindex, x, 2)
            test_rrule(getindex, x, 2, 1)
            test_rrule(getindex, x, 2, 2)

            test_rrule(getindex, x, CartesianIndex(2, 3))
        end

        @testset "slice/index postions" begin
            test_rrule(getindex, x, 2:3)
            test_rrule(getindex, x, 3:-1:2)
            test_rrule(getindex, x, [3,2])
            test_rrule(getindex, x, [2,3])

            test_rrule(getindex, x, 1:2, 2:3)
            test_rrule(getindex, x, (:), 2:3)

            test_rrule(getindex, x, 1:2, 1)
            test_rrule(getindex, x, 1, 1:2)

            test_rrule(getindex, x, 1:2, 2:3)
            test_rrule(getindex, x, (:), 2:3)

            test_rrule(getindex, x, (:), (:))
            test_rrule(getindex, x, (:))
        end

        @testset "masking" begin
            test_rrule(getindex, x, trues(size(x)))
            test_rrule(getindex, x, trues(length(x)))

            mask = falses(size(x))
            mask[2,3] = true
            mask[1,2] = true
            test_rrule(getindex, x, mask)

            test_rrule(getindex, x, [true, false], (:))
        end

        @testset "By position with repeated elements" begin
            test_rrule(getindex, x, [2, 2])
            test_rrule(getindex, x, [2, 2, 2])
            test_rrule(getindex, x, [2,2], [3,3])
        end
    end
end

@testset "view" begin
    test_frule(view, rand(3, 4), :, 1)
    test_frule(view, rand(3, 4), 2, [1, 1, 2])
    test_frule(view, rand(3, 4), 3, 4)
end

@testset "setindex!" begin
    test_frule(setindex!, rand(3, 4), rand(), 1, 2)
    test_frule(setindex!, rand(3, 4), [1,10,100.0], :, 3)
end

# Without these, you get errors, e.g.
# test_rrule: eachrow on Vector{Float64}: Error During Test at /Users/me/.julia/packages/ChainRulesTestUtils/8dFTY/src/testers.jl:195
#   Got exception outside of a @test
#   DimensionMismatch("second dimension of A, 6, does not match length of x, 5")
ChainRules.rrule(::typeof(collect∘eachrow), x) = rrule(eachrow, x)
ChainRules.rrule(::typeof(collect∘eachcol), x) = rrule(eachcol, x)
ChainRules.rrule(::typeof(collect∘eachslice), x; dims) = rrule(eachslice, x; dims=dims)

if VERSION >= v"1.6"
    # Without these, you get errors, e.g.
    # test_rrule: eachrow on Vector{Float64}: Error During Test at /Users/me/.julia/packages/ChainRulesTestUtils/8dFTY/src/testers.jl:195
    #   Got exception outside of a @test
    #   DimensionMismatch("second dimension of A, 6, does not match length of x, 5")
    ChainRules.rrule(::typeof(collect∘eachrow), x) = rrule(eachrow, x)
    ChainRules.rrule(::typeof(collect∘eachcol), x) = rrule(eachcol, x)
    ChainRules.rrule(::typeof(collect∘eachslice), x; dims) = rrule(eachslice, x; dims=dims)
    # And before Julia 1.6, `collect∘eachrow` is not a specific type.

    @testset "eachslice" begin
        test_rrule(collect∘eachrow, rand(5); check_inferred=false)
        test_rrule(collect∘eachrow, rand(3, 4); check_inferred=false)
        test_rrule(collect∘eachcol, rand(3, 4); check_inferred=false)
        test_rrule(collect∘eachslice, rand(3, 4, 5); fkwargs = (; dims = 3), check_inferred=false)
        test_rrule(collect∘eachslice, rand(3, 4, 5); fkwargs = (; dims = (2,)), check_inferred=false)
    end

end