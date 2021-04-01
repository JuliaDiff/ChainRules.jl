@testset "indexing.jl" begin
    @testset "getindex(::Matrix{<:Number},...)" begin
        x = [1.0 2.0 3.0; 10.0 20.0 30.0]

        @testset "single element" begin
            test_rrule(getindex, x, 2 ⊢ nothing)
            test_rrule(getindex, x, 2 ⊢ nothing, 1 ⊢ nothing)
            test_rrule(getindex, x, 2 ⊢ nothing, 2 ⊢ nothing)

            test_rrule(getindex, x, CartesianIndex(2, 3) ⊢ nothing)
        end

        @testset "slice/index postions" begin
            test_rrule(getindex, x, 2:3 ⊢ nothing)
            test_rrule(getindex, x, 3:-1:2 ⊢ nothing)
            test_rrule(getindex, x, [3,2] ⊢ nothing)
            test_rrule(getindex, x, [2,3] ⊢ nothing)

            test_rrule(getindex, x, 1:2 ⊢ nothing, 2:3 ⊢ nothing)
            test_rrule(getindex, x, (:) ⊢ nothing, 2:3 ⊢ nothing)

            test_rrule(getindex, x, 1:2 ⊢ nothing, 1 ⊢ nothing)
            test_rrule(getindex, x, 1 ⊢ nothing, 1:2 ⊢ nothing)

            test_rrule(getindex, x, 1:2 ⊢ nothing, 2:3 ⊢ nothing)
            test_rrule(getindex, x, (:) ⊢ nothing, 2:3 ⊢ nothing)

            test_rrule(getindex, x, (:) ⊢ nothing, (:) ⊢ nothing)
            test_rrule(getindex, x, (:) ⊢ nothing)
        end

        @testset "masking" begin
            test_rrule(getindex, x, trues(size(x)) ⊢ nothing)
            test_rrule(getindex, x, trues(length(x)) ⊢ nothing)

            mask = falses(size(x))
            mask[2,3] = true
            mask[1,2] = true
            test_rrule(getindex, x, mask ⊢ nothing)

            test_rrule(getindex, x, [true, false] ⊢ nothing, (:) ⊢ nothing)
        end

        @testset "By position with repeated elements" begin
            test_rrule(getindex, x, [2, 2] ⊢ nothing)
            test_rrule(getindex, x, [2, 2, 2] ⊢ nothing)
            test_rrule(getindex, x, [2,2] ⊢ nothing, [3,3] ⊢ nothing)
        end
    end

    @testset "getindex(::Matrix{Not a Number},...)" begin
        rrule_test(  # Vector of Vectors
            getindex, [[2.3,],], ([[1.0,],[2.0]], [[2.1,],[310]]), (2, nothing)
        )
        rrule_test( # Vector of Vectors of different sizes
            getindex, [[2.3, 1.1],], ([[1.0,],[2.0,3.0]], [[2.1,],[3.2, 2.1]]), (2, nothing)
        )

        # shorthand constructor for Composite with primal matching backing
        Compo(x...) = Composite{typeof(x)}(x...)
        rrule_test( # Vector of Tuples
            getindex,
            [Compo(2.5, 4.1),],
            (
                [(1.1, 2.3), (1.3, 2.5)], [Compo(2.5, 4.1), Compo(2.5, 4.1)]
            ),
            (2, nothing)
        )
        rrule_test( # Vector of Tuples of different sizes
            getindex,
            [Compo(2.5, 4.1),],
            (
                [(1.1, 2.3), (1.3, 2.5)], [Compo(2.5, 4.1), Compo(2.5, 4.1)]
            ),
            (2, nothing)
        )
    end
end
