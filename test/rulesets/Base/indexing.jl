@testset "indexing.jl" begin
    @testset "getindex(::Matrix{<:Number},...)" begin
        x = [1.0 2.0 3.0; 10.0 20.0 30.0]
        x̄ = [1.4 2.5 3.7; 10.5 20.1 30.2]
        full_ȳ = [7.4 5.5 2.7; 8.5 11.1 4.2]

        @testset "single element" begin
            rrule_test(getindex, 2.3, (x, x̄), (2, nothing))
            rrule_test(getindex, 2.3, (x, x̄), (2, nothing), (1, nothing))
            rrule_test(getindex, 2.3, (x, x̄), (2, nothing), (2, nothing))

            rrule_test(getindex, 2.3, (x, x̄), (CartesianIndex(2, 3), nothing))
        end

        @testset "slice/index postions" begin
            rrule_test(getindex, [2.3, 3.1], (x, x̄), (2:3, nothing))
            rrule_test(getindex, [2.3, 3.1], (x, x̄), (3:-1:2, nothing))
            rrule_test(getindex, [2.3, 3.1], (x, x̄), ([3,2], nothing))
            rrule_test(getindex, [2.3, 3.1], (x, x̄), ([2,3], nothing))

            rrule_test(getindex, [2.3 3.1; 4.1 5.1], (x, x̄), (1:2, nothing), (2:3, nothing))
            rrule_test(getindex, [2.3 3.1; 4.1 5.1], (x, x̄), (:, nothing), (2:3, nothing))

            rrule_test(getindex, [2.3, 3.1], (x, x̄), (1:2, nothing), (1, nothing))
            rrule_test(getindex, [2.3, 3.1], (x, x̄), (1, nothing), (1:2, nothing))

            rrule_test(getindex, [2.3 3.1; 4.1 5.1], (x, x̄), (1:2, nothing), (2:3, nothing))
            rrule_test(getindex, [2.3 3.1; 4.1 5.1], (x, x̄), (:, nothing), (2:3, nothing))


            rrule_test(getindex, full_ȳ, (x, x̄), (:, nothing), (:, nothing))
            rrule_test(getindex, full_ȳ[:], (x, x̄), (:, nothing))
        end

        @testset "masking" begin
            rrule_test(getindex, full_ȳ, (x, x̄), (trues(size(x)), nothing))
            rrule_test(getindex, full_ȳ[:], (x, x̄), (trues(length(x)), nothing))

            mask = falses(size(x))
            mask[2,3] = true
            mask[1,2] = true
            rrule_test(getindex, [2.3, 3.1], (x, x̄), (mask, nothing))

            rrule_test(
                getindex, full_ȳ[1,:], (x, x̄), ([true, false], nothing), (:, nothing)
            )
        end

        @testset "By position with repeated elements" begin
            rrule_test(getindex, [2.3, 3.1], (x, x̄), ([2, 2], nothing))
            rrule_test(getindex, [2.3, 3.1, 4.1], (x, x̄), ([2, 2, 2], nothing))
            rrule_test(
                getindex, [2.3 3.1; 4.1 5.1], (x, x̄), ([2,2], nothing), ([3,3], nothing)
            )
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
