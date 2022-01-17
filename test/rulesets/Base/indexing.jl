@testset "getindex" begin
    @testset "getindex(::Tuple, ...)" begin
        # Tuple of numbers
        test_rrule(getindex, Tuple(rand(5)), 2; check_inferred=false)
        test_rrule(getindex, Tuple(rand(5)), 2:4; check_inferred=false)
        test_rrule(getindex, Tuple(rand(5)), [3, 3, 4, 1, 4]; check_inferred=false)  # with repeats

        # Tuple of other things
        test_rrule(getindex, tuple(rand(2), rand(2)), 2; check_inferred=false)
        @test_skip test_rrule(getindex, tuple(Tuple(rand(2)), Tuple(rand(3))), 2; check_inferred=false)  # MethodError: (::ChainRulesTestUtils.var"#test_approx##kw")(::NamedTuple{(:rtol, :atol), Tuple{Float64, Float64}}, ::typeof(test_approx), ::NoTangent, ::Tangent{Tuple{Float64, Float64}, Tuple{Float64, Float64}}, ::String) is ambiguous.
    end
    @testset "getindex(::Matrix{<:Number}, ...)" begin
        x = [1.0 2.0 3.0; 10.0 20.0 30.0]

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

@testset "Base.setindex" begin
    @testset "setindex(::Tuple, ...)" begin
        test_rrule(Base.setindex, Tuple(rand(5)), rand(), 2; check_inferred=false)
        test_rrule(Base.setindex, Tuple(rand(5)), rand(2,2), 2; check_inferred=false)
    end
end
