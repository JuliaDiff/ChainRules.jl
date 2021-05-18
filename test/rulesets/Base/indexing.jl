@testset "getindex" begin
    @testset "getindex(::Matrix{<:Number},...)" begin
        x = [1.0 2.0 3.0; 10.0 20.0 30.0]

        @testset "single element" begin
            test_rrule(getindex, x, 2)
            test_rrule(getindex, x, 2, 1)
            test_rrule(getindex, x, 2, 2)

            test_rrule(getindex, x, CartesianIndex(2, 3) ⊢ NoTangent())
        end

        @testset "slice/index postions" begin
            test_rrule(getindex, x, 2:3 ⊢ NoTangent())
            test_rrule(getindex, x, 3:-1:2 ⊢ NoTangent())
            test_rrule(getindex, x, [3,2] ⊢ NoTangent())
            test_rrule(getindex, x, [2,3] ⊢ NoTangent())

            test_rrule(getindex, x, 1:2 ⊢ NoTangent(), 2:3 ⊢ NoTangent())
            test_rrule(getindex, x, (:) ⊢ NoTangent(), 2:3 ⊢ NoTangent())

            test_rrule(getindex, x, 1:2 ⊢ NoTangent(), 1)
            test_rrule(getindex, x, 1, 1:2 ⊢ NoTangent())

            test_rrule(getindex, x, 1:2 ⊢ NoTangent(), 2:3 ⊢ NoTangent())
            test_rrule(getindex, x, (:) ⊢ NoTangent(), 2:3 ⊢ NoTangent())

            test_rrule(getindex, x, (:) ⊢ NoTangent(), (:) ⊢ NoTangent())
            test_rrule(getindex, x, (:) ⊢ NoTangent())
        end

        @testset "masking" begin
            test_rrule(getindex, x, trues(size(x)) ⊢ NoTangent())
            test_rrule(getindex, x, trues(length(x)) ⊢ NoTangent())

            mask = falses(size(x))
            mask[2,3] = true
            mask[1,2] = true
            test_rrule(getindex, x, mask ⊢ NoTangent())

            test_rrule(getindex, x, [true, false] ⊢ NoTangent(), (:) ⊢ NoTangent())
        end

        @testset "By position with repeated elements" begin
            test_rrule(getindex, x, [2, 2] ⊢ NoTangent())
            test_rrule(getindex, x, [2, 2, 2] ⊢ NoTangent())
            test_rrule(getindex, x, [2,2] ⊢ NoTangent(), [3,3] ⊢ NoTangent())
        end
    end
end
