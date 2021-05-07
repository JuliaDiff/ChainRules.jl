@testset "getindex" begin
    @testset "getindex(::Matrix{<:Number},...)" begin
        x = [1.0 2.0 3.0; 10.0 20.0 30.0]

        @testset "single element" begin
            test_rrule(getindex, x, 2)
            test_rrule(getindex, x, 2, 1)
            test_rrule(getindex, x, 2, 2)

            test_rrule(getindex, x, CartesianIndex(2, 3) ⊢ DoesNotExist())
        end

        @testset "slice/index postions" begin
            test_rrule(getindex, x, 2:3 ⊢ DoesNotExist())
            test_rrule(getindex, x, 3:-1:2 ⊢ DoesNotExist())
            test_rrule(getindex, x, [3,2] ⊢ DoesNotExist())
            test_rrule(getindex, x, [2,3] ⊢ DoesNotExist())

            test_rrule(getindex, x, 1:2 ⊢ DoesNotExist(), 2:3 ⊢ DoesNotExist())
            test_rrule(getindex, x, (:) ⊢ DoesNotExist(), 2:3 ⊢ DoesNotExist())

            test_rrule(getindex, x, 1:2 ⊢ DoesNotExist(), 1)
            test_rrule(getindex, x, 1, 1:2 ⊢ DoesNotExist())

            test_rrule(getindex, x, 1:2 ⊢ DoesNotExist(), 2:3 ⊢ DoesNotExist())
            test_rrule(getindex, x, (:) ⊢ DoesNotExist(), 2:3 ⊢ DoesNotExist())

            test_rrule(getindex, x, (:) ⊢ DoesNotExist(), (:) ⊢ DoesNotExist())
            test_rrule(getindex, x, (:) ⊢ DoesNotExist())
        end

        @testset "masking" begin
            test_rrule(getindex, x, trues(size(x)) ⊢ DoesNotExist())
            test_rrule(getindex, x, trues(length(x)) ⊢ DoesNotExist())

            mask = falses(size(x))
            mask[2,3] = true
            mask[1,2] = true
            test_rrule(getindex, x, mask ⊢ DoesNotExist())

            test_rrule(getindex, x, [true, false] ⊢ DoesNotExist(), (:) ⊢ DoesNotExist())
        end

        @testset "By position with repeated elements" begin
            test_rrule(getindex, x, [2, 2] ⊢ DoesNotExist())
            test_rrule(getindex, x, [2, 2, 2] ⊢ DoesNotExist())
            test_rrule(getindex, x, [2,2] ⊢ DoesNotExist(), [3,3] ⊢ DoesNotExist())
        end
    end
end
