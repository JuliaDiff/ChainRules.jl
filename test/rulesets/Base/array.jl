@testset "reshape" begin
    test_rrule(reshape, rand(4, 5), (2, 10) ⊢ NoTangent())
    test_rrule(reshape, rand(4, 5), 2, 10)
    test_rrule(reshape, rand(4, 5), 2, :)
end

@testset "hcat" begin
    test_rrule(hcat, randn(3, 2), randn(3), randn(3, 3))
    test_rrule(hcat, rand(), rand(1,2), rand(1,2,1))
    test_rrule(hcat, rand(3,1,1,2), rand(3,3,1,2))
end

@testset "reduce hcat" begin
    A = randn(3, 2)
    B = randn(3, 1)
    C = randn(3, 3)
    test_rrule(reduce, hcat ⊢ NoTangent(), [A, B, C])
end

@testset "vcat" begin
    test_rrule(vcat, randn(2, 4), randn(1, 4), randn(3, 4))
    test_rrule(vcat, rand(), rand())
    test_rrule(vcat, rand(), rand(3), rand(3,1,1))
    test_rrule(vcat, rand(3,1,2), rand(4,1,2))
end

@testset "reduce vcat" begin
    A = randn(2, 4)
    B = randn(1, 4)
    C = randn(3, 4)
    test_rrule(reduce, vcat ⊢ NoTangent(), [A, B, C])
end

@testset "cat" begin
    test_rrule(cat, rand(2, 4), rand(1, 4); fkwargs=(dims=1,))
    test_rrule(cat, rand(2, 4), rand(2); fkwargs=(dims=Val(2),))
    test_rrule(cat, rand(), rand(2, 3); fkwargs=(dims=[1,2],))
    test_rrule(cat, rand(1), rand(3, 2, 1); fkwargs=(dims=(1,2),), check_inferred=false) # infers Tuple{Zero, Vector{Float64}, Any}
end

@testset "hvcat" begin
    test_rrule(hvcat, 2 ⊢ DoesNotExist(), rand(ComplexF64, 6)...)
    test_rrule(hvcat, (2, 1) ⊢ DoesNotExist(), rand(), rand(1,1), rand(2,2))
    test_rrule(hvcat, 1 ⊢ DoesNotExist(), rand(3)' ⊢ rand(1,3), transpose(rand(3)) ⊢ rand(1,3))
    test_rrule(hvcat, 1 ⊢ DoesNotExist(), rand(0,3), rand(2,3), rand(1,3,1))
end

@testset "fill" begin
    test_rrule(fill, 44.0, 4; check_inferred=false)
    test_rrule(fill, 2.0, (3, 3, 3) ⊢ NoTangent())
end
