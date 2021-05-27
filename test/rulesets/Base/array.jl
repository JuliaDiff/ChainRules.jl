@testset "reshape" begin
    test_rrule(reshape, rand(4, 5), (2, 10) ⊢ NoTangent())
    test_rrule(reshape, rand(4, 5), 2, 10)
end

@testset "hcat" begin
    A = randn(3, 2)
    B = randn(3)
    C = randn(3, 3)
    test_rrule(hcat, A, B, C; check_inferred=false)
end

@testset "reduce hcat" begin
    A = randn(3, 2)
    B = randn(3, 1)
    C = randn(3, 3)
    test_rrule(reduce, hcat ⊢ NoTangent(), [A, B, C])
end

@testset "vcat" begin
    A = randn(2, 4)
    B = randn(1, 4)
    C = randn(3, 4)
    test_rrule(vcat, A, B, C; check_inferred=false)
end

@testset "reduce vcat" begin
    A = randn(2, 4)
    B = randn(1, 4)
    C = randn(3, 4)
    test_rrule(reduce, vcat ⊢ NoTangent(), [A, B, C])
end

@testset "fill" begin
    test_rrule(fill, 44.0, 4; check_inferred=false)
    test_rrule(fill, 2.0, (3, 3, 3) ⊢ NoTangent())
end
