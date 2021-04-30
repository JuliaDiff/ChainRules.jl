@testset "reshape" begin
    test_rrule(reshape, rand(4, 5), (2, 10) ⊢ DoesNotExist())
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
    test_rrule(reduce, hcat ⊢ DoesNotExist(), [A, B, C])
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
    test_rrule(reduce, vcat ⊢ DoesNotExist(), [A, B, C])
end

@testset "fill" begin
    test_rrule(fill, 44.0, 4; check_inferred=false)
    test_rrule(fill, 2.0, (3, 3, 3) ⊢ DoesNotExist())
end

@testset "repeat" begin
    @testset "rrule" begin
        test_rrule(repeat, randn(5), 3)
        test_rrule(repeat, randn(5), 3, 3)
        test_rrule(repeat, randn(3, 3), 2)
        test_rrule(repeat, randn(5, 5), 2,5)
        test_rrule(repeat, randn(5, 4, 3); fkwargs=(inner=(2, 2, 1), outer=(1, 1, 3)))
        test_rrule(repeat, fill(4.0), 3)
    end

    @testset "frule" begin
        test_frule(repeat, randn(5), 3)
        test_frule(repeat, randn(5), 3,3)
        test_frule(repeat, randn(3, 3), 2)
        test_frule(repeat, randn(3, 3), 2,5)
        test_frule(repeat, randn(5, 4, 3); fkwargs=(inner=(2, 2, 1), outer=(1, 1, 3)))
        test_frule(repeat, fill(4.0), 3)
    end
end
