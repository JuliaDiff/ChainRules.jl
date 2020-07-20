@testset "reshape" begin
    A = rand(4, 5)
    B, pullback = rrule(reshape, A, (5, 4))
    @test B == reshape(A, (5, 4))
    Ȳ = randn(4, 5)

    (s̄, Ā, d̄) = pullback(Ȳ)
    @test s̄ == NO_FIELDS
    @test d̄ isa DoesNotExist
    @test extern(Ā) == reshape(Ȳ, (5, 4))

    B, pullback = rrule(reshape, A, 5, 4)
    @test B == reshape(A, 5, 4)

    Ȳ = randn(4, 5)
    (s̄, Ā, d̄1, d̄2) = pullback(Ȳ)
    @test s̄ == NO_FIELDS
    @test d̄1 isa DoesNotExist
    @test d̄2 isa DoesNotExist
    @test extern(Ā) == reshape(Ȳ, 5, 4)
end

@testset "hcat" begin
    A = randn(3, 2)
    B = randn(3)
    C = randn(3, 3)
    H, pullback = rrule(hcat, A, B, C)
    @test H == hcat(A, B, C)
    H̄ = randn(3, 6)
    (ds, dA, dB, dC) = pullback(H̄)
    @test ds == NO_FIELDS
    @test dA ≈ view(H̄, :, 1:2)
    @test dB ≈ view(H̄, :, 3)
    @test dC ≈ view(H̄, :, 4:6)
end

@testset "reduce hcat" begin
    A = randn(3, 2)
    B = randn(3, 1)
    C = randn(3, 3)
    x = [A, B, C]
    H, pullback = rrule(reduce, hcat, x)
    @test H == reduce(hcat, x)
    H̄ = randn(3, 6)
    x̄ = randn.(size.(x))
    rrule_test(reduce, H̄, (hcat, nothing), (x, x̄))
end

@testset "vcat" begin
    A = randn(2, 4)
    B = randn(1, 4)
    C = randn(3, 4)
    V, pullback = rrule(vcat, A, B, C)
    @test V == vcat(A, B, C)
    V̄ = randn(6, 4)
    (ds, dA, dB, dC) = pullback(V̄)
    @test ds == NO_FIELDS
    @test dA ≈ view(V̄, 1:2, :)
    @test dB ≈ view(V̄, 3:3, :)
    @test dC ≈ view(V̄, 4:6, :)
end

@testset "reduce vcat" begin
    A = randn(2, 4)
    B = randn(1, 4)
    C = randn(3, 4)
    x = [A, B, C]
    V, pullback = rrule(reduce, vcat, x)
    @test V == reduce(vcat, x)
    V̄ = randn(6, 4)
    x̄ = randn.(size.(x))
    rrule_test(reduce, V̄, (vcat, nothing), (x, x̄))
end

@testset "fill" begin
    y, pullback = rrule(fill, 44, 4)
    @test y == [44, 44, 44, 44]
    (ds, dv, dd) = pullback(ones(4))
    @test ds === NO_FIELDS
    @test dd isa DoesNotExist
    @test extern(dv) == 4

    y, pullback = rrule(fill, 2.0, (3, 3, 3))
    @test y == fill(2.0, (3, 3, 3))
    (ds, dv, dd) = pullback(ones(3, 3, 3))
    @test ds === NO_FIELDS
    @test dd isa DoesNotExist
    @test dv ≈ 27.0
end
