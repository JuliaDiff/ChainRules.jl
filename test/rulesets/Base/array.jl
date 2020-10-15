@testset "reshape" begin
    x = rand(4, 5)
    x̄ = rand(4, 5)

    ȳ = rand(2, 10)

    rrule_test(reshape, ȳ, (x, x̄), ((2, 10), nothing))
    rrule_test(reshape, ȳ, (x, x̄), (2, nothing), (10, nothing))
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

@testset "getindex" begin
    x = [1.0 2.0 3.0; 10.0 20.0 30.0]
    x̄ = [1.4 2.5 3.7; 10.5 20.1 30.2]
    rrule_test(getindex, 2.3, (x, x̄), (2, nothing))
    rrule_test(getindex, 2.3, (x, x̄), (2, nothing), (1, nothing))
end
