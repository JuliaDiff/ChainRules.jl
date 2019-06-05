@testset "reshape" begin
    rng = MersenneTwister(1)
    A = randn(rng, 4, 5)
    B, (dA, dd) = rrule(reshape, A, (5, 4))
    @test B == reshape(A, (5, 4))
    @test dd isa ChainRules.DNERule
    Ȳ = randn(rng, 4, 5)
    Ā = dA(Ȳ)
    @test Ā == reshape(Ȳ, (5, 4))

    B, (dA, dd1, dd2) = rrule(reshape, A, 5, 4)
    @test B == reshape(A, 5, 4)
    @test dd1 isa ChainRules.DNERule
    @test dd2 isa ChainRules.DNERule
    Ȳ = randn(rng, 4, 5)
    Ā = dA(Ȳ)
    @test Ā == reshape(Ȳ, 5, 4)
end

@testset "hcat" begin
    rng = MersenneTwister(2)
    A = randn(rng, 3, 2)
    B = randn(rng, 3)
    C = randn(rng, 3, 3)
    H, (dA, dB, dC) = rrule(hcat, A, B, C)
    @test H == hcat(A, B, C)
    H̄ = randn(rng, 3, 6)
    @test dA(H̄) ≈ view(H̄, :, 1:2)
    @test dB(H̄) ≈ view(H̄, :, 3)
    @test dC(H̄) ≈ view(H̄, :, 4:6)
end

@testset "vcat" begin
    rng = MersenneTwister(3)
    A = randn(rng, 2, 4)
    B = randn(rng, 1, 4)
    C = randn(rng, 3, 4)
    V, (dA, dB, dC) = rrule(vcat, A, B, C)
    @test V == vcat(A, B, C)
    V̄ = randn(rng, 6, 4)
    @test dA(V̄) ≈ view(V̄, 1:2, :)
    @test dB(V̄) ≈ view(V̄, 3:3, :)
    @test dC(V̄) ≈ view(V̄, 4:6, :)
end
