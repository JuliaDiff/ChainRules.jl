using LinearAlgebra.BLAS: gemm

@testset "BLAS" begin
    @testset "gemm" begin
        rng = MersenneTwister(1)
        dims = 3:5
        for m in dims, n in dims, p in dims, tA in ('N', 'T'), tB in ('N', 'T')
            α = randn(rng)
            A = randn(rng, tA === 'N' ? (m, n) : (n, m))
            B = randn(rng, tB === 'N' ? (n, p) : (p, n))
            C = gemm(tA, tB, α, A, B)
            fAB, (dtA, dtB, dα, dA, dB) = rrule(gemm, tA, tB, α, A, B)
            @test C ≈ fAB
            @test dtA isa ChainRules.DNERule
            @test dtB isa ChainRules.DNERule
            for (f, x, dx) in [(X->gemm(tA, tB, X, A, B), α, dα),
                               (X->gemm(tA, tB, α, X, B), A, dA),
                               (X->gemm(tA, tB, α, A, X), B, dB)]
                ȳ = randn(rng, size(C)...)
                x̄_ad = dx(ȳ)
                x̄_fd = j′vp(central_fdm(5, 1), f, ȳ, x)
                @test x̄_ad ≈ x̄_fd rtol=1e-9 atol=1e-9
            end
        end
    end
end
