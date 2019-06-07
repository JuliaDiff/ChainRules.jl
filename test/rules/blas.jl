@testset "BLAS" begin
    @testset "gemm" begin
        rng = MersenneTwister(1)
        dims = 3:5
        for m in dims, n in dims, p in dims, tA in ('N', 'T'), tB in ('N', 'T')
            α = randn(rng)
            A = randn(rng, tA === 'N' ? (m, n) : (n, m))
            B = randn(rng, tB === 'N' ? (n, p) : (p, n))
            C = gemm(tA, tB, α, A, B)
            ȳ = randn(rng, size(C)...)
            rrule_test(gemm, ȳ, (tA, nothing), (tB, nothing), (α, randn(rng)),
                       (A, randn(rng, size(A))), (B, randn(rng, size(B))))
        end
    end
    @testset "gemv" begin
        rng = MersenneTwister(2)
        for n in 3:5, m in 3:5, t in ('N', 'T')
            α = randn(rng)
            A = randn(rng, m, n)
            x = randn(rng, t === 'N' ? n : m)
            y = α * (t === 'N' ? A : A') * x
            ȳ = randn(rng, size(y)...)
            rrule_test(gemv, ȳ, (t, nothing), (α, randn(rng)), (A, randn(rng, size(A))),
                       (x, randn(rng, size(x))))
        end
    end
end
