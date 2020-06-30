@testset "BLAS" begin
    @testset "dot" begin
        @testset "all entries" begin
            n = 10
            x, y = randn(n), randn(n)
            ẋ, ẏ = randn(n), randn(n)
            x̄, ȳ = randn(n), randn(n)
            frule_test(BLAS.dot, (x, ẋ), (y, ẏ))
            rrule_test(BLAS.dot, randn(), (x, x̄), (y, ȳ))
        end

        @testset "over strides" begin
            n = 10
            stride1 = 2
            stride2 = 3
            x, y = randn(n * stride1), randn(n * stride2)
            x̄, ȳ = randn(n * stride1), randn(n * stride2)
            rrule_test(
                BLAS.dot,
                randn(),
                (n, nothing),
                (x, x̄),
                (stride1, nothing),
                (y, ȳ),
                (stride2, nothing),
            )
        end
    end

    @testset "nrm2" begin
        @testset "all entires" begin
            @testset "$T" for T in (Float64,ComplexF64)
                n = 10
                x, ẋ, x̄ = randn(T, n), randn(T, n), randn(T, n)
                frule_test(BLAS.nrm2, (x, ẋ))
                rrule_test(BLAS.nrm2, randn(), (x, x̄))
            end
        end

        @testset "over strides" begin
            dims = (5, 2, 3)
            stride = 2
            @testset "Array{$T,$N}" for N in 1:length(dims), T in (Float64,)
                s = (dims[1] * stride, dims[2:N]...)
                n = prod(s)
                x, x̄ = randn(T, s), randn(T, s)
                rrule_test(
                    BLAS.nrm2,
                    randn(),
                    (n, nothing),
                    (x, x̄),
                    (stride, nothing);
                    atol=0,
                    rtol=sqrt(eps(T)),
                )
            end
        end
    end

    @testset "gemm" begin
        dims = 3:5
        for m in dims, n in dims, p in dims, tA in ('N', 'T'), tB in ('N', 'T')
            α = randn()
            A = randn(tA === 'N' ? (m, n) : (n, m))
            B = randn(tB === 'N' ? (n, p) : (p, n))
            C = gemm(tA, tB, α, A, B)
            ȳ = randn(size(C)...)
            rrule_test(
                gemm,
                ȳ,
                (tA, nothing),
                (tB, nothing),
                (α, randn()),
                (A, randn(size(A))),
                (B, randn(size(B))),
            )
        end
    end
    @testset "gemv" begin
        for n in 3:5, m in 3:5, t in ('N', 'T')
            α = randn()
            A = randn(m, n)
            x = randn(t === 'N' ? n : m)
            y = α * (t === 'N' ? A : A') * x
            ȳ = randn(size(y)...)
            rrule_test(
                gemv,
                ȳ,
                (t, nothing),
                (α, randn()),
                (A, randn(size(A))),
                (x, randn(size(x))),
            )
        end
    end
end
