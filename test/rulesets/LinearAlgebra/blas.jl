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
            incx = 2
            incy = 3
            x, y = randn(n * incx), randn(n * incy)
            x̄, ȳ = randn(n * incx), randn(n * incy)
            rrule_test(
                BLAS.dot,
                randn(),
                (n, nothing),
                (x, x̄),
                (incx, nothing),
                (y, ȳ),
                (incy, nothing),
            )
        end
    end

    @testset "nrm2" begin
        @testset "all entries" begin
            @testset "$T" for T in (Float64,ComplexF64)
                n = 10
                x, ẋ, x̄ = randn(T, n), randn(T, n), randn(T, n)
                frule_test(BLAS.nrm2, (x, ẋ))
                rrule_test(BLAS.nrm2, randn(), (x, x̄); rtol=1e-7)
            end
        end

        @testset "over strides" begin
            dims = (3, 2, 1)
            incx = 2
            @testset "Array{$T,$N}" for N in 1:length(dims), T in (Float64,ComplexF64)
                s = (dims[1] * incx, dims[2:N]...)
                n = div(prod(s), incx)
                x, x̄ = randn(T, s), randn(T, s)
                rrule_test(
                    BLAS.nrm2,
                    randn(),
                    (n, nothing),
                    (x, x̄),
                    (incx, nothing);
                    atol=0,
                    rtol=1e-5,
                )
            end
        end
    end

    @testset "asum" begin
        @testset "all entries" begin
            @testset "$T" for T in (Float64,ComplexF64)
                n = 6
                x, ẋ, x̄ = randn(T, n), randn(T, n), randn(T, n)
                frule_test(BLAS.asum, (x, ẋ))
                rrule_test(BLAS.asum, randn(), (x, x̄))
            end
        end

        @testset "over strides" begin
            dims = (2, 2, 1)
            incx = 2
            @testset "Array{$T,$N}" for N in 1:length(dims), T in (Float64,ComplexF64)
                s = (dims[1] * incx, dims[2:N]...)
                n = div(prod(s), incx)
                x, x̄ = randn(T, s), randn(T, s)
                rrule_test( BLAS.asum, randn(), (n, nothing), (x, x̄), (incx, nothing))
            end
        end
    end

    @testset "gemm" begin
        dims = 3:5
        for m in dims, n in dims, p in dims, tA in ('N', 'C', 'T'), tB in ('N', 'C', 'T'), T in (Float64, ComplexF64)
            α = randn(T)
            A = randn(T, tA === 'N' ? (m, n) : (n, m))
            B = randn(T, tB === 'N' ? (n, p) : (p, n))
            C = gemm(tA, tB, α, A, B)
            ȳ = randn(T, size(C)...)
            rrule_test(
                gemm,
                ȳ,
                (tA, nothing),
                (tB, nothing),
                (α, randn(T)),
                (A, randn(T, size(A))),
                (B, randn(T, size(B))),
            )

            rrule_test(
                gemm,
                ȳ,
                (tA, nothing),
                (tB, nothing),
                (A, randn(T, size(A))),
                (B, randn(T, size(B))),
            )
        end
    end

    @testset "gemv" begin
        for n in 3:5, m in 3:5, t in ('N', 'C', 'T'), T in (Float64, ComplexF64)
            α = randn(T)
            A = randn(T, m, n)
            x = randn(T, t === 'N' ? n : m)
            y = gemv(t, α, A, x)
            ȳ = randn(T, size(y)...)
            rrule_test(
                gemv,
                ȳ,
                (t, nothing),
                (α, randn(T)),
                (A, randn(T, size(A))),
                (x, randn(T, size(x))),
            )
        end
    end
end
