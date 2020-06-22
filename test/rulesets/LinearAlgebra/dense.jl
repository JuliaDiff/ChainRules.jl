@testset "linalg" begin
    @testset "dot" begin
        @testset "Vector" begin
            M = 3
            x, y = randn(M), randn(M)
            ẋ, ẏ = randn(M), randn(M)
            x̄, ȳ = randn(M), randn(M)
            frule_test(dot, (x, ẋ), (y, ẏ))
            rrule_test(dot, randn(), (x, x̄), (y, ȳ))
        end
        @testset "Matrix" begin
            M, N = 3, 4
            x, y = randn(M, N), randn(M, N)
            ẋ, ẏ = randn(M, N), randn(M, N)
            x̄, ȳ = randn(M, N), randn(M, N)
            frule_test(dot, (x, ẋ), (y, ẏ))
            rrule_test(dot, randn(), (x, x̄), (y, ȳ))
        end
        @testset "Array{T, 3}" begin
            M, N, P = 3, 4, 5
            x, y = randn(M, N, P), randn(M, N, P)
            ẋ, ẏ = randn(M, N, P), randn(M, N, P)
            x̄, ȳ = randn(M, N, P), randn(M, N, P)
            frule_test(dot, (x, ẋ), (y, ẏ))
            rrule_test(dot, randn(), (x, x̄), (y, ȳ))
        end
    end
    @testset "cross" begin
        @testset "frule" begin
            @testset "$T" for T in (Float64, ComplexF64)
                n = 3
                x, y = randn(T, n), randn(T, n)
                ẋ, ẏ = randn(T, n), randn(T, n)
                frule_test(cross, (x, ẋ), (y, ẏ))
            end
        end
        @testset "rrule" begin
            n = 3
            x, y = randn(n), randn(n)
            x̄, ȳ = randn(n), randn(n)
            ΔΩ = randn(n)
            rrule_test(cross, ΔΩ, (x, x̄), (y, ȳ))
        end
    end
    @testset "inv" begin
        N = 3
        B = generate_well_conditioned_matrix(N)
        frule_test(inv, (B, randn(N, N)))
        rrule_test(inv, randn(N, N), (B, randn(N, N)))
    end
    @testset "det" begin
        N = 3
        B = generate_well_conditioned_matrix(N)
        frule_test(det, (B, randn(N, N)))
        rrule_test(det, randn(), (B, randn(N, N)))
    end
    @testset "logdet" begin
        N = 3
        B = generate_well_conditioned_matrix(N)
        frule_test(logdet, (B, randn(N, N)))
        rrule_test(logdet, randn(), (B, randn(N, N)))
    end
    @testset "tr" begin
        N = 4
        frule_test(tr, (randn(N, N), randn(N, N)))
        rrule_test(tr, randn(), (randn(N, N), randn(N, N)))
    end
    @testset "*" begin
        dims = [3,4,5]
        for n in dims, m in dims, p in dims
            n > 3 && n == m == p && continue  # don't need to test square case multiple times
            A = randn(m, n)
            B = randn(n, p)
            Ȳ = randn(m, p)
            rrule_test(*, Ȳ, (A, randn(m, n)), (B, randn(n, p)))
        end
    end
    @testset "$f" for f in [/, \]
        @testset "Matrix" begin
            for n in 3:5, m in 3:5
                A = randn(m, n)
                B = randn(m, n)
                Ȳ = randn(size(f(A, B)))
                rrule_test(f, Ȳ, (A, randn(m, n)), (B, randn(m, n)))
            end
        end
        @testset "Vector" begin
            x = randn(10)
            y = randn(10)
            ȳ = randn(size(f(x, y))...)
            rrule_test(f, ȳ, (x, randn(10)), (y, randn(10)))
        end
        if f == (/)
            @testset "$T on the RHS" for T in (Diagonal, UpperTriangular, LowerTriangular)
                RHS = T(randn(T == Diagonal ? 10 : (10, 10)))
                Y = randn(5, 10)
                Ȳ = randn(size(f(Y, RHS))...)
                rrule_test(f, Ȳ, (Y, randn(size(Y))), (RHS, randn(size(RHS))))
            end
        else
            @testset "$T on LHS" for T in (Diagonal, UpperTriangular, LowerTriangular)
                LHS = T(randn(T == Diagonal ? 10 : (10, 10)))
                y = randn(10)
                ȳ = randn(size(f(LHS, y))...)
                rrule_test(f, ȳ, (LHS, randn(size(LHS))), (y, randn(10)))
                Y = randn(10, 10)
                Ȳ = randn(10, 10)
                rrule_test(f, Ȳ, (LHS, randn(size(LHS))), (Y, randn(size(Y))))
            end
            @testset "Matrix $f Vector" begin
                X = randn(10, 4)
                y = randn(10)
                ȳ = randn(size(f(X, y))...)
                rrule_test(f, ȳ, (X, randn(size(X))), (y, randn(10)))
            end
            @testset "Vector $f Matrix" begin
                x = randn(10)
                Y = randn(10, 4)
                ȳ = randn(size(f(x, Y))...)
                rrule_test(f, ȳ, (x, randn(size(x))), (Y, randn(size(Y))))
            end
        end
    end
    @testset "norm" begin
        for dims in [(), (5,), (3, 2), (7, 3, 2)]
            A = randn(dims...)
            p = randn()
            ȳ = randn()
            rrule_test(norm, ȳ, (A, randn(dims...)), (p, randn()))
        end
    end
end
