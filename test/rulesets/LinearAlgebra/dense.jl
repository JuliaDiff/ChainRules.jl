function generate_well_conditioned_matrix(rng, N)
    A = randn(rng, N, N)
    return A * A' + I
end

@testset "linalg" begin
    @testset "dot" begin
        @testset "Vector" begin
            rng, M = MersenneTwister(123456), 3
            x, y = randn(rng, M), randn(rng, M)
            ẋ, ẏ = randn(rng, M), randn(rng, M)
            x̄, ȳ = randn(rng, M), randn(rng, M)
            frule_test(dot, (x, ẋ), (y, ẏ))
            rrule_test(dot, randn(rng), (x, x̄), (y, ȳ))
        end
        @testset "Matrix" begin
            rng, M, N = MersenneTwister(123456), 3, 4
            x, y = randn(rng, M, N), randn(rng, M, N)
            ẋ, ẏ = randn(rng, M, N), randn(rng, M, N)
            x̄, ȳ = randn(rng, M, N), randn(rng, M, N)
            frule_test(dot, (x, ẋ), (y, ẏ))
            rrule_test(dot, randn(rng), (x, x̄), (y, ȳ))
        end
        @testset "Array{T, 3}" begin
            rng, M, N, P = MersenneTwister(123456), 3, 4, 5
            x, y = randn(rng, M, N, P), randn(rng, M, N, P)
            ẋ, ẏ = randn(rng, M, N, P), randn(rng, M, N, P)
            x̄, ȳ = randn(rng, M, N, P), randn(rng, M, N, P)
            frule_test(dot, (x, ẋ), (y, ẏ))
            rrule_test(dot, randn(rng), (x, x̄), (y, ȳ))
        end
    end
    @testset "inv" begin
        rng, N = MersenneTwister(123456), 3
        B = generate_well_conditioned_matrix(rng, N)
        frule_test(inv, (B, randn(rng, N, N)))
        rrule_test(inv, randn(rng, N, N), (B, randn(rng, N, N)))
    end
    @testset "det" begin
        rng, N = MersenneTwister(123456), 3
        B = generate_well_conditioned_matrix(rng, N)
        frule_test(det, (B, randn(rng, N, N)))
        rrule_test(det, randn(rng), (B, randn(rng, N, N)))
    end
    @testset "logdet" begin
        rng, N = MersenneTwister(123456), 3
        B = generate_well_conditioned_matrix(rng, N)
        frule_test(logdet, (B, randn(rng, N, N)))
        rrule_test(logdet, randn(rng), (B, randn(rng, N, N)))
    end
    @testset "tr" begin
        rng, N = MersenneTwister(123456), 4
        frule_test(tr, (randn(rng, N, N), randn(rng, N, N)))
        rrule_test(tr, randn(rng), (randn(rng, N, N), randn(rng, N, N)))
    end
    @testset "*" begin
        rng = MersenneTwister(123456)
        dims = [3,4,5]
        for n in dims, m in dims, p in dims
            n > 3 && n == m == p && continue  # don't need to test square case multiple times
            A = randn(rng, m, n)
            B = randn(rng, n, p)
            Ȳ = randn(rng, m, p)
            rrule_test(*, Ȳ, (A, randn(rng, m, n)), (B, randn(rng, n, p)))
        end
    end
    @testset "$f" for f in [/, \]
        rng = MersenneTwister(42)
        @testset "Matrix" begin
            for n in 3:5, m in 3:5
                A = randn(rng, m, n)
                B = randn(rng, m, n)
                Ȳ = randn(rng, size(f(A, B)))
                rrule_test(f, Ȳ, (A, randn(rng, m, n)), (B, randn(rng, m, n)))
            end
        end
        @testset "Vector" begin
            x = randn(rng, 10)
            y = randn(rng, 10)
            ȳ = randn(rng, size(f(x, y))...)
            rrule_test(f, ȳ, (x, randn(rng, 10)), (y, randn(rng, 10)))
        end
        if f == (/)
            @testset "$T on the RHS" for T in (Diagonal, UpperTriangular, LowerTriangular)
                RHS = T(randn(rng, T == Diagonal ? 10 : (10, 10)))
                Y = randn(rng, 5, 10)
                Ȳ = randn(rng, size(f(Y, RHS))...)
                rrule_test(f, Ȳ, (Y, randn(rng, size(Y))), (RHS, randn(rng, size(RHS))))
            end
        else
            @testset "$T on LHS" for T in (Diagonal, UpperTriangular, LowerTriangular)
                LHS = T(randn(rng, T == Diagonal ? 10 : (10, 10)))
                y = randn(rng, 10)
                ȳ = randn(rng, size(f(LHS, y))...)
                rrule_test(f, ȳ, (LHS, randn(rng, size(LHS))), (y, randn(rng, 10)))
                Y = randn(rng, 10, 10)
                Ȳ = randn(rng, 10, 10)
                rrule_test(f, Ȳ, (LHS, randn(rng, size(LHS))), (Y, randn(rng, size(Y))))
            end
            @testset "Matrix $f Vector" begin
                X = randn(rng, 10, 4)
                y = randn(rng, 10)
                ȳ = randn(rng, size(f(X, y))...)
                rrule_test(f, ȳ, (X, randn(rng, size(X))), (y, randn(rng, 10)))
            end
            @testset "Vector $f Matrix" begin
                x = randn(rng, 10)
                Y = randn(rng, 10, 4)
                ȳ = randn(rng, size(f(x, Y))...)
                rrule_test(f, ȳ, (x, randn(rng, size(x))), (Y, randn(rng, size(Y))))
            end
        end
    end
    @testset "norm" begin
        rng = MersenneTwister(3)
        for dims in [(), (5,), (3, 2), (7, 3, 2)]
            A = randn(rng, dims...)
            p = randn(rng)
            ȳ = randn(rng)
            rrule_test(norm, ȳ, (A, randn(rng, dims...)), (p, randn(rng)))
        end
    end
end
