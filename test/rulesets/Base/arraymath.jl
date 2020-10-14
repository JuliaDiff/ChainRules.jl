@testset "arraymath" begin
    @testset "inv(::Matrix{$T})" for T in (Float64, ComplexF64)
        N = 3
        B = generate_well_conditioned_matrix(T, N)
        frule_test(inv, (B, randn(T, N, N)))
        rrule_test(inv, randn(T, N, N), (B, randn(T, N, N)))
    end

    @testset "*" begin
        @testset "Matrix-Matrix" begin
            dims = [3,4,5]
            for n in dims, m in dims, p in dims
                # don't need to test square case multiple times
                n > 3 && n == m == p && continue
                A = randn(m, n)
                B = randn(n, p)
                Ȳ = randn(m, p)
                rrule_test(*, Ȳ, (A, randn(m, n)), (B, randn(n, p)))
            end
        end
        @testset "Scalar-AbstractArray" begin
            for dims in ((3,), (5,4), (10,10), (2,3,4), (2,3,4,5))
                rrule_test(*, randn(dims), (1.5, 4.2), (randn(dims), randn(dims)))
                rrule_test(*, randn(dims), (randn(dims), randn(dims)), (1.5, 4.2))
            end
        end
    end

    @testset "$f" for f in (/, \)
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
        if f == (\)
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
    @testset "/ and \\ Scalar-AbstractArray" begin
        A = randn(3, 4, 5)
        Ā = randn(3, 4, 5)
        Ȳ = randn(3, 4, 5)
        rrule_test(/, Ȳ, (A, Ā), (7.2, 2.3))
        rrule_test(\, Ȳ, (7.2, 2.3), (A, Ā))
    end


    @testset "negation" begin
        A = randn(4, 4)
        Ā = randn(4, 4,)
        Ȳ = randn(4, 4,)
        rrule_test(-, Ȳ, (A, Ā))
        rrule_test(-, Diagonal(Ȳ), (Diagonal(A), Diagonal(Ā)))
    end
end
