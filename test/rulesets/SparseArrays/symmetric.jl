@testset "Hermitian/Symmetric sparse matrices" begin
    n = 10
    k = 3  # number of columns for matrix multiplication

    # Helper to create a random sparse Hermitian/Symmetric matrix
    function rand_hermsym_sparse(SymHerm, T, n, uplo; density=0.3)
        A = sprand(T, n, n, density)
        A = A + A'
        return SymHerm(A, uplo)
    end

    # Helper to create a random tangent with same sparsity pattern
    function rand_tangent_sparse(A::Union{Hermitian, Symmetric})
        dA = similar(A)
        rand!(nonzeros(parent(dA)))
        return dA
    end

    # Convert symmetric gradient to raw parent convention (double off-diagonals in stored triangle, zero non-stored)
    function symgrad_to_parent(∂A)
        P = copy(parent(∂A))

        for j in axes(P, 2)
            for p in SparseArrays.nzrange(P, j)
                i = rowvals(P)[p]

                if (∂A.uplo == 'U' && i < j) || (∂A.uplo == 'L' && i > j)
                    nonzeros(P)[p] *= 2
                elseif (∂A.uplo == 'U' && i > j) || (∂A.uplo == 'L' && i < j)
                    nonzeros(P)[p] = 0
                end
            end
        end

        return P
    end

    # Test rrule by pulling back through parent and comparing to FD on parent
    function test_sparse_rrule(f, A, args...; fdm=_fdm, rtol=1e-6, atol=1e-9)
        SymHerm = typeof(A).name.wrapper
        uplo = Symbol(A.uplo)

        # Compute rrule
        y, pb = rrule(f, A, args...)
        dy = rand_tangent(y)
        cotangents = pb(dy)
        ∂A = unthunk(cotangents[2])

        # Convert symmetric gradient to parent convention
        ∂data = symgrad_to_parent(∂A)

        # FD on parent data
        f_parent(data) = f(SymHerm(data, uplo), args...)
        ∂data_fd = j′vp(fdm, f_parent, dy, parent(A))[1]

        @test ∂data ≈ ∂data_fd rtol=rtol atol=atol
    end

    function test_sparse_rrule_right(f, X, A; fdm=_fdm, rtol=1e-6, atol=1e-9)
        SymHerm = typeof(A).name.wrapper
        uplo = Symbol(A.uplo)

        # Compute rrule
        y, pb = rrule(f, X, A)
        dy = rand_tangent(y)
        cotangents = pb(dy)
        ∂A = unthunk(cotangents[3])

        # Convert symmetric gradient to parent convention
        ∂data = symgrad_to_parent(∂A)

        # FD on parent data
        f_parent(data) = f(X, SymHerm(data, uplo))
        ∂data_fd = j′vp(fdm, f_parent, dy, parent(A))[1]

        @test ∂data ≈ ∂data_fd rtol=rtol atol=atol
    end

    function test_sparse_rrule_dot(A, x, y; fdm=_fdm, rtol=1e-6, atol=1e-9)
        SymHerm = typeof(A).name.wrapper
        uplo = Symbol(A.uplo)

        # Compute rrule
        val, pb = rrule(dot, x, A, y)
        dval = rand_tangent(val)
        cotangents = pb(dval)
        ∂A = unthunk(cotangents[3])

        # Convert symmetric gradient to parent convention
        ∂data = symgrad_to_parent(∂A)

        # FD on parent data
        f_parent(data) = dot(x, SymHerm(data, uplo), y)
        ∂data_fd = j′vp(fdm, f_parent, dval, parent(A))[1]

        @test ∂data ≈ ∂data_fd rtol=rtol atol=atol
    end

    @testset "$(SymHerm){$T} * DenseVecOrMat, uplo=:$uplo" for
        SymHerm in (Symmetric, Hermitian),
        T in (Float64, ComplexF64),
        uplo in (:U, :L)

        A = rand_hermsym_sparse(SymHerm, T, n, uplo)
        dA = rand_tangent_sparse(A)
        x = randn(T, n)
        X = randn(T, n, k)
        Xt = randn(T, k, n)  # for transpose/adjoint tests

        @testset "A * x (vector)" begin
            test_sparse_rrule(*, A, x)
            test_frule(*, A ⊢ dA, x)
        end

        @testset "A * X (matrix)" begin
            test_sparse_rrule(*, A, X)
            test_frule(*, A ⊢ dA, X)
        end

        @testset "A * X' (adjoint matrix)" begin
            test_sparse_rrule(*, A, Xt')
            test_frule(*, A ⊢ dA, Xt')
        end

        @testset "A * transpose(X) (transpose matrix)" begin
            test_sparse_rrule(*, A, transpose(Xt))
            test_frule(*, A ⊢ dA, transpose(Xt))
        end
    end

    @testset "DenseMat * $(SymHerm){$T}, uplo=:$uplo" for
        SymHerm in (Symmetric, Hermitian),
        T in (Float64, ComplexF64),
        uplo in (:U, :L)

        A = rand_hermsym_sparse(SymHerm, T, n, uplo)
        dA = rand_tangent_sparse(A)
        X = randn(T, k, n)
        Xt = randn(T, n, k)  # for transpose/adjoint tests
        x = randn(T, n)      # for row vector tests

        @testset "X * A (matrix)" begin
            test_sparse_rrule_right(*, X, A)
            test_frule(*, X, A ⊢ dA)
        end

        @testset "X' * A (adjoint matrix)" begin
            test_sparse_rrule_right(*, Xt', A)
            test_frule(*, Xt', A ⊢ dA)
        end

        @testset "transpose(X) * A (transpose matrix)" begin
            test_sparse_rrule_right(*, transpose(Xt), A)
            test_frule(*, transpose(Xt), A ⊢ dA)
        end

        @testset "x' * A (adjoint vector / row vector)" begin
            test_sparse_rrule_right(*, x', A)
            test_frule(*, x', A ⊢ dA)
        end

        @testset "transpose(x) * A (transpose vector / row vector)" begin
            test_sparse_rrule_right(*, transpose(x), A)
            test_frule(*, transpose(x), A ⊢ dA)
        end
    end

    # dot(x, A, y) - vectors only, matching upstream ChainRules
    @testset "dot(x, $(SymHerm){$T}, y), uplo=:$uplo" for
        SymHerm in (Symmetric, Hermitian),
        T in (Float64, ComplexF64),
        uplo in (:U, :L)

        A = rand_hermsym_sparse(SymHerm, T, n, uplo)
        dA = rand_tangent_sparse(A)
        x = randn(T, n)
        y = randn(T, n)

        @testset "dot(x, A, y)" begin
            test_sparse_rrule_dot(A, x, y)
            test_frule(dot, x, A ⊢ dA, y)
        end
    end
end
