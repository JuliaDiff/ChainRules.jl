@testset "Structured Matrices" begin
    @testset "/ and \\ on Square Matrixes" begin
        @testset "//, $T on the RHS" for T in (Diagonal, UpperTriangular, LowerTriangular)
            RHS = T(randn(T == Diagonal ? 10 : (10, 10)))
            Y = randn(5, 10)
            Ȳ = randn(size(/(Y, RHS))...)
            rrule_test(/, Ȳ, (Y, randn(size(Y))), (RHS, randn(size(RHS))))
        end

        @testset "\\ $T on LHS" for T in (Diagonal, UpperTriangular, LowerTriangular)
            LHS = T(randn(T == Diagonal ? 10 : (10, 10)))
            y = randn(10)
            ȳ = randn(size(\(LHS, y))...)
            rrule_test(\, ȳ, (LHS, randn(size(LHS))), (y, randn(10)))
            Y = randn(10, 10)
            Ȳ = randn(10, 10)
            rrule_test(\, Ȳ, (LHS, randn(size(LHS))), (Y, randn(size(Y))))
        end
    end

    @testset "Diagonal" begin
        N = 3
        rrule_test(Diagonal, randn(N, N), (randn(N), randn(N)))
        D = Diagonal(randn( N))
        rrule_test(Diagonal, D, (randn(N), randn(N)))
        # Concrete type instead of UnionAll
        rrule_test(typeof(D), D, (randn(N), randn(N)))

        # TODO: replace this with a `rrule_test` once we have that working
        # see https://github.com/JuliaDiff/ChainRulesTestUtils.jl/issues/24
        res, pb = rrule(Diagonal, [1, 4])
        @test pb(10*res) == (NO_FIELDS, [10, 40])
        comp = Composite{typeof(res)}(; diag=10*res.diag)  # this is the structure of Diagonal
        @test pb(comp) == (NO_FIELDS, [10, 40])
    end

    @testset "::Diagonal * ::AbstractVector" begin
        N = 3
        rrule_test(
            *,
            randn(N),
            (Diagonal(randn(N)), Diagonal(randn(N))),
            (randn(N), randn(N)),
        )
    end
    @testset "diag" begin
        N = 7
        rrule_test(diag, randn(N), (randn(N, N), randn(N, N)))
        rrule_test(diag, randn(N), (Diagonal(randn(N)), randn(N, N)))
        rrule_test(diag, randn(N), (randn(N, N), Diagonal(randn(N))))
        rrule_test(diag, randn(N), (Diagonal(randn(N)), Diagonal(randn(N))))
        VERSION ≥ v"1.3" && @testset "k=$k" for k in (-1, 0, 2)
            M = N - abs(k)
            rrule_test(diag, randn(M), (randn(N, N), randn(N, N)), (k, nothing))
        end
    end
    @testset "diagm" begin
        @testset "without size" begin
            M, N = 7, 9
            s = (8, 8)
            a, ā = randn(M), randn(M)
            b, b̄ = randn(M), randn(M)
            c, c̄ = randn(M - 1), randn(M - 1)
            ȳ = randn(s)
            ps = (0 => a, 1 => b, 0 => c)
            y, back = rrule(diagm, ps...)
            @test y == diagm(ps...)
            ∂self, ∂pa, ∂pb, ∂pc = back(ȳ)
            @test ∂self === NO_FIELDS
            ∂a_fd, ∂b_fd, ∂c_fd = j′vp(_fdm, (a, b, c) -> diagm(0 => a, 1 => b, 0 => c), ȳ, a, b, c)
            for (p, ∂px, ∂x_fd) in zip(ps, (∂pa, ∂pb, ∂pc), (∂a_fd, ∂b_fd, ∂c_fd))
                ∂px = unthunk(∂px)
                @test ∂px isa Composite{typeof(p)}
                @test ∂px.first isa AbstractZero
                @test ∂px.second ≈ ∂x_fd
            end
        end
        VERSION ≥ v"1.3" && @testset "with size" begin
            M, N = 7, 9
            a, ā = randn(M), randn(M)
            b, b̄ = randn(M), randn(M)
            c, c̄ = randn(M - 1), randn(M - 1)
            ȳ = randn(M, N)
            ps = (0 => a, 1 => b, 0 => c)
            y, back = rrule(diagm, M, N, ps...)
            @test y == diagm(M, N, ps...)
            ∂self, ∂M, ∂N, ∂pa, ∂pb, ∂pc = back(ȳ)
            @test ∂self === NO_FIELDS
            @test ∂M === DoesNotExist()
            @test ∂N === DoesNotExist()
            ∂a_fd, ∂b_fd, ∂c_fd = j′vp(_fdm, (a, b, c) -> diagm(M, N, 0 => a, 1 => b, 0 => c), ȳ, a, b, c)
            for (p, ∂px, ∂x_fd) in zip(ps, (∂pa, ∂pb, ∂pc), (∂a_fd, ∂b_fd, ∂c_fd))
                ∂px = unthunk(∂px)
                @test ∂px isa Composite{typeof(p)}
                @test ∂px.first isa AbstractZero
                @test ∂px.second ≈ ∂x_fd
            end
        end
    end
    @testset "$(SymHerm)(::AbstractMatrix{$T}, :$(uplo))" for
        SymHerm in (Symmetric, Hermitian),
        T in (Float64, ComplexF64),
        uplo in (:U, :L)

        N = 3
        @testset "frule" begin
            x = randn(T, N, N)
            Δx = randn(T, N, N)
            # can't use frule_test here because it doesn't yet ignore nothing tangents
            Ω = SymHerm(x, uplo)
            Ω_ad, ∂Ω_ad = frule((Zero(), Δx, Zero()), SymHerm, x, uplo)
            @test Ω_ad == Ω
            ∂Ω_fd = jvp(_fdm, z -> SymHerm(z, uplo), (x, Δx))
            @test ∂Ω_ad ≈ ∂Ω_fd
        end
        @testset "rrule" begin
            x = randn(T, N, N)
            ∂x = randn(T, N, N)
            ΔΩ = randn(T, N, N)
            @testset "back(::$MT)" for MT in (Matrix, LowerTriangular, UpperTriangular)
                rrule_test(SymHerm, MT(ΔΩ), (x, ∂x), (uplo, nothing))
            end
            @testset "back(::Diagonal)" begin
                rrule_test(SymHerm, Diagonal(ΔΩ), (x, Diagonal(∂x)), (uplo, nothing))
            end
        end
    end
    @testset "$(f)(::$(SymHerm){$T}) with uplo=:$uplo" for f in (Matrix, Array),
        SymHerm in (Symmetric, Hermitian),
        T in (Float64, ComplexF64),
        uplo in (:U, :L)

        N = 3
        x = SymHerm(randn(T, N, N), uplo)
        Δx = randn(T, N, N)
        ∂x = SymHerm(randn(T, N, N), uplo)
        ΔΩ = f(SymHerm(randn(T, N, N), uplo))
        frule_test(f, (x, Δx))
        frule_test(f, (x, SymHerm(Δx, uplo)))
        rrule_test(f, ΔΩ, (x, ∂x))
    end
    @testset "$f" for f in (Adjoint, adjoint, Transpose, transpose)
        n = 5
        m = 3
        rrule_test(f, randn(m, n), (randn(n, m), randn(n, m)))
        rrule_test(f, randn(1, n), (randn(n), randn(n)))
    end
    @testset "$T" for T in (UpperTriangular, LowerTriangular)
        n = 5
        rrule_test(T, T(randn(n, n)), (randn(n, n), randn(n, n)))
    end
    @testset "$Op" for Op in (triu, tril)
        n = 7
        rrule_test(Op, randn(n, n), (randn(n, n), randn(n, n)))
        @testset "k=$k" for k in -2:2
            rrule_test(Op, randn(n, n), (randn(n, n), randn(n, n)), (k, nothing))
        end
    end

    @testset "det and logdet $S" for S in (Diagonal, UpperTriangular, LowerTriangular)
        @testset "$op" for op in (det, logdet)
            @testset "$T" for T in (Float64, ComplexF64)
                n = 5
                # rand (not randn) so det will be postive, so logdet will be defined
                X = S(3*rand(T, (n, n)) .+ 1)
                X̄_acc = Diagonal(rand(T, (n, n)))  # sensitivity is always a diagonal for these types
                rrule_test(op, rand(T), (X, X̄_acc))
            end
            @testset "return type" begin
                X = S(3*rand(6, 6) .+ 1)
                _, op_pullback = rrule(op, X)
                X̄ = op_pullback(2.7)[2]
                @test X̄ isa Diagonal
            end
        end
    end
end
