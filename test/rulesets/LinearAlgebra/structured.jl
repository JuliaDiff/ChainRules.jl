@testset "Structured Matrices" begin
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
    @testset "Symmetric(::AbstractMatrix{$T})" for T in (Float64, ComplexF64)
        N = 3
        rrule_test(Symmetric, randn(T, N, N), (randn(T, N, N), randn(T, N, N)))
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

    @testset "eigen(::$T{<:Real})" for T in (Symmetric, Hermitian)
        n = 5
        @testset "frule" begin
            @testset for uplo in (:L, :U)
                A, ΔA = T(randn(n, n), uplo), T(randn(n, n), uplo)
                F = eigen(A)
                F_ad, ∂F_ad = frule((Zero(), ΔA), eigen, A)
                @test F_ad == F
                ∂F_ad = unthunk(∂F_ad)
                @test ∂F_ad isa Composite{typeof(F)}
                @test ∂F_ad.values ≈ jvp(_fdm, A -> eigen(T(A, uplo)).values, (A.data, ΔA.data))
                # TODO: adopt a deterministic sign convention to stabilize the FD estimate
                @test ∂F_ad.vectors ≈ jvp(_fdm, A -> eigen(T(A, uplo)).vectors, (A.data, ΔA.data))
            end
        end

        @testset "rrule" begin
            @testset for uplo in (:L, :U)
                A, ΔU, Δλ = T(randn(n, n), uplo), randn(n, n), randn(n)
                F = eigen(A)
                ΔFall = Composite{typeof(F)}(values = Δλ, vectors = ΔU)
                # NOTE: how can we test pulling back both Δλ and ΔU?
                # TODO: adopt a deterministic sign convention to stabilize the FD estimate
                @testset for p in (:values, :vectors)
                    F_ad, back = rrule(eigen, A)
                    @test F_ad == F
                    ΔFp = getproperty(ΔFall, p)
                    ΔF = Composite{typeof(F)}(; p => ΔFp)
                    ∂self, ∂A = back(ΔF)
                    @test ∂self === NO_FIELDS
                    ∂A = unthunk(∂A)
                    @test ∂A isa typeof(A)
                    @test ∂A.uplo == A.uplo
                    @test ∂A.data ≈ only(j′vp(_fdm, A -> getproperty(eigen(T(A, uplo)), p), ΔFp, A.data))
                end
            end
        end
    end

    @testset "eigvals(::$T{<:Real})" for T in (Symmetric, Hermitian)
        n = 5
        @testset "frule" begin
            @testset for uplo in (:L, :U)
                A, ΔA = T(randn(n, n), uplo), T(randn(n, n), uplo)
                λ = eigvals(A)
                λ_ad, ∂λ_ad = frule((Zero(), ΔA), eigvals, A)
                @test λ_ad ≈ λ # inexact because frule uses eigen not eigvals
                ∂λ_ad = unthunk(∂λ_ad)
                @test ∂λ_ad isa typeof(λ)
                @test ∂λ_ad ≈ jvp(_fdm, A -> eigvals(T(A, uplo)), (A.data, ΔA.data))
            end
        end

        @testset "rrule" begin
            @testset for uplo in (:L, :U)
                A, Δλ = T(randn(n, n), uplo), randn(n)
                λ = eigvals(A)
                λ_ad, back = rrule(eigvals, A)
                @test λ_ad ≈ λ # inexact because rrule uses eigen not eigvals
                ∂self, ∂A = back(Δλ)
                @test ∂self === NO_FIELDS
                ∂A = unthunk(∂A)
                @test ∂A isa typeof(A)
                @test ∂A.uplo == A.uplo
                @test ∂A.data ≈ only(j′vp(_fdm, A -> eigvals(T(A, uplo)), Δλ, A.data))
            end
        end
    end

    # TODO: add tests for
    #   - atanh
    #   - degenerate matrices
    #   - singular matrices
    @testset "Symmetric/Hermitian power series functions" begin
        @testset "^(::$T{<:Real}, $p::Integer)" for T in (Symmetric, Hermitian), p in -3:3
            n = 5
            @testset "frule" begin
                @testset for uplo in (:L, :U)
                    A, ΔA = T(randn(n, n), uplo), T(randn(n, n), uplo)
                    Y = A^p
                    Y_ad, ∂Y_ad = frule((Zero(), ΔA, Zero()), ^, A, p)
                    @test Y_ad ≈ Y # inexact because frule uses eigen not Base.power_by_squaring
                    ∂Y_ad = unthunk(∂Y_ad)
                    @test ∂Y_ad isa typeof(Y)
                    @test ∂Y_ad.uplo == Y.uplo
                    # lower tolerance because ∂A=0 for p=0 and numbers are large for p=±3
                    @test ∂Y_ad.data ≈ jvp(_fdm, x -> (T(x, uplo)^p).data, (A.data, ΔA.data)) rtol=1e-8 atol=1e-10
                end
            end

            @testset "rrule" begin
                @testset for uplo in (:L, :U)
                    A, ΔY = T(randn(n, n), uplo), T(randn(n, n), uplo)
                    Y = A^p
                    Y_ad, back = rrule(^, A, p)
                    @test Y_ad ≈ Y # inexact because rrule uses eigen not Base.power_by_squaring
                    ∂self, ∂A, ∂p = back(ΔY)
                    @test ∂self === NO_FIELDS
                    @test ∂p === DoesNotExist()
                    ∂A = unthunk(∂A)
                    @test ∂A isa typeof(A)
                    @test ∂A.uplo == A.uplo
                    # lower tolerance because ∂A=0 for p=0 and numbers are large for p=±3
                    @test ∂A.data ≈ only(j′vp(_fdm, A -> (T(A, uplo)^p).data, ΔY, A.data)) rtol=1e-8 atol=1e-10
                end
            end
        end

        @testset "$(f)(::$T{<:Real})" for f in (
                exp, cos, sin, tan, cosh, sinh, tanh, atan, asinh
            ), T in (Symmetric, Hermitian)
            n = 5
            @testset "frule" begin
                @testset for uplo in (:L, :U)
                    A, ΔA = T(randn(n, n), uplo), T(randn(n, n), uplo)
                    Y = f(A)
                    Y_ad, ∂Y_ad = frule((Zero(), ΔA), f, A)
                    @test Y_ad == Y
                    ∂Y_ad = unthunk(∂Y_ad)
                    @test ∂Y_ad isa typeof(Y)
                    @test ∂Y_ad.uplo == Y.uplo
                    @test ∂Y_ad.data ≈ jvp(_fdm, x -> f(T(x, uplo)).data, (A.data, ΔA.data))
                end
            end

            @testset "rrule" begin
                @testset for uplo in (:L, :U)
                    A, ΔY = T(randn(n, n), uplo), T(randn(n, n), uplo)
                    Y = f(A)
                    Y_ad, back = rrule(f, A)
                    @test Y_ad == Y
                    ∂self, ∂A = back(ΔY)
                    @test ∂self === NO_FIELDS
                    ∂A = unthunk(∂A)
                    @test ∂A isa typeof(A)
                    @test ∂A.uplo == A.uplo
                    @test ∂A.data ≈ only(j′vp(_fdm, A -> f(T(A, uplo)).data, ΔY, A.data))
                end
            end
        end
    end
end
