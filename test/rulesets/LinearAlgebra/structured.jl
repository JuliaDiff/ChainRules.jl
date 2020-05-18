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
    end
    @testset "Symmetric" begin
        N = 3
        rrule_test(Symmetric, randn(N, N), (randn(N, N), randn(N, N)))
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
end
