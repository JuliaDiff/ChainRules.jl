@testset "Symmetric/Hermitian rules" begin
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
