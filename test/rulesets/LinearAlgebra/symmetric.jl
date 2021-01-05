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
