@testset "matrix functions" begin
    @testset "LinearAlgebra.exp!(A::Matrix) frule" begin
        n = 10
        # each normalization hits a specific branch
        @testset "A::Matrix{$T}, opnorm(A,1)=$nrm" for T in (Float64, ComplexF64),
            nrm in (0.01, 0.1, 0.5, 1.5, 3.0, 6.0, 12.0)

            A = randn(T, n, n)
            ΔA = randn(T, n, n)
            A *= nrm / opnorm(A, 1)
            tols = nrm == 0.1 ? (atol=1e-8, rtol=1e-8) : NamedTuple()
            frule_test(LinearAlgebra.exp!, (A, ΔA); tols...)
        end
        @testset "imbalanced A" begin
            A = Float64[0 10 0 0; -1 0 0 0; 0 0 0 0; -2 0 0 0]
            ΔA = rand_tangent(A)
            frule_test(LinearAlgebra.exp!, (A, ΔA))
        end
        @testset "hermitian A, T=$T" for T in (Float64, ComplexF64)
            A = Matrix(Hermitian(randn(T, n, n)))
            ΔA = randn(T, n, n)
            frule_test(LinearAlgebra.exp!, (A, Matrix(Hermitian(ΔA))))
            frule_test(LinearAlgebra.exp!, (A, ΔA))
        end
    end

    @testset "exp(A::Matrix) rrule" begin
        n = 10
        # each normalization hits a specific branch
        @testset "A::Matrix{$T}, opnorm(A,1)=$nrm" for T in (Float64, ComplexF64),
            nrm in (0.01, 0.1, 0.5, 1.5, 3.0, 6.0, 12.0)

            A = randn(T, n, n)
            ΔA = randn(T, n, n)
            ΔY = randn(T, n, n)
            A *= nrm / opnorm(A, 1)
            # rrule is not inferrable, but pullback should be
            tols = nrm == 0.1 ? (atol=1e-8, rtol=1e-8) : NamedTuple()
            rrule_test(exp, ΔY, (A, ΔA); check_inferred=false, tols...)
            Y, back = rrule(exp, A)
            @inferred back(ΔY)
        end
        @testset "imbalanced A" begin
            A = Float64[0 10 0 0; -1 0 0 0; 0 0 0 0; -2 0 0 0]
            ΔA = rand_tangent(A)
            ΔY = rand_tangent(exp(A))
            rrule_test(exp, ΔY, (A, ΔA); check_inferred=false)
        end
        @testset "hermitian A, T=$T" for T in (Float64, ComplexF64)
            A = Matrix(Hermitian(randn(T, n, n)))
            ΔA = randn(T, n, n)
            ΔY = randn(T, n, n)
            rrule_test(exp, Matrix(Hermitian(ΔY)), (A, ΔA); check_inferred=false)
            rrule_test(exp, ΔY, (A, ΔA); check_inferred=false)
        end
    end
end
