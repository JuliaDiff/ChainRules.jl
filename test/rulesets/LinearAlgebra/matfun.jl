@testset "matrix functions" begin
    @testset "LinearAlgebra.exp!(A::Matrix) frule" begin
        n = 10
        @testset "A::Matrix{$T}, opnorm(A,1)=$nrm" for T in (Float64, ComplexF64),
            # choose normalization to hit specific branch
            nrm in (0.01, 0.1, 0.5, 1.5, 3.0, 6.0, 12.0)

            A, ΔA = randn(ComplexF64, n, n), randn(ComplexF64, n, n)
            A *= nrm / opnorm(A, 1)
            frule_test(LinearAlgebra.exp!, (A, ΔA))
        end
        @testset "imbalanced A" begin
            A = Float64[0 10 0 0; -1 0 0 0; 0 0 0 0; -2 0 0 0]
            ΔA = rand_tangent(A)
            frule_test(LinearAlgebra.exp!, (A, ΔA))
        end
        @testset "hermitian A" begin
            A = Matrix(Hermitian(randn(ComplexF64, n, n)))
            ΔA = randn(ComplexF64, n, n)
            frule_test(LinearAlgebra.exp!, (A, Matrix(Hermitian(ΔA))))
            frule_test(LinearAlgebra.exp!, (A, ΔA))
        end
    end

    @testset "exp(A::Matrix) rrule" begin
        n = 10
        @testset "A::Matrix{$T}, opnorm(A,1)=$nrm" for T in (Float64, ComplexF64),
            # choose normalization to hit specific branch
            nrm in (0.01, 0.1, 0.5, 1.5, 3.0, 6.0, 12.0)

            A, ΔA = randn(ComplexF64, n, n), randn(ComplexF64, n, n)
            ΔY = randn(ComplexF64, n, n)
            A *= nrm / opnorm(A, 1)
            # rrule is not inferrable, but pullback should be
            rrule_test(exp, ΔY, (A, ΔA); check_inferred=false)
            Y, back = rrule(exp, A)
            @inferred back(ΔY)
        end
        @testset "imbalanced A" begin
            A = Float64[0 10 0 0; -1 0 0 0; 0 0 0 0; -2 0 0 0]
            ΔA = rand_tangent(A)
            ΔY = rand_tangent(exp(A))
            rrule_test(exp, ΔY, (A, ΔA); check_inferred=false)
        end
        @testset "hermitian A" begin
            A, ΔA = Matrix(Hermitian(randn(ComplexF64, n, n))), randn(ComplexF64, n, n)
            ΔY = randn(ComplexF64, n, n)
            rrule_test(exp, Matrix(Hermitian(ΔY)), (A, ΔA); check_inferred=false)
            rrule_test(exp, ΔY, (A, ΔA); check_inferred=false)
        end
    end
end
