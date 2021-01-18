@testset "matrix functions" begin
    @testset "LinearAlgebra.exp!(A::Matrix) frule" begin
        n = 10
        @testset "A::Matrix{$T}, opnorm(A,1)=$nrm" for T in (Float64, ComplexF64),
nrm in (0.01, 0.1, 0.5, 1.5, 3.0, 6.0, 12.0)

            A, ΔA = randn(ComplexF64, n, n), randn(ComplexF64, n, n)
            # choose normalization to hit specific branch
            A *= nrm / opnorm(A, 1)
            frule_test(LinearAlgebra.exp!, (A, ΔA))
        end
        @testset "hermitian A" begin
            A, ΔA = Matrix(Hermitian(randn(ComplexF64, n, n))), randn(ComplexF64, n, n)
            frule_test(LinearAlgebra.exp!, (A, Matrix(Hermitian(ΔA))))
            frule_test(LinearAlgebra.exp!, (A, ΔA))
        end
    end

    @testset "exp(A::Matrix) rrule" begin
        n = 10
        @testset "A::Matrix{$T}, opnorm(A,1)=$nrm" for T in (Float64, ComplexF64),
nrm in (0.01, 0.1, 0.5, 1.5, 3.0, 6.0, 12.0)

            A, ΔA = randn(ComplexF64, n, n), randn(ComplexF64, n, n)
            ΔY = randn(ComplexF64, n, n)
            # choose normalization to hit specific branch
            A *= nrm / opnorm(A, 1)
            # rrule is not inferrable, but pullback should be
            rrule_test(exp, ΔY, (A, ΔA); check_inferred=false)
            Y, back = rrule(exp, A)
            @inferred back(ΔY)
        end
        @testset "hermitian A" begin
            A, ΔA = Matrix(Hermitian(randn(ComplexF64, n, n))), randn(ComplexF64, n, n)
            ΔY = randn(ComplexF64, n, n)
            rrule_test(exp, Matrix(Hermitian(ΔY)), (A, ΔA); check_inferred=false)
            rrule_test(exp, ΔY, (A, ΔA); check_inferred=false)
        end
    end
end
