@testset "matrix functions" begin
    @testset "LinearAlgebra.exp!(A::Matrix) frule" begin
        n = 10
        @testset "A::Matrix{$T}, opnorm(A,1)=$nrm" for T in (Float64, ComplexF64), nrm in (0.01, 0.1, 0.5, 1.5, 3.0, 6.0, 12.0)
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
end
