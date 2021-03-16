using SparseArrays

@testset "Sparse" begin
    r = sparse(rand(3,3))
    x, x̄ = rand(3,3), rand(3,3)
    test_rrule(SparseMatrixCSC, r)
    test_rrule(Matrix, r)
end
