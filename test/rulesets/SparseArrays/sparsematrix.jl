using SparseArrays

@testset "Sparse" begin
    r = sparse(rand(3,3))
    x, x̄ = rand(3,3), rand(3,3)
    rrule_test(sparse, r, (x, x̄))
end
