using SparseArrays

@testset "Sparse" begin
  x = rand(3,3)
  rrule_test(sparse, x)
end
