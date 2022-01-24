
@testset "sparse(I, J, V, m, n, +)" begin
    m, n = 3, 5
    s, t, w = [1,2], [2,3], [0.5,0.5]
    
    test_rrule(sparse, s, t, w, m, n, +)
end

@testset "SparseMatrixCSC(A)" begin
    A = rand(5, 3)
    test_rrule(SparseMatrixCSC, A)
    test_rrule(SparseMatrixCSC{Float32,Int}, A, rtol=1e-5)
end

@testset "SparseVector(v)" begin
    v = rand(5)
    test_rrule(SparseVector, v)
    test_rrule(SparseVector{Float32}, Float32.(v), rtol=1e-5)
end
