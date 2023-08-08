
@testset "sparse(I, J, V, m, n, +)" begin
    m, n = 3, 5
    s, t, w = [1,2], [2,3], [0.5,0.5]
    
    test_rrule(sparse, s, t, w, m, n, +)
end

@testset "SparseMatrixCSC(A)" begin
    A = rand(5, 3)
    test_rrule(SparseMatrixCSC, A)
    test_rrule(SparseMatrixCSC{Float32,Int}, A, rtol=1e-4)
end

@testset "SparseVector(v)" begin
    v = rand(5)
    test_rrule(SparseVector, v)
    test_rrule(SparseVector{Float32}, Float32.(v), rtol=1e-4)
end

@testset "spdiagm" begin
    @test 1 == 1
    m = 5
    n = 4
    v1 = ones(m)
    v2 = ones(n)
    test_rrule(spdiagm, m, n, 0 => v2)

    # test_rrule(spdiagm, 0 => v1)
    # test_rrule(spdiagm, v1)
end

@testset "findnz" begin
    A = sprand(5, 5, 0.5)
    dA = similar(A)
    rand!(dA.nzval)
    I, J, V = findnz(A)
    V̄ = rand!(similar(V))
    test_rrule(findnz, A ⊢ dA, output_tangent=(zeros(length(I)), zeros(length(J)), V̄))

    v = sprand(5, 0.5)
    dv = similar(v)
    rand!(dv.nzval)
    I, V = findnz(v)
    V̄ = rand!(similar(V))
    test_rrule(findnz, v ⊢ dv, output_tangent=(zeros(length(I)), V̄))
end
