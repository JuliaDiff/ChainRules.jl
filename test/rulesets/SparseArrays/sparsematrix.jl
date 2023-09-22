
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

# copied over from test/rulesets/LinearAlgebra/structured
@testset "spdiagm" begin
    @testset "without size" begin
        M, N = 7, 9
        s = (8, 8)
        a, ā = randn(M), randn(M)
        b, b̄ = randn(M), randn(M)
        c, c̄ = randn(M - 1), randn(M - 1)
        ȳ = randn(s)
        ps = (0 => a, 1 => b, 0 => c)
        y, back = rrule(spdiagm, ps...)
        @test y == spdiagm(ps...)
        ∂self, ∂pa, ∂pb, ∂pc = back(ȳ)
        @test ∂self === NoTangent()
        ∂a_fd, ∂b_fd, ∂c_fd = j′vp(_fdm, (a, b, c) -> spdiagm(0 => a, 1 => b, 0 => c), ȳ, a, b, c)
        for (p, ∂px, ∂x_fd) in zip(ps, (∂pa, ∂pb, ∂pc), (∂a_fd, ∂b_fd, ∂c_fd))
            ∂px = unthunk(∂px)
            @test ∂px isa Tangent{typeof(p)}
            @test ∂px.first isa AbstractZero
            @test ∂px.second ≈ ∂x_fd
        end
    end
    @testset "with size" begin
        M, N = 7, 9
        a, ā = randn(M), randn(M)
        b, b̄ = randn(M), randn(M)
        c, c̄ = randn(M - 1), randn(M - 1)
        ȳ = randn(M, N)
        ps = (0 => a, 1 => b, 0 => c)
        y, back = rrule(spdiagm, M, N, ps...)
        @test y == spdiagm(M, N, ps...)
        ∂self, ∂M, ∂N, ∂pa, ∂pb, ∂pc = back(ȳ)
        @test ∂self === NoTangent()
        @test ∂M === NoTangent()
        @test ∂N === NoTangent()
        ∂a_fd, ∂b_fd, ∂c_fd = j′vp(_fdm, (a, b, c) -> spdiagm(M, N, 0 => a, 1 => b, 0 => c), ȳ, a, b, c)
        for (p, ∂px, ∂x_fd) in zip(ps, (∂pa, ∂pb, ∂pc), (∂a_fd, ∂b_fd, ∂c_fd))
            ∂px = unthunk(∂px)
            @test ∂px isa Tangent{typeof(p)}
            @test ∂px.first isa AbstractZero
            @test ∂px.second ≈ ∂x_fd
        end
    end
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

@testset "[log[abs[det]]] SparseMatrixCSC" begin
    ii = [1:5; 2; 4]
    jj = [1:5; 4; 2]
    x = [ones(5); 0.1; 0.1]
    A = sparse(ii, jj, x)
    test_rrule(logabsdet, A)
    test_rrule(logdet, A)
    test_rrule(det, A)
end
