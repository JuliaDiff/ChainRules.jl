using ChainRules,
    ChainRulesCore,
    ChainRulesTestUtils,
    FiniteDifferences,
    LinearAlgebra,
    Random,
    SparseArrays,
    Test
using ChainRulesTestUtils: _fdm

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
        a = randn(M)
        b = randn(M)
        c = randn(M - 1)
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
        a = randn(M)
        b = randn(M)
        c = randn(M - 1)
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
    _, pb = rrule(findnz, A)
    @test pb(Tangent{typeof((I, J, V))}(@thunk(error()), @thunk(error()), @thunk(V̄)))[2] == sparse(I, J, V̄, 5, 5)

    v = sprand(5, 0.5)
    dv = similar(v)
    rand!(dv.nzval)
    I, V = findnz(v)
    V̄ = rand!(similar(V))
    test_rrule(findnz, v ⊢ dv, output_tangent=(zeros(length(I)), V̄))
    _, pb = rrule(findnz, v)
    @test pb(Tangent{typeof((I, V))}(@thunk(error()), @thunk(V̄)))[2] == sparsevec(I, V̄, 5)
end

if Base.USE_GPL_LIBS # these rrules don't work without CHOLMOD from SuiteSparse.jl
    @testset "[log[abs[det]]] SparseMatrixCSC" begin
        ii = [1:5; 2; 4]
        jj = [1:5; 4; 2]
        x = [ones(5); 0.1; 0.1]
        A = sparse(ii, jj, x)
        test_rrule(logabsdet, A)
        test_rrule(logdet, A)
        test_rrule(det, A)

        # tridiagonal: the inverse is dense, the cotangent is restricted to the pattern of `T`
        T = spdiagm(-1 => fill(-1.0, 4), 0 => fill(3.0, 5), 1 => fill(-1.0, 4))
        test_rrule(logabsdet, T)
        test_rrule(logdet, T)
        test_rrule(det, T)
    end

    @testset "sparse selected inverse" begin
        ext = Base.get_extension(ChainRules, :SparseArraysExt)
        function test_inv_on_pattern(x, F=first(ext._cholesky(x)))
            @test issuccess(F)
            Z = ext._inv_on_pattern(F, x)
            @test SparseArrays.getcolptr(Z) == SparseArrays.getcolptr(x)
            @test rowvals(Z) == rowvals(x)
            rows, cols, vals = findnz(Z)
            @test vals ≈ inv(Matrix(x))[CartesianIndex.(rows, cols)]
        end

        n = 200
        B = sprandn(n, n, 0.02)
        test_inv_on_pattern(B'B + n * I)

        C = sprandn(ComplexF64, n, n, 0.02)
        test_inv_on_pattern(sparse(Hermitian(C'C + n * I)))

        # explicitly stored zero in the strict upper triangle, ignored by CHOLMOD
        test_inv_on_pattern(sparse([1, 2, 3, 4, 1], [1, 2, 3, 4, 4], [2.0, 3, 4, 5, 0.0]))

        # fill-in `L[4, 3] = A[4, 3] - L[4, 1] * L[3, 1] - L[4, 2] * L[3, 2]` cancels exactly
        # but remains part of the pattern of `L`
        A = sparse([1, 3, 4, 2, 3, 4, 1, 2, 3, 1, 2, 4], [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
                   [1.0, 1, 1, 1, 1, -1, 1, 1, 3, 1, -1, 3])
        test_inv_on_pattern(A, cholesky(Hermitian(A, :L); perm=1:4))
    end

    @testset "[log[abs[det]]] of non-positive-definite SparseMatrixCSC" begin
        indefinite = sparse([1.0 2; 2 1])
        nonsymmetric = sparse([2.0 1; 0 2])
        for x in (indefinite, nonsymmetric)
            Ω, pb = rrule(logabsdet, x)
            @test Ω == logabsdet(x)
            @test_throws ArgumentError pb(Tangent{typeof(Ω)}(1.0, 0.0))
            for f in (logdet, det)
                x === indefinite && f === logdet && continue  # `logdet` of a negative det throws
                Ω, pb = rrule(f, x)
                @test Ω == f(x)
                @test_throws ArgumentError pb(1.0)
            end
        end
    end
end
