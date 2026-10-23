using ChainRules,
    ChainRulesCore, ChainRulesTestUtils, FiniteDifferences, LinearAlgebra, Random, Test
using ChainRulesTestUtils: _fdm

# TODO: move this to FiniteDifferences
function FiniteDifferences.to_vec(X::LU)
    x_vec, back = to_vec(Matrix(X.factors))
    function LU_from_vec(x_vec)
        return LU(back(x_vec), X.ipiv, X.info)
    end
    return x_vec, LU_from_vec
end

function FiniteDifferences.to_vec(C::Cholesky)
    C_vec, factors_from_vec = to_vec(C.factors)
    function cholesky_from_vec(v)
        return Cholesky(factors_from_vec(v), C.uplo, C.info)
    end
    return C_vec, cholesky_from_vec
end

# well-conditioned random n×n matrix with elements of type `T` for testing `eigen`
function rand_eigen(T::Type, n::Int)
    # uniform distribution over `(-1, 1)` / `(-1, 1)^2`
    _rand(T, n...) = rand(T, n...) .* rand(T <: Complex ? [1, im, -1, -im] : [1, -1], n...)

    # make sure that each eigenvector has one clearly defined entry with maximum magnitude
    # so that the complex phase of EVs is well defined
    V = _rand(T, n, n)
    for (col, i) in zip(eachcol(V), shuffle(1:n))
        col[i] += 3 * (T <: Complex ? cispi(2rand()) : rand([1, -1]))
        normalize!(col)
    end

    # make sure the sorting of eigenvalues is well defined
    λ = 10(_rand(T, n) .+ (0:3:3(n-1)))

    return V * Diagonal(λ) / V
end

# m×n matrix whose LU factorization has an exactly zero last pivot: since |L[i, j]| < 1,
# partial pivoting keeps the row order, and dyadic entries and power-of-two pivots make
# the factorization exact
function lu_singular_last_pivot(m, n)
    q = min(m, n)
    L = tril(rand(-3:3, m, q) / 4, -1)
    L[diagind(L)] .= 1
    U = triu(rand(-3:3, q, n))
    U[diagind(U)] = rand((-4, -2, -1, 1, 2, 4), q)
    U[q, q] = 0
    return L * U
end

@testset "Factorizations" begin
    @testset "lu decomposition" begin
        n = 10
        @testset "lu! frule" begin
            @testset "lu!(A::Matrix{$T}, $pivot) for size(A)=($m, $n)" for
                T in (Float64, ComplexF64),
                pivot in (RowMaximum(), NoPivot()),
                m in (7, 10, 13)

                test_frule(lu!, randn(T, m, n), pivot ⊢ NoTangent())
            end
            @testset "zero last pivot: lu!(A, $pivot) for size(A)=($m, $n)" for pivot in (
                    RowMaximum(), NoPivot()
                ),
                m in (7, 10, 13)

                A = lu_singular_last_pivot(m, n)
                @test iszero(last(diag(lu(A, pivot; check=false).U)))
                if m <= n
                    test_frule(lu!, A, pivot ⊢ NoTangent(); fkwargs=(; check=false))
                else  # not differentiable
                    ΔA = rand_tangent(A)
                    _, ∂F = frule((ZeroTangent(), ΔA), lu!, A, pivot; check=false)
                    @test all(isnan, ∂F.factors)
                end
            end
            @testset "check=false passed to primal function" begin
                Asingular = zeros(n, n)
                ΔAsingular = rand_tangent(Asingular)
                @test_throws SingularException frule(
                    (ZeroTangent(), copy(ΔAsingular)), lu!, copy(Asingular), RowMaximum()
                )
                _, ∂F = frule(
                    (ZeroTangent(), ΔAsingular), lu!, Asingular, RowMaximum(); check=false
                )
                @test all(isnan, ∂F.factors)  # not differentiable
            end
        end
        @testset "lu rrule" begin
            @testset "lu(A::Matrix{$T}, $pivot) for size(A)=($m, $n)" for
                T in (Float64, ComplexF64),
                pivot in (RowMaximum(), NoPivot()),
                m in (7, 10, 13)

                test_rrule(lu, randn(T, m, n), pivot ⊢ NoTangent())
            end
            @testset "zero last pivot: lu(A, $pivot) for size(A)=($m, $n)" for pivot in (
                    RowMaximum(), NoPivot()
                ),
                m in (7, 10, 13)

                A = lu_singular_last_pivot(m, n)
                @test iszero(last(diag(lu(A, pivot; check=false).U)))
                if m <= n
                    test_rrule(lu, A, pivot ⊢ NoTangent(); fkwargs=(; check=false))
                else  # not differentiable
                    F, back = rrule(lu, A, pivot; check=false)
                    _, ∂A, _ = back(Tangent{typeof(F)}(; factors=rand_tangent(F.factors)))
                    @test all(isnan, ∂A)
                end
            end
            @testset "check=false passed to primal function" begin
                Asingular = zeros(n, n)
                F = lu(Asingular, RowMaximum(); check=false)
                ΔF = Tangent{typeof(F)}(; factors=rand_tangent(F.factors))
                @test_throws SingularException rrule(lu, Asingular, RowMaximum())
                _, back = rrule(lu, Asingular, RowMaximum(); check=false)
                _, ∂A, _ = back(ΔF)
                @test all(isnan, ∂A)  # not differentiable
            end
        end
        @testset "LU" begin
            @testset "getproperty(::LU, k) rrule" begin
                # test that the getproperty rrule composes correctly with the lu rrule
                @testset "getproperty(lu(A::Matrix), :$k) for size(A)=($m, $n)" for
                    k in (:U, :L, :factors),
                    m in (7, 10, 13)

                    F = lu(randn(m, n))
                    test_rrule(getproperty, F, k; check_inferred=false)
                end
            end
            @testset "matrix inverse using LU" begin
                @testset "inv!(lu(::LU{$T,<:StridedMatrix}))" for T in (Float64,ComplexF64)
                    test_frule(LinearAlgebra.inv!, lu(randn(T, n, n), RowMaximum()))
                    test_rrule(inv, lu(randn(T, n, n), RowMaximum()))
                end
            end
        end
    end

    @testset "singular value decomposition" begin
        @testset "svd" begin
            for n in [4, 6, 10], m in [3, 5, 9]
                @testset "($n x $m) svd" begin
                    X = randn(n, m)
                    test_rrule(svd, X; atol=1e-6, rtol=1e-6)
                end
            end

            for n in [4, 6, 10], m in [3, 5, 10]
                @testset "($n x $m) getproperty" begin
                    X = randn(n, m)
                    F = svd(X)
                    rand_adj = adjoint(rand(reverse(size(F.V))...))

                    test_rrule(getproperty, F, :U; check_inferred=false)
                    test_rrule(getproperty, F, :S; check_inferred=false)
                    test_rrule(getproperty, F, :Vt; check_inferred=false)
                    test_rrule(
                        getproperty, F, :V; check_inferred=false, output_tangent=rand_adj
                    )
                end
            end

            @testset "Thunked inputs" begin
                X = randn(4, 3)
                F, dX_pullback = rrule(svd, X)
                for p in [:U, :S, :V, :Vt]
                    Y, dF_pullback = rrule(getproperty, F, p)
                    Ȳ = randn(size(Y)...)

                    _, dF_unthunked, _ = dF_pullback(Ȳ)

                    # helper to let us check how things are stored.
                    p_access = p == :V ? :Vt : p
                    backing_field(c, p) = getproperty(ChainRulesCore.backing(c), p_access)
                    @assert !(backing_field(dF_unthunked, p) isa AbstractThunk)

                    dF_thunked = map(f -> Thunk(() -> f), dF_unthunked)
                    @assert backing_field(dF_thunked, p) isa AbstractThunk

                    dself_thunked, dX_thunked = dX_pullback(dF_thunked)
                    dself_unthunked, dX_unthunked = dX_pullback(dF_unthunked)
                    @test dself_thunked == dself_unthunked
                    @test dX_thunked == dX_unthunked
                end
            end

            @testset "Helper functions" begin
                X = randn(10, 10)
                Y = randn(10, 10)
                @test ChainRules._mulsubtrans!!(copy(X), Y) ≈ Y .* (X - X')
                @test ChainRules._eyesubx!(copy(X)) ≈ I - X

                Z = randn(Float32, 10, 10)
                result = ChainRules._mulsubtrans!!(copy(Z), Y)
                @test result ≈ Y .* (Z - Z')
                @test eltype(result) == Float64
            end
        end

        @testset "svdvals" begin
            for n in [4, 6, 10]
                for m in [3, 5, 9]
                    @testset "($n x $m) svdvals" begin
                        X = randn(n, m)
                        test_rrule(svdvals, X; atol=1e-6, rtol=1e-6)
                    end
                end

                @testset "rrule for svdvals(::$SymHerm{$T}) ($n x $n, uplo=$uplo)" for SymHerm in
                                                                                       (
                        Symmetric, Hermitian
                    ),
                    T in (SymHerm === Symmetric ? (Float64,) : (Float64, ComplexF64)),
                    uplo in (:L, :U)

                    A, ΔS = randn(T, n, n), randn(n)
                    symA = SymHerm(A, uplo)

                    S = svdvals(symA)
                    S_ad, back = @inferred rrule(svdvals, symA)
                    @test S_ad ≈ S # inexact because rrule uses svd not svdvals
                    ∂self, ∂symA = @inferred back(ΔS)
                    @test ∂self === NoTangent()
                    @test ∂symA isa typeof(symA)
                    @test ∂symA.uplo == symA.uplo

                    # pull the cotangent back to A to test against finite differences
                    ∂A = rrule(SymHerm, A, uplo)[2](∂symA)[2]
                    @test ∂A ≈ j′vp(_fdm, A -> svdvals(SymHerm(A, uplo)), ΔS, A)[1]

                    @test @inferred(back(ZeroTangent())) == (NoTangent(), ZeroTangent())
                end
            end
        end
    end

    @testset "eigendecomposition" begin
        @testset "eigen/eigen!" begin
            # NOTE: eigen!/eigen are not type-stable, so neither are their frule/rrule

            # avoid implementing to_vec(::Eigen)
            asnt(E::Eigen) = (values=E.values, vectors=E.vectors)

            # NOTE: for unstructured matrices, low enough n, and this specific seed, finite
            # differences of eigen seems to be stable enough for direct comparison.
            # This allows us to directly check differential of normalization/phase
            # convention
            n = 10

            @testset "eigen!(::Matrix{$T}) frule" for T in (Float64,ComplexF64)
                X = rand_eigen(T, n)

                Ẋ = rand_tangent(X)
                F = eigen!(copy(X))
                F_fwd, Ḟ_ad = frule((ZeroTangent(), copy(Ẋ)), eigen!, copy(X))
                @test F_fwd == F
                @test Ḟ_ad isa Tangent{typeof(F)}
                Ḟ_fd = jvp(_fdm, asnt ∘ eigen! ∘ copy, (X, Ẋ))
                @test Ḟ_ad.values ≈ Ḟ_fd.values
                @test Ḟ_ad.vectors ≈ Ḟ_fd.vectors
                @test frule((ZeroTangent(), ZeroTangent()), eigen!, copy(X)) == (F, ZeroTangent())

                @testset "tangent types match outputs" begin
                    # hermitian matrices have real eigenvalues (of complex type for complex
                    # matrices on Julia >= 1.14) and, when real, real eigenvectors
                    X = Matrix(Hermitian(randn(T, n, n)))
                    F, Ḟ = frule((ZeroTangent(), rand_tangent(X)), eigen!, X)
                    @test eltype(Ḟ.values) == eltype(F.values)
                    @test eltype(Ḟ.vectors) == eltype(F.vectors)
                end
            end

            @testset "eigen(::Matrix{$T}) rrule" for T in (Float64, ComplexF64)
                X = rand_eigen(T, n)

                F = eigen(X)
                V̄ = rand_tangent(F.vectors)
                λ̄ = rand_tangent(F.values)
                CT = Tangent{typeof(F)}
                F_rev, back = rrule(eigen, X)
                @test F_rev == F
                # NOTE: eigen is not type-stable, so neither are is its rrule
                _, X̄_values_ad = @maybe_inferred back(CT(values = λ̄))
                @test X̄_values_ad ≈ j′vp(_fdm, x -> eigen(x).values, λ̄, X)[1]
                _, X̄_vectors_ad = @maybe_inferred back(CT(vectors = V̄))
                # need the conversion to `complex` here, since FiniteDiff is currently buggy if functions
                # return arrays of either real or complex values based solely on the input values (not the
                # input types). See https://github.com/JuliaLang/julia/issues/41243
                @test X̄_vectors_ad ≈ j′vp(_fdm, x -> complex.(eigen(x).vectors), complex.(V̄), X)[1] rtol=1e-6
                F̄ = CT(values = λ̄, vectors = V̄)
                s̄elf, X̄_ad = @maybe_inferred back(F̄)
                @test s̄elf === NoTangent()
                X̄_fd = j′vp(_fdm, asnt ∘ eigen, F̄, X)[1]
                @test X̄_ad ≈ X̄_fd rtol=1e-4
                @test @maybe_inferred(back(ZeroTangent())) === (NoTangent(), ZeroTangent())
                F̄zero = CT(values = ZeroTangent(), vectors = ZeroTangent())
                @test @maybe_inferred(back(F̄zero)) === (NoTangent(), ZeroTangent())

                T <: Real && @testset "cotangent is real when input is" begin
                    X = randn(T, n, n)
                    Ẋ = rand_tangent(X)

                    F = eigen(X)
                    V̄ = rand_tangent(F.vectors)
                    λ̄ = rand_tangent(F.values)
                    F̄ = Tangent{typeof(F)}(values = λ̄, vectors = V̄)
                    X̄ = rrule(eigen, X)[2](F̄)[2]
                    @test eltype(X̄) <: Real
                end
            end

            @testset "normalization/phase functions are idempotent" for T in (Float64,ComplexF64)
                # this is as much a math check as a code check. because normalization when
                # applied repeatedly is idempotent, repeated pushforward/pullback should
                # leave the (co)tangent unchanged
                X = randn(T, n, n)
                Ẋ = rand_tangent(X)
                F = eigen(X)

                V̇ = rand_tangent(F.vectors)
                V̇proj = ChainRules._eigen_norm_phase_fwd!(copy(V̇), X, F.vectors)
                @test !isapprox(V̇, V̇proj)
                V̇proj2 = ChainRules._eigen_norm_phase_fwd!(copy(V̇proj), X, F.vectors)
                @test V̇proj2 ≈ V̇proj

                V̄ = rand_tangent(F.vectors)
                V̄proj = ChainRules._eigen_norm_phase_rev!(copy(V̄), X, F.vectors)
                @test !isapprox(V̄, V̄proj)
                V̄proj2 = ChainRules._eigen_norm_phase_rev!(copy(V̄proj), X, F.vectors)
                @test V̄proj2 ≈ V̄proj
            end

            # below tests adapted from /test/rulesets/LinearAlgebra/symmetric.jl
            @testset "hermitian matrices" begin
                function _eigvecs_stabilize_mat(vectors)
                    Ui = @view(vectors[end, :])
                    return Diagonal(conj.(sign.(Ui)))
                end

                function _eigen_stable(A)
                    F = eigen(A)
                    rmul!(F.vectors, _eigvecs_stabilize_mat(F.vectors))
                    return F
                end

                # eigenvectors are only differentiable along Hermitian directions
                _eigvecs_stable(x) = _eigen_stable(Matrix(hermitianpart(x))).vectors

                n = 10
                @testset "eigen!(::Matrix{$T})" for T in (Float64, ComplexF64)
                    A, ΔA = Matrix(Hermitian(randn(T, n, n))), randn(T, n, n)

                    F = eigen!(copy(A))
                    @test frule((ZeroTangent(), ZeroTangent()), eigen!, copy(A)) == (F, ZeroTangent())
                    F_ad, ∂F_ad = frule((ZeroTangent(), copy(ΔA)), eigen!, copy(A))
                    @test F_ad == F
                    @test ∂F_ad isa Tangent{typeof(F)}
                    @test ∂F_ad.values isa typeof(F.values)
                    @test ∂F_ad.vectors isa typeof(F.vectors)

                    # eigenvalues are differentiable in all directions their type admits
                    f_values = eltype(F.values) == T ? eigvals : x -> eigvals(Matrix(hermitianpart(x)))
                    @test ∂F_ad.values ≈ jvp(_fdm, f_values, (A, ΔA))

                    C = _eigvecs_stabilize_mat(F.vectors)
                    @test ∂F_ad.vectors * C ≈ jvp(_fdm, _eigvecs_stable, (A, ΔA))
                end

                @testset "eigen(::Matrix{$T})" for T in (Float64, ComplexF64)
                    A = Matrix(Hermitian(randn(T, n, n)))
                    F = eigen(A)
                    ΔU, Δλ = randn(T, n, n), randn(eltype(F.values), n)
                    ΔF = Tangent{typeof(F)}(; values=Δλ, vectors=ΔU)
                    F_ad, back = rrule(eigen, A)
                    @test F_ad == F

                    C = _eigvecs_stabilize_mat(F.vectors)
                    CT = Tangent{typeof(F)}

                    # eigenvalues are differentiable in all directions their type admits
                    f_values = eltype(F.values) == T ? eigvals : x -> eigvals(Matrix(hermitianpart(x)))

                    @testset for nzprops in ([:values], [:vectors], [:values, :vectors])
                        ∂F = CT(; [s => getproperty(ΔF, s) for s in nzprops]...)
                        ∂F_stable = (; [s => copy(getproperty(ΔF, s)) for s in nzprops]...)
                        :vectors in nzprops && rmul!(∂F_stable.vectors, C)

                        f_stable = function(x)
                            F_ = (values=f_values(x), vectors=_eigvecs_stable(x))
                            return (; (s => getproperty(F_, s) for s in nzprops)...)
                        end

                        ∂self, ∂A = @maybe_inferred back(∂F)
                        @test ∂self === NoTangent()
                        @test ∂A isa typeof(A)
                        ∂A_fd = j′vp(_fdm, f_stable, ∂F_stable, A)[1]
                        @test ∂A ≈ ∂A_fd
                        @test @maybe_inferred(back(@thunk(∂F))) == (∂self, ∂A)
                    end
                end
            end
        end

        @testset "eigvals/eigvals!" begin
            # NOTE: eigvals!/eigvals are not type-stable, so neither are their frule/rrule
            @testset "eigvals!(::Matrix{$T}) frule" for T in (Float64,ComplexF64)
                n = 10
                X = randn(T, n, n)
                test_frule(eigvals!, X)
                @test frule((ZeroTangent(), ZeroTangent()), eigvals!, copy(X))[2] == ZeroTangent()

                @testset "tangent types match outputs" begin
                    # hermitian matrices have real eigenvalues (of complex type for complex
                    # matrices on Julia >= 1.14)
                    X = Matrix(Hermitian(randn(T, n, n)))
                    λ, λ̇ = frule((ZeroTangent(), rand_tangent(X)), eigvals!, X)
                    @test eltype(λ̇) == eltype(λ)
                end
            end

            @testset "eigvals(::Matrix{$T}) rrule" for T in (Float64,ComplexF64)
                n = 10
                test_rrule(eigvals, randn(T, n, n))

                λ, back = rrule(eigvals, randn(T, n, n))
                _, X̄ = @maybe_inferred back(rand_tangent(λ))
                @test @maybe_inferred(back(ZeroTangent())) === (NoTangent(), ZeroTangent())

                T <: Real && @testset "cotangent is real when input is" begin
                    @test eltype(X̄) <: Real
                end
            end

            # below tests adapted from /test/rulesets/LinearAlgebra/symmetric.jl
            @testset "hermitian matrices" begin
                n = 10
                @testset "eigvals!(::Matrix{$T})" for T in (Float64, ComplexF64)
                    A, ΔA = Matrix(Hermitian(randn(T, n, n))), randn(T, n, n)
                    λ = eigvals!(copy(A))
                    λ_ad, ∂λ_ad = frule((ZeroTangent(), copy(ΔA)), eigvals!, copy(A))
                    @test λ_ad ≈ λ # inexact because frule uses eigen not eigvals
                    @test ∂λ_ad isa typeof(λ)
                    # eigenvalues are differentiable in all directions their type admits
                    f = eltype(λ) == T ? eigvals : x -> eigvals(Matrix(hermitianpart(x)))
                    @test ∂λ_ad ≈ jvp(_fdm, f, (A, ΔA))
                end

                @testset "eigvals(::Matrix{$T})" for T in (Float64, ComplexF64)
                    A = Matrix(Hermitian(randn(T, n, n)))
                    λ = eigvals(A)
                    Δλ = randn(eltype(λ), n)
                    λ_ad, back = rrule(eigvals, A)
                    @test λ_ad ≈ λ # inexact because rrule uses eigen not eigvals
                    ∂self, ∂A = @maybe_inferred back(Δλ)
                    @test ∂self === NoTangent()
                    @test ∂A isa typeof(A)
                    # eigenvalues are differentiable in all directions their type admits
                    f = eltype(λ) == T ? eigvals : x -> eigvals(Matrix(hermitianpart(x)))
                    @test ∂A ≈ j′vp(_fdm, f, Δλ, A)[1]
                    @test @maybe_inferred(back(ZeroTangent())) == (NoTangent(), ZeroTangent())

                    # gradient of `sum ∘ eigvals = tr` (FluxML/Zygote.jl#1369)
                    @test back(ones(eltype(λ), n))[2] ≈ I
                end
            end
        end
    end

    # These tests are generally a bit tricky to write because FiniteDifferences doesn't
    # have fantastic support for this stuff at the minute.
    # also we might be missing some overloads for different tangent-types in the rules
    @testset "cholesky" begin
        @testset "Number" begin
            @testset "uplo=$uplo" for uplo in (:U, :L)
                test_rrule(cholesky, 0.8, uplo)
                test_rrule(cholesky, -0.3, uplo)
                test_rrule(cholesky, 0.23 + 0im, uplo)
                test_rrule(cholesky, 0.78 + 0.5im, uplo)
                test_rrule(cholesky, -0.34 + 0.1im, uplo)
            end
        end

        @testset "Diagonal" begin
            @testset "Diagonal{<:Real}" begin
                test_rrule(cholesky, Diagonal([0.3, 0.2, 0.5, 0.6, 0.9]), NoPivot())
            end
            @testset "Diagonal{<:Complex}" begin
                # finite differences in general will produce matrices with non-real
                # diagonals, which cause factorization to fail. If we turn off the check and
                # ensure the cotangent is real, then test_rrule still works.
                D = Diagonal([0.3 + 0im, 0.2, 0.5, 0.6, 0.9])
                C = cholesky(D)
                test_rrule(
                    cholesky,
                    D,
                    NoPivot();
                    output_tangent=Tangent{typeof(C)}(factors=complex(randn(5, 5))),
                    fkwargs=(; check=false),
                )
            end
            @testset "check has correct default and passed to primal" begin
                @test_throws Exception rrule(cholesky, Diagonal(-rand(5)), NoPivot())
                rrule(cholesky, Diagonal(-rand(5)), NoPivot(); check=false)
            end
            @testset "failed factorization" begin
                A = Diagonal(vcat(rand(4), -rand(4), rand(4)))
                test_rrule(cholesky, A, NoPivot(); fkwargs=(; check=false))
            end
        end

        @testset "StridedMatrix" begin
            @testset "Matrix{$T}" for T in (Float64, ComplexF64)
                X = generate_well_conditioned_matrix(T, 10)
                V = generate_well_conditioned_matrix(T, 10)
                F, dX_pullback = rrule(cholesky, X, NoPivot())
                @testset "uplo=$p, cotangent eltype=$T" for p in [:U, :L], S in unique([T, complex(T)])
                    Y, dF_pullback = rrule(getproperty, F, p)
                    Ȳ = randn(S, size(Y))
                    (dself, dF, dp) = dF_pullback(Ȳ)
                    @test dself === NoTangent()
                    @test dp === NoTangent()

                    # NOTE: We're doing Nabla-style testing here and avoiding using the `j′vp`
                    # machinery from FiniteDifferences because that isn't set up to respect
                    # necessary special properties of the input. In the case of the Cholesky
                    # factorization, we need the input to be Hermitian.
                    ΔF = unthunk(dF)
                    _, dX, darg2 = dX_pullback(ΔF)
                    @test darg2 === NoTangent()
                    X̄_ad = real(dot(unthunk(dX), V))
                    X̄_fd = central_fdm(5, 1)(0.000_0001) do ε
                        real(dot(Ȳ, getproperty(cholesky(X .+ ε .* V), p)))
                    end
                    @test X̄_ad ≈ X̄_fd rtol=1e-4
                end
            end
            @testset "check has correct default and passed to primal" begin
                # this will almost certainly be a non-PD matrix
                X = Matrix(Symmetric(randn(10, 10)))
                @test_throws Exception rrule(cholesky, X, NoPivot())
                rrule(cholesky, X, NoPivot(); check=false)  # just check it doesn't throw
            end
        end

        # Ensure that cotangents of cholesky(::StridedMatrix) and
        # (cholesky ∘ Symmetric)(::StridedMatrix) are equal.
        @testset "Symmetric" begin
            X = generate_well_conditioned_matrix(10)
            F, dX_pullback = rrule(cholesky, X, NoPivot())
            ΔU = randn(size(X))
            ΔF = Tangent{typeof(F)}(; factors=ΔU)

            @testset for uplo in (:L, :U)
                X_symmetric, sym_back = rrule(Symmetric, X, uplo)
                C, chol_back_sym = rrule(cholesky, X_symmetric, NoPivot())

                ΔC = Tangent{typeof(C)}(; factors=(uplo === :U ? ΔU : ΔU'))
                ΔX_symmetric = chol_back_sym(ΔC)[2]
                if uplo === :U
                    @test sym_back(ΔX_symmetric)[2] ≈ dX_pullback(ΔF)[2]
                else
                    @test sym_back(ΔX_symmetric)[2] ≈ dX_pullback(ΔF)[2]'
                end
            end
        end

        # Ensure that cotangents of cholesky(::StridedMatrix) and
        # (cholesky ∘ Hermitian)(::StridedMatrix) are equal.
        @testset "Hermitian" begin
            @testset "Hermitian{$T}" for T in (Float64, ComplexF64)
                X = generate_well_conditioned_matrix(T, 10)
                F, dX_pullback = rrule(cholesky, X, NoPivot())
                ΔU = randn(T, size(X))
                ΔF = Tangent{typeof(F)}(; factors=ΔU)

                @testset for uplo in (:L, :U)
                    X_hermitian, herm_back = rrule(Hermitian, X, uplo)
                    C, chol_back_herm = rrule(cholesky, X_hermitian, NoPivot())

                    ΔC = Tangent{typeof(C)}(; factors=(uplo === :U ? ΔU : ΔU'))
                    ΔX_hermitian = chol_back_herm(ΔC)[2]
                    if uplo === :U
                        @test herm_back(ΔX_hermitian)[2] ≈ dX_pullback(ΔF)[2]
                    else
                        @test herm_back(ΔX_hermitian)[2] ≈ dX_pullback(ΔF)[2]'
                    end
                end
            end
            @testset "check has correct default and passed to primal" begin
                # this will almost certainly be a non-PD matrix
                X = Hermitian(randn(10, 10))
                @test_throws Exception rrule(cholesky, X, NoPivot())
                rrule(cholesky, X, NoPivot(); check=false)
            end
        end

        @testset "det and logdet (uplo=$p)" for p in (:U, :L)
            @testset "$op" for op in (det, logdet)
                @testset "$T" for T in (Float64, ComplexF64)
                    n = 5
                    # rand (not randn) so det will be positive, so logdet will be defined
                    A = 3 * rand(T, (n, n))
                    X = Cholesky(A * A' + I, p, 0)
                    X̄_acc = Tangent{typeof(X)}(; factors=Diagonal(randn(T, n))) # sensitivity is always a diagonal
                    test_rrule(op, X ⊢ X̄_acc)

                    # return type
                    _, op_pullback = rrule(op, X)
                    X̄ = op_pullback(2.7)[2]
                    @test X̄ isa Tangent{<:Cholesky}
                    @test X̄.factors isa Diagonal

                    # zero co-tangent
                    X̄ = op_pullback(0.0)[2]
                    @test all(iszero, X̄.factors)
                end
            end

            @testset "singular ($T)" for T in (Float64, ComplexF64)
                n = 5
                L = LowerTriangular(randn(T, (n, n)))
                L[1, 1] = zero(T)
                X = cholesky(L * L'; check=false)
                detX, det_pullback = rrule(det, X)
                ΔX = det_pullback(rand())[2]
                @test iszero(detX)
                @test ΔX.factors isa Diagonal && all(iszero, ΔX.factors)
            end
        end

        @testset "\\(::Cholesky, ::AbstractVecOrMat)" begin
            n = 10
            for T in (Float64, ComplexF64), sz in (n, (n, 5))
                A = generate_well_conditioned_matrix(T, n)
                C = cholesky(A)
                B = randn(T, sz)
                # because the rule calls the rrule for getproperty, its rrule is not
                # completely type-inferable
                test_rrule(\, C, B; check_inferred=false)
            end
        end

        @testset "/(::AbstractMatrix, ::Cholesky)" begin
            n = 10
            for T in (Float64, ComplexF64)
                A = generate_well_conditioned_matrix(T, n)
                C = cholesky(A)
                B = randn(T, 5, n)
                # because the rule calls the rrule for getproperty, its rrule is not
                # completely type-inferable
                test_rrule(/, B, C; check_inferred=false)
            end
        end
    end
end
