function FiniteDifferences.to_vec(C::Cholesky)
    C_vec, factors_from_vec = to_vec(C.factors)
    function cholesky_from_vec(v)
        return Cholesky(factors_from_vec(v), C.uplo, C.info)
    end
    return C_vec, cholesky_from_vec
end

function FiniteDifferences.to_vec(x::Val)
    Val_from_vec(v) = x
    return Bool[], Val_from_vec
end

@testset "Factorizations" begin
    @testset "svd" begin
        for n in [4, 6, 10], m in [3, 5, 10]
            X = randn(n, m)
            F, dX_pullback = rrule(svd, X)
            for p in [:U, :S, :V]
                Y, dF_pullback = rrule(getproperty, F, p)
                Ȳ = randn(size(Y)...)

                dself1, dF, dp = dF_pullback(Ȳ)
                @test dself1 === NO_FIELDS
                @test dp === DoesNotExist()

                dself2, dX = dX_pullback(dF)
                @test dself2 === NO_FIELDS
                X̄_ad = unthunk(dX)
                X̄_fd = only(j′vp(central_fdm(5, 1), X->getproperty(svd(X), p), Ȳ, X))
                @test all(isapprox.(X̄_ad, X̄_fd; rtol=1e-6, atol=1e-6))
            end
            @testset "Vt" begin
                Y, dF_pullback = rrule(getproperty, F, :Vt)
                Ȳ = randn(size(Y)...)
                @test_throws ArgumentError dF_pullback(Ȳ)
            end
        end

        @testset "Thunked inputs" begin
            X = randn(4, 3)
            F, dX_pullback = rrule(svd, X)
            for p in [:U, :S, :V]
                Y, dF_pullback = rrule(getproperty, F, p)
                Ȳ = randn(size(Y)...)

                _, dF_unthunked, _ = dF_pullback(Ȳ)

                # helper to let us check how things are stored.
                backing_field(c, p) = getproperty(ChainRulesCore.backing(c), p)
                @assert !(backing_field(dF_unthunked, p) isa AbstractThunk)

                dF_thunked = map(f->Thunk(()->f), dF_unthunked)
                @assert backing_field(dF_thunked, p) isa AbstractThunk

                dself_thunked, dX_thunked = dX_pullback(dF_thunked)
                dself_unthunked, dX_unthunked = dX_pullback(dF_unthunked)
                @test dself_thunked == dself_unthunked
                @test dX_thunked == dX_unthunked
            end
        end

        @testset "+" begin
            X = [1.0 2.0; 3.0 4.0; 5.0 6.0]
            F, dX_pullback = rrule(svd, X)
            X̄ = Composite{typeof(F)}(U=zeros(3, 2), S=zeros(2), V=zeros(2, 2))
            for p in [:U, :S, :V]
                Y, dF_pullback = rrule(getproperty, F, p)
                Ȳ = ones(size(Y)...)
                dself, dF, dp = dF_pullback(Ȳ)
                @test dself === NO_FIELDS
                @test dp === DoesNotExist()
                X̄ += dF
            end
            @test X̄.U ≈ ones(3, 2) atol=1e-6
            @test X̄.S ≈ ones(2) atol=1e-6
            @test X̄.V ≈ ones(2, 2) atol=1e-6
        end

        @testset "Helper functions" begin
            X = randn(10, 10)
            Y = randn(10, 10)
            @test ChainRules._mulsubtrans!!(copy(X), Y) ≈ Y .* (X - X')
            @test ChainRules._eyesubx!(copy(X)) ≈ I - X
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
                X = randn(T, n, n)
                Ẋ = rand_tangent(X)
                F = eigen!(copy(X))
                F_fwd, Ḟ_ad = frule((Zero(), copy(Ẋ)), eigen!, copy(X))
                @test F_fwd == F
                @test Ḟ_ad isa Composite{typeof(F)}
                Ḟ_fd = jvp(_fdm, asnt ∘ eigen! ∘ copy, (X, Ẋ))
                @test Ḟ_ad.values ≈ Ḟ_fd.values
                @test Ḟ_ad.vectors ≈ Ḟ_fd.vectors
                @test frule((Zero(), Zero()), eigen!, copy(X)) == (F, Zero())

                @testset "tangents are real when outputs are" begin
                    # hermitian matrices have real eigenvalues and, when real, real eigenvectors
                    X = Matrix(Hermitian(randn(T, n, n)))
                    Ẋ = Matrix(Hermitian(rand_tangent(X)))
                    _, Ḟ = frule((Zero(), Ẋ), eigen!, X)
                    @test eltype(Ḟ.values) <: Real
                    T <: Real && @test eltype(Ḟ.vectors) <: Real
                end
            end

            @testset "eigen(::Matrix{$T}) rrule" for T in (Float64,ComplexF64)
                # NOTE: eigen is not type-stable, so neither are is its rrule
                X = randn(T, n, n)
                F = eigen(X)
                V̄ = rand_tangent(F.vectors)
                λ̄ = rand_tangent(F.values)
                CT = Composite{typeof(F)}
                F_rev, back = rrule(eigen, X)
                @test F_rev == F
                _, X̄_values_ad = @inferred back(CT(values = λ̄))
                @test X̄_values_ad ≈ j′vp(_fdm, x -> eigen(x).values, λ̄, X)[1]
                _, X̄_vectors_ad = @inferred back(CT(vectors = V̄))
                @test X̄_vectors_ad ≈ j′vp(_fdm, x -> eigen(x).vectors, V̄, X)[1]
                F̄ = CT(values = λ̄, vectors = V̄)
                s̄elf, X̄_ad = @inferred back(F̄)
                @test s̄elf === NO_FIELDS
                X̄_fd = j′vp(_fdm, asnt ∘ eigen, F̄, X)[1]
                @test X̄_ad ≈ X̄_fd
                @test @inferred(back(Zero())) === (NO_FIELDS, Zero())
                F̄zero = CT(values = Zero(), vectors = Zero())
                @test @inferred(back(F̄zero)) === (NO_FIELDS, Zero())

                T <: Real && @testset "cotangent is real when input is" begin
                    X = randn(T, n, n)
                    Ẋ = rand_tangent(X)

                    F = eigen(X)
                    V̄ = rand_tangent(F.vectors)
                    λ̄ = rand_tangent(F.values)
                    F̄ = Composite{typeof(F)}(values = λ̄, vectors = V̄)
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

                n = 10
                @testset "eigen!(::Matrix{$T})" for T in (Float64, ComplexF64)
                    A, ΔA = Matrix(Hermitian(randn(T, n, n))), Matrix(Hermitian(randn(T, n, n)))

                    F = eigen!(copy(A))
                    @test frule((Zero(), Zero()), eigen!, copy(A)) == (F, Zero())
                    F_ad, ∂F_ad = frule((Zero(), copy(ΔA)), eigen!, copy(A))
                    @test F_ad == F
                    @test ∂F_ad isa Composite{typeof(F)}
                    @test ∂F_ad.values isa typeof(F.values)
                    @test ∂F_ad.vectors isa typeof(F.vectors)

                    f = x -> asnt(eigen(Matrix(Hermitian(x))))
                    ∂F_fd = jvp(_fdm, f, (A, ΔA))
                    @test ∂F_ad.values ≈ ∂F_fd.values

                    f_stable = x -> asnt(_eigen_stable(Matrix(Hermitian(x))))
                    F_stable = f_stable(A)
                    ∂F_stable_fd = jvp(_fdm, f_stable, (A, ΔA))
                    C = _eigvecs_stabilize_mat(F.vectors)
                    @test ∂F_ad.vectors * C ≈ ∂F_stable_fd.vectors
                end

                @testset "eigen(::Matrix{$T})" for T in (Float64, ComplexF64)
                    A, ΔU, Δλ = Matrix(Hermitian(randn(T, n, n))), randn(T, n, n), randn(n)

                    F = eigen(A)
                    ΔF = Composite{typeof(F)}(; values=Δλ, vectors=ΔU)
                    F_ad, back = rrule(eigen, A)
                    @test F_ad == F

                    C = _eigvecs_stabilize_mat(F.vectors)
                    CT = Composite{typeof(F)}

                    @testset for nzprops in ([:values], [:vectors], [:values, :vectors])
                        ∂F = CT(; [s => getproperty(ΔF, s) for s in nzprops]...)
                        ∂F_stable = (; [s => copy(getproperty(ΔF, s)) for s in nzprops]...)
                        :vectors in nzprops && rmul!(∂F_stable.vectors, C)

                        f_stable = function(x)
                            F_ = _eigen_stable(Matrix(Hermitian(x)))
                            return (; (s => getproperty(F_, s) for s in nzprops)...)
                        end

                        ∂self, ∂A = @inferred back(∂F)
                        @test ∂self === NO_FIELDS
                        @test ∂A isa typeof(A)
                        ∂A_fd = j′vp(_fdm, f_stable, ∂F_stable, A)[1]
                        @test ∂A ≈ ∂A_fd
                    end
                end
            end
        end

        @testset "eigvals/eigvals!" begin
            # NOTE: eigvals!/eigvals are not type-stable, so neither are their frule/rrule
            @testset "eigvals!(::Matrix{$T}) frule" for T in (Float64,ComplexF64)
                n = 10
                X = randn(T, n, n)
                λ = eigvals!(copy(X))
                Ẋ = rand_tangent(X)
                frule_test(eigvals!, (X, Ẋ))
                @test frule((Zero(), Zero()), eigvals!, copy(X)) == (λ, Zero())

                @testset "tangents are real when outputs are" begin
                    # hermitian matrices have real eigenvalues
                    X = Matrix(Hermitian(randn(T, n, n)))
                    Ẋ = Matrix(Hermitian(rand_tangent(X)))
                    _, λ̇ = frule((Zero(), Ẋ), eigvals!, X)
                    @test eltype(λ̇) <: Real
                end
            end

            @testset "eigvals(::Matrix{$T}) rrule" for T in (Float64,ComplexF64)
                n = 10
                X = randn(T, n, n)
                X̄ = rand_tangent(X)
                λ̄ = rand_tangent(eigvals(X))
                rrule_test(eigvals, λ̄, (X, X̄))
                back = rrule(eigvals, X)[2]
                @inferred back(λ̄)
                @test @inferred(back(Zero())) === (NO_FIELDS, Zero())

                T <: Real && @testset "cotangent is real when input is" begin
                    X = randn(T, n, n)
                    λ = eigvals(X)
                    λ̄ = rand_tangent(λ)
                    X̄ = rrule(eigvals, X)[2](λ̄)[2]
                    @test eltype(X̄) <: Real
                end
            end

            # below tests adapted from /test/rulesets/LinearAlgebra/symmetric.jl
            @testset "hermitian matrices" begin
                n = 10
                @testset "eigvals!(::Matrix{$T})" for T in (Float64, ComplexF64)
                    A, ΔA = Matrix(Hermitian(randn(T, n, n))), Matrix(Hermitian(randn(T, n, n)))
                    λ = eigvals!(copy(A))
                    λ_ad, ∂λ_ad = frule((Zero(), copy(ΔA)), eigvals!, copy(A))
                    @test λ_ad ≈ λ # inexact because frule uses eigen not eigvals
                    @test ∂λ_ad isa typeof(λ)
                    @test ∂λ_ad ≈ jvp(_fdm, A -> eigvals(Matrix(Hermitian(A))), (A, ΔA))
                end

                @testset "eigvals(::Matrix{$T})" for T in (Float64, ComplexF64)
                    A, Δλ = Matrix(Hermitian(randn(T, n, n))), randn(n)
                    λ = eigvals(A)
                    λ_ad, back = rrule(eigvals, A)
                    @test λ_ad ≈ λ # inexact because rrule uses eigen not eigvals
                    ∂self, ∂A = @inferred back(Δλ)
                    @test ∂self === NO_FIELDS
                    @test ∂A isa typeof(A)
                    @test ∂A ≈ j′vp(_fdm, A -> eigvals(Matrix(Hermitian(A))), Δλ, A)[1]
                    @test @inferred(back(Zero())) == (NO_FIELDS, Zero())
                end
            end
        end
    end

    # These tests are generally a bit tricky to write because FiniteDifferences doesn't
    # have fantastic support for this stuff at the minute.
    @testset "cholesky" begin
        @testset "Real" begin
            C = cholesky(rand() + 0.1)
            ΔC = Composite{typeof(C)}((factors=rand_tangent(C.factors)))
            rrule_test(cholesky, ΔC, (rand() + 0.1, randn()))
        end
        @testset "Diagonal{<:Real}" begin
            D = Diagonal(rand(5) .+ 0.1)
            C = cholesky(D)
            ΔC = Composite{typeof(C)}((factors=Diagonal(randn(5))))
            rrule_test(cholesky, ΔC, (D, Diagonal(randn(5))), (Val(false), nothing))
        end

        X = generate_well_conditioned_matrix(10)
        V = generate_well_conditioned_matrix(10)
        F, dX_pullback = rrule(cholesky, X, Val(false))
        @testset "uplo=$p" for p in [:U, :L]
            Y, dF_pullback = rrule(getproperty, F, p)
            Ȳ = (p === :U ? UpperTriangular : LowerTriangular)(randn(size(Y)))
            (dself, dF, dp) = dF_pullback(Ȳ)
            @test dself === NO_FIELDS
            @test dp === DoesNotExist()

            # NOTE: We're doing Nabla-style testing here and avoiding using the `j′vp`
            # machinery from FiniteDifferences because that isn't set up to respect
            # necessary special properties of the input. In the case of the Cholesky
            # factorization, we need the input to be Hermitian.
            ΔF = unthunk(dF)
            _, dX = dX_pullback(ΔF)
            X̄_ad = dot(unthunk(dX), V)
            X̄_fd = central_fdm(5, 1)(0.000_001) do ε
                dot(Ȳ, getproperty(cholesky(X .+ ε .* V), p))
            end
            @test X̄_ad ≈ X̄_fd rtol=1e-4
        end

        # Ensure that cotangents of cholesky(::StridedMatrix) and
        # (cholesky ∘ Symmetric)(::StridedMatrix) are equal.
        @testset "Symmetric" begin
            X_symmetric, sym_back = rrule(Symmetric, X, :U)
            C, chol_back_sym = rrule(cholesky, X_symmetric, Val(false))

            Δ = Composite{typeof(C)}((U=UpperTriangular(randn(size(X)))))
            ΔX_symmetric = chol_back_sym(Δ)[2]
            @test sym_back(ΔX_symmetric)[2] ≈ dX_pullback(Δ)[2]
        end
    end
end
