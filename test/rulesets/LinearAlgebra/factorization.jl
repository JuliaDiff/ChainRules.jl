using ChainRules: level2partition, level3partition, chol_blocked_rev, chol_unblocked_rev

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
        @testset "eigen" begin
            # avoid implementing to_vec(::Eigen)
            f(E::Eigen) = (values=E.values, vectors=E.vectors)

            # NOTE: for unstructured matrices, low enough n, and this specific seed, finite
            # differences of eigen seems to be stable enough for direct comparison.
            # This allows us to directly check differential of normalization/phase
            # convention
            n = 10

            @testset "eigen(::Matrix{$T})" for T in (Float64,ComplexF64)
                # NOTE: eigen is not type-stable, so neither are its frule and rrule
                @testset "frule" begin
                    X = randn(T, n, n)
                    Ẋ = rand_tangent(X)
                    F = eigen(X)
                    F_fwd, Ḟ_ad = frule((Zero(), Ẋ), eigen, X)
                    @test F_fwd == F
                    @test Ḟ_ad isa Composite{typeof(F)}
                    Ḟ_fd = jvp(_fdm, f ∘ eigen, (X, Ẋ))
                    @test Ḟ_ad.values ≈ Ḟ_fd.values
                    @test Ḟ_ad.vectors ≈ Ḟ_fd.vectors
                    @test frule((Zero(), Zero()), eigen, X) == (F, Zero())
                end

                @testset "rrule" begin
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
                    X̄_fd = j′vp(_fdm, f ∘ eigen, F̄, X)[1]
                    @test X̄_ad ≈ X̄_fd
                    @test @inferred(back(Zero())) === (NO_FIELDS, Zero())
                    F̄zero = CT(values = Zero(), vectors = Zero())
                    @test @inferred(back(F̄zero)) === (NO_FIELDS, Zero())
                end
            end

            @testset "normalization/phase functions are idempotent" begin
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
        end

        @testset "eigvals" begin
            @testset "eigvals(::Matrix{$T})" for T in (Float64,ComplexF64)
                # NOTE: eigvals is not type-stable, so neither are its frule and rrule
                n = 10
                X = randn(T, n, n)
                λ = eigvals(X)
                Ẋ = rand_tangent(X)
                frule_test(eigvals, (X, Ẋ))
                @test frule((Zero(), Zero()), eigvals, X) == (λ, Zero())

                X̄ = rand_tangent(X)
                λ̄ = rand_tangent(eigvals(X))
                rrule_test(eigvals, λ̄, (X, X̄))
                back = rrule(eigvals, X)[2]
                @inferred back(λ̄)
                @test @inferred(back(Zero())) === (NO_FIELDS, Zero())
            end
        end
    end
    @testset "cholesky" begin
        @testset "the thing" begin
            X = generate_well_conditioned_matrix(10)
            V = generate_well_conditioned_matrix(10)
            F, dX_pullback = rrule(cholesky, X)
            for p in [:U, :L]
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
                X̄_fd = _fdm(0.0) do ε
                    dot(Ȳ, getproperty(cholesky(X .+ ε .* V), p))
                end
                @test X̄_ad ≈ X̄_fd rtol=1e-6 atol=1e-6
            end
        end
        @testset "helper functions" begin
            A = randn(5, 5)
            r, d, B2, c = level2partition(A, 4, false)
            R, D, B3, C = level3partition(A, 4, 4, false)
            @test all(r .== R')
            @test all(d .== D)
            @test B2[1] == B3[1]
            @test all(c .== C)

            # Check that level 2 partition with `upper == true` is consistent with `false`
            rᵀ, dᵀ, B2ᵀ, cᵀ = level2partition(transpose(A), 4, true)
            @test r == rᵀ
            @test d == dᵀ
            @test B2' == B2ᵀ
            @test c == cᵀ

            # Check that level 3 partition with `upper == true` is consistent with `false`
            R, D, B3, C = level3partition(A, 2, 4, false)
            Rᵀ, Dᵀ, B3ᵀ, Cᵀ = level3partition(transpose(A), 2, 4, true)
            @test transpose(R) == Rᵀ
            @test transpose(D) == Dᵀ
            @test transpose(B3) == B3ᵀ
            @test transpose(C) == Cᵀ

            A = Matrix(LowerTriangular(randn(10, 10)))
            Ā = Matrix(LowerTriangular(randn(10, 10)))
            # NOTE: BLAS gets angry if we don't materialize the Transpose objects first
            B = Matrix(transpose(A))
            B̄ = Matrix(transpose(Ā))
            @test chol_unblocked_rev(Ā, A, false) ≈ chol_blocked_rev(Ā, A, 1, false)
            @test chol_unblocked_rev(Ā, A, false) ≈ chol_blocked_rev(Ā, A, 3, false)
            @test chol_unblocked_rev(Ā, A, false) ≈ chol_blocked_rev(Ā, A, 5, false)
            @test chol_unblocked_rev(Ā, A, false) ≈ chol_blocked_rev(Ā, A, 10, false)
            @test chol_unblocked_rev(Ā, A, false) ≈ transpose(chol_unblocked_rev(B̄, B, true))

            @test chol_unblocked_rev(B̄, B, true) ≈ chol_blocked_rev(B̄, B, 1, true)
            @test chol_unblocked_rev(B̄, B, true) ≈ chol_blocked_rev(B̄, B, 5, true)
            @test chol_unblocked_rev(B̄, B, true) ≈ chol_blocked_rev(B̄, B, 10, true)
        end
    end
end
