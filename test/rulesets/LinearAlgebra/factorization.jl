using ChainRules: level2partition, level3partition, chol_blocked_rev, chol_unblocked_rev

@testset "Factorizations" begin
    @testset "svd" begin
        rng = MersenneTwister(2)
        for n in [4, 6, 10], m in [3, 5, 10]
            X = randn(rng, n, m)
            F, dX = rrule(svd, X)
            for p in [:U, :S, :V]
                Y, (dF, dp) = rrule(getproperty, F, p)
                @test dp isa ChainRules.DNERule
                Ȳ = randn(rng, size(Y)...)
                X̄_ad = dX(dF(Ȳ))
                X̄_fd = j′vp(central_fdm(5, 1), X->getproperty(svd(X), p), Ȳ, X)
                @test X̄_ad ≈ X̄_fd rtol=1e-6 atol=1e-6
            end
            @test_throws ArgumentError rrule(getproperty, F, :Vt)
        end
        @testset "accumulate!" begin
            X = [1.0 2.0; 3.0 4.0; 5.0 6.0]
            F, dX = rrule(svd, X)
            X̄ = (U=zeros(3, 2), S=zeros(2), V=zeros(2, 2))
            for p in [:U, :S, :V]
                Y, (dF, _) = rrule(getproperty, F, p)
                Ȳ = ones(size(Y)...)
                ChainRules.accumulate!(X̄, dF, Ȳ)
            end
            @test X̄.U ≈ ones(3, 2) atol=1e-6
            @test X̄.S ≈ ones(2) atol=1e-6
            @test X̄.V ≈ ones(2, 2) atol=1e-6
        end
        @testset "Helper functions" begin
            X = randn(rng, 10, 10)
            Y = randn(rng, 10, 10)
            @test ChainRules._mulsubtrans!(copy(X), Y) ≈ Y .* (X - X')
            @test ChainRules._eyesubx!(copy(X)) ≈ I - X
            @test ChainRules._add!(copy(X), Y) ≈ X + Y
        end
    end
    @testset "cholesky" begin
        rng = MersenneTwister(4)
        @testset "the thing" begin
            X = generate_well_conditioned_matrix(rng, 10)
            V = generate_well_conditioned_matrix(rng, 10)
            F, dX = rrule(cholesky, X)
            for p in [:U, :L]
                Y, (dF, dp) = rrule(getproperty, F, p)
                @test dp isa ChainRules.DNERule
                Ȳ = (p === :U ? UpperTriangular : LowerTriangular)(randn(rng, size(Y)))
                # NOTE: We're doing Nabla-style testing here and avoiding using the `j′vp`
                # machinery from FiniteDifferences because that isn't set up to respect
                # necessary special properties of the input. In the case of the Cholesky
                # factorization, we need the input to be Hermitian.
                X̄_ad = dot(dX(dF(Ȳ)), V)
                X̄_fd = central_fdm(5, 1)() do ε
                    dot(Ȳ, getproperty(cholesky(X .+ ε .* V), p))
                end
                @test X̄_ad ≈ X̄_fd rtol=1e-6 atol=1e-6
            end
        end
        @testset "helper functions" begin
            A = randn(rng, 5, 5)
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

            A = Matrix(LowerTriangular(randn(rng, 10, 10)))
            Ā = Matrix(LowerTriangular(randn(rng, 10, 10)))
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
