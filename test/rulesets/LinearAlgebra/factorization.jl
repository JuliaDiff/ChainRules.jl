function FiniteDifferences.to_vec(C::Cholesky)
    C_vec, factors_from_vec = to_vec(C.factors)
    function cholesky_from_vec(v)
        return Cholesky(factors_from_vec(v), C.uplo, C.info)
    end
    return C_vec, cholesky_from_vec
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
            rrule_test(cholesky, ΔC, (D, Diagonal(randn(5))))  
        end


        X = generate_well_conditioned_matrix(10)
        V = generate_well_conditioned_matrix(10)
        F, dX_pullback = rrule(cholesky, X)
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
            C, chol_back_sym = rrule(cholesky, X_symmetric)

            Δ = Composite{typeof(C)}((U=UpperTriangular(randn(size(X)))))
            ΔX_symmetric = chol_back_sym(Δ)[2]
            @test sym_back(ΔX_symmetric)[2] ≈ dX_pullback(Δ)[2]
        end
    end
end
