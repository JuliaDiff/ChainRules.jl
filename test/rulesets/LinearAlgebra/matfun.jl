@testset "matrix functions" begin
    @testset "LinearAlgebra.exp!(A::Matrix) frule" begin
        n = 10
        # each normalization hits a specific branch
        @testset "A::Matrix{$T}, opnorm(A,1)=$nrm" for T in (Float64, ComplexF64),
            nrm in (0.01, 0.1, 0.5, 1.5, 3.0, 6.0, 12.0)

            A = randn(T, n, n)
            A *= nrm / opnorm(A, 1)
            tols = nrm == 0.1 ? (atol=1e-8, rtol=1e-8) : NamedTuple()
            test_frule(LinearAlgebra.exp!, A; tols...)
        end
        @testset "imbalanced A" begin
            A = Float64[0 10 0 0; -1 0 0 0; 0 0 0 0; -2 0 0 0]
            test_frule(LinearAlgebra.exp!, A)
        end
        @testset "imbalanced A with no squaring" begin
            # https://github.com/JuliaDiff/ChainRules.jl/issues/595
            A = [
                -0.007623430669065629 -0.567237096385192  0.4419041897734335;
                 2.090838913114862    -1.254084243281689 -0.04145771190198238;
                 2.3397892123412833   -0.6650489083959324 0.6387266010923911
                ]
            test_frule(LinearAlgebra.exp!, A)
        end
        @testset "exhaustive test" begin
            # added to ensure we never hit truncation error
            # https://github.com/JuliaDiff/ChainRules.jl/issues/595
            rng = MersenneTwister(1)
            for _ in 1:100
                A = randn(rng, 3, 3)
                test_frule(LinearAlgebra.exp!, A)
            end
        end
        @testset "hermitian A, T=$T" for T in (Float64, ComplexF64)
            A = Matrix(Hermitian(randn(T, n, n)))
            test_frule(LinearAlgebra.exp!, A)
            test_frule(LinearAlgebra.exp!, A ⊢ Matrix(Hermitian(randn(T, n, n))))
        end
    end

    @testset "exp(A::Matrix) rrule" begin
        n = 10
        # each normalization hits a specific branch
        @testset "A::Matrix{$T}, opnorm(A,1)=$nrm" for T in (Float64, ComplexF64),
            nrm in (0.01, 0.1, 0.5, 1.5, 3.0, 6.0, 12.0)

            A = randn(T, n, n)
            A *= nrm / opnorm(A, 1)
            # rrule is not inferable, but pullback should be
            tols = nrm == 0.1 ? (atol=1e-8, rtol=1e-8) : NamedTuple()
            test_rrule(exp, A; check_inferred=false, tols...)
            Y, back = rrule(exp, A)
            @maybe_inferred back(rand_tangent(Y))
        end
        @testset "cotangent not mutated" begin
            # https://github.com/JuliaDiff/ChainRules.jl/issues/512
            A = [1.0 2.0; 3.0 4.0]
            Y, back = rrule(exp, A)
            ΔY′ = rand_tangent(Y)'
            ΔY′copy = copy(ΔY′)
            back(ΔY′)
            @test ΔY′ == ΔY′copy
        end
        @testset "imbalanced A" begin
            A = Float64[0 10 0 0; -1 0 0 0; 0 0 0 0; -2 0 0 0]
            test_rrule(exp, A; check_inferred=false)
        end
        @testset "imbalanced A with no squaring" begin
            # https://github.com/JuliaDiff/ChainRules.jl/issues/595
            A = [
                -0.007623430669065629 -0.567237096385192  0.4419041897734335;
                 2.090838913114862    -1.254084243281689 -0.04145771190198238;
                 2.3397892123412833   -0.6650489083959324 0.6387266010923911
                ]
            test_rrule(LinearAlgebra.exp, A; check_inferred=false)
        end
        @testset "exhaustive test" begin
            # added to ensure we never hit truncation error
            # https://github.com/JuliaDiff/ChainRules.jl/issues/595
            rng = MersenneTwister(1)
            for _ in 1:100
                A = randn(rng, 3, 3)
                test_rrule(LinearAlgebra.exp, A; check_inferred=false)
            end
        end
        @testset "hermitian A, T=$T" for T in (Float64, ComplexF64)
            A = Matrix(Hermitian(randn(T, n, n)))
            test_rrule(exp, A; check_inferred=false)
            test_rrule(
                exp, A;
                check_inferred=false, output_tangent=Matrix(Hermitian(randn(T, n, n)))
            )
        end
    end
end
