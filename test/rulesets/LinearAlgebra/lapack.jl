@testset "LAPACK" begin
    @testset "trsyl!" begin
        @testset "T=$T, m=$m, n=$n, transa='$transa', transb='$transb', isgn=$isgn" for
            T in (Float64, ComplexF64),
            transa in (T <: Real ? ('N', 'C', 'T') : ('N', 'C')),
            transb in (T <: Real ? ('N', 'C', 'T') : ('N', 'C')),
            m in (2, 3),
            n in (1, 3),
            isgn in (1, -1)

            # make A and B quasi upper-triangular (or upper-triangular for complex)
            # and their tangents have the same sparsity pattern
            A = schur(randn(T, m, m)).T
            B = schur(randn(T, n, n)).T
            C = randn(T, m, n)
            ΔA = rand_tangent(A) .* (!iszero).(A)
            ΔB = rand_tangent(B) .* (!iszero).(B)

            test_frule(
                LAPACK.trsyl!,
                transa ⊢ nothing,
                transb ⊢ nothing,
                A ⊢ ΔA,
                B ⊢ ΔB,
                C,
                isgn ⊢ nothing,
            )
        end
    end
end
