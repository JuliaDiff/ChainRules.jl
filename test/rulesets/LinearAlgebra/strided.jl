@testset "strided.jl" begin
    @testset "Matrix-Matrix" begin
        dims = [3]#,4,5]

        ⋆(a, b) = rand(-9.0:9.0, a, b)  # Helper to generate random matrix
        ⋆₂(a, b) = (a⋆b, a⋆b)  # Helper to generate random matrix and its cotangent
        @testset "n=$n, m=$m, p=$p" for n in dims, m in dims, p in dims
            rrule_test(*, n⋆p, (n⋆₂m), (m⋆₂p))

            rrule_test(*, n⋆p, Transpose.(m⋆₂n), Transpose.(p⋆₂m))
            rrule_test(*, n⋆p, Adjoint.(m⋆₂n), Adjoint.(p⋆₂m))

            rrule_test(*, n⋆p, Transpose.(m⋆₂n), (m⋆₂p))
            rrule_test(*, n⋆p, Adjoint.(m⋆₂n), (m⋆₂p))

            rrule_test(*, n⋆p, (n⋆₂m), Transpose.(p⋆₂m))
            rrule_test(*, n⋆p, (n⋆₂m), Adjoint.(p⋆₂m))
        end
    end
end
