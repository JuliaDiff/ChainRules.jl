@testset "arraymath" begin
    @testset "inv(::Matrix{$T})" for T in (Float64, ComplexF64)
        N = 3
        B = generate_well_conditioned_matrix(T, N)
        frule_test(inv, (B, randn(T, N, N)))
        rrule_test(inv, randn(T, N, N), (B, randn(T, N, N)))
    end

    @testset "*: $T" for T in (Float64, ComplexF64)
        ⋆(a) = round.(5*randn(T, a))  # Helper to generate nice random values
        ⋆(a, b) = ⋆((a, b))  # matrix
        ⋆() = only(⋆(()))  # scalar

        ⋆₂(a) = (⋆(a), ⋆(a)) # Helper to generate random matrix and its cotangent
        ⋆₂(a, b) = ⋆₂((a, b))  #matrix
        ⋆₂() = ⋆₂(())  # scalar

        ⋆₃(a) = (⋆(n), ⋆(n, 1))

        @testset "Scalar-Array $dims" for dims in ((3,), (5,4), (2, 3, 4, 5))
            rrule_test(*, ⋆(dims), ⋆₂(), ⋆₂(dims))
            rrule_test(*, ⋆(dims), ⋆₂(dims), ⋆₂())
        end

        @testset "AbstractMatrix-AbstractMatrix" begin
            # Matrix-Vector product
            @testset "n=$n, m=$m" for n in (2, 3), m in (4, 5)
                @testset "Array" begin
                    rrule_test(*, ⋆(n), ⋆₂(n, m), ⋆₂(m))
                end
            end

            @testset "n=$n, m=$m" for n in (2, 3), m in (4, 5)
                @testset "Array" begin
                    rrule_test(*, ⋆(n, m), ⋆₃(n), ⋆₂(1, m))
                end
            end

            @testset "n=$n, m=$m, p=$p" for n in (2, 5), m in (2, 4), p in (2, 3)
                @testset "Array" begin
                    rrule_test(*, n⋆p, (n⋆₂m), (m⋆₂p))
                end

                @testset "SubArray - $indexname" for (indexname, m_index) in (
                    ("fast", :), ("slow", Ref(m:-1:1))
                )
                    rrule_test(*, n⋆p, view.(n⋆₂m, :, m_index), view.(m⋆₂p, m_index, :))
                    rrule_test(*, n⋆p, n⋆₂m, view.(m⋆₂p, m_index, :))
                    rrule_test(*, n⋆p, view.(n⋆₂m, :, m_index), m⋆₂p)
                end

                @testset "Adjoints and Transposes" begin
                    rrule_test(*, n⋆p, Transpose.(m⋆₂n), Transpose.(p⋆₂m))
                    rrule_test(*, n⋆p, Adjoint.(m⋆₂n), Adjoint.(p⋆₂m))

                    rrule_test(*, n⋆p, Transpose.(m⋆₂n), (m⋆₂p))
                    rrule_test(*, n⋆p, Adjoint.(m⋆₂n), (m⋆₂p))

                    rrule_test(*, n⋆p, (n⋆₂m), Transpose.(p⋆₂m))
                    rrule_test(*, n⋆p, (n⋆₂m), Adjoint.(p⋆₂m))
                end
            end
        end
    end


    @testset "$f" for f in (/, \)
        @testset "Matrix" begin
            for n in 3:5, m in 3:5
                A = randn(m, n)
                B = randn(m, n)
                Ȳ = randn(size(f(A, B)))
                rrule_test(f, Ȳ, (A, randn(m, n)), (B, randn(m, n)))
            end
        end
        @testset "Vector" begin
            x = randn(10)
            y = randn(10)
            ȳ = randn(size(f(x, y))...)
            rrule_test(f, ȳ, (x, randn(10)), (y, randn(10)))
        end
        if f == (\)
            @testset "Matrix $f Vector" begin
                X = randn(10, 4)
                y = randn(10)
                ȳ = randn(size(f(X, y))...)
                rrule_test(f, ȳ, (X, randn(size(X))), (y, randn(10)))
            end
            @testset "Vector $f Matrix" begin
                x = randn(10)
                Y = randn(10, 4)
                ȳ = randn(size(f(x, Y))...)
                rrule_test(f, ȳ, (x, randn(size(x))), (Y, randn(size(Y))))
            end
        end
    end
    @testset "/ and \\ Scalar-AbstractArray" begin
        A = randn(3, 4, 5)
        Ā = randn(3, 4, 5)
        Ȳ = randn(3, 4, 5)
        rrule_test(/, Ȳ, (A, Ā), (7.2, 2.3))
        rrule_test(\, Ȳ, (7.2, 2.3), (A, Ā))
    end


    @testset "negation" begin
        A = randn(4, 4)
        Ā = randn(4, 4)
        Ȳ = randn(4, 4)
        rrule_test(-, Ȳ, (A, Ā))
        rrule_test(-, Diagonal(Ȳ), (Diagonal(A), Diagonal(Ā)))
    end
end
