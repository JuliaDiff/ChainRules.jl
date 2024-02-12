@testset "arraymath.jl" begin
    @testset "inv(::Matrix{$T})" for T in (Float64, ComplexF64)
        B = generate_well_conditioned_matrix(T, 3)
        if v"1.7" <= VERSION < v"1.9"
          @gpu test_frule(inv, B)
          @gpu test_rrule(inv, B)
        else
          @gpu_broken test_frule(inv, B)
          @gpu_broken test_rrule(inv, B)
        end
    end

    @testset "*: $T" for T in (Float64, ComplexF64)
        ⋆(a) = round.(5*randn(T, a))  # Helper to generate nice random values
        ⋆(a, b) = ⋆((a, b))  # matrix
        ⋆() = only(⋆(()))  # scalar

        @testset "Scalar-Array $dims" for dims in ((3,), (2, 3, 4))
            @gpu test_frule(*, ⋆(), ⋆(dims))
            @gpu test_frule(*, ⋆(dims), ⋆())

            @gpu test_rrule(*, ⋆(), ⋆(dims))
            @gpu test_rrule(*, ⋆(dims), ⋆())
        end

        @testset "AbstractMatrix-AbstractVector n=$n, m=$m" for n in (2, 3), m in (4, 5)
            @testset "Array" begin
                test_frule(*, n ⋆ m, ⋆(m))
                test_rrule(*, n ⋆ m, ⋆(m))
            end
        end

        @testset "AbstractVector-AbstractMatrix n=$n, m=$m" for n in (2, 3), m in (4, 5)
            @testset "Array" begin
                test_frule(*, ⋆(n), 1 ⋆ m)
                test_rrule(*, ⋆(n), 1 ⋆ m)
            end
        end

        @testset "dense-matrix n=$n, m=$m, p=$p" for n in (2, 5), m in (2, 4), p in (2, 3)
            @testset "Array" begin
                test_frule(*, (n⋆m), (m⋆p))
                test_rrule(*, (n⋆m), (m⋆p))
            end

            @testset "SubArray - $indexname" for (indexname, m_index) in (
                ("fast", :), ("slow", m:-1:1)
            )
                test_rrule(*, view(n⋆m, :, m_index), view(m⋆p, m_index, :))
                test_rrule(*, n⋆m, view(m⋆p, m_index, :))
                test_rrule(*, view(n⋆m, :, m_index), m⋆p)
            end

            @testset "Adjoints and Transposes" begin
                test_rrule(*, Transpose(m⋆n) ⊢ Transpose(m⋆n), Transpose(p⋆m) ⊢ Transpose(p⋆m))
                test_rrule(*, Adjoint(m⋆n) ⊢ Adjoint(m⋆n), Adjoint(p⋆m) ⊢ Adjoint(p⋆m))

                test_rrule(*, Transpose(m⋆n) ⊢ Transpose(m⋆n), (m⋆p))
                test_rrule(*, Adjoint(m⋆n) ⊢ Adjoint(m⋆n), (m⋆p))

                test_rrule(*, (n⋆m), Transpose(p⋆m) ⊢ Transpose(p⋆m))
                test_rrule(*, (n⋆m), Adjoint(p⋆m) ⊢ Adjoint(p⋆m))
            end
        end

        @testset "Diagonal" begin
            # fwd
            @gpu test_frule(*, Diagonal([1.0, 2.0, 3.0]), Diagonal([4.0, 5.0, 6.0]))
            @gpu test_frule(*, Diagonal([1.0, 2.0, 3.0]), rand(3))

            # rev
            @gpu test_rrule(*, Diagonal([1.0, 2.0, 3.0]), Diagonal([4.0, 5.0, 6.0]))
            @gpu test_rrule(*, Diagonal([1.0, 2.0, 3.0]), rand(3))

            # Needs to not try and inplace, as `mul!` will do wrong.
            # see https://github.com/JuliaDiff/ChainRulesCore.jl/issues/411
            @gpu test_rrule(*, Diagonal([1.0, 2.0, 3.0]), rand(3,3))
        end

        @testset "$adj * Vector" for adj in (adjoint, transpose)
            # This should be same as dot product and give a scalar
            test_rrule(*, adj(⋆(5)) ⊢ adj(⋆(5)), ⋆(5))
        end
    end

    @testset "muladd: $T" for T in (Float64, ComplexF64)
        @testset "add $(typeof(z))" for z in [rand(), rand(T, 3), rand(T, 3, 3), false]
            @testset "matrix * matrix" begin
                A = rand(T, 3, 3)
                B = rand(T, 3, 3)
                @gpu test_rrule(muladd, A, B, z)
                @gpu test_rrule(muladd, A', B, z)
                @gpu test_rrule(muladd, A , B', z)
                @gpu test_frule(muladd, A, B, z)
                @gpu test_frule(muladd, A', B, z)
                @gpu test_frule(muladd, A , B', z)

                C = rand(T, 3, 5)
                D = rand(T, 5, 3)
                @gpu test_rrule(muladd, C, D, z)
                @gpu test_frule(muladd, C, D, z)
            end
            if ndims(z) <= 1
                @testset "matrix * vector" begin
                    A, B = rand(T, 3, 3), rand(T, 3)
                    test_rrule(muladd, A, B, z)
                    test_rrule(muladd, A, B ⊢ rand(T, 3,1), z)
                    test_frule(muladd, A, B, z)
                end
                @testset "adjoint * matrix" begin
                    At, B = rand(T, 3)', rand(T, 3, 3)
                    test_rrule(muladd, At, B, z')
                    test_rrule(muladd, At ⊢ rand(T,1,3), B, z')
                    test_frule(muladd, At, B, z')
                end
            end
            if ndims(z) == 0
                @testset "adjoint * vector" begin # like dot
                    A, B = rand(T, 3)', rand(T, 3)
                    test_rrule(muladd, A, B, z)
                    test_rrule(muladd, A ⊢ rand(T,1,3), B, z')
                    test_frule(muladd, A, B, z)
                end
            end
            if ndims(z) == 2 # other dims lead to e.g. muladd(ones(4), ones(1,4), 1)
                @testset "vector * adjoint" begin # outer product
                    A, B = rand(T, 3), rand(T, 3)'
                    test_rrule(muladd, A, B, z)
                    test_rrule(muladd, A, B ⊢ rand(T,1,3), z)
                    test_frule(muladd, A, B, z)
                end
            end
        end
    end

    if VERSION > v"1.7.0-"
        @eval using LinearAlgebra: mat_mat_scalar, mat_vec_scalar, StridedMaybeAdjOrTransMat

        @testset "3-arg *, $T" for T in [Float64, ComplexF64]

            test_rrule(mat_mat_scalar, rand(T,4,4), rand(T,4,4), rand(T))
            test_rrule(mat_mat_scalar, rand(T,4,4), rand(T,4,4), 0.0)
            test_rrule(mat_mat_scalar, rand(T,4,4)' ⊢ rand(T,4,4), rand(T,4,4), rand(T))

            test_rrule(mat_vec_scalar, rand(T,4,4), rand(T,4), rand(T))
            test_rrule(mat_vec_scalar, rand(T,4,4), rand(T,4), 0.0)

            T == ComplexF64 && continue
            # Test with γ of a wider type
            A, B, b, γ = rand(3,3), rand(3,3), rand(3), rand()
            dZ, dz = rand(3,3), rand(3)
            unthunk(rrule(mat_mat_scalar, A, B, γ + 0im)[2](dZ)[4]) ≈ unthunk(rrule(mat_mat_scalar, A, B, γ)[2](dZ)[4])
            unthunk(rrule(mat_mat_scalar, A, B, 0 + 0im)[2](dZ)[4]) ≈ unthunk(rrule(mat_mat_scalar, A, B, 0)[2](dZ)[4])
            unthunk(rrule(mat_vec_scalar, A, b, γ + 0im)[2](dz)[4]) ≈ unthunk(rrule(mat_vec_scalar, A, b, γ)[2](dz)[4])
        end
    end # VERSION

    @testset "$f" for f in (/, \)
        @testset "Matrix" begin
            for n in 3:5, m in 3:5
                A = randn(m, n)
                B = randn(m, n)
                test_rrule(f, A, B; check_inferred=false) # ChainRulesCore #407
            end
        end
        @testset "Vector" begin
            x = randn(10)
            y = randn(10)
            test_rrule(f, x, y; check_inferred=false) # ChainRulesCore #407
        end
        if f == (\)
            @testset "Matrix $f Vector" begin
                X = randn(10, 4)
                y = randn(10)
                test_rrule(f, X, y; check_inferred=false)
            end
            @testset "Vector $f Matrix" begin
                x = randn(10)
                Y = randn(10, 4)
                test_rrule(f, x, Y; output_tangent=Transpose(rand(4)), check_inferred=false)
            end
        else
            A = rand(2, 4)
            B = rand(4, 4)
            test_rrule(f, A, B; check_inferred=false) # ChainRulesCore #407
        end
    end

    @testset "/ and \\ Scalar-AbstractArray" begin
        A = round.(10 .* randn(3, 4, 5), digits=1)
        # fwd
        @gpu test_frule(/, A, 7.2)
        @gpu test_frule(\, 7.2, A)
        # rev
        @gpu test_rrule(/, A, 7.2)
        @gpu test_rrule(\, 7.2, A)

        C = round.(10 .* randn(6) .+ im .* 10 .* randn(6), digits=1)
        @gpu test_rrule(/, C, 7.2+8.3im)
        @gpu test_rrule(\, 7.2+8.3im, C)
    end

    @testset "negation" begin
        A = randn(4, 4)
        Ā = randn(4, 4)
        # fwd
        @gpu test_frule(-, A)
        # rev
        @gpu test_rrule(-, A)
        @gpu test_rrule(-, Diagonal(A); output_tangent=Diagonal(Ā))
    end

    @testset "addition" begin
        # fwd
        @gpu test_frule(+, randn(2), randn(2), randn(2))
        # rev
        @gpu test_rrule(+, randn(4, 4), randn(4, 4), randn(4, 4))
        @gpu test_rrule(+, randn(3), randn(3,1), randn(3,1,1))
    end

    @testset "subtraction" begin
        # fwd
        @gpu test_frule(-, randn(2), randn(2))
        # rev
        @gpu test_rrule(-, randn(4, 4), randn(4, 4))
        @gpu test_rrule(-, randn(4), randn(ComplexF64, 4))
        @gpu test_rrule(-, randn(3), randn(3, 1))
    end
end
