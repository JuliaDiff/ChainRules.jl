@testset "dense" begin
    @testset "dot" begin
        @testset "Vector{$T}" for T in (Float64, ComplexF64)
            test_frule(dot, randn(T, 3), randn(T, 3))
            test_rrule(dot, randn(T, 3), randn(T, 3))
        end
        @testset "Matrix{$T}" for T in (Float64, ComplexF64)
            test_frule(dot, randn(T, 3, 4), randn(T, 3, 4))
            test_rrule(dot, randn(T, 3, 4), randn(T, 3, 4))
        end
        @testset "Array{$T, 3}" for T in (Float64, ComplexF64)
            test_frule(dot, randn(T, 3, 4, 5), randn(T, 3, 4, 5))
            test_rrule(dot, randn(T, 3, 4, 5), randn(T, 3, 4, 5))
        end
        @testset "3-arg dot, Array{$T}" for T in (Float64, ComplexF64)
            test_frule(dot, randn(T, 3), randn(T, 3, 4), randn(T, 4))
            test_rrule(dot, randn(T, 3), randn(T, 3, 4), randn(T, 4))
        end
        permuteddimsarray(A) = PermutedDimsArray(A, (2,1))
        @testset "3-arg dot, $F{$T}" for T in (Float32, ComplexF32), F in (adjoint, permuteddimsarray)
            A = F(rand(T, 4, 3)) ⊢ F(rand(T, 4, 3))
            test_frule(dot, rand(T, 3), A, rand(T, 4); rtol=1f-3)
            test_rrule(dot, rand(T, 3), A, rand(T, 4); rtol=1f-3)
        end
    end

    @testset "cross"
        test_frule(cross, randn(3), randn(3))
        test_frule(cross, randn(ComplexF64, 3), randn(ComplexF64, 3))
        test_rrule(cross, randn(3), randn(3))
        # No complex support for rrule(cross,...
    end
    @testset "pinv" begin
        @testset "$T" for T in (Float64, ComplexF64)
            test_scalar(pinv, randn(T))
            @test frule((Zero(), randn(T)), pinv, zero(T))[2] ≈ zero(T)
            @test rrule(pinv, zero(T))[2](randn(T))[2] ≈ zero(T)
        end
        @testset "Vector{$T}" for T in (Float64, ComplexF64)
            test_frule(pinv, randn(T, 3), 0.0)
            test_frule(pinv, randn(T, 3), 0.0)

            # Checking types. TODO do we still need this?
            x = randn(T, 3)
            ẋ = randn(T, 3)
            Δy = copyto!(similar(pinv(x)), randn(T, 3))
            @test frule((Zero(), ẋ), pinv, x)[2] isa typeof(pinv(x))
            @test rrule(pinv, x)[2](Δy)[2] isa typeof(x)
        end
        #TODO Everything after this point
        @testset "$F{Vector{$T}}" for T in (Float64, ComplexF64), F in (Transpose, Adjoint)
            x, ẋ, x̄ = F(randn(T, 3)), F(randn(T, 3)), F(randn(T, 3))
            y = pinv(x)
            Δy = copyto!(similar(y), randn(T, 3))
            test_frule(pinv, (x, ẋ))
            y_fwd, ∂y_fwd = frule((Zero(),  ẋ), pinv, x)
            @test y_fwd isa typeof(y)
            @test ∂y_fwd isa typeof(y)
            test_rrule(pinv, Δy, (x, x̄))
            y_rev, back = rrule(pinv, x)
            @test y_rev isa typeof(y)
            @test back(Δy)[2] isa typeof(x)
        end
        @testset "Matrix{$T} with size ($m,$3)" for T in (Float64, ComplexF64),
            m in 1:3,
            3 in 1:3

            X, Ẋ, X̄ = randn(T, m, 3), randn(T, m, 3), randn(T, m, 3)
            ΔY = randn(T, size(pinv(X))...)
            test_frule(pinv, (X, Ẋ))
            test_rrule(pinv, ΔY, (X, X̄))
        end
    end
    @testset "$f" for f in (det, logdet)
        @testset "$f(::$T)" for T in (Float64, ComplexF64)
            b = (f === logdet && T <: Real) ? abs(randn(T)) : randn(T)
            test_scalar(f, b)
        end
        @testset "$f(::Matrix{$T})" for T in (Float64, ComplexF64)
            if f === logdet && float(T) <: Float32
                kwargs = (atol=1e-5, rtol=1e-5)
            else
                kwargs = NamedTuple()
            end
            B = generate_well_conditioned_matrix(T, 4)
            test_frule(f, (B, randn(T, 4, 4)); kwargs...)
            test_rrule(f, randn(T), (B, randn(T, 4, 4)); kwargs...)
        end
    end
    @testset "logabsdet(::Matrix{$T})" for T in (Float64, ComplexF64)
        B = randn(T, 4, 4)
        test_frule(logabsdet, (B, randn(T, 4, 4)))
        test_rrule(logabsdet, (randn(), randn(T)), (B, randn(T, 4, 4)))
        # test for opposite sign of determinant
        test_frule(logabsdet, (-B, randn(T, 4, 4)))
        test_rrule(logabsdet, (randn(), randn(T)), (-B, randn(T, 4, 4)))
    end
    @testset "tr" begin
        test_frule(tr, (randn(4, 4), randn(4, 4)))
        test_rrule(tr, randn(), (randn(4, 4), randn(4, 4)))
    end
    ==#
end
