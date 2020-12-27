@testset "norm functions" begin
    @testset "$fnorm(x::Array{$T,$(length(sz))})" for
        fnorm in (
            LinearAlgebra.norm1,
            LinearAlgebra.norm2,
            LinearAlgebra.normInf,
            LinearAlgebra.normMinusInf,
        ),
        T in (Float64, ComplexF64),
        sz in [(3,), (3, 3), (3, 2, 1)]

        x = randn(T, sz)
        # finite differences is unstable if maxabs (minabs) values are not well
        # separated from other values
        if fnorm === LinearAlgebra.normInf
            x[end] = 1000rand(T)
            kwargs = (atol=1e-5, rtol=1e-5)
        elseif fnorm == LinearAlgebra.normMinusInf
            x .*= 1000
            x[end] = rand(T)
            kwargs = (atol=1e-5, rtol=1e-5)
        else
            kwargs = NamedTuple()
        end

        fnorm === LinearAlgebra.norm2 && @testset "frule" begin
            test_frule(fnorm, x)
        end
        @testset "rrule" begin
            test_rrule(fnorm, x; kwargs...)
            x isa Matrix && @testset "$MT" for MT in (Diagonal, UpperTriangular, LowerTriangular)
                test_rrule(fnorm, MT(x); kwargs...)
            end

            ȳ = rand_tangent(fnorm(x))
            @test extern(rrule(fnorm, zero(x))[2](ȳ)[2]) ≈ zero(x)
            @test rrule(fnorm, x)[2](Zero())[2] isa Zero
        end
        ndims(x) > 1 && @testset "non-strided" begin
            xp = if x isa Matrix
                view(x, [1,2,3], 1:3)
            elseif x isa Array{T,3}
                PermutedDimsArray(x, (1,2,3))
            end
            @test !(xp isa StridedArray)
            y = fnorm(x)
            # ẋ = rand(T, size(xp)) # rand_tangent(xp)
            x̄ = rand(T, size(xp)) # rand_tangent(xp)
            ȳ = rand_tangent(y)
            # frule_test(fnorm, (xp, ẋ))
            rrule_test(fnorm, ȳ, (xp, x̄))
        end
        T == Float64 && ndims(x) == 1 && @testset "Integer input" begin
            x = [1,2,3]
            int_fwd, int_back = rrule(fnorm, x)
            float_fwd, float_back = rrule(fnorm, float(x))
            @test int_fwd ≈ float_fwd
            @test unthunk(int_back(1.0)[2]) ≈ unthunk(float_back(1.0)[2])
        end
    end
    @testset "norm(x::Array{$T,$(length(sz))})" for
        T in (Float64, ComplexF64),
        sz in [(0,), (3,), (3, 3), (3, 2, 1)]

        x = randn(T, sz)

        @testset "frule" begin
            test_frule(norm, x)
            @test frule((Zero(), Zero()), norm, x)[2] isa Zero

            ẋ = rand_tangent(x)
            @test iszero(frule((Zero(), ẋ), norm, zero(x))[2])
        end
        @testset "rrule" begin
            test_rrule(norm, x)
            x isa Matrix && @testset "$MT" for MT in (Diagonal, UpperTriangular, LowerTriangular)
                # we don't check inference on older julia versions. Improvements to
                # inference mean on 1.5+ it works, and that is good enough
                test_rrule(norm, MT(x); check_inferred=VERSION>=v"1.5")
            end

            ȳ = rand_tangent(norm(x))
            @test extern(rrule(norm, zero(x))[2](ȳ)[2]) ≈ zero(x)
            @test rrule(norm, x)[2](Zero())[2] isa Zero
        end
        ndims(x) > 1 && @testset "non-strided" begin
            xp = if x isa Matrix
                view(x, [1,2,3], 1:3)
            elseif x isa Array{T,3}
                PermutedDimsArray(x, (1,2,3))
            end
            @test !(xp isa StridedArray)
            y = norm(x)
            ẋ = rand(T, size(xp)) # rand_tangent(xp)
            x̄ = rand(T, size(xp)) # rand_tangent(xp)
            ȳ = rand_tangent(y)
            frule_test(norm, (xp, ẋ))
            rrule_test(norm, ȳ, (xp, x̄))
        end
    end
    @testset "$fnorm(x::Array{$T,$(length(sz))}, $p) with size $sz" for
        fnorm in (norm, LinearAlgebra.normp),
        p in (1.0, 2.0, Inf, -Inf, 2.5),
        T in (Float64, ComplexF64),
        sz in (fnorm === norm ? [(0,), (3,), (3, 3), (3, 2, 1)] : [(3,), (3, 3), (3, 2, 1)])

        x = randn(T, sz)
        # finite differences is unstable if maxabs (minabs) values are not well
        # separated from other values
        if p == Inf
            if !isempty(x)
                x[end] = 1000rand(T)
            end
            kwargs = (atol=1e-5, rtol=1e-5)
        elseif p == -Inf
            if !isempty(x)
                x .*= 1000
                x[end] = rand(T)
            end
            kwargs = (atol=1e-5, rtol=1e-5)
        else
            kwargs = NamedTuple()
        end


        test_rrule(fnorm, x, p; kwargs...)
        x isa Matrix && @testset "$MT" for MT in (Diagonal, UpperTriangular, LowerTriangular)
            test_rrule(fnorm, MT(x), p;
                #Don't check inference on old julia, what matters is that works on new
                check_inferred=VERSION>=v"1.5", kwargs...
            )
        end

        ȳ = rand_tangent(fnorm(x, p))
        @test extern(rrule(fnorm, zero(x), p)[2](ȳ)[2]) ≈ zero(x)
        @test rrule(fnorm, x, p)[2](Zero())[2] isa Zero
        T == Float64 && sz == (3,) && @testset "Integer input, p=$p" begin
            x = [1,2,3]
            int_fwd, int_back = rrule(fnorm, x, p)
            float_fwd, float_back = rrule(fnorm, float(x), p)
            @test int_fwd ≈ float_fwd
            @test unthunk(unthunk(int_back(1.0)[2])) ≈ unthunk(unthunk(float_back(1.0)[2]))
        end
    end
    @testset "norm($fdual(::Vector{$T}), p)" for
        T in (Float64, ComplexF64),
        fdual in (adjoint, transpose)

        x = fdual(randn(T, 3))
        p = 2.5

        test_rrule(norm, x, p)
        ȳ = rand_tangent(norm(x, p))
        @test extern(rrule(norm, x, p)[2](ȳ)[2]) isa typeof(x)
    end
    @testset "norm(x::$T, p)" for T in (Float64, ComplexF64)
        @testset "p = $p" for p in (-1.0, 2.0, 2.5)
            test_frule(norm, randn(T), p)
            test_rrule(norm, randn(T), p)

            _, back = rrule(norm, randn(T), p)
            @test back(Zero()) == (NO_FIELDS, Zero(), Zero())
        end
        @testset "p = 0" begin
            p = 0.0
            x = randn(T)
            y = norm(x, p)
            ẋ, ṗ = rand_tangent.((x, p))
            x̄, p̄, ȳ = rand_tangent.((x, p, y))
            y_fwd, ẏ = frule((Zero(), ẋ, ṗ), norm, x, p)
            @test y_fwd == y
            @test iszero(ẏ)
            y_rev, back = rrule(norm, x, p)
            @test y_rev == y
            @test back(ȳ) == (NO_FIELDS, zero(x), Zero())
            @test back(Zero()) == (NO_FIELDS, Zero(), Zero())
        end
    end
end

@testset "normalize" begin
    @testset "x::Vector{$T}" for T in (Float64, ComplexF64)
        x = randn(T, 3)
        test_rrule(normalize, x)
        @test rrule(normalize, x)[2](Zero()) === (NO_FIELDS, Zero())
    end
    @testset "x::Vector{$T}, p=$p" for T in (Float64, ComplexF64),
        p in (1.0, 2.0, -Inf, Inf, 2.5) # skip p=0, since FD is unstable
        x = randn(T, 3)
        test_rrule(normalize, x, p)
        @test rrule(normalize, x, p)[2](Zero()) === (NO_FIELDS, Zero(), Zero())
    end
end
