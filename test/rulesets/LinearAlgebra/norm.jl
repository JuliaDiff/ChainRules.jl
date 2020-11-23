@testset "norm functions" begin
    @testset "$fnorm(x::Array{$T,$(length(sz))})" for
        fnorm in (
            LinearAlgebra.norm1,
            LinearAlgebra.norm2,
            LinearAlgebra.normInf,
            LinearAlgebra.normMinusInf,
        ),
        T in (Float64, ComplexF64),
        sz in [(3,), (3, 2), (3, 2, 1)]

        x = randn(T, sz)
        # finite differences is unstable if maxabs (minabs) values are not well
        # separated from other values
        if fnorm === LinearAlgebra.normInf
            x[3] = 1000rand(T)
        elseif fnorm == LinearAlgebra.normMinusInf
            x .*= 1000
            x[3] = rand(T)
        end

        y = fnorm(x)
        x̄ = rand_tangent(x)
        ȳ = rand_tangent(y)

        # fd has stability issues for the norm functions, so lower the required precision
        rrule_test(fnorm, ȳ, (x, x̄); rtol=1e-6)
        @test extern(rrule(fnorm, zero(x))[2](ȳ)[2]) ≈ zero(x)
        @test rrule(fnorm, x)[2](Zero())[2] isa Zero
    end
    @testset "norm(x::Array{$T,$(length(sz))})" for
        T in (Float64, ComplexF64),
        sz in [(3,), (3, 2), (3, 2, 1)]

        x = randn(T, sz)
        y = norm(x)
        ẋ = rand_tangent(x)
        x̄ = rand_tangent(x)
        ȳ = rand_tangent(y)

        frule_test(norm, (x, ẋ))
        @test frule((Zero(), Zero()), norm, x)[2] isa Zero
        @test iszero(frule((Zero(), ẋ), norm, zero(x))[2])
        rrule_test(norm, ȳ, (x, x̄))
        @test extern(rrule(norm, zero(x))[2](ȳ)[2]) ≈ zero(x)
        @test rrule(norm, x)[2](Zero())[2] isa Zero
    end
    @testset "$fnorm(x::Array{$T,$(length(sz))}, $p) with size $sz" for
        fnorm in (norm, LinearAlgebra.normp),
        p in (1.0, 2.0, Inf, -Inf, 1.5),
        T in (Float64, ComplexF64),
        sz in (fnorm === norm ? [(0,), (3,), (3, 2), (3, 2, 1)] : [(3,), (3, 2), (3, 2, 1)])

        x = randn(T, sz)
        !isempty(x) && if p == Inf
            x[3] = 1000rand(T)
        elseif p == -Inf
            x .*= 1000
            x[3] = rand(T)
        end

        y = fnorm(x, p)
        x̄ = rand_tangent(x)
        ȳ = rand_tangent(y)
        p̄ = rand_tangent(p)

        rrule_test(fnorm, ȳ, (x, x̄), (p, p̄); rtol=1e-6)
        @test extern(rrule(fnorm, zero(x), p)[2](ȳ)[2]) ≈ zero(x)
        @test rrule(fnorm, x, p)[2](Zero())[2] isa Zero
    end
    @testset "norm($fdual(::Vector{$T}), p)" for
        T in (Float64, ComplexF64),
        fdual in (adjoint, transpose)
        p = 1.5
        n = 3
        x = fdual(randn(T, n))
        y = norm(x, p)
        x̄ = rand_tangent(x)
        ȳ = rand_tangent(y)
        p̄ = rand_tangent(p)
        rrule_test(norm, ȳ, (x, x̄), (p, p̄))
        @test extern(rrule(norm, x, p)[2](ȳ)[2]) isa typeof(x)
    end
    @testset "norm(x::$T, p)" for T in (Float64, ComplexF64)
        @testset "p = $p" for p in (-1.0, 1.5, 2.0)
            x = randn(T)
            y = norm(x, p)
            ẋ, ṗ = rand_tangent.((x, p))
            x̄, p̄, ȳ = rand_tangent.((x, p, y))
            frule_test(norm, (x, ẋ), (p, ṗ))
            rrule_test(norm, ȳ, (x, x̄), (p, p̄))
            _, back = rrule(norm, x, p)
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
        n = 3
        x = randn(T, n)
        y = normalize(x)
        x̄ = rand_tangent(x)
        ȳ = rand_tangent(y)
        rrule_test(normalize, ȳ, (x, x̄))
    end
    @testset "x::Vector{$T}, p=$p" for T in (Float64, ComplexF64),
        p in (1.0, 2.0, -Inf, Inf, 1.5) # skip p=0, since FD is unstable
        n = 3
        x = randn(T, n)
        y = normalize(x, p)
        x̄ = rand_tangent(x)
        ȳ = rand_tangent(y)
        p̄ = rand_tangent(p)
        rrule_test(normalize, ȳ, (x, x̄), (p, p̄))
    end
end
