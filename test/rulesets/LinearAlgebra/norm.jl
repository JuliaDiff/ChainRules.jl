@testset "norm functions" begin
    @testset "$fnorm(x, $(p...)) with eltype $T and dimension $dims" for
        fnorm in (
            norm,
            LinearAlgebra.normp,
            LinearAlgebra.norm1,
            LinearAlgebra.norm2,
            LinearAlgebra.normInf,
            LinearAlgebra.normMinusInf,
        ),
        p in ((), 1.0, 2.0, Inf, -Inf, 1.5),
        T in (Float64, ComplexF64),
        dims in [(0,), (3,), (3, 2), (3, 2, 1)]

        # there is no default p for normp
        fnorm === LinearAlgebra.normp && isempty(p) && continue
        # the specialized norm functions don't take a p
        fnorm !== norm && fnorm !== LinearAlgebra.normp && !isempty(p) && continue
        # only norm can take empty iterators
        prod(dims) == 0 && fnorm !== norm && continue

        x = randn(T, dims...)
        # finite differences is unstable if maxabs (minabs) values are not well
        # separated from other values
        !isempty(x) && if p == Inf
            x[3] = 1000rand(T)
        elseif p == -Inf
            x .*= 1000
            x[3] = rand(T)
        end

        y = fnorm(x, p...)
        x̄ = rand_tangent(x)
        ȳ = rand_tangent(y)
        pp̄ = isempty(p) ? () : ((p, rand_tangent(p)),)

        # fd has stability issues for the norm functions, so we use high order 10
        rrule_test(norm, ȳ, (x, x̄), pp̄...; fdm = central_fdm(10, 1))
        @test extern(rrule(norm, zero(x), p...)[2](ȳ)[2]) ≈ zero(x)
        @test rrule(norm, x, p...)[2](Zero())[2] isa Zero
    end
    @testset "norm(x::$T[, p])" for T in (Float64, ComplexF64)
        @testset "norm(x::$T)" for T in (Float64, ComplexF64)
            test_scalar(norm, randn(T))
            test_scalar(norm, zero(T))
        end
        @testset "p = $p" for p in (-1.0, 1.5, 2.0)
            x = randn(T)
            y = norm(x, p)
            x̄, p̄, ȳ = rand_tangent.((x, p, y))
            rrule_test(norm, ȳ, (x, x̄), (p, p̄))
            _, back = rrule(norm, x, p)
            @test back(Zero()) == (NO_FIELDS, Zero(), Zero())
        end
        @testset "p = 0" begin
            p = 0.0
            x = randn(T)
            y = norm(x, p)
            x̄, p̄, ȳ = rand_tangent.((x, p, y))
            y_rev, back = rrule(norm, x, p)
            @test y_rev == y
            @test back(ȳ) == (NO_FIELDS, zero(x), Zero())
            @test back(Zero()) == (NO_FIELDS, Zero(), Zero())
        end
    end
end
@testset "normalize with T=$T, p=$p" for T in (Float64, ComplexF64),
    p in ((), 1.0, 2.0, -Inf, Inf, 1.5) # skip p=0, since FD is unstable
    n = 3
    x = randn(T, n)
    y = normalize(copy(x), p...)
    x̄ = rand_tangent(x)
    ȳ = rand_tangent(y)
    pp̄ = isempty(p) ? () : ((p, rand_tangent(p)),)
    rrule_test(normalize, ȳ, (x, x̄), pp̄...)
end
