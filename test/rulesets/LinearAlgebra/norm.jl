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

        @testset "frule" begin
            xiter = TestIterator(x, Base.HasLength(), Base.HasEltype())
            ẋ = rand_tangent(x)
            ẋiter = rand_tangent(xiter)
            ṗ = rand_tangent(p)
            pṗ = isempty(p) ? () : ((p, ṗ),)

            if isempty(x) # finite differences can't handle empty x
                @test frule((Zero(), ẋ, ṗ), fnorm, x, p...) == (zero(T), zero(T))
                @testset "iterator" begin
                    @test frule(
                        (Zero(), ẋiter, ṗ),
                        fnorm,
                        xiter,
                        p...,
                    ) == (zero(T), zero(T))
                end
                @test frule(
                    (Zero(), Zero(), Zero()),
                    fnorm,
                    x,
                    p...,
                ) == (zero(T), Zero())
                continue
            end

            frule_test(norm, (x, ẋ), pṗ...)
            @testset "iterator" begin
                frule_test(norm, (xiter, ẋiter), pṗ...)
            end
            @test frule((Zero(), ẋ, ṗ), norm, zero(x), p...)[2] ≈ 0
            @test frule((Zero(), Zero(), Zero()), norm, x, p...)[2] isa Zero
        end
        @testset "rrule" begin
            y = fnorm(x, p...)
            x̄ = rand_tangent(x)
            ȳ = rand_tangent(y)
            pp̄ = isempty(p) ? () : ((p, rand_tangent(p)),)

            # fd has stability issues for the norm functions, so we use high order 10
            rrule_test(norm, ȳ, (x, x̄), pp̄...; fdm = central_fdm(10, 1))
            @test extern(rrule(norm, zero(x), p...)[2](ȳ)[2]) ≈ zero(x)
            @test rrule(norm, x, p...)[2](Zero())[2] isa Zero
        end
    end
    @testset "norm(x::$T[, p])" for T in (Float64, ComplexF64)
        @testset "norm(x::$T)" for T in (Float64, ComplexF64)
            test_scalar(norm, randn(T))
            test_scalar(norm, zero(T))
        end
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
            y_fwd, ẏ_fwd = frule((Zero(), ẋ, ṗ), norm, x, p)
            @test y_fwd == y
            @test iszero(ẏ_fwd)
            y_rev, back = rrule(norm, x, p)
            @test y_rev == y
            @test back(ȳ) == (NO_FIELDS, zero(x), Zero())
            @test back(Zero()) == (NO_FIELDS, Zero(), Zero())
        end
    end
end
@testset "$f with p=$p" for f in (normalize, normalize!), T in (Float64, ComplexF64),
    p in ((), 1.0, 2.0, -Inf, Inf, 1.5) # skip p=0, since FD is unstable

    n = 3
    @testset "frule" begin
        x = randn(T, n)
        ẋ = rand_tangent(x)
        ṗ = rand_tangent(p)
        pṗ = isempty(p) ? () : ((p, ṗ),)
        y = f(copy(x), p...)
        # `frule_test` doesn't handle mutating functions yet
        xcopy = copy(x)
        ẋcopy = copy(ẋ)
        (y_fwd, ẏ_fwd) = frule((Zero(), ẋcopy, ṗ), f, xcopy, p...)
        if f === normalize!
            @test y_fwd === xcopy
            @test ẏ_fwd === ẋcopy
        end
        @test y_fwd ≈ y
        ẏ_fd = jvp(_fdm, (x, p...) -> f(copy(x), p...), (x, ẋ), pṗ...)
        @test ẏ_fwd ≈ ẏ_fd
    end
    f === normalize && @testset "rrule" begin
        x = randn(T, n)
        y = f(copy(x), p...)
        x̄ = rand_tangent(x)
        ȳ = rand_tangent(y)
        pp̄ = isempty(p) ? () : ((p, rand_tangent(p)),)
        rrule_test(f, ȳ, (x, x̄), pp̄...)
    end
end
