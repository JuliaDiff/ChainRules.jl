@testset "linalg" begin
    @testset "dot" begin
        @testset "Vector{$T}" for T in (Float64, ComplexF64)
            M = 3
            x, y = randn(T, M), randn(T, M)
            ẋ, ẏ = randn(T, M), randn(T, M)
            x̄, ȳ = randn(T, M), randn(T, M)
            frule_test(dot, (x, ẋ), (y, ẏ))
            rrule_test(dot, randn(T), (x, x̄), (y, ȳ))
        end
        @testset "Matrix{$T}" for T in (Float64, ComplexF64)
            M, N = 3, 4
            x, y = randn(T, M, N), randn(T, M, N)
            ẋ, ẏ = randn(T, M, N), randn(T, M, N)
            x̄, ȳ = randn(T, M, N), randn(T, M, N)
            frule_test(dot, (x, ẋ), (y, ẏ))
            rrule_test(dot, randn(T), (x, x̄), (y, ȳ))
        end
        @testset "Array{$T, 3}" for T in (Float64, ComplexF64)
            M, N, P = 3, 4, 5
            x, y = randn(T, M, N, P), randn(T, M, N, P)
            ẋ, ẏ = randn(T, M, N, P), randn(T, M, N, P)
            x̄, ȳ = randn(T, M, N, P), randn(T, M, N, P)
            frule_test(dot, (x, ẋ), (y, ẏ))
            rrule_test(dot, randn(T), (x, x̄), (y, ȳ))
        end
    end
    @testset "cross" begin
        @testset "frule" begin
            @testset "$T" for T in (Float64, ComplexF64)
                n = 3
                x, y = randn(T, n), randn(T, n)
                ẋ, ẏ = randn(T, n), randn(T, n)
                frule_test(cross, (x, ẋ), (y, ẏ))
            end
        end
        @testset "rrule" begin
            n = 3
            x, y = randn(n), randn(n)
            x̄, ȳ = randn(n), randn(n)
            ΔΩ = randn(n)
            rrule_test(cross, ΔΩ, (x, x̄), (y, ȳ))
        end
    end
    @testset "inv(::Matrix{$T})" for T in (Float64, ComplexF64)
        N = 3
        B = generate_well_conditioned_matrix(T, N)
        frule_test(inv, (B, randn(T, N, N)))
        rrule_test(inv, randn(T, N, N), (B, randn(T, N, N)))
    end
    @testset "pinv" begin
        @testset "$T" for T in (Float64, ComplexF64)
            test_scalar(pinv, randn(T))
            @test frule((Zero(), randn(T)), pinv, zero(T))[2] ≈ zero(T)
            @test rrule(pinv, zero(T))[2](randn(T))[2] ≈ zero(T)
        end
        @testset "Vector{$T}" for T in (Float64, ComplexF64)
            n = 3
            x, ẋ, x̄ = randn(T, n), randn(T, n), randn(T, n)
            tol, ṫol, t̄ol = 0.0, randn(), randn()
            Δy = copyto!(similar(pinv(x)), randn(T, n))
            frule_test(pinv, (x, ẋ), (tol, ṫol))
            @test frule((Zero(), ẋ), pinv, x)[2] isa typeof(pinv(x))
            rrule_test(pinv, Δy, (x, x̄), (tol, t̄ol))
            @test rrule(pinv, x)[2](Δy)[2] isa typeof(x)
        end
        @testset "$F{Vector{$T}}" for T in (Float64, ComplexF64), F in (Transpose, Adjoint)
            n = 3
            x, ẋ, x̄ = F(randn(T, n)), F(randn(T, n)), F(randn(T, n))
            y = pinv(x)
            Δy = copyto!(similar(y), randn(T, n))
            frule_test(pinv, (x, ẋ))
            y_fwd, ∂y_fwd = frule((Zero(),  ẋ), pinv, x)
            @test y_fwd isa typeof(y)
            @test ∂y_fwd isa typeof(y)
            rrule_test(pinv, Δy, (x, x̄))
            y_rev, back = rrule(pinv, x)
            @test y_rev isa typeof(y)
            @test back(Δy)[2] isa typeof(x)
        end
        @testset "Matrix{$T} with size ($m,$n)" for T in (Float64, ComplexF64),
            m in 1:3,
            n in 1:3

            X, Ẋ, X̄ = randn(T, m, n), randn(T, m, n), randn(T, m, n)
            ΔY = randn(T, size(pinv(X))...)
            frule_test(pinv, (X, Ẋ))
            rrule_test(pinv, ΔY, (X, X̄))
        end
    end
    @testset "det(::Matrix{$T})" for T in (Float64, ComplexF64)
        N = 3
        B = generate_well_conditioned_matrix(T, N)
        frule_test(det, (B, randn(T, N, N)))
        rrule_test(det, randn(T), (B, randn(T, N, N)))
    end
    @testset "logdet(::Matrix{$T})" for T in (Float64, ComplexF64)
        N = 3
        B = generate_well_conditioned_matrix(T, N)
        frule_test(logdet, (B, randn(T, N, N)))
        rrule_test(logdet, randn(T), (B, randn(T, N, N)))
    end
    @testset "tr" begin
        N = 4
        frule_test(tr, (randn(N, N), randn(N, N)))
        rrule_test(tr, randn(), (randn(N, N), randn(N, N)))
    end
    @testset "*" begin
        dims = [3,4,5]
        for n in dims, m in dims, p in dims
            n > 3 && n == m == p && continue  # don't need to test square case multiple times
            A = randn(m, n)
            B = randn(n, p)
            Ȳ = randn(m, p)
            rrule_test(*, Ȳ, (A, randn(m, n)), (B, randn(n, p)))
        end
    end
    @testset "$f" for f in [/, \]
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
        if f == (/)
            @testset "$T on the RHS" for T in (Diagonal, UpperTriangular, LowerTriangular)
                RHS = T(randn(T == Diagonal ? 10 : (10, 10)))
                Y = randn(5, 10)
                Ȳ = randn(size(f(Y, RHS))...)
                rrule_test(f, Ȳ, (Y, randn(size(Y))), (RHS, randn(size(RHS))))
            end
        else
            @testset "$T on LHS" for T in (Diagonal, UpperTriangular, LowerTriangular)
                LHS = T(randn(T == Diagonal ? 10 : (10, 10)))
                y = randn(10)
                ȳ = randn(size(f(LHS, y))...)
                rrule_test(f, ȳ, (LHS, randn(size(LHS))), (y, randn(10)))
                Y = randn(10, 10)
                Ȳ = randn(10, 10)
                rrule_test(f, Ȳ, (LHS, randn(size(LHS))), (Y, randn(size(Y))))
            end
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
end
