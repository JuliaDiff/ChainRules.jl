# for sum(xs, weights) (#522)
Base.sum(xs::AbstractArray, weights::AbstractArray) = dot(xs, weights)
struct SumRuleConfig <: RuleConfig{Union{HasReverseMode}} end

@testset "Reductions" begin
    @testset "sum(::Tuple)" begin
        test_frule(sum, Tuple(rand(5)))
    end
    @testset "sum(x; dims=$dims)" for dims in (:, 2, (1,3))
        # Forward
        test_frule(sum, rand(5); fkwargs=(;dims=dims))
        test_frule(sum, rand(ComplexF64, 2,3,4); fkwargs=(;dims=dims))

        # Reverse
        test_rrule(sum, rand(5); fkwargs=(;dims=dims))
        test_rrule(sum, rand(ComplexF64, 2,3,4); fkwargs=(;dims=dims))

        # Structured matrices
        test_rrule(sum, rand(5)'; fkwargs=(;dims=dims))
        y, back = rrule(sum, UpperTriangular(rand(5,5)); dims=dims)
        unthunk(back(y*(1+im))[2]) isa UpperTriangular{Float64}
        @test_skip test_rrule(sum, UpperTriangular(rand(5,5)) ⊢ randn(5,5); fkwargs=(;dims=dims), check_inferred=false) # Problem: in add!!  Evaluated: isapprox

        # Boolean -- via @non_differentiable
        test_rrule(sum, randn(5) .> 0; fkwargs=(;dims=dims))
        
        # Function allowing for 2nd derivatives
        for x in (rand(5), rand(2,3,4))
            dy = maximum(x; dims=dims)
            test_frule(ChainRules._unsum, x, dy, dims)
            test_rrule(ChainRules._unsum, x, dy, dims)
        end

        # Arrays of arrays
        for x  in ([rand(ComplexF64, 3) for _ in 1:4], [rand(3) for _ in 1:2, _ in 1:3, _ in 1:4])
            test_rrule(sum, x; fkwargs=(;dims=dims), check_inferred=false)

            dy = sum(x; dims=dims)
            ddy = rrule(ChainRules._unsum, x, dy, dims)[2](x)[3]
            @test size(ddy) == size(dy)
        end
    end

    @testset "sum abs2" begin
        sizes = (3, 4, 7)
        @testset "dims = $dims" for dims in (:, 1)
            @testset "Array{$N, $T}" for N in eachindex(sizes), T in (Float64, ComplexF64)
                x = randn(T, sizes[1:N]...)
                test_frule(sum, abs2, x; fkwargs=(;dims=dims))
                test_rrule(sum, abs2, x; fkwargs=(;dims=dims))
            end

            # Boolean -- via @non_differentiable, test that this isn't ambiguous
            test_rrule(sum, abs2, randn(5) .> 0; fkwargs=(;dims=dims))
        end
    end  # sum abs2

    @testset "sum(f, xs)" begin
        # This calls back into AD
        test_rrule(sum, abs, [-4.0, 2.0, 2.0])
        test_rrule(sum, cbrt, randn(5))
        test_rrule(sum, Multiplier(2.0), [2.0, 4.0, 8.0])

        # Complex numbers
        test_rrule(sum, sqrt, rand(ComplexF64, 5))
        test_rrule(sum, abs, rand(ComplexF64, 3, 4))  # complex -> real

        # inference fails for array of arrays
        test_rrule(sum, sum, [[2.0, 4.0], [4.0,1.9]]; check_inferred=false)
        
        # dims kwarg
        test_rrule(sum, abs, [-2.0 4.0; 5.0 1.9]; fkwargs=(;dims=1))
        test_rrule(sum, abs, [-2.0 4.0; 5.0 1.9]; fkwargs=(;dims=2))
        test_rrule(sum, sqrt, rand(ComplexF64, 3, 4); fkwargs=(;dims=(1,)))

        test_rrule(sum, abs, @SVector[1.0, -3.0])

        # covectors
        x = [-4.0 2.0; 2.0 -1.0]
        test_rrule(sum, inv, x[1, :]')
        test_rrule(sum, inv, x[1:1, :]')
        test_rrule(sum, inv, transpose(view(x, 1, :)))

        # Make sure we preserve type for StaticArrays
        ADviaRuleConfig = ChainRulesTestUtils.ADviaRuleConfig
        _, pb = rrule(ADviaRuleConfig(), sum, abs, @SVector[1.0, -3.0])
        @test pb(1.0) isa Tuple{NoTangent, NoTangent, SVector{2, Float64}}
      
        # make sure we preserve type for Diagonal
        _, pb = rrule(ADviaRuleConfig(), sum, abs, Diagonal([1.0, -3.0]))
        @test pb(1.0)[3] isa Diagonal

        # Boolean -- via @non_differentiable, test that this isn't ambiguous
        test_rrule(sum, sqrt, randn(5) .> 0) 
        test_rrule(sum, sqrt, randn(5,5) .> 0; fkwargs=(;dims=1))
        # ... and Bool produced by function
        @test_skip test_rrule(sum, iszero, randn(5))  # DimensionMismatch("second dimension of A, 1, does not match length of x, 0")


        # Functions that return a Vector
        # see https://github.com/FluxML/Zygote.jl/issues/1074
        test_rrule(sum, make_two_vec, [1.0, 3.0, 5.0, 7.0])
        test_rrule(sum, make_two_vec, [1.0 2.0; 3.0 4.0])
        test_rrule(sum, make_two_vec, [1.0 2.0; 3.0 4.0]; fkwargs=(;dims=2))
        test_rrule(sum, make_two_vec, [1.0 2.0; 3.0 4.0]; fkwargs=(;dims=1))
        test_rrule(sum, make_two_vec, [1.0 2.0; 3.0 4.0]; fkwargs=(;dims=(3, 4)))
    end

    # https://github.com/JuliaDiff/ChainRules.jl/issues/522
    @testset "sum(xs, weights) (#522)" begin
        xs = rand(5)
        weights = rand(5)

        @test rrule(SumRuleConfig(), Base.sum, xs, weights) isa Nothing
    end

    @testset "prod" begin
        @testset "Array{$T}" for T in [Float64, ComplexF64]
            @testset "size = $sz, dims = $dims" for (sz, dims) in [
                ((12,), :), ((12,), 1),
                ((3,4), 1), ((3,4), 2), ((3,4), :), ((3,4), [1,2]),
                ((3,4,1), 1), ((3,2,2), 3), ((3,2,2), 2:3),
                ]
                x = randn(T, sz)
                test_rrule(prod, x; fkwargs=(dims=dims,), check_inferred=true)
                x[1] = 0
                test_rrule(prod, x; fkwargs=(dims=dims,), check_inferred=true)
                x[5] = 0
                test_rrule(prod, x; fkwargs=(dims=dims,), check_inferred=true)
                x[3] = x[7] = 0  # two zeros along some slice, for any dims
                test_rrule(prod, x; fkwargs=(dims=dims,), check_inferred=true)

                if ndims(x) == 3
                    xp = PermutedDimsArray(x, (3,2,1))  # not a StridedArray
                    test_rrule(prod, xp; fkwargs=(dims=dims,), check_inferred=true)
                end
            end

            @testset "structured wrappers" begin
                # Adjoint -- like PermutedDimsArray this may actually be used
                xa = adjoint(rand(T,4,4))
                test_rrule(prod, xa)
                test_rrule(prod, xa, fkwargs=(dims=2,))
                # use Adjoint for tangent
                test_rrule(prod, xa ⊢ rand(T,4,4)')
                test_rrule(prod, xa ⊢ rand(T,4,4)', fkwargs=(dims=2,))
                @test unthunk(rrule(prod, adjoint(rand(T,3,3)))[2](1.0)[2]) isa Matrix
                @test unthunk(rrule(prod, adjoint(rand(T,3,3)), dims=1)[2](ones(1,3))[2]) isa Matrix

                # Diagonal -- a stupid thing to do, product of zeros! Shouldn't be an error though:
                @test iszero(unthunk(rrule(prod, Diagonal(rand(T,3)))[2](1.0)[2]))
                @test iszero(unthunk(rrule(prod, Diagonal(rand(T,3)), dims=1)[2](ones(1,3))[2]))
                # does a division for the complex case, so is not necessarily exact
                @test isapprox(
                    unthunk(rrule(prod, Diagonal(rand(T,1)))[2](1.0)[2]), # 1x1 sparse matrix
                    hcat(1);
                    rtol=T <: Complex ? 2eps() : 0.0,
                )
                @test unthunk(rrule(prod, Diagonal(ones(T,2)), dims=1)[2](ones(1,2))[2]) == Diagonal([0.0 1; 1 0])

                # Triangular -- almost equally stupud
                @test iszero(unthunk(rrule(prod, UpperTriangular(rand(T,3,3)))[2](1.0)[2]))
                @test unthunk(rrule(prod, UpperTriangular(ones(T,2,2)))[2](1.0)[2]) == UpperTriangular([0.0 0; 1 0])

                # Symmetric -- at least this doesn't have zeros, still an unlikely combination
                xs = Symmetric(rand(T,4,4))
                @test unthunk(rrule(prod, Symmetric(T[1 2; -333 4]))[2](1.0)[2]) == [16 8; 8 4]
                # TODO debug why these fail  https://github.com/JuliaDiff/ChainRules.jl/issues/475
                @test_skip test_rrule(prod, xs)
                @test_skip test_rrule(prod, xs, fkwargs=(dims=2,))
            end
        end
        @testset "Array{Float32}, no zero entries" begin
            v = [1f-5, 1f-10, 1f-15, 1f-20]
            @test prod(v) == 0
            @test unthunk(rrule(prod, v)[2](1f0)[2]) == zeros(4)
            test_rrule(prod, v)
        end
    end # prod

    @testset "foldl(f, ::Array)" begin
        CFG = ChainRulesTestUtils.ADviaRuleConfig()

        # Simple
        y1, b1 = rrule(CFG, foldl, *, [1,2,3]; init=1)
        @test y1 == 6
        b1(7) == (NoTangent(), NoTangent(), [42, 21, 14])

        y2, b2 = rrule(CFG, foldl, *, [1 2; 0 4])  # without init, needs vcat
        @test y2 == 0
        b2(8) == (NoTangent(), NoTangent(), [0 0; 64 0])  # matrix, needs reshape

        # Test execution order
        c5 = Counter()
        y5, b5 = rrule(CFG, foldl, c5, [5,7,11])
        @test c5 == Counter(2)
        @test y5 == ((5 + 7)*1 + 11)*2 == foldl(Counter(), [5,7,11])
        @test b5(1) == (NoTangent(), NoTangent(), [12*32, 12*42, 22])
        @test c5 == Counter(42)

        c6 = Counter()
        y6, b6 = rrule(CFG, foldl, c6, [5,7,11], init=3)
        @test c6 == Counter(3)
        @test y6 == (((3 + 5)*1 + 7)*2 + 11)*3 == foldl(Counter(), [5,7,11], init=3)
        @test b6(1) == (NoTangent(), NoTangent(), [63*33*13, 43*13, 23])
        @test c6 == Counter(63)

        # Test gradient of function
        y7, b7 = rrule(CFG, foldl, Multiplier(3), [5,7,11])
        @test y7 == foldl((x,y)->x*y*3, [5,7,11])
        @test b7(1) == (NoTangent(), Tangent{Multiplier{Int64}}(x = 2310,), [693, 495, 315])
        # ForwardDiff.derivative(z -> foldl((x,y)->x*y*z, [5,7,11]), 3) == 2310
        # ForwardDiff.gradient(z -> foldl((x,y)->x*y*3, z), [5,7,11]) |> string

        y8, b8 = rrule(CFG, foldl, Multiplier(13), [5,7,11], init=3)
        @test y8 == 2_537_535 == foldl((x,y)->x*y*13, [5,7,11], init=3)
        @test b8(1) == (NoTangent(), Tangent{Multiplier{Int64}}(x = 585585,), [507507, 362505, 230685])
        # ForwardDiff.derivative(z -> foldl((x,y)->x*y*z, [5,7,11], init=3), 13)
        # ForwardDiff.gradient(z -> foldl((x,y)->x*y*13, z, init=3), [5,7,11]) |> string

        # Finite differencing
        test_rrule(foldl, /, 1 .+ rand(3,4))
        test_rrule(foldl, *, rand(ComplexF64,3,4); fkwargs=(; init=rand()), check_inferred=false)
        test_rrule(foldl, max, rand(3); fkwargs=(; init=999))
    end
    @testset "foldl(f, ::Tuple)" begin
        y1, b1 = rrule(CFG, foldl, *, (1,2,3); init=1)
        @test y1 == 6
        b1(7) == (NoTangent(), NoTangent(), Tangent{NTuple{3,Int}}(42, 21, 14))

        y2, b2 = rrule(CFG, foldl, *, (1,2,0,4))
        @test y2 == 0
        b2(8) == (NoTangent(), NoTangent(), Tangent{NTuple{4,Int}}(0, 0, 64, 0))

        # Finite differencing
        test_rrule(foldl, /, Tuple(1 .+ rand(5)))
        test_rrule(foldl, *, Tuple(rand(ComplexF64, 5)))
    end
end

@testset "Accumulations" begin
    @testset "cumprod" begin
        v = round.(10 .* randn(9), sigdigits=3)
        test_rrule(cumprod, v)
        v[3] = 0
        test_rrule(cumprod, v)
        v[6] = 0
        test_rrule(cumprod, v)

        @testset "higher dimensions, dims=$dims" for dims in (1,2,3)
            m = round.(10 .* randn(4,5), sigdigits=3)
            test_rrule(cumprod, m; fkwargs=(;dims=dims), atol=0.1)
            m[2,2] = 0
            m[2,4] = 0
            test_rrule(cumprod, m; fkwargs=(;dims=dims))

            t = round.(10 .* randn(3,3,3), sigdigits=3)
            test_rrule(cumprod, t; fkwargs=(;dims=dims))
            t[2,2,2] = 0
            t[2,3,3] = 0
            test_rrule(cumprod, t; fkwargs=(;dims=dims))
        end

        @testset "types" begin
            back = rrule(cumprod, [1, 2, 3])[2]  # rule allows integer input, but test_rrule does not
            @test unthunk(back(fill(0.5, 3))[2]) == [9/2, 2, 1]

            back = rrule(cumprod, PermutedDimsArray([1 2; 3 4], (2,1)); dims=1)[2]
            @test unthunk(back(ones(Float32, 2,2))[2]) == [3 5; 1 3]

            @test_throws Exception cumprod(Symmetric([1 2; 3 4]), dims=1) # forward pass fails, so can't test gradient

            back = rrule(cumprod, Diagonal([1, 2]); dims=1)[2]
            @test unthunk(back(fill(0.5, 2, 2))[2]) ≈ [1/2 0; 0 0]  # ProjectTo'd to Diagonal now
        end
    end
    @testset "accumulate(f, ::Array)" begin
        CFG = ChainRulesTestUtils.ADviaRuleConfig()

        # Simple
        y1, b1 = rrule(CFG, accumulate, *, [1,2,3,4]; init=1)
        @test y1 == [1, 2, 6, 24]
        @test b1([1,1,1,1]) == (NoTangent(), NoTangent(), [33, 16, 10, 6])

        y2, b2 = rrule(CFG, accumulate, /, [1 2; 3 4])
        @test y2 ≈ accumulate(/, [1 2; 3 4])
        @test b2(ones(2,2))[3] ≈ [1.5416666 -0.104166664; -0.18055555 -0.010416667]  atol=1e-6
        # ForwardDiff.gradient(x -> sum(accumulate(/,x)), Float32[1 2; 3 4]) |> string

        # Test execution order
        c3 = Counter()
        y3, b3 = rrule(CFG, accumulate, c3, [5,7,11]; init=3)
        @test c3 == Counter(3)
        @test y3 == [8, 30, 123] == accumulate(Counter(), [5,7,11]; init=3)
        @test b3([1,1,1]) == (NoTangent(), NoTangent(), [29169, 602, 23]) # the 23 is clear!

        c4 = Counter()
        y4, b4 = rrule(CFG, accumulate, c4, [5,7,11])
        @test c4 == Counter(2)
        @test y4 == [5, (5+7)*1, ((5+7)*1 + 11)*2] == accumulate(Counter(), [5,7,11])
        @test b4([1,1,1]) == (NoTangent(), NoTangent(), [417, 42*(1 + 12), 22])

        # Test gradient of function
        y7, b7 = rrule(CFG, accumulate, Multiplier(3), [5,7,11])
        @test y7 == accumulate((x,y)->x*y*3, [5,7,11])
        @test b7([1,1,1]) == (NoTangent(), Tangent{Multiplier{Int64}}(x = 2345,), [715, 510, 315])
        # ForwardDiff.derivative(z -> sum(accumulate((x,y)->x*y*z, [5,7,11])), 3) == 2345
        # ForwardDiff.gradient(z -> sum(accumulate((x,y)->x*y*3, z)), [5,7,11]) |> string

        y8, b8 = rrule(CFG, accumulate, Multiplier(13), [5,7,11], init=3)
        @test y8 == [195, 17745, 2537535] == accumulate((x,y)->x*y*13, [5,7,11], init=3)
        @test b8([1,1,1]) == (NoTangent(), Tangent{Multiplier{Int64}}(x = 588330,), [511095, 365040, 230685])
        # ForwardDiff.derivative(z -> sum(accumulate((x,y)->x*y*z, [5,7,11], init=3)), 13)
        # ForwardDiff.gradient(z -> sum(accumulate((x,y)->x*y*13, z, init=3)), [5,7,11]) |> string

        # Finite differencing
        test_rrule(accumulate, *, randn(5); fkwargs=(; init=rand()))
        test_rrule(accumulate, /, 1 .+ rand(3,4))
        test_rrule(accumulate, ^, 1 .+ rand(2,3); fkwargs=(; init=rand()))
    end
    @testset "accumulate(f, ::Tuple)" begin
        # Simple
        y1, b1 = rrule(CFG, accumulate, *, (1,2,3,4); init=1)
        @test y1 == (1, 2, 6, 24)
        @test b1((1,1,1,1)) == (NoTangent(), NoTangent(), Tangent{NTuple{4,Int}}(33, 16, 10, 6))

        # Finite differencing
        test_rrule(accumulate, *, Tuple(randn(5)); fkwargs=(; init=rand()))
        test_rrule(accumulate, /, Tuple(1 .+ rand(5)); check_inferred=false)
    end
end
