struct FooTwoField
    x::Float64
    y::Float64
end
 

@testset "getfield" begin
   test_frule(getfield, FooTwoField(1.5, 2.5), :x, check_inferred=false)
    
    test_frule(getfield, (; a=1.5, b=2.5), :a, check_inferred=false)
    test_frule(getfield, (; a=1.5, b=2.5), 2)

    test_frule(getfield, (1.5, 2.5), 2)
    test_frule(getfield, (1.5, 2.5), 2, true)
end

@testset "getindex" begin
    @testset "getindex(::Tuple, ...)" begin
        x = (1.2, 3.4, 5.6)
        x2 = (rand(2), (a=1.0, b=x))
        
        # don't test Forward because this will be handled by lowering to getfield
        # Reverse
        test_rrule(getindex, x, 2)
        @test_skip test_rrule(getindex, x2, 1, check_inferred=false)  # method ambiguity, maybe fixed by https://github.com/JuliaDiff/ChainRulesTestUtils.jl/pull/253
    
        test_rrule(getindex, x, 2:3; check_inferred=false)
        test_rrule(getindex, x, [1, 1, 2], check_inferred=false)
        test_rrule(getindex, x2, 1:2, check_inferred=false)
        
        test_rrule(getindex, x, :)
    end
    
    @testset "getindex(::Matrix{<:Number}, ...)" begin
        x = [1.0 2.0 3.0; 10.0 20.0 30.0]
        
        @testset "forward mode" begin
            test_frule(getindex, x, 2)
            test_frule(getindex, x, 2, 1)
            test_frule(getindex, x, CartesianIndex(2, 3))

            test_frule(getindex, x, 2:3)
            test_frule(getindex, x, (:), 2:3)
        end

        @testset "single element" begin
            test_rrule(getindex, x, 2)
            test_rrule(getindex, x, 2, 1; check_inferred=false)
            test_rrule(getindex, x, 2, 2; check_inferred=false)

            test_rrule(getindex, x, CartesianIndex(2, 3); check_inferred=false)
        end

        @testset "slice/index positions" begin
            test_rrule(getindex, x, 2:3)
            test_rrule(getindex, x, 3:-1:2)
            test_rrule(getindex, x, [3,2])
            test_rrule(getindex, x, [2,3])

            test_rrule(getindex, x, 1:2, 2:3)
            test_rrule(getindex, x, (:), 2:3)

            test_rrule(getindex, x, 1:2, 1)
            test_rrule(getindex, x, 1, 1:2)

            test_rrule(getindex, x, 1:2, 2:3)
            test_rrule(getindex, x, (:), 2:3)

            test_rrule(getindex, x, (:), (:))
            test_rrule(getindex, x, (:))
        end

        @testset "masking" begin
            test_rrule(getindex, x, trues(size(x)))
            test_rrule(getindex, x, trues(length(x)))

            mask = falses(size(x))
            mask[2,3] = true
            mask[1,2] = true
            test_rrule(getindex, x, mask)

            test_rrule(getindex, x, [true, false], (:))
        end

        @testset "By position with repeated elements" begin
            test_rrule(getindex, x, [2, 2])
            test_rrule(getindex, x, [2, 2, 2])
            test_rrule(getindex, x, [2,2], [3,3])
        end
    end
    
    @testset "getindex for structured arrays" begin
        test_frule(getindex, Diagonal(rand(3)), 1)
        test_frule(getindex, Symmetric(rand(3, 3)), 2, 3)
        
        test_rrule(getindex, Diagonal(rand(3)), 1)
        @test_skip test_rrule(getindex, Diagonal(rand(3)), 2, :)  # https://github.com/JuliaDiff/ChainRulesTestUtils.jl/issues/260
        dgrad = rrule(getindex, Diagonal(rand(3)), 2, :)[2]([1,2,3])[2]
        @test unthunk(dgrad) ≈ Diagonal([0, 2, 0])
        
        test_rrule(getindex, Symmetric(rand(3, 3)), 2, 2; check_inferred=false)  # Infers to Any
        sgrad = rrule(getindex, Symmetric(rand(3, 3)), 2, 3)[2](1.0)[2]
        @test unthunk(sgrad) ≈ [0 0 0; 0 0 1/2; 0 1/2 0]
    end
    
    @testset "getindex(::Array{<:Array})" begin
        test_frule(getindex, [rand(2) for _ in 1:3], 1)
        test_frule(getindex, [rand(2) for _ in 1:3], 2:3)
        test_frule(getindex, [rand(2) for _ in 1:3], [true, false, true])
        
        test_rrule(getindex, [rand(2) for _ in 1:3], 1; check_inferred=false)
        test_rrule(getindex, [rand(2) for _ in 1:3], 2:3; check_inferred=false)
        test_frule(getindex, [rand(2) for _ in 1:3], [true, false, true]; check_inferred=false)
    end
    
    @testset "getindex(::Array{<:Weird})" begin
        xfix = [Base.Fix1(*, pi), Base.Fix1(^, ℯ), Base.Fix1(/, -1)]
        dxfix = [Tangent{Base.Fix1}(; x = i/10) for i in 1:3]
        # test_frule(getindex, xfix ⊢ dxfix, 1)
        # test_rrule(getindex, xfix ⊢ dxfix, 1)
        
        dx1 = unthunk(rrule(getindex, xfix, 1)[2](dxfix[1])[2])
        @test dx1[1] == dxfix[1]
        @test iszero(dx1[2])
        
        dx23 = unthunk(rrule(getindex, xfix, 2:3)[2](dxfix[2:3])[2])
        @test iszero(dx23[1])
        @test dx23[3] == dxfix[3]
    end

    @testset "second derivatives: ∇getindex" begin
        @eval using ChainRules: ∇getindex
        # Forward, scalar result
        test_frule(∇getindex, rand(2, 3), rand(), 3)
        test_frule(∇getindex, rand(2, 3), rand()+im, 2, 1)
        # array result
        test_frule(∇getindex, rand(2, 3), rand(2), 4:5)
        test_frule(∇getindex, rand(2, 3), rand(3), 1, :)
        test_frule(∇getindex, rand(2, 3), rand(1, 2), [CartesianIndex(2, 1) CartesianIndex(2, 2)]  ⊢ NoTangent())
        test_frule(∇getindex, rand(2, 3), rand(3), Bool[1 0 1; 0 1 0])
        # arrays of arrays
        test_frule(∇getindex, [rand(2) for _ in 1:3], rand(2), 3)
        test_frule(∇getindex, [rand(2) for _ in 1:3], [rand(2), rand(2)], 1:2)

        # Reverse, scalar result
        test_rrule(∇getindex, rand(2, 3), rand(), 3; check_inferred=false)
        test_rrule(∇getindex, rand(2, 3), rand()+im, 2, 1; check_inferred=false)
        # array result
        test_rrule(∇getindex, rand(2, 3), rand(2), 4:5; check_inferred=false)
        test_rrule(∇getindex, rand(2, 3), rand(3), 1, :; check_inferred=false)
        test_rrule(∇getindex, rand(2, 3), rand(1, 2), [CartesianIndex(2, 1) CartesianIndex(2, 2)]  ⊢ NoTangent(); check_inferred=false)
        test_rrule(∇getindex, rand(2, 3), rand(3), Bool[1 0 1; 0 1 0]; check_inferred=false)
        # arrays of arrays
        test_rrule(∇getindex, [rand(2) for _ in 1:3], rand(2), 3; check_inferred=false)
        test_rrule(∇getindex, [rand(2) for _ in 1:3], [rand(2), rand(2)], 1:2; check_inferred=false)
    end

    @testset "getindex(::AbstractGPUArray)" begin
        x_23_gpu = jl(rand(2, 3))  # using JLArrays, loaded for @gpu in test_helpers.jl
    
        # Scalar indexing, copied from:  @macroexpand @allowscalar A[i]
        @test_skip begin  # This gives 
          y1, bk1 = rrule(CFG, Base.task_local_storage, () -> x_23_gpu[1], :ScalarIndexing, ScalarAllowed)
          @test y1 == @allowscalar x_23_gpu[1]
        end
        @test_skip begin
          bk1(1.0) # This gives a StackOverflowError! Or gives zero in global scope.
          true
        end
        # But this works, and calls the rule:
        # Zygote.gradient(x -> @allowscalar(x[1]), jl(rand(3)))[1]

        y2, bk2 = rrule(getindex, x_23_gpu, :, 2:3)  # fast path, just broadcast .+=
        @test unthunk(bk2(jl(ones(2,2)))[2]) == jl([0 1 1; 0 1 1])

        y3, bk3 = rrule(getindex, x_23_gpu, 1, [1,1,2])  # slow path, copy to CPU
        @test Array(y3) == Array(x_23_gpu)[1, [1,1,2]]
        @test unthunk(bk3(jl(ones(3)))[2]) == jl([2 1 0; 0 0 0])
    end

    @testset "getindex(::Array{<:AbstractGPUArray})" begin
        x_gpu = jl(rand(1))
        y, back = rrule(getindex, [x_gpu], 1)
        @test y === x_gpu
        dxs_gpu = unthunk(back(jl([1.0]))[2])
        @test dxs_gpu == [jl([1.0])]
    end
end

# first & tail handled by getfield rules

@testset "view" begin
    test_frule(view, rand(3, 4), :, 1)
    test_frule(view, rand(3, 4), 2, [1, 1, 2])
    test_frule(view, rand(3, 4), 3, 4)

    test_rrule(view, rand(3, 4), :, 1)
    test_rrule(view, rand(3, 4), 2, [1, 1, 2])
    test_rrule(view, rand(3, 4), 3, 4)
end

@testset "setindex!" begin
    test_frule(setindex!, rand(3, 4), rand(), 1, 2)
    test_frule(setindex!, rand(3, 4), [1,10,100.0], :, 3)
end

@testset "unsafe_getindex" begin
    # In real life this is called only on some AbstractRanges, but easier to test on Array:
    test_frule(Base.unsafe_getindex, collect(1:0.1:2), 3)
    test_rrule(Base.unsafe_getindex, collect(1:0.1:2), 3)
end

@testset "eachslice" begin
    # Testing eachrow not collect∘eachrow leads to errors, e.g.
    # test_rrule: eachrow on Vector{Float64}: Error During Test at /Users/me/.julia/packages/ChainRulesTestUtils/8dFTY/src/testers.jl:195
    #   Got exception outside of a @test
    #   DimensionMismatch("second dimension of A, 6, does not match length of x, 5")
    # Probably similar to https://github.com/JuliaDiff/ChainRulesTestUtils.jl/issues/234 (about Broadcasted not Generator)

    # Inference on 1.6 sometimes fails, so don't enforce there.
    test_rrule(collect ∘ eachrow, rand(5); check_inferred=(VERSION >= v"1.7"))
    test_rrule(collect ∘ eachrow, rand(3, 4); check_inferred=(VERSION >= v"1.7"))

    test_rrule(collect ∘ eachcol, rand(3, 4); check_inferred=(VERSION >= v"1.7"))
    @test_skip test_rrule(collect ∘ eachcol, Diagonal(rand(5)))  # works locally!

    if VERSION >= v"1.7"
        # On 1.6, ComposedFunction doesn't take keywords. Only affects this testing strategy, not real use.
        test_rrule(collect ∘ eachslice, rand(3, 4, 5); fkwargs=(; dims=3))
        test_rrule(collect ∘ eachslice, rand(3, 4, 5); fkwargs=(; dims=(2,)))

        test_rrule(
            collect ∘ eachslice,
            FooTwoField.(rand(3, 4, 5), rand(3, 4, 5));
            check_inferred=false,
            fkwargs=(; dims=3),
        )
    end

    # Make sure pulling back an array that mixes some AbstractZeros in works right
    _, back = rrule(eachcol, rand(3, 4))
    @test back([1:3, ZeroTangent(), 7:9, NoTangent()]) == (NoTangent(), [1 0 7 0; 2 0 8 0; 3 0 9 0])
    @test back([1:3, ZeroTangent(), 7:9, NoTangent()])[2] isa Matrix{Float64}
    @test back([ZeroTangent(), ZeroTangent(), NoTangent(), NoTangent()]) == (NoTangent(), [0 0 0 0; 0 0 0 0; 0 0 0 0])

    _, back = ChainRules.rrule(
        eachslice, FooTwoField.(rand(2, 3, 2), rand(2, 3, 2)); dims=3
    )
    @test back([fill(Tangent{Any}(; x=0.0, y=1.0), 2, 3), fill(ZeroTangent(), 2, 3)]) == (
        NoTangent(),
        cat(fill(Tangent{Any}(; x=0.0, y=1.0), 2, 3), fill(ZeroTangent(), 2, 3); dims=3),
    )

    # Second derivative rule
    test_rrule(ChainRules.∇eachslice, [rand(4) for _ in 1:3], rand(3, 4), Val(1))
    test_rrule(ChainRules.∇eachslice, [rand(3) for _ in 1:4], rand(3, 4), Val(2))
    test_rrule(
        ChainRules.∇eachslice,
        [rand(2, 3) for _ in 1:4],
        rand(2, 3, 4),
        Val(3);
        check_inferred=(VERSION >= v"1.7"),
    )

    # eachslice: Make sure pulling back an array of thunks unthunks them and does not return all zeros.
    x = ones(Float32, 3)
    Δ = ones(Float32, 1)
    _, norm_back = ChainRules.rrule(norm, x)
    dx = norm_back(Δ)[2]
    @test dx isa AbstractThunk

    x = ones(Float32, 3, 1)
    _, eachcol_back = ChainRules.rrule(eachcol, x)
    Δ2 = [dx]
    dx2 = eachcol_back(Δ2)[2]
    @test all(dx2 .≉ 0f0)
end
