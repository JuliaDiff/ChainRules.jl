@testset "getindex" begin
    @testset "getindex(::Tuple, ...)" begin
        x = (1.2, 3.4, 5.6)
        x2 = (rand(2), (a=1.0, b=x))
        
        # Forward
        test_frule(getindex, x, 2)
        test_frule(getindex, x2, 1)
        test_frule(getindex, x, 1:2)
        test_frule(getindex, x2, :)
        
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
            test_rrule(getindex, x, 2, 1)
            test_rrule(getindex, x, 2, 2)

            test_rrule(getindex, x, CartesianIndex(2, 3))
        end

        @testset "slice/index postions" begin
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
        @test_skip test_rrule(getindex, Diagonal(rand(3)), 2, :)  # in-place update of off-diagonal entries
        dgrad = rrule(getindex, Diagonal(rand(3)), 2, :)[2]([1,2,3])[2]
        @test unthunk(dgrad) ≈ Diagonal([0, 2, 0])
        
        test_rrule(getindex, Symmetric(rand(3, 3)), 2, 2)
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
end

@testset "first & tail" begin
    x = (1.2, 3.4, 5.6)
    x2 = (rand(2), (a=1.0, b=x))

    test_frule(first, x)
    test_frule(first, x2)

    test_rrule(first, x)
    # test_rrule(first, x2) # MethodError: (::ChainRulesTestUtils.var"#test_approx##kw")(::NamedTuple{(:rtol, :atol), Tuple{Float64, Float64}}, ::typeof(test_approx), ::NoTangent, ::Tangent{NamedTuple{(:a, :b), Tuple{Float64, Tuple{Float64, Float64, Float64}}}, NamedTuple{(:a, :b), Tuple{Float64, Tangent{Tuple{Float64, Float64, Float64}, Tuple{Float64, Float64, Float64}}}}}, ::String) is ambiguous

    test_frule(Base.tail, x, check_inferred=false) # return type Tuple{Tuple{Float64, Float64}, Tangent{Tuple{Float64, Float64}, Tuple{Float64, Float64}}} does not match inferred return type Tuple{Tuple{Float64, Float64}, Tangent{Tuple{Float64, Float64}}}
    test_frule(Base.tail, x2, check_inferred=false)

    test_rrule(Base.tail, x)
    test_rrule(Base.tail, x2)
end

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

@testset "eachslice" begin
    # Testing eachrow not collect∘eachrow leads to errors, e.g.
    # test_rrule: eachrow on Vector{Float64}: Error During Test at /Users/me/.julia/packages/ChainRulesTestUtils/8dFTY/src/testers.jl:195
    #   Got exception outside of a @test
    #   DimensionMismatch("second dimension of A, 6, does not match length of x, 5")
    # Probably similar to https://github.com/JuliaDiff/ChainRulesTestUtils.jl/issues/234 (about Broadcasted not Generator)

    test_rrule(collect∘eachrow, rand(5))
    test_rrule(collect∘eachrow, rand(3, 4))

    test_rrule(collect∘eachcol, rand(3, 4))
    @test_skip test_rrule(collect∘eachcol, Diagonal(rand(5)))  # works locally!

    if VERSION >= v"1.7"
        # On 1.6, ComposedFunction doesn't take keywords. Only affects this testing strategy, not real use.
        test_rrule(collect∘eachslice, rand(3, 4, 5); fkwargs = (; dims = 3))
        test_rrule(collect∘eachslice, rand(3, 4, 5); fkwargs = (; dims = (2,)))
    end

    # Make sure pulling back an array that mixes some AbstractZeros in works right
    _, back = rrule(eachcol, rand(3, 4))
    @test back([1:3, ZeroTangent(), 7:9, NoTangent()]) == (NoTangent(), [1 0 7 0; 2 0 8 0; 3 0 9 0])
    @test back([1:3, ZeroTangent(), 7:9, NoTangent()])[2] isa Matrix{Float64}
    @test back([ZeroTangent(), ZeroTangent(), NoTangent(), NoTangent()]) == (NoTangent(), [0 0 0 0; 0 0 0 0; 0 0 0 0])

    # Second derivative rule
    test_rrule(ChainRules.∇eachslice, [rand(4) for _ in 1:3], rand(3, 4), Val(1))
    test_rrule(ChainRules.∇eachslice, [rand(3) for _ in 1:4], rand(3, 4), Val(2))
    test_rrule(ChainRules.∇eachslice, [rand(2, 3) for _ in 1:4], rand(2, 3, 4), Val(3), check_inferred=false)
end
