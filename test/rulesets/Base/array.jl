@testset "Array constructors" begin
    @testset "undef" begin
    # We can't use test_rrule here (as it's currently implemented) because the elements of
    # the array have arbitrary values. The only thing we can do is ensure that we're getting
    # `ZeroTangent`s back, and that the forwards pass produces the correct thing still.
    # Issue: https://github.com/JuliaDiff/ChainRulesTestUtils.jl/issues/202
        val, pullback = rrule(Array{Float64}, undef, 5)
        @test size(val) == (5, )
        @test val isa Array{Float64, 1}
        @test pullback(randn(5)) == (NoTangent(), NoTangent(), NoTangent())
    end
    @testset "from existing array" begin
        # fwd
        test_frule(Array, randn(2, 5))
        test_frule(Array, Diagonal(randn(5)))
        # rev
        test_rrule(Array, randn(2, 5))
        test_rrule(Array, Diagonal(randn(5)))
        test_rrule(Matrix, Diagonal(randn(5)))
        test_rrule(Matrix, transpose(randn(4)))
        test_rrule(Array{ComplexF64}, randn(3))
    end
end
@testset "AbstractArray constructors" begin
    # These are what float(x) calls, but it's trivial with floating point numbers:
    test_frule(AbstractArray{Float32}, rand(3); atol=0.01)
    test_frule(AbstractArray{Float32}, Diagonal(rand(4)); atol=0.01)
    # rev
    test_rrule(AbstractArray{Float32}, rand(3); atol=0.01)
    test_rrule(AbstractArray{Float32}, Diagonal(rand(4)); atol=0.01)
    # Check with integers:
    rrule(AbstractArray{Float64}, [1, 2, 3])[2]([1, 10, 100]) == (NoTangent(), [1.0, 10.0, 100.0])
end

@testset "vect" begin
    test_rrule(Base.vect)
    @testset "homogeneous type" begin
        test_rrule(Base.vect, (5.0,), (4.0,))
        test_frule(Base.vect, (5.0,), (4.0,))
        test_rrule(Base.vect, 5.0, 4.0, 3.0)
        test_frule(Base.vect, 5.0, 4.0, 3.0)
        test_rrule(Base.vect, randn(2, 2), randn(3, 3))
        test_frule(Base.vect, randn(2, 2), randn(3, 3))

        # Nonnumber types
        test_frule(Base.vect, (1.0, 2.0), (1.0, 2.0))
        test_rrule(Base.vect, (1.0, 2.0), (1.0, 2.0))
    end
    @testset "inhomogeneous type" begin
        # fwd
        test_frule(Base.vect, 5.0, 3f0)
        # rev
        test_rrule(
            Base.vect, 5.0, 3f0;
            atol=1e-6, rtol=1e-6,
        ) # tolerance due to Float32.
        test_rrule(Base.vect, 5.0, randn(3, 3); check_inferred=false)
        test_rrule(Base.vect, (5.0, 4.0), (y=randn(3),); check_inferred=false)
    end
    @testset "_instantiate_zeros" begin
        # This is an internal function also used for `cat` etc.
        _instantiate_zeros = ChainRules._instantiate_zeros
        # Check these hit the fast path, unrealistic input so that map would fail:
        @test _instantiate_zeros((true, 2 , 3.0), ()) == (1, 2, 3)
        @test _instantiate_zeros((1:2, [3, 4]), ()) == (1:2, 3:4)
    end
end

@testset "copyto!" begin
    test_frule(copyto!, rand(5), rand(5))
    test_frule(copyto!, rand(10), 3, rand(5))
    test_frule(copyto!, rand(10), 2, rand(5), 2)
    test_frule(copyto!, rand(10), 2, rand(5), 2, 4)
end

@testset "reshape" begin
    # Forward
    @gpu test_frule(reshape, rand(4, 3), 2, :)
    test_frule(reshape, rand(4, 3), axes(rand(6, 2)))
    @test_skip test_frule(reshape, Diagonal(rand(4)), 2, :) # https://github.com/JuliaDiff/ChainRulesTestUtils.jl/issues/239

    # Reverse
    @gpu test_rrule(reshape, rand(4, 5), (2, 10))
    test_rrule(reshape, rand(4, 5), 2, 10)
    test_rrule(reshape, rand(4, 5), 2, :)
    test_rrule(reshape, rand(4, 5), axes(rand(10, 2)))
    # structured
    test_rrule(reshape, transpose(rand(4)), :)
    test_rrule(reshape, adjoint(rand(ComplexF64, 4)), :)
    @test rrule(reshape, adjoint(rand(ComplexF64, 4)), :)[2](rand(4))[2] isa Adjoint{ComplexF64}
    @test rrule(reshape, Diagonal(rand(4)), (2, :))[2](ones(2,8))[2] isa Diagonal
    @test_skip test_rrule(reshape, Diagonal(rand(4)), 2, :)  # DimensionMismatch("second dimension of A, 22, does not match length of x, 16")
    @test_skip test_rrule(reshape, UpperTriangular(rand(4,4)), (8, 2)) # https://github.com/JuliaDiff/ChainRulesTestUtils.jl/issues/239
end

@testset "dropdims" begin
    # fwd
    test_frule(dropdims, rand(4, 1); fkwargs=(; dims=2))
    # rev
    test_rrule(dropdims, rand(4, 1); fkwargs=(; dims=2))
    test_rrule(dropdims, transpose(rand(4)); fkwargs=(; dims=1))
    test_rrule(dropdims, adjoint(rand(ComplexF64, 4)); fkwargs=(; dims=1))
    @test rrule(dropdims, adjoint(rand(ComplexF64, 4)); dims=1)[2](rand(4))[2] isa Adjoint{ComplexF64}
end

@testset "permutedims + PermutedDimsArray" begin
    # Forward
    @gpu test_frule(permutedims, rand(5))
    @gpu test_frule(permutedims, rand(3, 4), (2, 1))
    test_frule(permutedims!, rand(4,3), rand(3, 4), (2, 1))
    test_frule(PermutedDimsArray, rand(3, 4, 5), (3, 1, 2))

    # Reverse
    @gpu test_rrule(permutedims, rand(5))
    @gpu test_rrule(permutedims, rand(3, 4), (2, 1))
    test_rrule(permutedims, Diagonal(rand(5)), (2, 1))
    # Note BTW that permutedims(Diagonal(rand(5))) does not use the rule at all

    @test invperm((3, 1, 2)) != (3, 1, 2)
    test_rrule(permutedims, rand(3, 4, 5), (3, 1, 2))

    @test_skip test_rrule(PermutedDimsArray, rand(3, 4, 5), (3, 1, 2))  # https://github.com/JuliaDiff/ChainRulesTestUtils.jl/issues/240
    x = rand(2, 3, 4)
    dy = rand(4, 2, 3)
    @test rrule(permutedims, x, (3, 1, 2))[2](dy)[2] == rrule(PermutedDimsArray, x, (3, 1, 2))[2](dy)[2]
end

@testset "repeat" begin
    # forward
    test_frule(repeat, rand(4), 2)
    test_frule(repeat, rand(2, 3); fkwargs = (inner=(1,2), outer=(1,3)))

    # reverse
    test_rrule(repeat, rand(4, ))
    test_rrule(repeat, rand(4, 5))
    test_rrule(repeat, rand(4, 5); fkwargs = (outer=(1,2),))
    @gpu_broken test_rrule(repeat, rand(4, 5); fkwargs = (inner=(1,2), outer=(1,3)))
    @gpu_broken test_rrule(repeat, rand(4, 5); fkwargs = (outer=2,))

    @gpu test_rrule(repeat, rand(4, ), 2)
    @gpu test_rrule(repeat, rand(4, 5), 2)
    @gpu test_rrule(repeat, rand(4, 5), 2, 3)
    test_rrule(repeat, rand(1,2,3), 2,3,4; check_inferred=VERSION>v"1.6")
    test_rrule(repeat, rand(0,2,3), 2,0,4; check_inferred=VERSION>v"1.6")
    test_rrule(repeat, rand(1,1,1,1), 2,3,4,5; check_inferred=VERSION>v"1.6")
    
    # These need Julia 1.6
    test_rrule(repeat, rand(4, 5); fkwargs = (inner=(2,4), outer=(1,1,1,3)))
    test_rrule(repeat, rand(1,2,3), 2,3)
    test_rrule(repeat, rand(1,2,3), 2,3,4,2)
    test_rrule(repeat, fill(1.0), 2)
    test_rrule(repeat, fill(1.0), 2, 3)

    # These fail for other v1.0 related issues (add!!)
    # v"1.0": fill(1.0) + fill(1.0) != fill(2.0)
    # v"1.6: fill(1.0) + fill(1.0) == fill(2.0) # Expected
    test_rrule(repeat, fill(1.0); fkwargs = (inner=2,))
    test_rrule(repeat, fill(1.0); fkwargs = (inner=2, outer=3,))


    @test rrule(repeat, [1,2,3], 4)[2](ones(12))[2] == [4,4,4]
    @test rrule(repeat, [1,2,3], outer=4)[2](ones(12))[2] == [4,4,4]

    test_rrule(repeat, [true, false], 3)
end

@testset "hcat" begin
    # forward
    @gpu test_frule(hcat, randn(3, 2), randn(3))
    @gpu test_frule(hcat, randn(), randn(1,3))

    # reverse
    @gpu test_rrule(hcat, randn(3, 2), randn(3), randn(3, 3))
    @gpu test_rrule(hcat, rand(1,2), rand(), rand(1,3))
    test_rrule(hcat, rand(), rand(1,2), rand(1,2,1))
    test_rrule(hcat, rand(3,1,1,2), rand(3,3,1,2))

    # mix types
    test_rrule(hcat, rand(1, 3), rand(2)')
    test_rrule(hcat, rand(1), (nothing, rand()), check_inferred=false)
end

@testset "reduce hcat" begin
    mats = [randn(3, 2), randn(3, 1), randn(3, 3)]
    test_frule(reduce, hcat, mats)
    test_rrule(reduce, hcat, mats)
    
    vecs = [rand(3) for _ in 1:4]
    test_frule(reduce, hcat, vecs)
    test_rrule(reduce, hcat, vecs)
    
    mix = AbstractVecOrMat[rand(4,2), rand(4)]  # this is weird, but does hit the fast path
    test_rrule(reduce, hcat, mix)

    adjs = vec([randn(2, 4), randn(1, 4), randn(3, 4)]')  # not a Vector
    # test_rrule(reduce, hcat, adjs ⊢ map(m -> rand(size(m)), adjs))
    dy = 1 ./ reduce(hcat, adjs)
    @test rrule(reduce, hcat, adjs)[2](dy)[3] ≈ rrule(reduce, hcat, collect.(adjs))[2](dy)[3]

    # mix types
    mats = [randn(2, 2), rand(2, 2)']
    test_rrule(reduce, hcat, mats)
end

@testset "vcat" begin
    # forward
    test_frule(vcat, randn(), randn(3), rand())
    @gpu test_frule(vcat, randn(3), rand(), randn(3))
    @gpu test_frule(vcat, randn(3, 1), randn(3))

    # reverse
    @gpu test_rrule(vcat, randn(3), rand(), randn(3))
    @gpu test_rrule(vcat, randn(2, 4), randn(1, 4), randn(3, 4))
    test_rrule(vcat, rand(), rand())
    test_rrule(vcat, rand(), rand(3), rand(3,1,1))
    test_rrule(vcat, rand(3,1,2), rand(4,1,2))

    # mix types
    test_rrule(vcat, rand(2, 2), rand(2, 2)')
    test_rrule(vcat, rand(), rand() => rand(); check_inferred=false)
    test_rrule(vcat, rand(3), (rand(), nothing), pi/2; check_inferred=false)
end

@testset "reduce vcat" begin
    mats = [randn(2, 4), randn(1, 4), randn(3, 4)]
    test_frule(reduce, vcat, mats)
    test_rrule(reduce, vcat, mats)

    vecs = [rand(2), rand(3), rand(4)]
    test_frule(reduce, vcat, vecs)
    test_rrule(reduce, vcat, vecs)

    mix = AbstractVecOrMat[rand(4,1), rand(4)]
    test_rrule(reduce, vcat, mix)
end

@testset "cat" begin
    # forward
    test_frule(cat, rand(2, 4), rand(1, 4); fkwargs=(dims=1,))
    test_frule(cat, rand(), rand(2,3); fkwargs=(dims=(1,2),))

    # reverse
    @gpu test_rrule(cat, rand(2, 4), rand(1, 4); fkwargs=(dims=1,))
    @gpu test_rrule(cat, rand(2, 4), rand(2); fkwargs=(dims=Val(2),))
    test_rrule(cat, rand(), rand(2, 3); fkwargs=(dims=[1,2],))
    test_rrule(cat, rand(1), rand(3, 2, 1); fkwargs=(dims=(1,2),), check_inferred=false) # infers Tuple{Zero, Vector{Float64}, Any}
    
    if VERSION ≥ v"1.8" # Val(tuple) dims support was added in v1.8
        test_rrule(cat, randn(3,2,4), randn(3,2,4); fkwargs=(dims=Val((1,2)),)) #678
    end

    test_rrule(cat, rand(2, 2), rand(2, 2)'; fkwargs=(dims=1,))
    # inference on exotic array types
    test_rrule(cat, @SArray(rand(3, 2, 1)), @SArray(rand(3, 2, 1)); fkwargs=(dims=Val(2),))
    test_rrule(cat, pi/2, rand(1,3), (4.5,); fkwargs=(;dims=(2,)), check_inferred=false)
end

@testset "hvcat" begin
    # forward
    test_frule(hvcat, 2, rand(6)...)

    # reverse
    test_rrule(hvcat, 2, rand(ComplexF64, 6)...)
    test_rrule(hvcat, (2, 1), rand(), rand(1,1), rand(2,2))
    test_rrule(hvcat, 1, rand(3)' ⊢ rand(1,3), transpose(rand(3)) ⊢ rand(1,3))
    test_rrule(hvcat, 1, rand(0,3), rand(2,3), rand(1,3,1))

    # mix types (adjoint and transpose)
    test_rrule(hvcat, 1, rand(3)', transpose(rand(3)) ⊢ rand(1,3))
    test_rrule(hvcat, (1,2), rand(2)', (3.4, 5.6), 7.8; check_inferred=false)
end

@testset "reverse" begin
    @testset "Tuple" begin
        test_frule(reverse, Tuple(rand(10)))
        @test_skip test_rrule(reverse, Tuple(rand(10)))  # Ambiguity in isapprox, https://github.com/JuliaDiff/ChainRulesTestUtils.jl/issues/229
    end
    @testset "Array" begin
        # Forward
        @gpu_broken test_frule(reverse, rand(5))
        test_frule(reverse, rand(5), 2, 4)
        test_frule(reverse, rand(5), fkwargs=(dims=1,))
        test_frule(reverse, rand(3,4), fkwargs=(dims=2,))
        test_frule(reverse, rand(3,4))
        test_frule(reverse, rand(3,4,5), fkwargs=(dims=(1,3),))

        test_frule(reverse!, rand(5))
        test_frule(reverse!, rand(5), 2, 4)
        test_frule(reverse!, rand(3,4), fkwargs=(dims=2,))

        # Reverse
        @gpu_broken test_rrule(reverse, rand(5))
        test_rrule(reverse, rand(5), 2, 4)
        test_rrule(reverse, rand(5), fkwargs=(dims=1,))

        test_rrule(reverse, rand(3,4), fkwargs=(dims=2,))
        test_rrule(reverse, rand(3,4))
        test_rrule(reverse, rand(3,4,5), fkwargs=(dims=(1,3),))

        # Structured
        y, pb = rrule(reverse, Diagonal([1,2,3]))
        # We only preserve structure in this case if given structured tangent (no ProjectTo)
        @test unthunk(pb(Diagonal([1.1, 2.1, 3.1]))[2]) isa Diagonal
        @test unthunk(pb(rand(3, 3))[2]) isa AbstractArray
    end
end

@testset "circshift" begin
    # Forward
    @gpu test_frule(circshift, rand(10), 1)
    test_frule(circshift, rand(10), (1,))
    test_frule(circshift, rand(3,4), (-7,2))

    test_frule(circshift!, rand(10), rand(10), 1)
    test_frule(circshift!, rand(3,4), rand(3,4), (-7,2))

    # Reverse
    @gpu test_rrule(circshift, rand(10), 1)
    test_rrule(circshift, rand(10) .+ im, -2)
    test_rrule(circshift, rand(10), (1,))
    test_rrule(circshift, rand(3,4), (-7,2))
end

@testset "fill" begin
    # Forward
    test_frule(fill, 12.3, 4)
    test_frule(fill, 5.0, (6, 7))

    test_frule(fill!, rand(2, 3), rand())

    # Reverse
    test_rrule(fill, 44.4, 4)
    test_rrule(fill, 55 + 0.5im, 5)
    test_rrule(fill, 3.3, (3, 3, 3))
end

@testset "filter" begin
    @testset "Array" begin
        # Random numbers will confuse finite differencing here, as it may perturb across the boundary.
        x5 = [0.0, 1.0, 0.2, 0.9, 0.7]
        x34 = Float64[-113  124   -37   12
                        96  -89   103  119
                        91  -21  -110   10]

        # Forward
        test_frule(filter, >(0.5) ⊢ NoTangent(), x5)
        test_frule(filter, <(0), x34)
        test_frule(filter, >(100), x5)

        # Reverse
        test_rrule(filter, >(0.5) ⊢ NoTangent(), x5)  # Without ⊢, MethodError: zero(::Base.Fix2{typeof(>), Float64}) -- https://github.com/JuliaDiff/ChainRulesTestUtils.jl/issues/231
        test_rrule(filter, <(0), x34)
        test_rrule(filter, >(100), x5)  # fixed in https://github.com/JuliaDiff/ChainRulesCore.jl/pull/534
        @test unthunk(rrule(filter, >(100), x5)[2](Int[])[3]) == zero(x5)
    end
end

@testset "findmin & findmax" begin
    # Forward
    test_frule(findmin, rand(10))
    test_frule(findmax, rand(10))
    @test @inferred(frule((nothing, rand(3,4)), findmin, rand(3,4))) isa Tuple{Tuple{Float64, CartesianIndex}, Tangent}
    @test @inferred(frule((nothing, rand(3,4)), findmin, rand(3,4), dims=1)) isa Tuple{Tuple{Matrix, Matrix}, Tangent}
    @test_skip test_frule(findmin, rand(3,4)) # error from test_approx(actual::CartesianIndex{2}, expected::CartesianIndex{2}
    @test_skip test_frule(findmin, rand(3,4), output_tangent = (rand(), NoTangent()))
    @test_skip test_frule(findmin, rand(3,4), fkwargs=(dims=1,))
    # These skipped tests might be fixed by https://github.com/JuliaDiff/FiniteDifferences.jl/issues/188
    # or by https://github.com/JuliaLang/julia/pull/48404

    # Reverse
    test_rrule(findmin, rand(10), output_tangent = (rand(), false))
    test_rrule(findmax, rand(10), output_tangent = (rand(), false))
    test_rrule(findmin, rand(5,3); check_inferred=false)
    test_rrule(findmax, rand(5,3); check_inferred=false)
    @test [0 0; 0 5] == unthunk(rrule(findmax, [1 2; 3 4])[2]((5.0, nothing))[2])
    @test [0 0; 0 5] == unthunk(rrule(findmax, [1 2; 3 4])[2]((5.0, NoTangent()))[2])

    # Reverse with dims:
    @test [0 0; 5 6] == @inferred unthunk(rrule(findmax, [1 2; 3 4], dims=1)[2](([5 6], nothing))[2])
    @test [5 0; 6 0] == @inferred unthunk(rrule(findmin, [1 2; 3 4], dims=2)[2]((hcat([5,6]), nothing))[2])
    test_rrule(findmin, rand(3,4), fkwargs=(dims=1,), output_tangent = (rand(1,4), NoTangent()))
    test_rrule(findmin, rand(3,4), fkwargs=(dims=2,))
    test_rrule(findmin, rand(3,4), fkwargs=(dims=(1,2),))
end

@testset "$imum" for imum in [maximum, minimum]
    # Forward
    test_frule(imum, rand(10))
    test_frule(imum, rand(3,4))
    @gpu test_frule(imum, rand(3,4), fkwargs=(dims=1,))
    test_frule(imum, [rand(2) for _ in 1:3])
    test_frule(imum, [rand(2) for _ in 1:3, _ in 1:4]; fkwargs=(dims=1,))

    # Reverse
    test_rrule(imum, rand(10))
    test_rrule(imum, rand(3,4); check_inferred=false)
    @gpu test_rrule(imum, rand(3,4), fkwargs=(dims=1,))
    test_rrule(imum, rand(3,4,5), fkwargs=(dims=(1,3),))

    # Arrays of arrays
    test_rrule(imum, [rand(2) for _ in 1:3]; check_inferred=false)
    test_rrule(imum, [rand(2) for _ in 1:3, _ in 1:4]; fkwargs=(dims=1,), check_inferred=false)

    # Case which attains max twice -- can't use FiniteDifferences for this
    res = imum == maximum ? [0,1,0,0,0,0] : [1,0,0,0,0,0]
    @test res == @inferred unthunk(rrule(imum, [1,2,1,2,1,2])[2](1.0)[2])

    # Structured matrix -- NB the minimum is a structral zero here
    @test unthunk(rrule(imum, Diagonal(rand(3) .+ 1))[2](5.5)[2]) isa Diagonal
    @test unthunk(rrule(imum, UpperTriangular(rand(3,3) .+ 1))[2](5.5)[2]) isa UpperTriangular{Float64}
    @test_skip test_rrule(imum, Diagonal(rand(3) .+ 1)) # MethodError: no method matching zero(::Type{Any}), from fill!(A::SparseArrays.SparseMatrixCSC{Any, Int64}, x::Bool)
end

@testset "extrema" begin
    test_rrule(extrema, rand(10), output_tangent = (rand(), rand()))
    test_rrule(extrema, rand(3,4), fkwargs=(dims=1,), output_tangent = collect(zip(rand(1,4), rand(1,4))))
    # Case where both extrema are the same index, to check accumulation:
    test_rrule(extrema, rand(1), output_tangent = (rand(), rand()))
    test_rrule(extrema, rand(1,1), fkwargs=(dims=2,), output_tangent = hcat((rand(), rand())))
    test_rrule(extrema, rand(3,1), fkwargs=(dims=2,), output_tangent = collect(zip(rand(3,1), rand(3,1))))
    # Double-check the forward pass
    A = randn(3,4,5)
    @test extrema(A, dims=(1,3)) == rrule(extrema, A, dims=(1,3))[1]
    B = hcat(A[:,:,1], A[:,:,1])
    @test extrema(B, dims=2) == rrule(extrema, B, dims=2)[1]
end

@testset "stack" begin
    # vector container
    xs = [rand(3, 4), rand(3, 4)]
    test_frule(stack, xs)
    test_frule(stack, xs; fkwargs=(dims=1,))

    test_rrule(stack, xs, check_inferred=false)
    test_rrule(stack, xs, fkwargs=(dims=1,), check_inferred=false)
    test_rrule(stack, xs, fkwargs=(dims=2,), check_inferred=false)
    test_rrule(stack, xs, fkwargs=(dims=3,), check_inferred=false)

    # multidimensional container
    ms = [rand(2,3) for _ in 1:4, _ in 1:5];

    if VERSION > v"1.9-"  # this needs new eachslice, not yet in Compat
        test_rrule(stack, ms, check_inferred=false)
    end
    test_rrule(stack, ms, fkwargs=(dims=1,), check_inferred=false)
    test_rrule(stack, ms, fkwargs=(dims=3,), check_inferred=false)
    
    # non-array inner objects
    ts = [Tuple(rand(3)) for _ in 1:4, _ in 1:2];

    if VERSION > v"1.9-"  
        test_rrule(stack, ts, check_inferred=false)
    end
    test_rrule(stack, ts, fkwargs=(dims=1,), check_inferred=false)
    test_rrule(stack, ts, fkwargs=(dims=2,), check_inferred=false)
end
