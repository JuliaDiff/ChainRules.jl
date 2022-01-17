@testset "Array constructors" begin

    # We can't use test_rrule here (as it's currently implemented) because the elements of
    # the array have arbitrary values. The only thing we can do is ensure that we're getting
    # `ZeroTangent`s back, and that the forwards pass produces the correct thing still.
    # Issue: https://github.com/JuliaDiff/ChainRulesTestUtils.jl/issues/202
    @testset "undef" begin
        val, pullback = rrule(Array{Float64}, undef, 5)
        @test size(val) == (5, )
        @test val isa Array{Float64, 1}
        @test pullback(randn(5)) == (NoTangent(), NoTangent(), NoTangent())
    end
    @testset "from existing array" begin
        test_rrule(Array, randn(2, 5))
        test_rrule(Array, Diagonal(randn(5)))
        test_rrule(Matrix, Diagonal(randn(5)))
        test_rrule(Matrix, transpose(randn(4)))
        test_rrule(Array{ComplexF64}, randn(3))
    end
end

@testset "vect" begin
    test_rrule(Base.vect)
    @testset "homogeneous type" begin
        test_rrule(Base.vect, (5.0, ), (4.0, ))
        test_rrule(Base.vect, 5.0, 4.0, 3.0)
        test_rrule(Base.vect, randn(2, 2), randn(3, 3))
    end
    @testset "inhomogeneous type" begin
        test_rrule(
            Base.vect, 5.0, 3f0;
            atol=1e-6, rtol=1e-6, check_inferred=VERSION>=v"1.6",
        ) # tolerance due to Float32.
        test_rrule(Base.vect, 5.0, randn(3, 3); check_inferred=false)
        test_rrule(Base.vect, (5.0, 4.0), (y=randn(3),); check_inferred=false)
    end
end

@testset "reshape" begin
    test_rrule(reshape, rand(4, 5), (2, 10))
    test_rrule(reshape, rand(4, 5), 2, 10)
    test_rrule(reshape, rand(4, 5), 2, :)
end

@testset "permutedims + PermutedDimsArray" begin
    test_rrule(permutedims, rand(5))

    test_rrule(permutedims, rand(3, 4), (2, 1))
    test_rrule(permutedims, Diagonal(rand(5)), (2, 1))
    # Note BTW that permutedims(Diagonal(rand(5))) does not use the rule at all

    @test invperm((3, 1, 2)) != (3, 1, 2)
    test_rrule(permutedims, rand(3, 4, 5), (3, 1, 2); check_inferred=VERSION>=v"1.1")

    @test_skip test_rrule(PermutedDimsArray, rand(3, 4, 5), (3, 1, 2))
    x = rand(2, 3, 4)
    dy = rand(4, 2, 3)
    @test rrule(permutedims, x, (3, 1, 2))[2](dy)[2] == rrule(PermutedDimsArray, x, (3, 1, 2))[2](dy)[2]
end

@testset "repeat" begin

    test_rrule(repeat, rand(4, ))
    test_rrule(repeat, rand(4, 5))
    test_rrule(repeat, rand(4, 5); fkwargs = (outer=(1,2),))
    test_rrule(repeat, rand(4, 5); fkwargs = (inner=(1,2), outer=(1,3)))

    test_rrule(repeat, rand(4, ), 2; check_inferred=VERSION>=v"1.6")
    test_rrule(repeat, rand(4, 5), 2; check_inferred=VERSION>=v"1.6")
    test_rrule(repeat, rand(4, 5), 2, 3; check_inferred=VERSION>=v"1.6")
    test_rrule(repeat, rand(1,2,3), 2,3,4; check_inferred=VERSION>v"1.6")
    test_rrule(repeat, rand(0,2,3), 2,0,4; check_inferred=VERSION>v"1.6")
    test_rrule(repeat, rand(1,1,1,1), 2,3,4,5; check_inferred=VERSION>v"1.6")
    

    if VERSION>=v"1.6"
        # These are cases where repeat itself fails in earlier versions
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

    end

    @test rrule(repeat, [1,2,3], 4)[2](ones(12))[2] == [4,4,4]
    @test rrule(repeat, [1,2,3], outer=4)[2](ones(12))[2] == [4,4,4]

end

@testset "hcat" begin
    test_rrule(hcat, randn(3, 2), randn(3), randn(3, 3); check_inferred=VERSION>v"1.1")
    test_rrule(hcat, rand(), rand(1,2), rand(1,2,1); check_inferred=VERSION>v"1.1")
    test_rrule(hcat, rand(3,1,1,2), rand(3,3,1,2); check_inferred=VERSION>v"1.1")

    # mix types
    test_rrule(hcat, rand(2, 2), rand(2, 2)'; check_inferred=VERSION>v"1.1")
end

@testset "reduce hcat" begin
    mats = [randn(3, 2), randn(3, 1), randn(3, 3)]
    test_rrule(reduce, hcat, mats)
    
    vecs = [rand(3) for _ in 1:4]
    test_rrule(reduce, hcat, vecs)
    
    mix = AbstractVecOrMat[rand(4,2), rand(4)]  # this is weird, but does hit the fast path
    test_rrule(reduce, hcat, mix)

    adjs = vec([randn(2, 4), randn(1, 4), randn(3, 4)]')  # not a Vector
    # test_rrule(reduce, hcat, adjs ⊢ map(m -> rand(size(m)), adjs))
    dy = 1 ./ reduce(hcat, adjs)
    @test rrule(reduce, hcat, adjs)[2](dy)[3] ≈ rrule(reduce, hcat, collect.(adjs))[2](dy)[3]

    # mix types
    mats = [randn(2, 2), rand(2, 2)']
    test_rrule(reduce, hcat, mats; check_inferred=VERSION>v"1.1")
end

@testset "vcat" begin
    test_rrule(vcat, randn(2, 4), randn(1, 4), randn(3, 4); check_inferred=VERSION>v"1.1")
    test_rrule(vcat, rand(), rand(); check_inferred=VERSION>v"1.1")
    test_rrule(vcat, rand(), rand(3), rand(3,1,1); check_inferred=VERSION>v"1.1")
    test_rrule(vcat, rand(3,1,2), rand(4,1,2); check_inferred=VERSION>v"1.1")

    # mix types
    test_rrule(vcat, rand(2, 2), rand(2, 2)'; check_inferred=VERSION>v"1.1")
end

@testset "reduce vcat" begin
    mats = [randn(2, 4), randn(1, 4), randn(3, 4)]
    test_rrule(reduce, vcat, mats)

    vecs = [rand(2), rand(3), rand(4)]
    test_rrule(reduce, vcat, vecs)

    mix = AbstractVecOrMat[rand(4,1), rand(4)]
    test_rrule(reduce, vcat, mix)
end

@testset "cat" begin
    test_rrule(cat, rand(2, 4), rand(1, 4); fkwargs=(dims=1,), check_inferred=VERSION>v"1.1")
    test_rrule(cat, rand(2, 4), rand(2); fkwargs=(dims=Val(2),), check_inferred=VERSION>v"1.1")
    test_rrule(cat, rand(), rand(2, 3); fkwargs=(dims=[1,2],), check_inferred=VERSION>v"1.1")
    test_rrule(cat, rand(1), rand(3, 2, 1); fkwargs=(dims=(1,2),), check_inferred=false) # infers Tuple{Zero, Vector{Float64}, Any}

    test_rrule(cat, rand(2, 2), rand(2, 2)'; fkwargs=(dims=1,), check_inferred=VERSION>v"1.1")
end

@testset "hvcat" begin
    test_rrule(hvcat, 2, rand(ComplexF64, 6)...; check_inferred=VERSION>v"1.1")
    test_rrule(hvcat, (2, 1), rand(), rand(1,1), rand(2,2); check_inferred=VERSION>v"1.1")
    test_rrule(hvcat, 1, rand(3)' ⊢ rand(1,3), transpose(rand(3)) ⊢ rand(1,3); check_inferred=VERSION>v"1.1")
    test_rrule(hvcat, 1, rand(0,3), rand(2,3), rand(1,3,1); check_inferred=VERSION>v"1.1")

    # mix types (adjoint and transpose)
    test_rrule(hvcat, 1, rand(3)', transpose(rand(3)) ⊢ rand(1,3); check_inferred=VERSION>v"1.1")
end

@testset "reverse" begin
    @testset "Tuple" begin
        test_frule(reverse, Tuple(rand(10)))
        test_rrule(reverse, Tuple(rand(10)))
    end
    @testset "Array" begin
        # Forward
        test_frule(reverse, rand(5))
        test_frule(reverse, rand(5), 2, 4)
        test_frule(reverse, rand(5), fkwargs=(dims=1,))

        test_frule(reverse, rand(3,4), fkwargs=(dims=2,))
        if VERSION >= v"1.6"
            test_frule(reverse, rand(3,4))
            test_frule(reverse, rand(3,4,5), fkwargs=(dims=(1,3),))
        end

        # Reverse
        test_rrule(reverse, rand(5))
        test_rrule(reverse, rand(5), 2, 4)
        test_rrule(reverse, rand(5), fkwargs=(dims=1,))

        test_rrule(reverse, rand(3,4), fkwargs=(dims=2,))
        if VERSION >= v"1.6"
            test_rrule(reverse, rand(3,4))
            test_rrule(reverse, rand(3,4,5), fkwargs=(dims=(1,3),))

            # Structured
            y, pb = rrule(reverse, Diagonal([1,2,3]))
            # We only preserve structure in this case if given structured tangent (no ProjectTo)
            @test unthunk(pb(Diagonal([1.1, 2.1, 3.1]))[2]) isa Diagonal
            @test unthunk(pb(rand(3, 3))[2]) isa AbstractArray
        end
    end
end

@testset "circshift" begin
    # Forward
    test_frule(circshift, rand(10), 1)
    test_frule(circshift, rand(10), (1,))
    test_frule(circshift, rand(3,4), (-7,2))

    # Reverse
    test_rrule(circshift, rand(10), 1)
    test_rrule(circshift, rand(10) .+ im, -2)
    test_rrule(circshift, rand(10), (1,))
    test_rrule(circshift, rand(3,4), (-7,2))
end

@testset "fill" begin
    test_frule(fill, 12.3, 4)
    test_frule(fill, 5.0, (6, 7))

    test_rrule(fill, 44.4, 4)
    test_rrule(fill, 55 + 0.5im, 5)
    test_rrule(fill, 3.3, (3, 3, 3))
end

@testset "filter" begin
    @testset "Tuple" begin
        test_frule(filter, >(0.5), Tuple(rand(10)))
        test_rrule(filter, >(0.5), Tuple(rand(10)))
    end
    @testset "Array" begin
        test_frule(filter, >(0.5), rand(10))
        test_frule(filter, <(0), rand(3, 4))

        test_rrule(filter, >(0.5), rand(10))
        test_rrule(filter, <(0), rand(3, 4))
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

    # Reverse
    test_rrule(findmin, rand(10), output_tangent = (rand(), false))
    test_rrule(findmax, rand(10), output_tangent = (rand(), false))
    test_rrule(findmin, rand(5,3))
    test_rrule(findmax, rand(5,3))
    @test [0 0; 0 5] == @inferred unthunk(rrule(findmax, [1 2; 3 4])[2]((5.0, nothing))[2])
    @test [0 0; 0 5] == @inferred unthunk(rrule(findmax, [1 2; 3 4])[2]((5.0, NoTangent()))[2])
    
    # Reverse with dims:
    @test [0 0; 5 6] == @inferred unthunk(rrule(findmax, [1 2; 3 4], dims=1)[2](([5 6], nothing))[2])
    @test [5 0; 6 0] == @inferred unthunk(rrule(findmin, [1 2; 3 4], dims=2)[2]((hcat([5,6]), nothing))[2])
    test_rrule(findmin, rand(3,4), fkwargs=(dims=1,), output_tangent = (rand(1,4), NoTangent()))
    test_rrule(findmin, rand(3,4), fkwargs=(dims=2,))

    # Second derivatives
    test_frule(ChainRules._zerolike_writeat, rand(2, 2), 5.0, :, CartesianIndex(2, 2))
    test_rrule(ChainRules._zerolike_writeat, rand(2, 2), 5.0, :, CartesianIndex(2, 2))
    @test_skip test_rrule(ChainRules._zerolike_writeat, rand(2, 2), 5.0, 1, [CartesianIndex(2, 1) CartesianIndex(2, 2)]  ⊢ NoTangent())  # MethodError: no method matching isapprox(::Matrix{Float64}, ::Float64; rtol=1.0e-9, atol=1.0e-9)
    y, bk = rrule(ChainRules._zerolike_writeat, rand(2, 2), 5.0, 1, [CartesianIndex(2, 1) CartesianIndex(2, 2)])
    @test y == [0 0; 5 5]
    @test bk([1 2; 3 4]) == (NoTangent(), NoTangent(), [3 4], NoTangent(), NoTangent())
end

@testset "$imum" for imum in [maximum, minimum]
    # Forward
    test_frule(imum, rand(10))
    test_frule(imum, rand(3,4))
    test_frule(imum, rand(3,4), fkwargs=(dims=1,))
    test_frule(imum, [rand(2) for _ in 1:3])
    test_frule(imum, [rand(2) for _ in 1:3, _ in 1:4]; fkwargs=(dims=1,))

    # Reverse
    test_rrule(imum, rand(10))
    test_rrule(imum, rand(3,4))
    test_rrule(imum, rand(3,4), fkwargs=(dims=1,))
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
