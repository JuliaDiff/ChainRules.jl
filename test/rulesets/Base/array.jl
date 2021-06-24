@testset "reshape" begin
    test_rrule(reshape, rand(4, 5), (2, 10) ⊢ NoTangent())
    test_rrule(reshape, rand(4, 5), 2, 10)
    test_rrule(reshape, rand(4, 5), 2, :)
end

@testset "hcat" begin
    test_rrule(hcat, randn(3, 2), randn(3), randn(3, 3); check_inferred=VERSION>v"1.1")
    test_rrule(hcat, rand(), rand(1,2), rand(1,2,1); check_inferred=VERSION>v"1.1")
    test_rrule(hcat, rand(3,1,1,2), rand(3,3,1,2); check_inferred=VERSION>v"1.1")
end

@testset "reduce hcat" begin
    mats = [randn(3, 2), randn(3, 1), randn(3, 3)]
    test_rrule(reduce, hcat ⊢ NoTangent(), mats)
    
    vecs = [rand(3) for _ in 1:4]
    test_rrule(reduce, hcat ⊢ NoTangent(), vecs)
    
    mix = AbstractVecOrMat[rand(4,2), rand(4)]  # this is weird, but does hit the fast path
    test_rrule(reduce, hcat ⊢ NoTangent(), mix)

    adjs = vec([randn(2, 4), randn(1, 4), randn(3, 4)]')  # not a Vector
    # test_rrule(reduce, hcat ⊢ NoTangent(), adjs ⊢ map(m -> rand(size(m)), adjs))
    dy = 1 ./ reduce(hcat, adjs)
    @test rrule(reduce, hcat, adjs)[2](dy)[3] ≈ rrule(reduce, hcat, collect.(adjs))[2](dy)[3]
end

@testset "vcat" begin
    test_rrule(vcat, randn(2, 4), randn(1, 4), randn(3, 4); check_inferred=VERSION>v"1.1")
    test_rrule(vcat, rand(), rand(); check_inferred=VERSION>v"1.1")
    test_rrule(vcat, rand(), rand(3), rand(3,1,1); check_inferred=VERSION>v"1.1")
    test_rrule(vcat, rand(3,1,2), rand(4,1,2); check_inferred=VERSION>v"1.1")
end

@testset "reduce vcat" begin
    mats = [randn(2, 4), randn(1, 4), randn(3, 4)]
    test_rrule(reduce, vcat ⊢ NoTangent(), mats)

    vecs = [rand(2), rand(3), rand(4)]
    test_rrule(reduce, vcat ⊢ NoTangent(), vecs)

    mix = AbstractVecOrMat[rand(4,1), rand(4)]
    test_rrule(reduce, vcat ⊢ NoTangent(), mix)
end

@testset "cat" begin
    test_rrule(cat, rand(2, 4), rand(1, 4); fkwargs=(dims=1,), check_inferred=VERSION>v"1.1")
    test_rrule(cat, rand(2, 4), rand(2); fkwargs=(dims=Val(2),), check_inferred=VERSION>v"1.1")
    test_rrule(cat, rand(), rand(2, 3); fkwargs=(dims=[1,2],), check_inferred=VERSION>v"1.1")
    test_rrule(cat, rand(1), rand(3, 2, 1); fkwargs=(dims=(1,2),), check_inferred=false) # infers Tuple{Zero, Vector{Float64}, Any}
end

@testset "hvcat" begin
    test_rrule(hvcat, 2 ⊢ NoTangent(), rand(ComplexF64, 6)...; check_inferred=VERSION>v"1.1")
    test_rrule(hvcat, (2, 1) ⊢ NoTangent(), rand(), rand(1,1), rand(2,2); check_inferred=VERSION>v"1.1")
    test_rrule(hvcat, 1 ⊢ NoTangent(), rand(3)' ⊢ rand(1,3), transpose(rand(3)) ⊢ rand(1,3); check_inferred=VERSION>v"1.1")
    test_rrule(hvcat, 1 ⊢ NoTangent(), rand(0,3), rand(2,3), rand(1,3,1); check_inferred=VERSION>v"1.1")
end

@testset "fill" begin
    test_rrule(fill, 44.0, 4; check_inferred=false)
    test_rrule(fill, 2.0, (3, 3, 3) ⊢ NoTangent())
end
