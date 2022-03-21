@testset "pmap" begin

    # basic tests
    
    test_rrule(pmap, inv, default_worker_pool(), rand(10), check_inferred=false) # automatic test
    y1, b1 = rrule(CFG, pmap, inv, default_worker_pool(), [1, 2, 3]) # manual test
    @test y1 ≈ 1 ./ [1, 2, 3] 
    @test b1([3,4,7])[4] ≈ [-3 / 1^2, -4 / 2^2, -7 / 3^2]
    @test b1([3, 4, 7])[1:3] == (NoTangent(), NoTangent(), NoTangent())

    # test different worker pools
    
    test_rrule(pmap, inv, WorkerPool(), rand(10), check_inferred=false) # empty worker pool

    # test different inputs
    
    test_rrule(pmap, inv, default_worker_pool(), rand(4, 4), check_inferred=false) # test matrix input

    # test keywords
    
    test_rrule(pmap, inv, default_worker_pool(), rand(10), fkwargs=(batch_size=2,), check_inferred=false) # batch_size > 1
    test_rrule(pmap, inv, default_worker_pool(), rand(10), fkwargs=(distributed=false,), check_inferred=false) # distributed = false

    # test adjoint of function input
    
    test_rrule(pmap, Multiplier(2.0), default_worker_pool(), rand(3), check_inferred=false)

end
