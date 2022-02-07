@testset "pmap" begin
    test_rrule(pmap, inv, WorkerPool(), rand(10), check_inferred=false) # test empty worker pool
    test_rrule(pmap, inv, default_worker_pool(), rand(10), check_inferred=false)
    test_rrule(pmap, inv, default_worker_pool(), rand(10), fkwargs=(batch_size=2,), check_inferred=false)
    # TODO: test other collections, e.g. zip(rand(10), rand(10)) or 1:10 or 1.:10.
end
