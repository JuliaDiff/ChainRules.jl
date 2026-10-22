using Test

# Rules for these packages live in extensions, so loading ChainRules must not load them.
@testset "load time dependencies" begin
    code = """
    using ChainRules
    print(join(sort([pkg.name for pkg in keys(Base.loaded_modules) if pkg.name in ("Distributed", "SparseArrays", "Statistics")]), ","))
    """
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`
    @test readchomp(cmd) == ""
end
