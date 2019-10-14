@testset "Package version checking" begin
    # This test needs to be updated when we allow a new version of ChainRulesCore
    @test ChainRules.pkg_version(ChainRulesCore) ∈ ChainRules.version_spec"0.3"
end
