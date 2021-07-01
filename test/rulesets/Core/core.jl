@testset "typeassert" begin
    test_rrule(typeassert, 1.0, Float64 ⊢ NoTangent())
    test_frule(typeassert, 1.0, Float64 ⊢ NoTangent())
end

@testset "ifelse" begin
    test_rrule(typeassert, true ⊢ NoTangent(), 1.0, 2.0)
    test_frule(typeassert, false ⊢ NoTangent(), 1.0, 2.0)
end