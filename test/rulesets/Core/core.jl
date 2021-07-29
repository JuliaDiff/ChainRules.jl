@testset "typeassert" begin
    test_rrule(typeassert, 1.0, Float64 ⊢ NoTangent())
    test_frule(typeassert, 1.0, Float64 ⊢ NoTangent())
end

@testset "ifelse" begin
    test_rrule(ifelse, true ⊢ NoTangent(), 1.0, 2.0)
    test_frule(ifelse, false ⊢ NoTangent(), 1.0, 2.0)

    test_rrule(ifelse, true ⊢ NoTangent(), [1.0], [2.0]; check_inferred=false)
    test_frule(ifelse, false ⊢ NoTangent(), [1.0], [2.0]; check_inferred=false)
end
