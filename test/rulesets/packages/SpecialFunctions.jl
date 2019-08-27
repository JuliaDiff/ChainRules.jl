using SpecialFunctions

@testset "SpecialFunctions" for x in (1, -1, 0, 0.5, 10, -17.1, 1.5 + 0.7im)
    test_scalar(SpecialFunctions.erf, x)
    test_scalar(SpecialFunctions.erfc, x)
    test_scalar(SpecialFunctions.erfi, x)

    test_scalar(SpecialFunctions.airyai, x)
    test_scalar(SpecialFunctions.airyaiprime, x)
    test_scalar(SpecialFunctions.airybi, x)
    test_scalar(SpecialFunctions.airybiprime, x)

    test_scalar(SpecialFunctions.besselj0, x)
    test_scalar(SpecialFunctions.besselj1, x)

    test_scalar(SpecialFunctions.erfcx, x)
    test_scalar(SpecialFunctions.dawson, x)

    if x isa Real
        test_scalar(SpecialFunctions.invdigamma, x)
    end

    if x isa Real && 0 < x < 1
        test_scalar(SpecialFunctions.erfinv, x)
        test_scalar(SpecialFunctions.erfcinv, x)
    end

    if x isa Real && x > 0 || x isa Complex
        test_scalar(SpecialFunctions.bessely0, x)
        test_scalar(SpecialFunctions.bessely1, x)
        test_scalar(SpecialFunctions.gamma, x)
        test_scalar(SpecialFunctions.digamma, x)
        test_scalar(SpecialFunctions.trigamma, x)
        test_scalar(SpecialFunctions.lgamma, x)
    end
end
