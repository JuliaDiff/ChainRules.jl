@testset "general: single input" begin
    for x in (1.0, -1.0, 0.0, 0.5, 10.0, -17.1, 1.5 + 0.7im)
        test_scalar(SpecialFunctions.erf, x)
        test_scalar(SpecialFunctions.erfc, x)
        test_scalar(SpecialFunctions.erfi, x)

        test_scalar(SpecialFunctions.airyai, x)
        test_scalar(SpecialFunctions.airyaiprime, x)
        test_scalar(SpecialFunctions.airybi, x)
        test_scalar(SpecialFunctions.airybiprime, x)

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
            test_scalar(SpecialFunctions.gamma, x)
            test_scalar(SpecialFunctions.digamma, x)
            test_scalar(SpecialFunctions.trigamma, x)
        end
    end
end

@testset "Bessel functions" begin
    for x in (1.5, 2.5, 10.5, -0.6, -2.6, -3.3, 1.6 + 1.6im, 1.6 - 1.6im, -4.6 + 1.6im)
        test_scalar(SpecialFunctions.besselj0, x)
        test_scalar(SpecialFunctions.besselj1, x)

        isreal(x) && x < 0 && continue

        test_scalar(SpecialFunctions.bessely0, x)
        test_scalar(SpecialFunctions.bessely1, x)

        for nu in (-1.5, 2.2, 4.0)
            test_frule(SpecialFunctions.besseli, nu, x)
            test_rrule(SpecialFunctions.besseli, nu, x)

            test_frule(SpecialFunctions.besselj, nu, x)
            test_rrule(SpecialFunctions.besselj, nu, x)

            test_frule(SpecialFunctions.besselk, nu, x)
            test_rrule(SpecialFunctions.besselk, nu, x)

            test_frule(SpecialFunctions.bessely, nu, x)
            test_rrule(SpecialFunctions.bessely, nu, x)

            # use complex numbers in `rrule` for FiniteDifferences
            test_frule(SpecialFunctions.hankelh1, nu, x)
            test_rrule(SpecialFunctions.hankelh1, nu, complex(x))

            # use complex numbers in `rrule` for FiniteDifferences
            test_frule(SpecialFunctions.hankelh2, nu, x)
            test_rrule(SpecialFunctions.hankelh2, nu, complex(x))
        end
    end
end

@testset "beta and logbeta" begin
    test_points = (1.5, 2.5, 10.5, 1.6 + 1.6im, 1.6 - 1.6im, 4.6 + 1.6im)
    for _x in test_points, _y in test_points
        # ensure all complex if any complex for FiniteDifferences
        x, y = promote(_x, _y)
        test_frule(SpecialFunctions.beta, x, y)
        test_rrule(SpecialFunctions.beta, x, y)

        if isdefined(SpecialFunctions, :lbeta)
            test_frule(SpecialFunctions.lbeta, x, y)
            test_rrule(SpecialFunctions.lbeta, x, y)
        end

        if isdefined(SpecialFunctions, :logbeta)
            test_frule(SpecialFunctions.logbeta, x, y)
            test_rrule(SpecialFunctions.logbeta, x, y)
        end
    end
end

@testset "log gamma and co" begin
    # It is important that we have negative numbers with both odd and even integer parts
    for x in (1.5, 2.5, 10.5, -0.6, -2.6, -3.3, 1.6 + 1.6im, 1.6 - 1.6im, -4.6 + 1.6im)
        for m in (0, 1, 2, 3)
            test_frule(SpecialFunctions.polygamma, m, x)
            test_rrule(SpecialFunctions.polygamma, m, x)
        end

        if isdefined(SpecialFunctions, :lgamma)
            test_scalar(SpecialFunctions.lgamma, x)
        end

        if isdefined(SpecialFunctions, :loggamma)
            isreal(x) && x < 0 && continue
            test_scalar(SpecialFunctions.loggamma, x)
        end

        if isdefined(SpecialFunctions, :logabsgamma)
            isreal(x) || continue
            test_frule(SpecialFunctions.logabsgamma, x)
            test_rrule(SpecialFunctions.logabsgamma, x; output_tangent=(randn(), randn()))
        end
    end
end
