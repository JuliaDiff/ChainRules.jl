using FDM: jvp, j′vp

const _fdm = central_fdm(5, 1)

"""
    frule_test(f, (x, ẋ)...; rtol=1e-9, atol=1e-9, fdm=central_fdm(5, 1))

# Arguments
- `f`: Function for which the `frule` should be tested.
- `x`: input at which to evaluate `f` (should generally be set randomly).
- `ẋ`: differential w.r.t. `x` (should generally be set randomly).
"""
function frule_test(f, (x, ẋ); rtol=1e-9, atol=1e-9, fdm=_fdm)
    return frule_test(f, ((x, ẋ),); rtol=rtol, atol=atol, fdm=fdm)
end

function frule_test(f, xẋs::Tuple{Any, Any}...; rtol=1e-9, atol=1e-9, fdm=_fdm)
    xs, ẋs = collect(zip(xẋs...))
    Ω, dΩ_rule = ChainRules.frule(f, xs...)
    @test f(xs...) == Ω

    dΩ_ad, dΩ_fd = dΩ_rule(ẋs...), jvp(fdm, xs->f(xs...), (xs, ẋs))
    @test chain_rules_isapprox(dΩ_ad, dΩ_fd, rtol, atol)
end

"""
    rrule_test(f, ȳ, (x, x̄)...; rtol=1e-9, atol=1e-9, fdm=central_fdm(5, 1))

# Arguments
- `f`: Function to which rule should be applied.
- `ȳ`: adjoint w.r.t. output of `f` (should generally be set randomly).
- `x`: input at which to evaluate `f` (should generally be set randomly).
- `x̄`: currently accumulated adjoint (should generally be set randomly).
"""
function rrule_test(f, ȳ, (x, x̄)::Tuple{Any, Any}; rtol=1e-9, atol=1e-9, fdm=_fdm)
    # Check correctness of evaluation.
    fx, dx = ChainRules.rrule(f, x)
    @test fx ≈ f(x)

    # Correctness testing via finite differencing.
    x̄_ad, x̄_fd = dx(ȳ), j′vp(fdm, f, ȳ, x)
    @test cr_isapprox(x̄_ad, x̄_fd, rtol, atol)

    # Assuming x̄_ad to be correct, check that other ChainRules mechanisms are correct.
    test_adjoint!(x̄, dx, ȳ, x̄_ad)
end

function rrule_test(f, ȳ, xx̄s::Tuple{Any, Any}...; rtol=1e-9, atol=1e-9, fdm=_fdm)
    # Check correctness of evaluation.
    xs, x̄s = collect(zip(xx̄s...))
    Ω, Δx_rules = ChainRules.rrule(f, xs...)
    @test f(xs...) == Ω

    # Correctness testing via finite differencing.
    Δxs_ad, Δxs_fd = map(Δx_rule->Δx_rule(ȳ), Δx_rules), j′vp(fdm, f, ȳ, xs...)
    @test all(map((Δx_ad, Δx_fd)->cr_isapprox(Δx_ad, Δx_fd, rtol, atol), Δxs_ad, Δxs_fd))

    # Assuming the above to be correct, check that other ChainRules mechanisms are correct.
    map((x̄, Δx_rule, Δx_ad)->test_adjoint!(x̄, Δx_rule, ȳ, Δx_ad), x̄s, Δx_rules, Δxs_ad)
end

function cr_isapprox(d_ad, d_fd, rtol, atol)
    return isapprox(d_ad, d_fd; rtol=rtol, atol=atol)
end
function cr_isapprox(ad::Wirtinger, fd, rtol, atol)
    error("Finite differencing with Wirtinger rules not implemented")
end
function cr_isapprox(d_ad::Casted, d_fd, rtol, atol)
    return all(isapprox.(extern(d_ad), d_fd; rtol=rtol, atol=atol))
end
function cr_isapprox(d_ad::DNE, d_fd, rtol, atol)
    error("Tried to differentiate w.r.t. a DNE")
end
function cr_isapprox(d_ad::Thunk, d_fd, rtol, atol)
    return isapprox(extern(d_ad), d_fd; rtol=rtol, atol=atol)
end

function test_adjoint!(x̄, dx, ȳ, partial)
    x̄_old = copy(x̄)
    x̄_zeros = zero.(x̄)

    @test all(accumulate(Zero(), dx, ȳ) .== accumulate(x̄_zeros, dx, ȳ))
    @test all(accumulate(x̄, dx, ȳ) .== (x̄ .+ partial))
    @test x̄ == x̄_old

    accumulate!(x̄, dx, ȳ)
    @test x̄ == (x̄_old .+ partial)
    x̄ .= x̄_old

    store!(x̄, dx, ȳ)
    @test all(x̄ .== partial)
    x̄ .= x̄_old

    return nothing
end
