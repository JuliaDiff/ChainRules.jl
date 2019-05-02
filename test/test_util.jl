using FDM: jvp, j′vp

const _fdm = central_fdm(5, 1)

"""
    frule_test(f, (x, ẋ)...; rtol=1e-9, atol=1e-9, fdm=central_fdm(5, 1), kwargs...)

# Arguments
- `f`: Function for which the `frule` should be tested.
- `x`: input at which to evaluate `f` (should generally be set randomly).
- `ẋ`: differential w.r.t. `x` (should generally be set randomly).

All keyword arguments except for `fdm` are passed to `isapprox`.
"""
function frule_test(f, (x, ẋ); rtol=1e-9, atol=1e-9, fdm=_fdm, kwargs...)
    return frule_test(f, ((x, ẋ),); rtol=rtol, atol=atol, fdm=fdm, kwargs...)
end

function frule_test(f, xẋs::Tuple{Any, Any}...; rtol=1e-9, atol=1e-9, fdm=_fdm, kwargs...)
    xs, ẋs = collect(zip(xẋs...))
    Ω, dΩ_rule = ChainRules.frule(f, xs...)
    @test f(xs...) == Ω

    dΩ_ad, dΩ_fd = dΩ_rule(ẋs...), jvp(fdm, xs->f(xs...), (xs, ẋs))
    @test isapprox(dΩ_ad, dΩ_fd; rtol=rtol, atol=atol, kwargs...)
end

"""
    rrule_test(f, ȳ, (x, x̄)...; rtol=1e-9, atol=1e-9, fdm=central_fdm(5, 1), kwargs...)

# Arguments
- `f`: Function to which rule should be applied.
- `ȳ`: adjoint w.r.t. output of `f` (should generally be set randomly).
- `x`: input at which to evaluate `f` (should generally be set randomly).
- `x̄`: currently accumulated adjoint (should generally be set randomly).

All keyword arguments except for `fdm` are passed to `isapprox`.
"""
function rrule_test(f, ȳ, (x, x̄)::Tuple{Any, Any}; rtol=1e-9, atol=1e-9, fdm=_fdm, kwargs...)
    # Check correctness of evaluation.
    fx, dx = ChainRules.rrule(f, x)
    @test fx ≈ f(x)

    # Correctness testing via finite differencing.
    x̄_ad, x̄_fd = dx(ȳ), j′vp(fdm, f, ȳ, x)
    @test isapprox(x̄_ad, x̄_fd; rtol=rtol, atol=atol, kwargs...)

    # Assuming x̄_ad to be correct, check that other ChainRules mechanisms are correct.
    test_accumulation(x̄, dx, ȳ, x̄_ad)
    test_accumulation(Zero(), dx, ȳ, x̄_ad)
end

function rrule_test(f, ȳ, xx̄s::Tuple{Any, Any}...; rtol=1e-9, atol=1e-9, fdm=_fdm, kwargs...)
    # Check correctness of evaluation.
    xs, x̄s = collect(zip(xx̄s...))
    Ω, Δx_rules = ChainRules.rrule(f, xs...)
    @test f(xs...) == Ω

    # Correctness testing via finite differencing.
    Δxs_ad = map(Δx_rule->Δx_rule(ȳ), Δx_rules)
    Δxs_fd = j′vp(fdm, f, ȳ, xs...)
    for (Δx_ad, Δx_fd) in zip(Δxs_ad, Δxs_fd)
        @test isapprox(Δx_ad, Δx_fd; rtol=rtol, atol=atol, kwargs...)
    end

    # Assuming the above to be correct, check that other ChainRules mechanisms are correct.
    for (x̄, Δx_rule, Δx_ad) in zip(x̄s, Δx_rules, Δxs_ad)
        test_accumulation(x̄, Δx_rule, ȳ, Δx_ad)
        test_accumulation(Zero(), Δx_rule, ȳ, Δx_ad)
    end
end

function Base.isapprox(ad::Wirtinger, fd; kwargs...)
    error("Finite differencing with Wirtinger rules not implemented")
end
function Base.isapprox(d_ad::Casted, d_fd; kwargs...)
    return all(isapprox.(extern(d_ad), d_fd; kwargs...))
end
function Base.isapprox(d_ad::DNE, d_fd; kwargs...)
    error("Tried to differentiate w.r.t. a DNE")
end
function Base.isapprox(d_ad::Thunk, d_fd; kwargs...)
    return isapprox(extern(d_ad), d_fd; kwargs...)
end

function test_accumulation(x̄, dx, ȳ, partial)
    @test all(extern(ChainRules.add(x̄, partial)) .== extern(x̄) .+ extern(partial))
    test_accumulate(x̄, dx, ȳ, partial)
    test_accumulate!(x̄, dx, ȳ, partial)
    test_store!(x̄, dx, ȳ, partial)
    return nothing
end

function test_accumulate(x̄::Zero, dx, ȳ, partial)
    @test extern(accumulate(x̄, dx, ȳ)) == extern(partial)
    return nothing
end

function test_accumulate(x̄::Number, dx, ȳ, partial)
    @test extern(accumulate(x̄, dx, ȳ)) == extern(x̄) + extern(partial)
    return nothing
end

function test_accumulate(x̄::AbstractArray, dx, ȳ, partial)
    x̄_old = copy(x̄)
    @test all(extern(accumulate(x̄, dx, ȳ)) .== (extern(x̄) .+ extern(partial)))
    @test x̄ == x̄_old
    return nothing
end

test_accumulate!(x̄::Zero, dx, ȳ, partial) = nothing

function test_accumulate!(x̄::Number, dx, ȳ, partial)
    @test accumulate!(x̄, dx, ȳ) == accumulate(x̄, dx, ȳ)
    return nothing
end

function test_accumulate!(x̄::AbstractArray, dx, ȳ, partial)
    x̄_copy = copy(x̄)
    accumulate!(x̄_copy, dx, ȳ)
    @test extern(x̄_copy) == (extern(x̄) .+ extern(partial))
    return nothing
end

test_store!(x̄::Zero, dx, ȳ, partial) = nothing
test_store!(x̄::Number, dx, ȳ, partial) = nothing

function test_store!(x̄::AbstractArray, dx, ȳ, partial)
    x̄_copy = copy(x̄)
    store!(x̄_copy, dx, ȳ)
    @test all(x̄_copy .== extern(partial))
    return nothing
end
