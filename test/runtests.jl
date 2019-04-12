# TODO: more tests!

using ChainRules, Test
using ChainRules: One, Zero, rrule, frule, extern, cast, accumulate, accumulate!, store!

cool(x) = x + 1
@testset "frule and rrule" begin
    @test frule(cool, 1) === nothing
    @test rrule(cool, 1) === nothing
    ChainRules.@scalar_rule(Main.cool(x), one(x))
    frx, fr = frule(cool, 1)
    @test frx == 2
    @test fr(1) == 1
    rrx, rr = rrule(cool, 1)
    @test rrx == 2
    @test rr(1) == 1
end

#####
##### `*(x, y)`
#####

function test_adjoint!(x̄, dx, ȳ, partial)
    x̄_old = copy(x̄)
    x̄_zeros = zero.(x̄)

    @test extern(accumulate(Zero(), dx, ȳ)) == extern(accumulate(x̄_zeros, dx, ȳ))
    @test extern(accumulate(x̄, dx, ȳ)) == (x̄ .+ partial)
    @test x̄ == x̄_old

    accumulate!(x̄, dx, ȳ)
    @test x̄ == (x̄_old .+ partial)
    x̄ .= x̄_old

    store!(x̄, dx, ȳ)
    @test x̄ == partial
    x̄ .= x̄_old

    return nothing
end

x, y = rand(3, 2), rand(2, 5)
z, (dx, dy) = rrule(*, x, y)

@test z == x * y

z̄ = rand(3, 5)

@test dx(z̄) == extern(accumulate(zeros(3, 2), dx, z̄))
@test dy(z̄) == extern(accumulate(zeros(2, 5), dy, z̄))

test_adjoint!(rand(3, 2), dx, z̄, z̄ * y')
test_adjoint!(rand(2, 5), dy, z̄, x' * z̄)

#####
##### `sin.(x)`
#####

x = rand(3, 3)
y, (dsin, dx) = rrule(broadcast, sin, x)

@test y == sin.(x)
@test extern(dx(One())) == cos.(x)

x̄, ȳ = rand(), rand()
@test extern(accumulate(x̄, dx, ȳ)) == x̄ .+ ȳ .* cos.(x)

x̄, ȳ = Zero(), rand(3, 3)
@test extern(accumulate(x̄, dx, ȳ)) == ȳ .* cos.(x)

x̄, ȳ = Zero(), cast(rand(3, 3))
@test extern(accumulate(x̄, dx, ȳ)) == extern(ȳ) .* cos.(x)

#####
##### `hypot(x, y)`
#####

x, y = rand(2)
h, dxy = frule(hypot, x, y)

@test extern(dxy(One(), Zero())) === y / h
@test extern(dxy(Zero(), One())) === x / h

cx, cy = cast((One(), Zero())), cast((Zero(), One()))
dx, dy = extern(dxy(cx, cy))
@test dx === y / h
@test dy === x / h

cx, cy = cast((rand(), Zero())), cast((Zero(), rand()))
dx, dy = extern(dxy(cx, cy))
@test dx === y / h * cx.value[1]
@test dy === x / h * cy.value[2]

#####
##### More!
#####

include(joinpath("rules", "base.jl"))
