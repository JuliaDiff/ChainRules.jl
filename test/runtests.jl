# TODO: more tests!

using ChainRules, Test
using ChainRules: One, Zero, rrule, frule, extern, cast, Accumulate

#####
##### `*(x, y)`
#####

function test_adjoint!(x̄, dx, ȳ, partial)
    x̄_old = copy(x̄)
    x̄_zeros = zero.(x̄)

    @test extern(dx(Zero(), ȳ)) == extern(dx(x̄_zeros, ȳ))
    @test extern(dx(x̄, ȳ)) == (x̄ .+ partial)
    @test x̄ == x̄_old

    dx(Accumulate(x̄), ȳ)
    @test x̄ == (x̄_old .+ partial)
    x̄ .= x̄_old

    dx(Accumulate(x̄, false), ȳ)
    @test x̄ == partial
    x̄ .= x̄_old

    return nothing
end

x, y = rand(3, 2), rand(2, 5)
z, (dx, dy) = rrule(*, x, y)

@test z == x * y

z̄ = rand(3, 5)

@test dx(Zero(), z̄) == extern(dx(zeros(3, 2), z̄))
@test dy(Zero(), z̄) == extern(dy(zeros(2, 5), z̄))

test_adjoint!(rand(3, 2), dx, z̄, z̄ * y')
test_adjoint!(rand(2, 5), dy, z̄, x' * z̄)

#####
##### `sin.(x)`
#####

x = rand(3, 3)
y, (dsin, dx) = rrule(broadcast, sin, x)

@test y == sin.(x)
@test extern(dx(Zero(), One())) == cos.(x)

x̄, ȳ = rand(), rand()
@test extern(dx(x̄, ȳ)) == x̄ .+ ȳ .* cos.(x)

x̄, ȳ = Zero(), rand(3, 3)
@test extern(dx(x̄, ȳ)) == ȳ .* cos.(x)

x̄, ȳ = Zero(), cast(rand(3, 3))
@test extern(dx(x̄, ȳ)) == extern(ȳ) .* cos.(x)

#####
##### `hypot(x, y)`
#####

x, y = rand(2)
h, dxy = frule(hypot, x, y)

@test extern(dxy(Zero(), One(), Zero())) === y / h
@test extern(dxy(Zero(), Zero(), One())) === x / h

cx, cy = cast((One(), Zero())), cast((Zero(), One()))
dx, dy = extern(dxy(Zero(), cx, cy))
@test dx === y / h
@test dy === x / h

cx, cy = cast((rand(), Zero())), cast((Zero(), rand()))
dx, dy = extern(dxy(Zero(), cx, cy))
@test dx === y / h * cx.value[1]
@test dy === x / h * cy.value[2]
