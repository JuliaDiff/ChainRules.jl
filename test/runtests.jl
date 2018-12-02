# TODO: more tests!

using ChainRules, Test
using ChainRules: One, Zero, MaterializeInto, rrule, frule, materialize, casted

#####
##### `*(x, y)`
#####

function test_adjoint!(x̄, dx, ȳ, partial)
    x̄_old = copy(x̄)
    x̄_zeros = zero.(x̄)

    @test materialize(dx(Zero(), ȳ)) == materialize(dx(x̄_zeros, ȳ))
    @test materialize(dx(x̄, ȳ)) == (x̄ .+ partial)
    @test x̄ == x̄_old

    dx(MaterializeInto(x̄), ȳ)
    @test x̄ == (x̄_old .+ partial)
    x̄ .= x̄_old

    dx(MaterializeInto(x̄, false), ȳ)
    @test x̄ == partial
    x̄ .= x̄_old

    return nothing
end

x, y = rand(3, 2), rand(2, 5)
z, (dx, dy) = rrule(*, x, y)

@test z == x * y

z̄ = rand(3, 5)

@test dx(Zero(), z̄) == materialize(dx(zeros(3, 2), z̄))
@test dy(Zero(), z̄) == materialize(dy(zeros(2, 5), z̄))

test_adjoint!(rand(3, 2), dx, z̄, z̄ * y')
test_adjoint!(rand(2, 5), dy, z̄, x' * z̄)

#####
##### `sin.(x)`
#####

x = rand(3, 3)
y, (dsin, dx) = rrule(broadcast, sin, x)

@test y == sin.(x)
@test materialize(dx(Zero(), One())) == cos.(x)

x̄, ȳ = rand(), rand()
@test materialize(dx(x̄, ȳ)) == x̄ .+ ȳ .* cos.(x)

x̄, ȳ = Zero(), rand(3, 3)
@test materialize(dx(x̄, ȳ)) == ȳ .* cos.(x)

x̄, ȳ = Zero(), casted(rand(3, 3))
@test materialize(dx(x̄, ȳ)) == materialize(ȳ) .* cos.(x)

#####
##### `hypot(x, y)`
#####

x, y = rand(2)
h, dxy = frule(hypot, x, y)

@test materialize(dxy(Zero(), One(), Zero())) === y / h
@test materialize(dxy(Zero(), Zero(), One())) === x / h

cx, cy = casted((One(), Zero())), casted((Zero(), One()))
dx, dy = materialize(dxy(Zero(), cx, cy))
@test dx === y / h
@test dy === x / h

cx, cy = casted((rand(), Zero())), casted((Zero(), rand()))
dx, dy = materialize(dxy(Zero(), cx, cy))
@test dx === y / h * cx.value[1]
@test dy === x / h * cy.value[2]
