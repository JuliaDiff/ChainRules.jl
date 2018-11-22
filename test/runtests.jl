# TODO: more tests!

using ChainRules, Test
using ChainRules: rrule, @domain, Seed

function test_adjoint!(x̄, dx, ȳ, partial)
    x̄_old = copy(x̄)

    @test all(dx(x̄, ȳ) .== x̄ .+ partial)
    @test x̄ == x̄_old
    @test dx(Seed(x̄; store_into = true), ȳ) === x̄
    @test x̄ == x̄_old .+ partial

    copyto!(x̄, x̄_old)

    s = Seed(x̄; increment_adjoint = false)
    @test all(dx(s, ȳ) .== partial)
    @test x̄ == x̄_old
    s = Seed(x̄; store_into = true, increment_adjoint = false)
    @test dx(s, ȳ) === x̄
    @test all(x̄ .== partial)
end

#####
##### `*(x, y)`
#####

x, y = rand(3, 2), rand(2, 5)
z, (dx, dy) = rrule(@domain(R×R → R), *, x, y)

@test z == x * y

z̄ = rand(3, 5)

@test dx(nothing, z̄) == false
@test dy(nothing, z̄) == false

test_adjoint!(rand(3, 2), dx, z̄, z̄ * y')
test_adjoint!(rand(2, 5), dy, z̄, x' * z̄)

#####
##### `sin.(x)`
#####

x = rand(3, 3)
fx, dx = rrule(@domain(_×R → R), broadcast, sin, x)

@test fx == sin.(x)
@test dx(false, true) == cos.(x)

for (x̄, ȳ) in [(false, true),
               (rand(), rand()),
               (rand(), rand(3, 3)),
               (rand(), rand(3))]
    @test dx(x̄, ȳ) == x̄ .+ ȳ .* cos.(x)
end

for (x̄, ȳ) in [(rand(3, 3), rand()),
               (rand(3, 3), rand(3, 3)),
               (rand(3, 3), rand(3))]
    test_adjoint!(x̄, dx, ȳ, ȳ .* cos.(x))
end
