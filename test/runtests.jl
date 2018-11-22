# TODO: more tests!

using ChainRules, Test
using ChainRules: rrule, @domain

#####
##### `*(x, y)`
#####

x, y = rand(3, 2), rand(2, 5)
z, (dx!, dy!) = rrule(@domain(R×R → R), *, x, y)

@test z == x * y

z̄ = rand(3, 5)

@test dx!(nothing, z̄) == false
@test dy!(nothing, z̄) == false

x̄ = rand(3, 2)
x̄_should_be = x̄ .+ (z̄ * y')
@test dx!(x̄, z̄) === x̄
@test x̄ == x̄_should_be

ȳ = rand(2, 5)
ȳ_should_be = ȳ .+ (x' * z̄)
@test dy!(ȳ, z̄) === ȳ
@test ȳ == ȳ_should_be

#####
##### `sin.(x)`
#####

x = rand(3, 3)
fx, dx! = rrule(@domain(_×R → R), broadcast, sin, x)

@test fx == sin.(x)

x̄, z̄ = false, true
@test dx!(x̄, z̄) == cos.(x)

for (x̄, z̄) in [(false, true),
               (rand(), rand()),
               (rand(), rand(3, 3)),
               (rand(), rand(3))]
    @test dx!(x̄, z̄) == x̄ .+ z̄ .* cos.(x)
end

for (x̄, z̄) in [(rand(3, 3), rand()),
               (rand(3, 3), rand(3, 3)),
               (rand(3, 3), rand(3))]
    x̄_should_be = x̄ .+ z̄ .* cos.(x)
    dx!(x̄, z̄)
    @test x̄ == x̄_should_be
end
