# TODO: more tests!

using ChainRules, Test

x = rand(3, 3)
sig = ChainRules.Signature((sin, x), x)
fx, dx! = ChainRules.reverse_rule(sig, broadcast, sin, x)

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
    should_be = x̄ .+ z̄ .* cos.(x)
    dx!(x̄, z̄)
    @test x̄ == should_be
end
