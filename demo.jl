using Pkg: @pkg_str
pkg"activate ."

using ChainRules, LinearAlgebra
#==
f1, dX1 = rrule(sum, randn(4, 4));
@time dX1(f1)
@time dX1(f1)
==#
F, dX = rrule(svd, randn(4, 4));
nt = (U=F.U, S=F.S, V=F.V);

@time dX(nt)

@time dX(nt)


#=
## Original, Cassette
3.128563 seconds (11.08 M allocations: 577.564 MiB
0.005609 seconds (2.19 k allocations: 152.170 KiB)

# No Overdubs (Best Case Senario)
0.794261 seconds (2.49 M allocations: 120.038 MiB, 7.86% gc time)
0.000015 seconds (23 allocations: 3.641 KiB)

# IRTools, with precompile.jl
0.881375 seconds (2.98 M allocations: 146.369 MiB, 6.39% gc time)
0.000031 seconds (33 allocations: 4.188 KiB)

### Cassette with precompile.jl
2.838656 seconds (11.13 M allocations: 580.525 MiB, 7.72% gc time)
0.004810 seconds (2.19 k allocations: 152.170 KiB)
=#
