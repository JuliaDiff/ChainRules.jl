using Base.Broadcast: broadcastable
using ChainRules
using ChainRulesCore
using ChainRulesTestUtils
using ChainRulesTestUtils: rand_tangent, _fdm
using Compat: hasproperty, only
using FiniteDifferences
using FiniteDifferences: rand_tangent
using SpecialFunctions
using LinearAlgebra
using LinearAlgebra.BLAS
using LinearAlgebra: dot
using Random
using Statistics
using Test

Random.seed!(1) # Set seed that all testsets should reset to.
BLAS.set_num_threads(1)

function include_test(path)
    println("Testing $path:")  # print so TravisCI doesn't timeout due to no output
    @time include(path)  # show basic timing, (this will print a newline at end)
end

if VERSION >= v"1.3"
    using Base.Threads: @spawn, @sync
else
    macro spawn(ex) esc(ex) end
    macro sync(ex) esc(ex) end
end

println("Testing ChainRules.jl")
@testset "ChainRules" begin
    @testset "rulesets" begin
        @sync begin
            @testset "Base" begin
                @sync begin
                    @spawn include_test("rulesets/Base/base.jl")
                    @spawn include_test("rulesets/Base/fastmath_able.jl")
                    @spawn include_test("rulesets/Base/evalpoly.jl")
                    @spawn include_test("rulesets/Base/array.jl")
                    @spawn include_test("rulesets/Base/arraymath.jl")
                    @spawn include_test("rulesets/Base/indexing.jl")
                    @spawn include_test("rulesets/Base/mapreduce.jl")
                    @spawn include_test("rulesets/Base/sort.jl")
                end
            end
            println()

            Threads.@spawn @testset "Statistics" begin
                include_test("rulesets/Statistics/statistics.jl")
            end
            println()

            @testset "LinearAlgebra" begin
                @sync begin
                    @spawn include_test("rulesets/LinearAlgebra/dense.jl")
                    @spawn include_test("rulesets/LinearAlgebra/norm.jl")
                    @spawn include_test("rulesets/LinearAlgebra/matfun.jl")
                    @spawn include_test("rulesets/LinearAlgebra/structured.jl")
                    @spawn include_test("rulesets/LinearAlgebra/symmetric.jl")
                    @spawn include_test("rulesets/LinearAlgebra/factorization.jl")
                    @spawn include_test("rulesets/LinearAlgebra/blas.jl")
                    @spawn include_test("rulesets/LinearAlgebra/lapack.jl")
                end
            end
            println()

            @spawn @testset "Random" begin
                include_test("rulesets/Random/random.jl")
            end
            println()
        end
    end
end
