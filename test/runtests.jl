using Test, ChainRulesCore, ChainRulesTestUtils

@nospecialize

using Base.Broadcast: broadcastable
using ChainRules
using ChainRulesCore
using ChainRulesTestUtils
using ChainRulesTestUtils: rand_tangent, _fdm
using FiniteDifferences
using LinearAlgebra
using LinearAlgebra.BLAS
using LinearAlgebra: dot
using Random
using SparseArrays
using StaticArrays
using Statistics
using Test
using JuliaInterpreter

union!(JuliaInterpreter.compiled_modules, Any[Base, Base.Broadcast, LinearAlgebra, Random, StaticArrays, Statistics])

Random.seed!(1) # Set seed that all testsets should reset to.

function include_test(path)
    if isempty(ARGS) || any(occursin(a, path) for a in ARGS)
        println("Testing $path:")  # print so TravisCI doesn't timeout due to no output
        @time Base.include(@__MODULE__(), path) do ex
            Meta.isexpr(ex, :macrocall) && ex.args[1] == Symbol("@testset") || return ex
            return :(@interpret (() -> $ex)())  # interpret testsets using JuliaInterpreter
        end
    else
        # If you provide ARGS like so, then it runs only matching testsets: 
        # Pkg.test("ChainRules", test_args = ["index", "LinearAlgebra"])
        println("(Not testing $path)")
    end
end

if isempty(ARGS)
    println("Testing ChainRules.jl")
else
    println("Testing ChainRules.jl with test_args = ", ARGS)
end

@testset "ChainRules" begin  # One overall @testset ensures it keeps going after failures
    include("test_helpers.jl")  # This can't be skipped
    println()

    test_method_tables()  # Check the global method tables are consistent

    # Each file puts all tests inside one or more @testset blocks
    include_test("rulesets/Base/base.jl")
    include_test("rulesets/Base/fastmath_able.jl")
    include_test("rulesets/Base/evalpoly.jl")
    include_test("rulesets/Base/array.jl")
    include_test("rulesets/Base/arraymath.jl")
    include_test("rulesets/Base/indexing.jl")
    include_test("rulesets/Base/mapreduce.jl")
    include_test("rulesets/Base/sort.jl")
    include_test("rulesets/Base/broadcast.jl")

    include_test("unzipped.jl")  # used primarily for broadcast

    println()

    include_test("rulesets/Statistics/statistics.jl")

    println()

    include_test("rulesets/LinearAlgebra/dense.jl")
    include_test("rulesets/LinearAlgebra/norm.jl")
    include_test("rulesets/LinearAlgebra/matfun.jl")
    include_test("rulesets/LinearAlgebra/structured.jl")
    include_test("rulesets/LinearAlgebra/symmetric.jl")
    include_test("rulesets/LinearAlgebra/factorization.jl")
    include_test("rulesets/LinearAlgebra/blas.jl")
    include_test("rulesets/LinearAlgebra/lapack.jl")
    include_test("rulesets/LinearAlgebra/uniformscaling.jl")

    println()

    include_test("rulesets/SparseArrays/sparsematrix.jl")

    println()

    include_test("rulesets/Random/random.jl")
    println()
end
