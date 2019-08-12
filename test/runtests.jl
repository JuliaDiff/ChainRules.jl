using Base.Broadcast: broadcastable
using ChainRules
using ChainRulesCore
using FiniteDifferences
using LinearAlgebra
using LinearAlgebra.BLAS
using LinearAlgebra: dot
using Random
using Statistics
using Test

# For testing purposes we use a lot of
using ChainRulesCore: add, cast, extern, accumulate, accumulate!, store!, @scalar_rule,
    Wirtinger, wirtinger_primal, wirtinger_conjugate, add_wirtinger, mul_wirtinger,
    Zero, add_zero, mul_zero, One, add_one, mul_one, Casted, cast, add_casted, mul_casted,
    DNE, Thunk, Casted, DNERule

include("test_util.jl")

@testset "ChainRules" begin
    include("helper_functions.jl")
    @testset "rulesets" begin
        @testset "Base" begin
            include(joinpath("rulesets", "Base", "base.jl"))
            include(joinpath("rulesets", "Base", "array.jl"))
            include(joinpath("rulesets", "Base", "mapreduce.jl"))
            include(joinpath("rulesets", "Base", "broadcast.jl"))
        end

        @testset "LinearAlgebra" begin
            include(joinpath("rulesets", "LinearAlgebra", "dense.jl"))
            include(joinpath("rulesets", "LinearAlgebra", "structured.jl"))
            include(joinpath("rulesets", "LinearAlgebra", "factorization.jl"))
            include(joinpath("rulesets", "LinearAlgebra", "blas.jl"))
        end

        @testset "packages" begin
            include(joinpath("rulesets", "packages", "NaNMath.jl"))
            include(joinpath("rulesets", "packages", "SpecialFunctions.jl"))
        end
    end
end
