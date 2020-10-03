using Base.Broadcast: broadcastable
using ChainRules
using ChainRulesCore
using ChainRulesTestUtils
using ChainRulesTestUtils: _fdm
using Compat: only
using FiniteDifferences
using LinearAlgebra
using LinearAlgebra.BLAS
using LinearAlgebra: dot
using Random
using Statistics
using Test

Random.seed!(1) # Set seed that all testsets should reset to.

println("Testing ChainRules.jl")
@testset "ChainRules" begin
    @testset "rulesets" begin
        @testset "Base" begin
            include(joinpath("rulesets", "Base", "nondiff.jl"))
            include(joinpath("rulesets", "Base", "base.jl"))
            include(joinpath("rulesets", "Base", "fastmath_able.jl"))
            include(joinpath("rulesets", "Base", "evalpoly.jl"))
            include(joinpath("rulesets", "Base", "array.jl"))
            include(joinpath("rulesets", "Base", "arraymath.jl"))
            include(joinpath("rulesets", "Base", "mapreduce.jl"))
        end

        print(" ")

        @testset "Statistics" begin
            include(joinpath("rulesets", "Statistics", "statistics.jl"))
        end

        print(" ")

        @testset "LinearAlgebra" begin
            include(joinpath("rulesets", "LinearAlgebra", "dense.jl"))
            include(joinpath("rulesets", "LinearAlgebra", "structured.jl"))
            include(joinpath("rulesets", "LinearAlgebra", "factorization.jl"))
            include(joinpath("rulesets", "LinearAlgebra", "blas.jl"))
        end

        print(" ")

        @testset "Random" begin
            include(joinpath("rulesets", "Random", "random.jl"))
        end

        print(" ")

        @testset "packages" begin
            include(joinpath("rulesets", "packages", "NaNMath.jl"))
            include(joinpath("rulesets", "packages", "SpecialFunctions.jl"))
        end
    end
end
