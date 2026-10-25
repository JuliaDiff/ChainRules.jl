using ChainRules
using ParallelTestRunner

# Shared test helpers are defined once per worker process. This includes some piracy
# (`Adapt` methods for the mock GPU array), which should only be evaluated once.
const init_worker_code = quote
    include($(joinpath(@__DIR__, "testutils.jl")))
end

# Each test file runs in its own sandbox module and loads the packages it needs itself.
const init_code = quote
    using Main.ChainRulesTestHelpers
end

# Every file under `test/` (except this one) is a test, named by its path, e.g. "rulesets/Base/array".
# Select tests via `Pkg.test("ChainRules"; test_args=["rulesets/LinearAlgebra"])`,
# exclude them with a leading `!`, and list all with `test_args=["--list"]`.
const testsuite = find_tests(@__DIR__)
delete!(testsuite, "testutils")

runtests(ChainRules, ARGS; testsuite, init_code, init_worker_code)
