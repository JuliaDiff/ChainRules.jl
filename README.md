# ChainRules

[![Travis](https://travis-ci.org/JuliaDiff/ChainRules.jl.svg?branch=master)](https://travis-ci.org/JuliaDiff/ChainRules.jl)
[![Coveralls](https://coveralls.io/repos/github/JuliaDiff/ChainRules.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaDiff/ChainRules.jl?branch=master)

**Docs:**
[![](https://img.shields.io/badge/docs-master-blue.svg)](https://JuliaDiff.github.io/ChainRules.jl/dev)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaDiff.github.io/ChainRules.jl/stable)

The ChainRules package provides a variety of common utilities that can be used by downstream automatic differentiation (AD) tools to define and execute forward-, reverse-, and mixed-mode primitives.

The Core logic of ChainRules is implemented in [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl).
If adding ChainRules support to your package (defining `rrule`s or `ffules` you only need to depend on that very light-weight package.
This repo: ChainRules.jl is what people actually use, it reexports all the ChainRules functionality,
and has all the rules for the julia standard library.


Here are some of the core features of the package:

- Mixed-mode composability without being coupled to a specific AD implementation.
- Extensible rules: package authors can add rules (and thus AD support) to the functions in their packages in their packages. They don't need to edit the ChainRules repo.
- Control-inverted design: rule authors can fully specify derivatives in a concise manner while naturally allowing the caller to compute only what they need.
- Propagation semantics built-in, with default implementations that allow rule authors to easily opt-in to common optimizations (fusion, increment elision, memoization, etc.).


The ChainRules source code follows the [YASGuide](https://github.com/jrevels/YASGuide).
