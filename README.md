# ChainRules

[![Travis](https://travis-ci.org/JuliaDiff/ChainRules.jl.svg?branch=master)](https://travis-ci.org/JuliaDiff/ChainRules.jl)
[![Coveralls](https://coveralls.io/repos/github/JuliaDiff/ChainRules.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaDiff/ChainRules.jl?branch=master)

**Docs:* 
[![](https://img.shields.io/badge/docs-master-blue.svg)](https://JuliaDiff.github.io/ChainRules.jl/dev)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaDiff.github.io/ChainRules.jl/stable)

The ChainRules package provides a variety of common utilities that can be used by downstream automatic differentiation (AD) tools to define and execute forward-, reverse-, and mixed-mode primitives.

This package is a WIP; the framework is essentially there, but there are a bunch of TODOs, virtually no tests, etc. PRs welcome! Documentation is incoming, which should help if you'd like to contribute.

Here are some of the basic goals for the package:

- First-class support for complex differentiation via Wirtinger derivatives.

- Mixed-mode composability without being coupled to a specific AD implementation.

- Propagation semantics built-in, with default implementations that allow rule authors to easily opt-in to common optimizations (fusion, increment elision, memoization, etc.).

- Control-inverted design: rule authors can fully specify derivatives in a concise manner while naturally allowing the caller to compute only what they need.

The ChainRules source code follows the [YASGuide](https://github.com/jrevels/YASGuide).
