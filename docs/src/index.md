```@meta
DocTestSetup = :(using ChainRules)
CurrentModule = ChainRules
```

# ChainRules.jl

The ChainRules package provides a framework of common utilities and derivative definitions
for use by upstream automatic differentiation packages.
Its primary design goals include:

* Mixed-mode composability without being tied to a particular AD implementation

* Built-in propagation semantics with default implementations that allow derivative rule
  authors to easily opt into common optimizations, like broadcast fusion, increment
  elision, and memoization

* First-class support for differentiating functions of complex numbers using Wirtinger
  derivatives

* Control-inverted design: Rules authors can fully specify derivatives in a concise manner
  while allowing the caller to compute only what they need

ChainRules is currently a work in progress, and contributions are welcome!

## Contents

```@contents
Pages = [
    "differentials.md",
    "rules.md",
]
Depth = 1
```
