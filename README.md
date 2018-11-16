# ChainRules

The ChainRules package provides a variety of common utilities that can be used
by downstream automatic differentiation (AD) tools to define and execute
forward-, reverse-, and mixed-mode primitives.

This package is a WIP; the framework is essentially there, but there are only a
few toy rules right now, a bunch of TODOs, virtually no tests, etc. PRs welcome!
Documentation is incoming, which should help if you'd like to contribute.

Here are some of the basic goals for the package:

- First-class support for complex differentiation via Wirtinger derivatives.

- Mixed-mode composability without being coupled to a specific AD implementation.

- Propagation semantics built-in, with default implementations that allow rule
authors to easily opt-in to common optimizations (fusion, increment elision, etc.).

- Control-inverted design: rule authors can fully specify derivatives in
a concise manner while naturally allowing the caller to compute only what they
need.

- Genericity/Overloadability: rules are well-specified independently of target
function's input/output values' types, though these types can be specialized
on when desired. Furthermore, properties like storage device, tensor shape,
domain etc. can be specified by callers (and thus exploited by rule authors)
independently of these types.

The ChainRules source code follows the [YASGuide](https://github.com/jrevels/YASGuide).
