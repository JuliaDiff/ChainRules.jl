# Differentials

```@docs
ChainRules.AbstractDifferential
ChainRules.extern
```

## Wirtinger Differentials

```@docs
ChainRules.Wirtinger
```

Note that multiplication of two `Wirtinger` objects is intentionally not defined.
This is because application of the chain rule often expands into a non-commutative
operation in the Wirtinger calculus.
That is, simply given two `Wirtinger` objects and no other information, we can't know
which components to conjugate in order to implement the chain rule.
We could pick a convention, e.g. we could define `*(a::Wirtinger, b::Wirtinger)` such
that we assume the chain rule application is of the form ``f_a \circ f_b`` instead of
``f_b \circ f_a``.
However, picking such a convention is likely to lead to silently incorrect derivatives
due to commutativity assumptions made in downstream general code that deals with real
numbers.
Thus ChainRules opts to make this operation an error.
