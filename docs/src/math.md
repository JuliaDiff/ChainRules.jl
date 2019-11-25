# On the Math of ChainRules.jl
This page explain the mathematical underpinnings of ChainRulesCore.jl.
Primarily in the abstract.
It should not be read as documentation on how things work,
but rather on why things work the way they do.

It is written around the Julia idea of types.
Which do not nessicarily correspond too closely with the idea of Programming Language Theory (static) types,
nor with type-theory types.
The key feature of the the Julia type system is that it enables multiple dispatch.

As Julia is a dynamic programming language functions can return different types of values depending on the values of their inputs (i.e. not be type stable).

Bacause of multiple dispatch,
all functions are able to be define different _methods_
dependencing on the types of the inputs.
We thus can have defined different ``+``
for different input types, and so will not distinguish between them.
The functions thus stand alone from the types,
except that a set of input type may special case them.
Where as most similar definitions might decribe a object as a type and some operations on the type,
we can consider them seperately.

!!! terminology Notation:
    - ``d::\mathcal D``, a value ``d`` of type ``\mathcal D``, or the assertion that the value ``d`` has type ``\mathcal D``

## Part 1: What is a Differential ?

We begin by considering some function we would like to differentiate.
For the sake of convention,
we will consider all functions as having 1 input and 1 output;
but that input and output may be a composite object, such as a struct or tuple.
Any valid input or output of a function, or any component there of, has a type.
Such a type we will call a **Primal Type**.

Roughly speaking, the **Differential Type** is a type that represents the difference between two primal type value.
For a given Primal type there will often be multiple
valid Differential types.
For example, for the Primal Type of `DateTime`, the valid Differential Types include: `Millisecond`, `Second`, `Hour`, `Day` etc.

!!! note
    There will always be one (the **Zero**).
    If there is only that one then one normally says that the type is not differentiable.
    For example booleans, and integers except when they are being uses as special cases of real numbers; as a computational optimization.


### Direct Differential Type
Formally speaking:
For a given Primal Type ``\mathcal P``,
qnd a type ``\mathcal D``.
If for all values ``d::\mathcal D``
and for all values ``p::\mathcal P``
there exists a ``q::\mathcal P`` such that
``p + d = d + p = q``.
Then we say that  ``\mathcal D`` is a direct-differential type for the primal type ``\mathcal P``.
Which we write as ``\mathcal D \triangleleft \mathcal P``.

### Valid Differential Types
As a stronger condition.
For a Primal Type ``\mathcal P``,
qnd a set of types ``\mathbb D = \lbrace \mathcal D_1,\; \mathcal D_2,\; \ldots\rbrace`` all of which are direct-differentials for ``\mathcal P``.

If for all ``\mathcal D_i,\; \mathcal D_j\; \in \mathbb D``,
there exists ``d_k :: \mathcal D_k`` for ``\mathcal D_k\in \mathbb D``
with ``d_i :: \mathcal D_i``, ``d_j :: \mathcal D_j``,
such that ``d_i + d_j = d_j + d_i = d_k``.

Then we say that ``\mathbb D`` is a set of valid differentials for ``\mathcal P``.
And we say that each ``\mathcal D_i \in \mathbb D`` is a valid differential type for ``\mathcal P``.
And we write ``D_i \blacktriangleleft \mathcal P``.

The informal version of this if the direct-differentials are closed under addition to all the direct-differentials, then they are all valid differentials.

TODO: CHeck the order of all the foralls and exists inm the above
