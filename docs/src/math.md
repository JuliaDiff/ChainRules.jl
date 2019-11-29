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
depending on the types of the inputs.
We thus can have defined different ``+``
for different input types, and so will not distinguish between them.
The functions thus stand alone from the types,
except that a set of input types may special case them.
Where as most similar definitions might decribe a object as a type and some operations on the type,
we can consider them seperately.

!!! terminology "Notation"
    We use some notation here based closely off of how JuliaLang indicates type relationships. </br>
    - ``d::\mathcal D``: a value ``d`` of type ``\mathcal D``, or the assertion that the value ``d`` has type ``\mathcal D`` </br>
    - ``\mathcal D <: \mathbb D``: the type ``\mathcal D`` is a subtype of the type ``\mathbb D``. In particular, we are only concerned with the case of ``\mathbb D`` being a type-union. So in this case it can be seen as also saying that ``\mathcal D`` is a member of the type-union ``\mathbb D``.

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

### Differential Type

Consider a Primal Type ``\mathcal P``.
Consider some type ``\mathcal D``.


 - If there exists a type-union ``\mathbb U``, with ``\mathcal D <: \mathbb U``,
 - if for all ``u :: \mathbb U`` and for all ``p :: \mathcal P``, there exists a ``q :: \mathcal P`` such that `u + p = p + u = q`
 - and for all ``d :: \mathcal D``, and for all ``x :: \mathbb U``,  exists ``s :: \mathbb U`` such that ``d + x = x + d = s``
(!!# TODO should this be on just ``d :: \mathcal D`` or on all of U)


then we say that ``\mathcal D`` is a (valid) differential type for ``\mathcal P``.
And we write this as ``\mathcal D \triangleleft \mathcal P``.

Note: in this case it is also true that every other type in the type-union ``\mathbb D``, and and indeed ``\mathbb D`` itself are also valid differential types for ``\mathcal P``.


The short version of this is that a differential type is one that is a member of some type-union that is closed  under addition, and that can be added to the primal type to give an element of the primal type

The important take away is you can add instances of all valid differential types for ``\mathcal P``,
and know you will always get an instance of valid differential type back.
