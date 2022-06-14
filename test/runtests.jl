using LazyArrays: eltype
using ForwardDiff: derivative, hessian
using Base: beginsym
using AlgebraicCurveOrthogonalPolynomials, LinearAlgebra, BlockBandedMatrices, BlockArrays, FillArrays, Test

include("test_circle.jl")
include("test_arc.jl")
include("test_annulus.jl")
include("test_hermlaurent.jl")