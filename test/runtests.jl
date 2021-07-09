using ForwardDiff: derivative
using Base: beginsym
using AlgebraicCurveOrthogonalPolynomials, LinearAlgebra, BlockBandedMatrices, BlockArrays, FillArrays, Test

include("test_arc.jl")
include("test_annulus.jl")