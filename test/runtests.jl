using OrthogonalPolynomialsAlgebraicCurves, BlockBandedMatrices, BlockArrays, Test


X,Y = quarticjacobi(10)
@test X[Block(5,5)] == # TODO: correct entries