module OrthogonalPolynomialsAlgebraicCurves
using GaussQuadrature, FastGaussQuadrature, SpecialFunctions, LinearAlgebra, BlockBandedMatrices

export quarticjacobi

function quarticjacobi(::Type{T}, N) where T
	X = BlockBandedMatrix{T}(undef, [1:3; fill(4,N-3)], [1:3; fill(4,N-3)], (1,1))
	Y = BlockBandedMatrix{T}(undef, [1:3; fill(4,N-3)], [1:3; fill(4,N-3)], (1,1))	
	
	# TODO: generate entries
	
	X,Y
end

quarticjacobi(N) = quarticjacobi(BigFloat, N)


end # module
