using OrthogonalPolynomialsQuasi, FillArrays, InfiniteArrays, LazyBandedMatrices, BlockBandedMatrices

T = Chebyshev()
x = axes(T,1)
X = T \ (x .* T)
KronTrav(X,Eye(∞))
KronTrav(Eye(∞),X)