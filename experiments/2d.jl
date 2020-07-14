using OrthogonalPolynomialsQuasi, FillArrays, InfiniteArrays, LazyBandedMatrices, BlockBandedMatrices

T = Chebyshev()
x = axes(T,1)
X = T \ (x .* T)
KronTrav(X,Eye(∞)) # (T[w]z + T[w̄]z̄)/2
KronTrav(Eye(∞),X) # (z + z̄)/2

# Is two-variable symbols the right approach?
X = (z,w) -> (w*z + conj(w)conj(z))/2
Y = (z,w) -> (z + conj(z))/2

