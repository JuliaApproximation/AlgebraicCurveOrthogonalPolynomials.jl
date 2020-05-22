using OrthogonalPolynomialsQuasi

T = Chebyshev()
x = axes(T,1)
X = T \ (x .* T)