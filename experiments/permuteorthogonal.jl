using RandomMatrices

N = 8
Q = rand(Haar(1),N)
W = (I-Q) / (I+Q)
K = nullspace(Matrix(cmjac(W)))
B = reshape(K * randn(size(K,2)), N, N); B = B- B';
V = (I-B) / (I+B)
cm(Q,V)
cm(Q',V)
cm(Q,V')

Bˣ = Q/2
Bʸ = V/2

X = z -> Bˣ/z + z*Bˣ'
Y = z -> Bʸ/z + z*Bʸ'

cm(X(z),Y(z))

scatter(vec(specgrid(X,Y)); legend=:false)




L = log(Q)
K = nullspace(Matrix(cmjac(L)))
B = reshape(K * nullspace(imag(K)), 3, 3)
B
det(Q)

B = reshape(K * randn(size(K,2)), 3, 3); B = B - B';
V = exp(B)

