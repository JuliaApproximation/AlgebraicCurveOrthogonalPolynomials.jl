
using RandomMatrices, ForwardDiff, BlockBandedMatrices, BlockArrays, FillArrays


###
# rand orthog
###


import ForwardDiff: jacobian

B = rand(Haar(1),2)/2



A = 0randn(2,2)


z = 1.2+eps()im
eq = X -> X + B*inv(X)*B' - (A - z * I)
eq2 = function(x)
    X = [x[1] x[2]; x[2] x[3]] + im  *  [x[4] x[5]; x[5] x[6]]
    ret = eq(X)
    [real(ret[1,1]); real(ret[1,2]); real(ret[2,2]); imag(ret[1,1]); imag(ret[1,2]); imag(ret[2,2])]
end

x = randn(6)
jacobian(eq2,x)
# x = randn(8)
x = x - jacobian(eq2,x) \ eq2(x)
X = [x[1] x[2]; x[2] x[3]] + im  *  [x[4] x[5]; x[5] x[6]]




X = randn(2,2)
X = eq(X)



N = 10; T = BlockBandedMatrix(mortar(Tridiagonal(fill(Matrix(B'),N-1), fill(zeros(2,2),N), fill(B,N-1))))

z = 1.2
(T - z*I) \ Matrix(I, size(T,1),2)


###
# Explicit construction of UL
#
# We know L has the structure
#
# […    …
# [… A  B
# [  B' A   B
# [     B'  X
# [         B' X
#
# where apply elimination we multply the interface rows by 
#
#  [ I -B*inv(X)
#  [ 0  I]
# 
#  Which leads to the result that X satisfies
#
#   A - B*inv(X)*B' = X
###



d = 2
A = randn(d,d); A = A + A'
B = tril(randn(d,d))

N = 20; T = BlockBandedMatrix(mortar(Tridiagonal(fill(Matrix(B'),N-1), fill(A,N), fill(B,N-1))))

z = 5.1

X = A
for _=1:1000
    global X
    X = (A-z*I) - B*inv(X)*B'
end

Y = -inv(X)
@test (A-z*I) + B*Y*B' ≈ -inv(Y)

Z = sqrt(Y) * B * sqrt(Y)
@test sqrt(Y)*(A-z*I)*sqrt(Y) + Z*Z' ≈ -I 
@test sqrt(Y)*(A-z*I)*sqrt(Y) + Z*Z' ≈ -I 

Z

sqrt(Y) * B * Y * B' * sqrt(Y)

X
@test ((T -z*I) \ [Matrix(I,d,d); zeros(size(T,1)-d,d)])[Block(1,1)] ≈ inv(X)

A[1,1] - z  - B[1,1]^2*inv(X)[1,1]
λ,Q = eigen(Symmetric(X))
Λ = Diagonal(λ)
(A-z*I) - B*Q*inv(Λ)*Q'*B' ≈ Q*Λ*Q'
Q'*(A-z*I)*Q - Q'*B*Q*inv(Λ)*Q'*B'*Q ≈ Λ

X
-B


N = 6000; T = BlockBandedMatrix(mortar(Tridiagonal(fill(Matrix(B'),N-1), fill(zeros(2,2),N), fill(B,N-1))))

z = 3.2
((T - z*I) \ Matrix(I, size(T,1),2))[Block(1,1)]


@time (T - z*I) \ [1; zeros(size(T,1)-1)];





