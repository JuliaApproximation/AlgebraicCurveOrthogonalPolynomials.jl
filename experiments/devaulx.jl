using BlockBandedMatrices, BlockArrays, InfiniteLinearAlgebra

B = randn(2,2)
A = randn(2,2) - 10I #; A = A + A'
C = Matrix(B')

N = 501; J = mortar(Tridiagonal(fill(C,N-1), fill(A,N), fill(B,N-1)))

function ul(A)
    L,U = lu(A[end:-1:1,end:-1:1], Val(false))
    L[end:-1:1,end:-1:1] , U[end:-1:1,end:-1:1]
end

U,L = ul(Matrix(J))

X = U[1:2,1:2]*L[1:2,1:2]

@test A - B*inv(X)*C ≈ X
@test inv(J)[Block(1,1)] ≈ inv(X)
R = inv(X)*C
@test A - B*R ≈ C*inv(R)
z,V = eigen(R)
@test norm((A-z[1]*B - C/z[1])V[:,1]) ≤ 1E-10
@test det(A-z[1]*B - C/z[1]) ≤ 1E-10
@test det(z[1]*(A)-z[1]^2*B - C) ≤ 1E-10
@test det(z[1]*(A)/B-z[1]^2 *I - C/B) ≤ 1E-10
@test det(z[1]^2 *I - z[1]*(A)/B + C/B) ≤ 1E-10

λ = filter!(λ -> abs(λ) ≤ 1, eigvals([zeros(2,2) Matrix(I,2,2); -C/B (A)/B]))




# inv(X) == R*inv(B)
R = -inv(X)*B
B'*R + A - x*I + B*inv(R)

B*R + A - x*I + B'*R

function op(J, x)
    N = blocksize(J,1)
    PseudoBlockArray([Matrix(I,2,2) zeros(2,2N-2); (J-x*I)[Block.(1:N-1),:]] \ [Matrix(I,2,2); zeros(2N-2,2)], fill(2,N), [2])
end
N = 110; J = mortar(Tridiagonal(fill(C,N-1), fill(A,N), fill(B,N-1)))
P = op(BigFloat.(J), 10)
R = P[Block(N-1,1)]/(P[Block(N,1)])

B'R + A - x*I + B*inv(R)

X̃ = -inv(R*inv(B))
@test A-x*I - B*inv(X̃)*C ≈ X̃
inv(X)

R = -inv(X)*B

## triangular

B = tril(randn(2,2))
x = 10
A = randn(2,2); A = A + A' - x*I
C = Matrix(B')

N = 501; J = mortar(Tridiagonal(fill(C,N-1), fill(A,N), fill(B,N-1)))

function ul(A)
    L,U = lu(A[end:-1:1,end:-1:1], Val(false))
    L[end:-1:1,end:-1:1] , U[end:-1:1,end:-1:1]
end



U,L = ul(Matrix(J - x*I))

