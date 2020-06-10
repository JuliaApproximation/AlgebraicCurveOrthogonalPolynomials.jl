using OrthogonalPolynomialsAlgebraicCurves, ForwardDiff, Test, Plots
import OrthogonalPolynomialsAlgebraicCurves: symunroll, comunroll, cm, speccurvemat, spec2alg, evalmonbasis, symroll
import ForwardDiff: jacobian

# explore 2 x 2 commuting Laurent polynomials,
# begining with A_x = A_y = 0, that is,
#
# X(z) = A_x + B_x/z + z*B_x'
# Y(z) = A_y + B_y/z + z*B_y'
# 
# These satisfy
# 
# X(z) Y(z) = Y(z) X(z)
# 
# i.e.
#
# 0 = [B_x,B_y]/z^2 + ([A_x,B_y] + [B_x,A_y])/z + [B_x,B_y'] + [B_x',B_y] + [A_x,A_y]  + c.c.
#
# First we need B_x and B_y to commute. Is that enough? No.

N = 3
Aˣ, (λˣ, V), κʸ = Symmetric(randn(N,N)), (randn(N),randn(N,N)), randn(3)
X, Y = speccurve(Aˣ, (λˣ, V), κʸ)

z = exp(0.1im)
@test X(z)Y(z) ≈ Y(z)X(z)

c = spec2alg(X,Y)
p = (x,y) -> evalmonbasis(N, x+im*y)'*c

x = y = range(-5,5; length=50)
contour(x,y, p.(x',y); nlevels=500)
scatter!(vec(specgrid(X,Y)))

z = exp(0.1im)
@test X(z)Y(z) ≈ Y(z)X(z)

@test norm(p.(real.(vec(specgrid(X,Y))), imag.(vec(specgrid(X,Y))))) ≤ 1E-12


# try half-arc
# X^2 = B_x^2/z^2 + (A_x*B_x + B_x*A_x)/z + A_x^2 + B_x*B_x' + B_x'*B_x + cc
# Y^2 = B_y^2/z^2 + (A_y*B_y + B_y*A_y)/z + A_y^2 + B_y*B_y' + B_y'*B_y + cc
# So we have
#
# 1) B_x^2 + B_y^2 = 0
# 2) A_x*B_x + B_x*A_x + A_y*B_y + B_y*A_y = 0
# 3) A_x^2 + B_x*B_x' + B_x'*B_x + A_y^2 + B_y*B_y' + B_y'*B_y = I
#

conds = function(p)
    N = round(Int,(-3 + sqrt(3)sqrt(-21 + 8length(p))) / 6)
    @assert sum(1:N) + N + N^2 + 3 == length(p)
    Aˣ = Symmetric(symunroll(p[1:sum(1:N)]))
    λˣ = p[sum(1:N)+1:sum(1:N)+N]
    V  = reshape(p[sum(1:N)+N+1:sum(1:N)+N+N^2],N,N)
    κʸ = p[sum(1:N)+N+N^2+1:end]
    (Aˣ,Bˣ),(Aʸ,Bʸ) = speccurvemat(Aˣ, (λˣ, V), κʸ)
    [vec(Bˣ^2 + Bʸ^2); 
    vec(Aˣ*Bˣ + Bˣ*Aˣ + Aʸ*Bʸ + Bʸ*Aʸ); 
    vec(Aˣ^2 + Bˣ*Bˣ' + Bˣ'*Bˣ + Aʸ^2 + Bʸ*Bʸ' + Bʸ'*Bʸ);
    eigvals(Symmetric(Aˣ + Bˣ + Bˣ')) - [1; 1];
    eigvals(Symmetric(Aʸ - Bʸ - Bʸ')) - [-1; 1]
    ]
end

p = [[0.5; 0; 0.5]; [0.25; 0.25]; ]


N = 2
p = randn(sum(1:N) + N + N^2 + 3)
J = jacobian(conds,p); p = p - (J \ conds(p)); conds(p)

jacobian(conds,p)
h = 0.01; p̃ = copy(p); p̃[1] += h;  (conds(p̃) - conds(p))/h





###
# Don't try to make commuting apriori
###

conds_enough = function(p)
    m = length(p)
    N = round(Int,(-1 + sqrt(1 + 12m))/6)
    @assert 2sum(1:N) + 2N^2 == length(p)
    Aˣ = Symmetric(symunroll(p[1:sum(1:N)]))
    Aʸ = Symmetric(symunroll(p[sum(1:N)+1:2sum(1:N)]))
    Bˣ = reshape(p[2sum(1:N)+1:2sum(1:N)+N^2],N,N)
    Bʸ = reshape(p[2sum(1:N)+N^2+1:end],N,N)
    [vec(Bˣ^2 + Bʸ^2); 
    vec(Aˣ*Bˣ + Bˣ*Aˣ + Aʸ*Bʸ + Bʸ*Aʸ); 
    vec(Aˣ^2 + Bˣ*Bˣ' + Bˣ'*Bˣ + Aʸ^2 + Bʸ*Bʸ' + Bʸ'*Bʸ - I);
    # eigvals(Symmetric(Aˣ + Bˣ + Bˣ')) - [1; 1];
    # eigvals(Symmetric(Aʸ - Bʸ - Bʸ')) - [-1; 1];
    # Aʸ[1,1];
    # Aʸ[2,2]
    vec(Aˣ + Bˣ + Bˣ' - [1 0; 0 1]);
    vec(Aˣ - Bˣ - Bˣ');
    vec(Aʸ + Bʸ + Bʸ');
    vec(Aʸ - Bʸ - Bʸ' - [0 -1; -1 0]);
    ]
end


conds = function(p)
    m = length(p)
    N = round(Int,(-1 + sqrt(1 + 12m))/6)
    @assert 2sum(1:N) + 2N^2 == length(p)
    Aˣ = Symmetric(symunroll(p[1:sum(1:N)]))
    Aʸ = Symmetric(symunroll(p[sum(1:N)+1:2sum(1:N)]))
    Bˣ = reshape(p[2sum(1:N)+1:2sum(1:N)+N^2],N,N)
    Bʸ = reshape(p[2sum(1:N)+N^2+1:end],N,N)
    # cond1 = (Aʸ,Bʸ) -> cm(Aˣ,Bʸ) + cm(Bˣ,Aʸ)
    # cond0 = (Aʸ,Bʸ) -> cm(Bˣ,Bʸ') + cm(Bˣ',Bʸ) + cm(Aˣ, Aʸ)
    [
        vec(cm(Bˣ,Bʸ));
        vec(cm(Aˣ,Bʸ) + cm(Bˣ,Aʸ));
        vec(cm(Bˣ,Bʸ') + cm(Bˣ',Bʸ) + cm(Aˣ, Aʸ));
    vec(Bˣ^2 + Bʸ^2); 
    vec(Aˣ*Bˣ + Bˣ*Aˣ + Aʸ*Bʸ + Bʸ*Aʸ); 
    vec(Aˣ^2 + Bˣ*Bˣ' + Bˣ'*Bˣ + Aʸ^2 + Bʸ*Bʸ' + Bʸ'*Bʸ - I);
    eigvals(Symmetric(Aˣ + Bˣ + Bˣ')) - [1;1];
    eigvals(Symmetric(Aˣ - Bˣ - Bˣ'));
    eigvals(Symmetric(Aʸ + Bʸ + Bʸ'));
    eigvals(Symmetric(Aʸ - Bʸ - Bʸ')) - [-1; 1];
    Aʸ[1,2]
    ]
end

N = 2
a₁₂ = (1 + sqrt(2))/4
a₂₁ = (1 - sqrt(2))/4
p = [[0.5,0,0.5]; [0,-0.5,0]; [0.25,0,0,0.25]; vec([0 a₁₂; a₂₁ 0])]

Aˣ = Symmetric(symunroll(p[1:sum(1:N)]))
Aʸ = Symmetric(symunroll(p[sum(1:N)+1:2sum(1:N)]))
Bˣ = reshape(p[2sum(1:N)+1:2sum(1:N)+N^2],N,N)
Bʸ = reshape(p[2sum(1:N)+N^2+1:end],N,N)

_,Q = eigen(Aʸ)
Aˣ = Q'Aˣ*Q
Aʸ = Q'Aʸ*Q
Bˣ = Q'Bˣ*Q
Bʸ = Q'Bʸ*Q

p_ex = [[Aˣ[1,1]; Aˣ[:,2]]; [Aʸ[1,1]; Aʸ[:,2]]; vec(Bˣ); vec(Bʸ)]


@test norm(conds(p)) ≤ 1E-15

# 
p .+= 0.01randn.()

p = randn(2(sum(1:N) + N^2))
p = p - jacobian(conds,p) \ conds(p); norm(conds(p))



###
# 2 arcs
###

conds = function(p)
    m = length(p)
    N = round(Int,(-1 + sqrt(1 + 12m))/6)
    @assert 2sum(1:N) + 2N^2 == length(p)
    Aˣ = Symmetric(symunroll(p[1:sum(1:N)]))
    Aʸ = Symmetric(symunroll(p[sum(1:N)+1:2sum(1:N)]))
    Bˣ = reshape(p[2sum(1:N)+1:2sum(1:N)+N^2],N,N)
    Bʸ = reshape(p[2sum(1:N)+N^2+1:end],N,N)
    # cond1 = (Aʸ,Bʸ) -> cm(Aˣ,Bʸ) + cm(Bˣ,Aʸ)
    # cond0 = (Aʸ,Bʸ) -> cm(Bˣ,Bʸ') + cm(Bˣ',Bʸ) + cm(Aˣ, Aʸ)

    x0,x1 = [-1.,-1.],[1.0,1.0]
    y0,y1 = (-1,1) .* sqrt.(1 .- x0.^2), (-1,1) .* sqrt.(1 .- x1.^2)

    [
        vec(cm(Bˣ,Bʸ));
        vec(cm(Aˣ,Bʸ) + cm(Bˣ,Aʸ));
        vec(cm(Bˣ,Bʸ') + cm(Bˣ',Bʸ) + cm(Aˣ, Aʸ));
    vec(Bˣ^2 + Bʸ^2); 
    vec(Aˣ*Bˣ + Bˣ*Aˣ + Aʸ*Bʸ + Bʸ*Aʸ); 
    vec(Aˣ^2 + Bˣ*Bˣ' + Bˣ'*Bˣ + Aʸ^2 + Bʸ*Bʸ' + Bʸ'*Bʸ - I);
    eigvals(Symmetric(Aˣ - Bˣ - Bˣ')) - x0;
    eigvals(Symmetric(Aʸ - Bʸ - Bʸ')) - y0;
    eigvals(Symmetric(Aˣ + Bˣ + Bˣ')) - x1;
    eigvals(Symmetric(Aʸ + Bʸ + Bʸ')) - y1;
    Aʸ[1,2]
    ]
end

# p = randn(2(sum(1:N) + N^2))
p = [[Aˣ[1,1]; Aˣ[:,2]]; [Aʸ[1,1]; Aʸ[:,2]]; vec(Bˣ); vec(Bʸ)]
p = p - jacobian(conds,p) \ conds(p); norm(conds(p))


##
# N = 4
##
conds = function(p)
    m = length(p)
    N = round(Int,(-1 + sqrt(1 + 12m))/6)
    @assert 2sum(1:N) + 2N^2 == length(p)
    Aˣ = Symmetric(symunroll(p[1:sum(1:N)]))
    Aʸ = Symmetric(symunroll(p[sum(1:N)+1:2sum(1:N)]))
    Bˣ = reshape(p[2sum(1:N)+1:2sum(1:N)+N^2],N,N)
    Bʸ = reshape(p[2sum(1:N)+N^2+1:end],N,N)
    # cond1 = (Aʸ,Bʸ) -> cm(Aˣ,Bʸ) + cm(Bˣ,Aʸ)
    # cond0 = (Aʸ,Bʸ) -> cm(Bˣ,Bʸ') + cm(Bˣ',Bʸ) + cm(Aˣ, Aʸ)

    x0,x1 = zeros(N),fill(1,N)
    y0,y1 = [-1,-1,1,1],zeros(N)

    [
        vec(cm(Bˣ,Bʸ));
        vec(cm(Aˣ,Bʸ) + cm(Bˣ,Aʸ));
        vec(cm(Bˣ,Bʸ') + cm(Bˣ',Bʸ) + cm(Aˣ, Aʸ));
    vec(Bˣ^2 + Bʸ^2); 
    vec(Aˣ*Bˣ + Bˣ*Aˣ + Aʸ*Bʸ + Bʸ*Aʸ); 
    vec(Aˣ^2 + Bˣ*Bˣ' + Bˣ'*Bˣ + Aʸ^2 + Bʸ*Bʸ' + Bʸ'*Bʸ - I);
    eigvals(Symmetric(Aˣ - Bˣ - Bˣ')) - x0;
    eigvals(Symmetric(Aʸ - Bʸ - Bʸ')) - y0;
    eigvals(Symmetric(Aˣ + Bˣ + Bˣ')) - x1;
    eigvals(Symmetric(Aʸ + Bʸ + Bʸ')) - y1;
    Aʸ[1,2];
    Aʸ[1:2,3];
    Aʸ[1:3,4];
    ]
end

N = 4
# p = randn(2(sum(1:N) + N^2))

p .+= 0.0001randn.()
p = p - jacobian(conds,p) \ conds(p); norm(conds(p))

p_ex = [

Aˣ4 = mortar(Diagonal(Matrix{Float64}[Aˣ, Aˣ]))
Aʸ4 = mortar(Diagonal(Matrix{Float64}[Aʸ, Aʸ]))
Bˣ4 = mortar(Diagonal([Bˣ, Bˣ]))
Bʸ4 = mortar(Diagonal([Bʸ, Bʸ]))

p_ex4 = [symroll(Aˣ4); symroll(Aʸ4); vec(Bˣ4); vec(Bʸ4)]
p = p_ex4


norm(conds(p))


###
# X^2 = B^2/z^2 + 
#    (A*B + B*A)/z +
#    A^2 + B*B' + B'*B + …
# X^4 = B^4/z^4 + 
#    (B^3*A + B^2*A*B + B*A*B^2 + A*B^3)/z^3 +
#    (A*B*A*B + A*B^2*A + B*A^2*B + B*A*B*A + B^2*A^2 + A^2*B^2 + B^3*B' + B^2*B'*B + B*B'*B^2 + B'*B^3)/z^2 +
#    (B^2*B'*A' + B^2*A'*B' + A*B*A^2 + A*B^2*B' + A*B*B'*B + B*A^3 + B*A*B*B' + B*A*B'*B + 
# 
###

"""
 Squares B/z + A + B'*z 
"""
lrntsquare(B, A) = (B^2, B*A+A*B, B*B'+A^2 + B'B)
"""
 Squares C/z^2 + B/z + A + B'*z + C'*z^2
"""
lrntsquare(C, B, A) = (C^2, C*B + B*C, C*A + B^2 + A*C, C*B' + B*A + A*B + B'*C, C*C' + B*B' + A^2 + B'B + C'C)

function jointeigen(A::Symmetric, B::Symmetric)
    _,Q = eigen(A + 1.23409304233B)
    x,y = diag(Q'A*Q), diag(Q'B*Q)
    p = sortperm(x)
    x[p],y[p],Q
end

conds = function(p)
    m = length(p)
    N = round(Int,(-1 + sqrt(1 + 12m))/6)
    @assert 2sum(1:N) + 2N^2 == length(p)
    Aˣ = Symmetric(symunroll(p[1:sum(1:N)]))
    Aʸ = Symmetric(symunroll(p[sum(1:N)+1:2sum(1:N)]))
    Bˣ = reshape(p[2sum(1:N)+1:2sum(1:N)+N^2],N,N)
    Bʸ = reshape(p[2sum(1:N)+N^2+1:end],N,N)
    X² = lrntsquare(Bˣ, Aˣ); X⁴ = lrntsquare(X²...)
    Y² = lrntsquare(Bʸ, Aʸ); Y⁴ = lrntsquare(Y²...)
    # x0,x1 = zeros(N),fill(1,N)
    # y0,y1 = [-1,-1,1,1],zeros(N)
    x1,y1,_ = jointeigen(Symmetric(Aˣ - Bˣ - Bˣ'), Symmetric(Aʸ - Bʸ - Bʸ'))
    x2,y2,_ = jointeigen(Symmetric(Aˣ + Bˣ + Bˣ'), Symmetric(Aʸ + Bʸ + Bʸ'))
    [vec(cm(Bˣ,Bʸ));
        vec(cm(Aˣ,Bʸ) + cm(Bˣ,Aʸ));
        vec(cm(Bˣ,Bʸ') + cm(Bˣ',Bʸ) + cm(Aˣ, Aʸ));
        vec.(X⁴ .+ Y⁴ .- (0I,0I,0I,0I,I))...;
        (x1 .- [-1,0,0,1])...;
        (y1 .- [0,1,-1,0])...;
        (x2 .- [-1,0,0,1])...;
        (y2 .- [0,1,-1,0])...;
    # eigvals(Symmetric(Aˣ - Bˣ - Bˣ'))[1] - x0;
    # eigvals(Symmetric(Aˣ - Bˣ - Bˣ'))[1] + 1;
    # eigvals(Symmetric(Aˣ - Bˣ - Bˣ'))[2];
    # eigvals(Symmetric(Aˣ - Bˣ - Bˣ'))[3];
    # eigvals(Symmetric(Aˣ - Bˣ - Bˣ'))[end] - 1;
    # eigvals(Symmetric(Aʸ - Bʸ - Bʸ'))[1] + 1;
    # eigvals(Symmetric(Aʸ - Bʸ - Bʸ'))[2];
    # eigvals(Symmetric(Aʸ - Bʸ - Bʸ'))[3];
    # eigvals(Symmetric(Aʸ - Bʸ - Bʸ'))[end] - 1;
    # eigvals(Hermitian(Aˣ + Bˣ/im + im*Bˣ'))[1] + 1;
    # eigvals(Hermitian(Aˣ + Bˣ/im + im*Bˣ'))[2];
    # eigvals(Hermitian(Aˣ + Bˣ/im + im*Bˣ'))[3];
    # eigvals(Hermitian(Aˣ + Bˣ/im + im*Bˣ'))[4] - 1;
    # eigvals(Hermitian(Aʸ + Bʸ/im + im*Bʸ'))[1] + 1;
    # eigvals(Hermitian(Aʸ + Bʸ/im + im*Bʸ'))[2];
    # eigvals(Hermitian(Aʸ + Bʸ/im + im*Bʸ'))[3];
    # eigvals(Hermitian(Aʸ + Bʸ/im + im*Bʸ'))[4] - 1;
    # # eigvals(Symmetric(Aˣ + Bˣ + Bˣ')) - x1;
    # eigvals(Symmetric(Aʸ + Bʸ + Bʸ'))[1] + 1;
    # eigvals(Symmetric(Aʸ + Bʸ + Bʸ'))[2];
    # eigvals(Symmetric(Aʸ + Bʸ + Bʸ'))[3];
    # eigvals(Symmetric(Aʸ + Bʸ + Bʸ'))[end] - 1;
    # eigvals(Symmetric(Aˣ + Bˣ + Bˣ'))[1] + 1;
    # eigvals(Symmetric(Aˣ + Bˣ + Bˣ'))[2];
    # eigvals(Symmetric(Aˣ + Bˣ + Bˣ'))[3];
    # eigvals(Symmetric(Aˣ + Bˣ + Bˣ'))[end] - 1;    
    Aʸ[1,2];
    Aʸ[1:2,3];
    Aʸ[1:3,4]]
end

p = randn(52)
p = p - jacobian(conds,p) \ conds(p); norm(conds(p))

Aˣ = Symmetric(symunroll(p[1:sum(1:N)]))
Aʸ = Symmetric(symunroll(p[sum(1:N)+1:2sum(1:N)]))
Bˣ = reshape(p[2sum(1:N)+1:2sum(1:N)+N^2],N,N)
Bʸ = reshape(p[2sum(1:N)+N^2+1:end],N,N)

X = z -> Aˣ + Bˣ/z + z*Bˣ'
Y = z -> Aʸ + Bʸ/z + z*Bʸ'

scatter(vec(specgrid(X,Y)))

z
X(z)^4 + Y(z)^4

eigvals(X(z))
eigvals(Y(z))
X(z)Y(z)
Y(z)X(z)
