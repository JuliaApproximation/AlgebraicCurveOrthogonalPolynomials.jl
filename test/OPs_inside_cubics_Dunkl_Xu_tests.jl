using ClassicalOrthogonalPolynomials, ContinuumArrays, SemiclassicalOrthogonalPolynomials, Plots, LinearAlgebra, BlockBandedMatrices, BandedMatrices, GenericSVD, GenericLinearAlgebra, Test

# For more details, see experiments\OPs_inside_cubics_Lanczos.ipynb

include("OPs_inside_cubics_Dunkl_Xu.jl")

@testset "Multiplication by x" begin
    nmax = 7
    b = 2.0
    X = XP(nmax,b)
    r, c = size(X)
    # evaluate Pnk at xv and yv
    xv = -0.5
    yv = 0.3
    # test a few of the k = 0 cases
    k = 0
    n = 0
    # generate the required polynomials:
    P = Normalized(SemiclassicalJacobi((b+1)/2, k+0.5, k+0.5, k+0.5))
    lhs = xv*P[(xv + 1)/2,n+1]
    Pnk = (n,x,y) -> P[(x + 1)/2,n+1]
    lhs = xv*Pnk(n,xv,yv)
    rhs = X[1,1]*Pnk(0,xv,yv) + X[1,2]*Pnk(1,xv,yv)
    @test lhs ≈ rhs
    n = 3
    lhs = xv*Pnk(n,xv,yv)
    rhs = X[7,4]*Pnk(n-1,xv,yv) + X[7,7]*Pnk(n,xv,yv) + X[7,11]*Pnk(n+1,xv,yv)
    @test lhs ≈ rhs
    n = nmax
    lhs = xv*Pnk(n,xv,yv)
    rhs = X[r-nmax,r-2nmax]*Pnk(n-1,xv,yv) + X[r-nmax,r-nmax]*Pnk(n,xv,yv) + X[r-nmax,r+1]*Pnk(n+1,xv,yv)
    @test lhs ≈ rhs
    # test a few k = 2 cases
    k = 2
    P = Normalized(SemiclassicalJacobi((b+1)/2, k+0.5, k+0.5, k+0.5))
    n = 2
    Pn2 = (n,x,y) -> P[(x + 1)/2,n-2+1]
    lhs = xv*Pn2(n,xv,yv)
    rhs = X[6,6]*Pn2(n,xv,yv) + X[6,9]*Pn2(n+1,xv,yv)
    @test lhs ≈  rhs
    n = 4
    lhs = xv*Pn2(n,xv,yv)
    rhs = X[13,9]*Pn2(n-1,xv,yv) + X[13,13]*Pn2(n,xv,yv) + X[13,18]*Pn2(n+1,xv,yv)
    @test lhs ≈ rhs
    n = nmax
    lhs = xv*Pn2(n,xv,yv)
    rhs = X[r-nmax+2,r-2nmax+2]*Pn2(n-1,xv,yv) + X[r-nmax+2,r-nmax+2]*Pn2(n,xv,yv) + X[r-nmax+2,r+3]*Pn2(n+1,xv,yv)
    @test lhs ≈ rhs
    # test a few k = 1 cases
    k = 1
    P = Normalized(SemiclassicalJacobi((b+1)/2, k+0.5, k+0.5, k+0.5))
    Pnk = (n,x,y) -> P[(x + 1)/2,n-k+1]
    n = 1
    lhs = xv*Pnk(n,xv,yv)
    rhs = X[3,3]*Pnk(n,xv,yv) + X[3,5]*Pnk(n+1,xv,yv)
    @test lhs ≈  rhs
    n = 3
    lhs = xv*Pnk(n,xv,yv)
    rhs = X[8,8-n]*Pnk(n-1,xv,yv) + X[8,8]*Pnk(n,xv,yv) + X[8,8+n+1]*Pnk(n+1,xv,yv)
    @test lhs ≈ rhs
    n = nmax
    lhs = xv*Pnk(n,xv,yv)
    rhs = X[r-nmax+1,r-2nmax+1]*Pnk(n-1,xv,yv) + X[r-nmax+1,r-nmax+1]*Pnk(n,xv,yv) + X[r-nmax+1,r+2]*Pnk(n+1,xv,yv)
    @test lhs ≈ rhs
    # Test a few k = nmax-1 cases
    k = nmax - 1
    P = Normalized(SemiclassicalJacobi((b+1)/2, k+0.5, k+0.5, k+0.5))
    Pnk = (n,x,y) -> P[(x + 1)/2,n-k+1]
    n = nmax - 1
    lhs = xv*Pnk(n,xv,yv)
    rhs = X[r-nmax-1,r-nmax-1]*Pnk(n,xv,yv) + X[r-nmax-1,r-1]*Pnk(n+1,xv,yv)
    @test lhs ≈ rhs
    n = nmax
    lhs = xv*Pnk(n,xv,yv)
    rhs = X[r-1,r-nmax-1]*Pnk(n-1,xv,yv) + X[r-1,r-1]*Pnk(n,xv,yv) + X[r-1,r+nmax]*Pnk(n+1,xv,yv)
    @test lhs ≈ rhs
    # Test the k = nmax case
    k = nmax
    P = Normalized(SemiclassicalJacobi((b+1)/2, k+0.5, k+0.5, k+0.5))
    Pnk = (n,x,y) -> P[(x + 1)/2,n-k+1]
    n = nmax
    lhs = xv*Pnk(n,xv,yv)
    rhs = X[r,r]*Pnk(n,xv,yv) + X[r,r+nmax+1]*Pnk(n+1,xv,yv)
    @test lhs ≈ rhs
end

@testset "Multiplication by y" begin
    nmax = 7
    b = 2.0
    Y = YP(nmax,b)
    r, c = size(Y)
    xv = -0.5
    yv = 0.3
    ϕ = x -> (1 - x^2)*(b - x)
    # Test a few of the k = 0 cases
    k = 0
    q = Normalized(ClassicalOrthogonalPolynomials.Legendre())
    Pk = Normalized(SemiclassicalJacobi((b+1)/2, k+0.5, k+0.5, k+0.5))
    Pkp1 = Normalized(SemiclassicalJacobi((b+1)/2, k+1.5, k+1.5, k+1.5))
    Pnk = (n,x,y) -> Pk[(x + 1)/2,n-k+1]*sqrt(ϕ(x))^k*q[y/sqrt(ϕ(x)),k+1]/2^(5/4+3k/2)
    Pnkp1 = (n,x,y) -> Pkp1[(x + 1)/2,n-k]*sqrt(ϕ(x))^(k+1)*q[y/sqrt(ϕ(x)),k+2]/2^(5/4+3(k+1)/2)
    n = 0
    lhs = yv*Pnk(n,xv,yv)
    rhs = Y[1,3]*Pnkp1(n+1,xv,yv)
    @test lhs ≈ rhs;
    n = 2
    lhs = yv*Pnk(n,xv,yv)
    rhs = Y[4,3]*Pnkp1(n-1,xv,yv) + Y[4,5]*Pnkp1(n,xv,yv) + Y[4,8]*Pnkp1(n+1,xv,yv)
    @test lhs ≈ rhs;
    n = 3
    lhs = yv*Pnk(n,xv,yv)
    rhs = Y[7,3]*Pnkp1(n-2,xv,yv) + Y[7,5]*Pnkp1(n-1,xv,yv) + Y[7,8]*Pnkp1(n,xv,yv) + Y[7,12]*Pnkp1(n+1,xv,yv)
    @test lhs ≈ rhs;
    n = nmax
    lhs = yv*Pnk(n,xv,yv)
    rhs = Y[r-nmax,r-3nmax+2]*Pnkp1(n-2,xv,yv) + Y[r-nmax,r-2nmax+1]*Pnkp1(n-1,xv,yv) + Y[r-nmax,r-nmax+1]*Pnkp1(n,xv,yv) + Y[r-nmax,r+2]*Pnkp1(n+1,xv,yv)
    @test lhs ≈ rhs;
    # Test a few of the k = 1 cases
    k = 1
    Pkm1 = Normalized(SemiclassicalJacobi((b+1)/2, k-0.5, k-0.5, k-0.5))
    Pk = Normalized(SemiclassicalJacobi((b+1)/2, k+0.5, k+0.5, k+0.5))
    Pkp1 = Normalized(SemiclassicalJacobi((b+1)/2, k+1.5, k+1.5, k+1.5))
    Pnk = (n,x,y) -> Pk[(x + 1)/2,n-k+1]*sqrt(ϕ(x))^k*q[y/sqrt(ϕ(x)),k+1]/2^(5/4+3k/2)
    Pnkp1 = (n,x,y) -> Pkp1[(x + 1)/2,n-k]*sqrt(ϕ(x))^(k+1)*q[y/sqrt(ϕ(x)),k+2]/2^(5/4+3(k+1)/2)
    Pnkm1 = (n,x,y) -> Pkm1[(x + 1)/2,n-k+2]*sqrt(ϕ(x))^(k-1)*q[y/sqrt(ϕ(x)),k]/2^(5/4+3(k-1)/2)
    n = 1
    lhs = yv*Pnk(n,xv,yv)
    rhs = Y[3,1]*Pnkm1(n-1,xv,yv) + Y[3,2]*Pnkm1(n,xv,yv) + Y[3,4]*Pnkm1(n+1,xv,yv) + Y[3,6]*Pnkp1(n+1,xv,yv) + Y[3,7]*Pnkm1(n+2,xv,yv)
    @test lhs ≈ rhs;
    n = 4
    lhs = yv*Pnk(n,xv,yv)
    rhs = Y[12,6]*Pnkp1(n-2,xv,yv)+Y[12,7]*Pnkm1(n-1,xv,yv)+Y[12,9]*Pnkp1(n-1,xv,yv)+Y[12,11]*Pnkm1(n,xv,yv)+Y[12,13]*Pnkp1(n,xv,yv)+Y[12,16]*Pnkm1(n+1,xv,yv)+Y[12,18]*Pnkp1(n+1,xv,yv)+Y[12,22]*Pnkm1(n+2,xv,yv)
    @test lhs ≈ rhs;
    n = nmax
    lhs = yv*Pnk(n,xv,yv)
    rhs = Y[r-nmax+1,r-3nmax+3]*Pnkp1(n-2,xv,yv)+Y[r-nmax+1,r-2nmax]*Pnkm1(n-1,xv,yv)+Y[r-nmax+1,r-2nmax+2]*Pnkp1(n-1,xv,yv)+Y[r-nmax+1,r-nmax]*Pnkm1(n,xv,yv)+Y[r-nmax+1,r-nmax+2]*Pnkp1(n,xv,yv)+Y[r-nmax+1,r+1]*Pnkm1(n+1,xv,yv)+Y[r-nmax+1,r+3]*Pnkp1(n+1,xv,yv)+Y[r-nmax+1,r+nmax+3]*Pnkm1(n+2,xv,yv)
    @test lhs ≈ rhs;
    k = nmax
    Pkm1 = Normalized(SemiclassicalJacobi((b+1)/2, k-0.5, k-0.5, k-0.5))
    Pk = Normalized(SemiclassicalJacobi((b+1)/2, k+0.5, k+0.5, k+0.5))
    Pkp1 = Normalized(SemiclassicalJacobi((b+1)/2, k+1.5, k+1.5, k+1.5))
    Pnk = (n,x,y) -> Pk[(x + 1)/2,n-k+1]*sqrt(ϕ(x))^k*q[y/sqrt(ϕ(x)),k+1]/2^(5/4+3k/2)
    Pnkp1 = (n,x,y) -> Pkp1[(x + 1)/2,n-k]*sqrt(ϕ(x))^(k+1)*q[y/sqrt(ϕ(x)),k+2]/2^(5/4+3(k+1)/2)
    Pnkm1 = (n,x,y) -> Pkm1[(x + 1)/2,n-k+2]*sqrt(ϕ(x))^(k-1)*q[y/sqrt(ϕ(x)),k]/2^(5/4+3(k-1)/2)
    n = nmax
    lhs = yv*Pnk(n,xv,yv)
    rhs = Y[r,r-1-nmax]*Pnkm1(n-1,xv,yv) + Y[r,r-1]*Pnkm1(n,xv,yv) + Y[r,r+nmax]*Pnkm1(n+1,xv,yv) + Y[r,r+nmax+2]*Pnkp1(n+1,xv,yv) + Y[r,r+2nmax+2]*Pnkm1(n+2,xv,yv)
    #Y[r,r-1-nmax], Y[r,r-1], Y[r,r+nmax], Y[r,r+nmax+2], Y[r,r+2nmax+2]
    @test lhs ≈ rhs;
end;

@testset "Connection matrix" begin
    nmax = 13
    b  = 2.0
    C, Jx, Jy, Cinds = LanczosCubic(nmax,b)
    for degree = 0:nmax
        Cm = C[1:nk2ind(degree,degree),1:Cinds[nk2ind(degree,degree),2]]
        @test norm(Cm*Cm'-I) < 4E-12
    end
    commutator = Matrix(Jx*Jy-Jy*Jx)
    for degree = 0:nmax-1
        com = norm(commutator[1:nk2ind(degree,degree),1:nk2ind(degree,degree)])
        @test com < 4E-9
    end
end
