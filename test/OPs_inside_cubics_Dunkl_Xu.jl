function XP(n,t)

    # Generate the operator representing multiplication of the orthonormal polynomials P_{k,j} by x for 0 <= k <= n and 0 <= j <= k
    # The operator is block-tridiagonal with diagonal blocks

    l,u = 1,1          # block bandwidths
    rows = 1:n+1
    cols = 1:n+2
    X = BlockBandedMatrix(Zeros{typeof(t)}(sum(rows),sum(cols)), rows,cols, (l,u))

    row1 = 1
    row = 1
    for k = 0:n
        J = jacobimatrix(Normalized(SemiclassicalJacobi((t+1)/2, k+0.5, k+0.5, k+0.5)))
        row = row1
        jr = 1
        for d = k:n-1
            X[row,row] = 2J[jr,jr]-1
            X[row,row+d+1] = X[row+d+1,row] = 2J[jr,jr+1]
            row += d+1
            jr += 1
        end
        X[row,row] = 2J[jr,jr]-1
        X[row,row+n+1] = 2J[jr,jr+1]

        row1 += k+2
    end

    X

end

function YP(n,t)
    # Construct the block-pentadiagonal operator representing multiplication of the P_{k,j} by y for 0 <= k <= n, 0 <= j <= k

    l,u = 2,2
    rows = 1:n+1
    cols = 1:n+3
    Y = BlockBandedMatrix(Zeros{typeof(t)}(sum(rows),sum(cols)), rows,cols, (l,u))

    # Symmetric Jacobi operator of the orthonormal Legendre polynomials
    Jq = jacobimatrix(Normalized(ClassicalOrthogonalPolynomials.Legendre()))

    # First assign the entries for k = 0

    k = 0
    # Conversion operator
    Pk = Normalized(SemiclassicalJacobi((t+1)/2, k+0.5, k+0.5, k+0.5))
    Pk1 = Normalized(SemiclassicalJacobi((t+1)/2, k+1.5, k+1.5, k+1.5))
    Rk1 = Pk1\Pk

    con2 = 2^(3/2)*Jq[1,2]
    Y[1,3] = Y[3,1] = con2*Rk1[1,1]
    row1 = 1
    row = 1
    row += 1
    for d = 1:n-1
        Y[row,row+1] = Y[row+1,row] = con2*Rk1[d,d+1]
        Y[row,row+d+2] = Y[row+d+2,row] = con2*Rk1[d+1,d+1]
        row += d+1
    end
    # d = n
    Y[row,row+1] = Y[row+1,row] = con2*Rk1[n,n+1]
    Y[row,row+n+2] = con2*Rk1[n+1,n+1]

    # Assign entries for k = 1, …, n
    Rk = Zeros(Rk1)
    con1 = 0
    row1 += k+2
    for k = 1:n
        con1 = con2
        Rk = Rk1
        Pk = Pk1
        Pk1 = Normalized(SemiclassicalJacobi((t+1)/2, k+1.5, k+1.5, k+1.5))
        Rk1 = Pk1\Pk
        con2 = 2^(3/2)*Jq[k+1,k+2]
        row = row1
        # Assign first row, d = k = n, d = `degree'
        d = k
        if k <= n-2
            Y[row,row+d] = Y[row+d,row] = con1*Rk[1,3]
            Y[row,row+d+2] = Y[row+d+2,row] = con2*Rk1[1,1]
            Y[row,row+2d+2] = Y[row+2d+2,row] = con1*Rk[1,4]
        elseif k == n-1
            Y[row,row+d] = Y[row+d,row] = con1*Rk[1,3]
            Y[row,row+d+2] = Y[row+d+2,row] = con2*Rk1[1,1]
            Y[row,row+2d+2] = con1*Rk[1,4]
        else
            Y[row,row+d]  = con1*Rk[1,3]
            Y[row,row+d+2] = con2*Rk1[1,1]
            Y[row,row+2d+2] = con1*Rk[1,4]
        end
            row += d+1
        # Assign the remaining rows

        for d = k+1:n
            if d <= n-2
                Y[row,row+1] = Y[row+1,row] = con2*Rk1[d-k,d-k+1]
                Y[row,row+d] = Y[row+d,row] = con1*Rk[d-k+1,d-k+3]
                Y[row,row+d+2] = Y[row+d+2,row] = con2*Rk1[d-k+1,d-k+1]
                Y[row,row+2d+2] = Y[row+2d+2,row] = con1*Rk[d-k+1,d-k+4]
            elseif d == n-1
                Y[row,row+1] = Y[row+1,row] = con2*Rk1[d-k,d-k+1]
                Y[row,row+d] = Y[row+d,row] = con1*Rk[d-k+1,d-k+3]
                Y[row,row+d+2] = Y[row+d+2,row] = con2*Rk1[d-k+1,d-k+1]
                Y[row,row+2d+2] = con1*Rk[d-k+1,d-k+4]
            else
                Y[row,row+1] = Y[row+1,row] = con2*Rk1[d-k,d-k+1]
                Y[row,row+d] = con1*Rk[d-k+1,d-k+3]
                Y[row,row+d+2] = con2*Rk1[d-k+1,d-k+1]
                Y[row,row+2d+2]  = con1*Rk[d-k+1,d-k+4]
            end

            row += d+1
        end

            row1 += k + 2
    end

    Y

end

function nk2ind(n,k)
    # Convert the indices (n,k) of a polynomial to a linear index/its index in the vector of polynomials
    if k == n
        out = Int64((n+1)*(n+2)/2)
    else
        out = Int64(n*(n+1)/2)+k+1
    end
    out
end

function ind2nk(ind)
    # Convert linear index to the ordered degree indices (n,k)
    n1 = (-1+sqrt(1+8ind))/2
    if mod(n1,1) == 0
        out = Int64.([n1-1, n1-1])
    else
        n1 = Int64(floor(n1))
        out = [n1, ind-Int64((n1+1)n1/2)-1]
    end
    out
end

function LanczosCubic(nmax,b)

    d = nk2ind(nmax,nmax)
    # connection matrix
    C = zeros(d,3d) #still have to figure out the column index of the rightmost nonzero entry in a given row
    C[1:5,1:5] = Eye(5)
    # store the leftmost and rightmost nonzero columns of each row of C in Cinds
    Cinds = zeros(Int64,d,2)
    Cinds[1:5,1] = (collect(1:5))
    Cinds[1:5,2] = (collect(1:5))
    # Jacobi operators of the OPs
    l,u = 1,1
    rows = 1:nmax+1
    cols = 1:nmax+1
    Jx = BlockBandedMatrix(Zeros{typeof(b)}(sum(rows),sum(cols)), rows,cols, (l,u))
    Jy = BlockBandedMatrix(Zeros{typeof(b)}(sum(rows),sum(cols)), rows,cols, (l,u))

    # Lanczos for degrees 0, 1, 2, 3 are hard coded

    Y = YP(nmax+8,b) #Chose nmax+8 by trial-and-error
    X = XP(nmax+8,b)

    # since the first five OPs coincide with the P polynomials,
    A0x = X[Block(1,1)]
    B0x = X[Block(1,2)]
    A0y = Y[Block(1,1)]
    B0y = Y[Block(1,2)]
    A1x = X[Block(2,2)]
    A1y = Y[Block(2,2)]
    B1x = X[Block(2,3)]

    # Degree 2
    B1y = zeros(2,3)
    B1y[1,2] = Y[Block(2,3)][1,2]
    B1y[2,1] = Y[Block(2,3)][2,1]
    yQ11 = Y[nk2ind(1,1),1:nk2ind(3,0)]
    yQ11[nk2ind(0,0)] -= B0y[1,2]
    yQ11[nk2ind(1,0)] -= A1y[1,2]
    yQ11[nk2ind(2,0)] -= B1y[2,1]
    B1y[2,3] = sqrt(sum(yQ11.^2))
    C[nk2ind(2,2),1:nk2ind(3,0)] = yQ11/B1y[2,3]
    lind = findfirst(x-> abs(x)>1e-13,C[nk2ind(2,2),1:nk2ind(3,0)])
    Cinds[nk2ind(2,2),:] = [lind, nk2ind(3,0)]

    # Degree 3
    A2x = zeros(3,3)
    A2x[1,1] = X[4,4]
    A2x[1,3] = A2x[3,1] = dot(C[4,4]*X[4,6:7],C[6,6:7])
    A2x[2,2] = X[5,5]
    A2x[3,3] = C[6,6:7]'*X[6:7,6:7]*C[6,6:7]
    #C[nk2ind(2,0):nk2ind(2,2),nk2ind(2,0):nk2ind(3,0)]*X[nk2ind(2,0):nk2ind(3,0),nk2ind(2,0):nk2ind(3,0)]*(C[nk2ind(2,0):nk2ind(2,2),nk2ind(2,0):nk2ind(3,0)]')
    B2x = zeros(3,4)
    #cols = [nk2ind(1,0),nk2ind(2,0),nk2ind(3,0)]
    #xQ20 = X[nk2ind(2,0),cols]
    xQ20 = X[nk2ind(2,0),1:nk2ind(3,0)]
    xQ20[nk2ind(1,0)] -= B1x[1,1]
    xQ20[nk2ind(2,0)] -= A2x[1,1]
    xQ20[nk2ind(2,2):nk2ind(3,0)] -= A2x[1,3]*C[nk2ind(2,2),nk2ind(2,2):nk2ind(3,0)]
    B2x[1,1] = sqrt(sum(xQ20.^2))
    C[nk2ind(3,0),1:nk2ind(3,0)] = xQ20/B2x[1,1]
    lind = findfirst(x-> abs(x)>1e-13,C[nk2ind(3,0),1:nk2ind(3,0)])
    Cinds[nk2ind(3,0),:] = [lind nk2ind(3,0)]
    xQ21 = X[nk2ind(2,1),1:nk2ind(3,1)]
    xQ21[nk2ind(1,1)] -= B1x[2,2]
    xQ21[nk2ind(2,1)] -= A2x[2,2]
    B2x[2,2] = sqrt(sum(xQ21.^2))
    C[nk2ind(3,1),1:nk2ind(3,1)] = xQ21/B2x[2,2]
    lind = findfirst(x-> abs(x)>1e-13,C[nk2ind(3,1),1:nk2ind(3,1)])
    Cinds[nk2ind(3,1),:] = [lind nk2ind(3,1)]
    B2x[3,1] = C[nk2ind(2,2),nk2ind(2,2):nk2ind(3,0)]'*X[nk2ind(2,2):nk2ind(3,0),nk2ind(2,2):nk2ind(3,0)]*C[nk2ind(3,0),nk2ind(2,2):nk2ind(3,0)]
    xQ22 = C[nk2ind(2,2),nk2ind(2,2):nk2ind(3,0)]'*X[nk2ind(2,2):nk2ind(3,0),1:nk2ind(4,0)]
    xQ22[nk2ind(2,0)] -= A2x[3,1]
    xQ22[nk2ind(2,2):nk2ind(3,0)] -= A2x[3,3]*C[nk2ind(2,2),nk2ind(2,2):nk2ind(3,0)]
    xQ22[nk2ind(2,2):nk2ind(3,0)] -= B2x[3,1]*C[nk2ind(3,0),nk2ind(2,2):nk2ind(3,0)]
    B2x[3,3] = sqrt(sum(xQ22.^2))
    C[nk2ind(3,2),1:nk2ind(4,0)] = xQ22/B2x[3,3]
    lind = findfirst(x-> abs(x)>1e-13,C[nk2ind(3,2),1:nk2ind(4,0)])
    Cinds[nk2ind(3,2),:] = [lind nk2ind(4,0)]
    A2y = zeros(3,3)
    A2y[1,2] = A2y[2,1] = Y[4,5]
    A2y[2,3] = A2y[3,2] = dot(Y[5,6:7],C[6,6:7])
    B2y = zeros(3,4)
    B2y[1,2] = dot(Y[4,8],C[8,8])
    B2y[2,1] = dot(Y[5,6:7],C[7,6:7])
    B2y[2,3] = dot(Y[5,9:11],C[9,9:11])
    B2y[3,2] = dot(C[6,6:7]'*Y[6:7,8],C[8,8])
    yQ22 = C[6,6:7]'*Y[6:7,3:12]
    yQ22[1] -= B1y[2,3]
    yQ22[3] -= A2y[3,2]
    yQ22[6] -= B2y[3,2]
    B2y[3,4] = sqrt(sum(yQ22.^2))
    C[10,3:12] = yQ22/B2y[3,4]
    lind = findfirst(x-> abs(x)>1e-13,C[nk2ind(3,3),1:nk2ind(4,1)])
    Cinds[nk2ind(3,3),:] = [lind nk2ind(4,1)]

    Jx[Block(1,1)] = A0x
    Jx[Block(1,2)] = B0x
    Jx[Block(2,1)] = B0x'
    Jx[Block(2,2)] = A1x
    Jx[Block(2,3)] = B1x
    Jx[Block(3,2)] = B1x'
    Jx[Block(3,3)] = A2x
    Jx[Block(3,4)] = B2x
    Jx[Block(4,3)] = B2x'
    Jy[Block(1,1)] = A0y
    Jy[Block(1,2)] = B0y
    Jy[Block(2,1)] = B0y'
    Jy[Block(2,2)] = A1y
    Jy[Block(2,3)] = B1y
    Jy[Block(3,2)] = B1y'
    Jy[Block(3,3)] = A2y
    Jy[Block(3,4)] = B2y
    Jy[Block(4,3)] = B2y'

    tol = 1e-13
    for n = 4:nmax
        # Compute Ax
        Ax = zeros(n,n)
        # compute the diagonal elements
        for k = 0:n-1
            r = nk2ind(n-1,k)
            Ax[k+1,k+1] = C[r,Cinds[r,1]:Cinds[r,2]]'*X[Cinds[r,1]:Cinds[r,2],Cinds[r,1]:Cinds[r,2]]*C[r,Cinds[r,1]:Cinds[r,2]]
        end
        # off-diagonal elements
        for r = 1:n-2
            row = nk2ind(n-1,r-1)
            for c = r+2:2:n
                Ax[r,c] = Ax[c,r] = C[row,Cinds[row,1]:Cinds[row,2]]'*X[Cinds[row,1]:Cinds[row,2],Cinds[row+c-r,1]:Cinds[row+c-r,2]]*C[row+c-r,Cinds[row+c-r,1]:Cinds[row+c-r,2]]
            end
        end

        Jx[Block(n,n)] = Ax

        # Orthogonalize xQn-1 to form Qₙ₀, …, Qₙ,ₙ₋₁
        Bx = zeros(n,n+1)
        maxind = maximum(Cinds[nk2ind(n-1,0):nk2ind(n-1,n-1),2])
        for k = 0:n-1
            r = nk2ind(n-1,k)
            inds = ind2nk(Cinds[r,2])
            nv = inds[1]
            kv = inds[2]
            rind = nk2ind(nv+1,kv)
            if rind >= maxind
                right = rind
            else
                right = maxind
            end
            # Multiply  by x
            xQn = C[r,Cinds[r,1]:Cinds[r,2]]'*X[Cinds[r,1]:Cinds[r,2],1:right]
            # Subtract off components along lower degree OPs
            # first for OPs of degree n-2
            if k < n-1
                for c = k+1:2:n-1
                    l = nk2ind(n-2,c-1)
                    xQn[Cinds[l,1]:Cinds[l,2]] -= Jx[Block(n,n-1)][k+1,c]*C[l,Cinds[l,1]:Cinds[l,2]]
                end
            end
            # next for OPs of degree n-1
            if mod(k,2)==0
                for c = 1:2:n
                    l = nk2ind(n-1,c-1)
                    xQn[Cinds[l,1]:Cinds[l,2]] -= Jx[Block(n,n)][k+1,c]*C[l,Cinds[l,1]:Cinds[l,2]]
                end
            else
                for c = 2:2:n
                    l = nk2ind(n-1,c-1)
                    xQn[Cinds[l,1]:Cinds[l,2]] -= Jx[Block(n,n)][k+1,c]*C[l,Cinds[l,1]:Cinds[l,2]]
                end
            end
            # Finally OPs of degree n
            if k > 1
               if mod(k,2) == 0
                    for c = 1:2:k-1
                        l = nk2ind(n,c-1)
                        xQn[Cinds[l,1]:Cinds[l,2]] -= Bx[k+1,c]*C[l,Cinds[l,1]:Cinds[l,2]]
                    end
                else
                    for c = 2:2:k-1
                        l = nk2ind(n,c-1)
                        xQn[Cinds[l,1]:Cinds[l,2]] -= Bx[k+1,c]*C[l,Cinds[l,1]:Cinds[l,2]]
                    end
                end
            end
            # Normalise
            Bx[k+1,k+1] = sqrt(sum(xQn.^2))
            C[nk2ind(n,k),1:right] = xQn/Bx[k+1,k+1]
            #C[nk2ind(n,k),1:nk2ind(nv+1,kv)] = xQn/Bx[k+1,k+1]
            # find the leftmost nonzero entry
            #rind = nk2ind(nv+1,kv)
            #lind = findfirst(x-> abs(x)>1e-13,C[nk2ind(n,k),1:rind])
            #Cinds[nk2ind(n,k),:] = [lind rind]
            lind = findfirst(x-> abs(x)>tol,C[nk2ind(n,k),1:right])
            rind = findlast(x-> abs(x)>tol,C[nk2ind(n,k),1:right+1])
            Cinds[nk2ind(n,k),:] = [lind rind]

            # Compute the entries in column k+1 of Bx:

            if k < n-2
                for row = k+3:2:n
                    ind = nk2ind(n-1,row-1)
                    Bx[row,k+1] = C[ind,Cinds[ind,1]:Cinds[ind,2]]'*X[Cinds[ind,1]:Cinds[ind,2],lind:rind]*C[nk2ind(n,k),lind:rind]
                end
            end
        end
        Jx[Block(n,n+1)] = Bx
        Jx[Block(n+1,n)] = Bx'

        # Compute Ay
        Ay = zeros(n,n)
        for r = 1:n-1
            row = nk2ind(n-1,r-1)
            for c = r+1:2:n
                Ay[r,c] = Ay[c,r] = C[row,Cinds[row,1]:Cinds[row,2]]'*Y[Cinds[row,1]:Cinds[row,2],Cinds[row+c-r,1]:Cinds[row+c-r,2]]*C[row+c-r,Cinds[row+c-r,1]:Cinds[row+c-r,2]]
            end
        end
        Jy[Block(n,n)] = Ay

        # Compute By column by column
        By = zeros(n,n+1)
        ind = nk2ind(n,0)
        # first column
        for r = 2:2:n
            l = nk2ind(n-1,r-1)
            By[r,1] = C[l,Cinds[l,1]:Cinds[l,2]]'*Y[Cinds[l,1]:Cinds[l,2],Cinds[ind,1]:Cinds[ind,2]]*C[ind,Cinds[ind,1]:Cinds[ind,2]]
        end
        # rest of the columns
        for c = 2:n
            ind = nk2ind(n,c-1)
            for r = c-1:2:n
                l = nk2ind(n-1,r-1)
                By[r,c] = C[l,Cinds[l,1]:Cinds[l,2]]'*Y[Cinds[l,1]:Cinds[l,2],Cinds[ind,1]:Cinds[ind,2]]*C[ind,Cinds[ind,1]:Cinds[ind,2]]
            end
        end
        # orthogonalize yQ_{n-1,n-1} to find Q_{n,n}
        # Multiply Q_{n-1,n-1} by y
        r = nk2ind(n-1,n-1)
        inds = ind2nk(Cinds[r,2])
        nv = inds[1]
        kv = inds[2]
        yQn = C[r,Cinds[r,1]:Cinds[r,2]]'*Y[Cinds[r,1]:Cinds[r,2],1:nk2ind(nv+2,kv-1)]
        # subtract off OPs of degree n-2
        l = nk2ind(n-2,n-2)
        yQn[Cinds[l,1]:Cinds[l,2]] -= Jy[Block(n,n-1)][n,n-1]*C[l,Cinds[l,1]:Cinds[l,2]]
        # subtract off OPs of degree n-1
        if mod(n,2)==0
            for c = 1:2:n
                l = nk2ind(n-1,c-1)
                yQn[Cinds[l,1]:Cinds[l,2]] -= Jy[Block(n,n)][n,c]*C[l,Cinds[l,1]:Cinds[l,2]]
            end
        else
            for c = 2:2:n
                l = nk2ind(n-1,c-1)
                yQn[Cinds[l,1]:Cinds[l,2]] -= Jy[Block(n,n)][n,c]*C[l,Cinds[l,1]:Cinds[l,2]]
            end
        end
        # subtract off OPs of degree n
        if mod(n,2)==0
            for c = 1:2:n-1
                l = nk2ind(n,c-1)
                yQn[Cinds[l,1]:Cinds[l,2]] -= By[n,c]*C[l,Cinds[l,1]:Cinds[l,2]]
            end
        else
            for c = 2:2:n-1
                l = nk2ind(n,c-1)
                yQn[Cinds[l,1]:Cinds[l,2]] -= By[n,c]*C[l,Cinds[l,1]:Cinds[l,2]]
            end
        end

        # normalise
        By[n,n+1] = sqrt(sum(yQn.^2))

        C[nk2ind(n,n),1:nk2ind(nv+2,kv-1)] = yQn/By[n,n+1]
        # find the leftmost nonzero entry
        rind = findlast(x-> abs(x)>tol,C[nk2ind(n,n),1:nk2ind(nv+2,kv-1)+1])
        #rind = nk2ind(nv+2,kv-1)
        lind = findfirst(x-> abs(x)>tol,C[nk2ind(n,n),1:rind])
        Cinds[nk2ind(n,n),:] = [lind rind]
        Jy[Block(n,n+1)] = By
        Jy[Block(n+1,n)] = By'
    end

    Ax = zeros(nmax+1,nmax+1)
    # compute the diagonal elements
    for k = 0:nmax
        r = nk2ind(nmax,k)
        Ax[k+1,k+1] = C[r,Cinds[r,1]:Cinds[r,2]]'*X[Cinds[r,1]:Cinds[r,2],Cinds[r,1]:Cinds[r,2]]*C[r,Cinds[r,1]:Cinds[r,2]]
    end
    # off-diagonal elements
    for r = 1:nmax-1
        row = nk2ind(nmax,r-1)
        for c = r+2:2:nmax+1
            Ax[r,c] = Ax[c,r] = C[row,Cinds[row,1]:Cinds[row,2]]'*X[Cinds[row,1]:Cinds[row,2],Cinds[row+c-r,1]:Cinds[row+c-r,2]]*C[row+c-r,Cinds[row+c-r,1]:Cinds[row+c-r,2]]
        end
    end

    Jx[Block(nmax+1,nmax+1)] = Ax

    Ay = zeros(nmax+1,nmax+1)
    for r = 1:nmax
        row = nk2ind(nmax,r-1)
        for c = r+1:2:nmax+1
            Ay[r,c] = Ay[c,r] = C[row,Cinds[row,1]:Cinds[row,2]]'*Y[Cinds[row,1]:Cinds[row,2],Cinds[row+c-r,1]:Cinds[row+c-r,2]]*C[row+c-r,Cinds[row+c-r,1]:Cinds[row+c-r,2]]
        end
    end
    Jy[Block(nmax+1,nmax+1)] = Ay

    C, Jx, Jy, Cinds
end
