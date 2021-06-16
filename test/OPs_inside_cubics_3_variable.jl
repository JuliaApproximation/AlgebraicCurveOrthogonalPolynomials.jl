function XP(nmax,α₁,β₁,α₂,β₂,b)

    # Construct the operator representing the multiplication of the 3-variable polynomials by x for all polynomials of degree <= nmax
    # α₁ and β₁ are the parameters of the Jacobi weight of the boundary OPs
    # α₂ and β₂ are the parameters of the Jacobi weight of the OPs in z
    # b > 1 is the third root of the cubic ϕ = (1 - x²)(b - x)

    l,u = 1,1          # block bandwidths
    rows = vcat(1,3*(1:nmax+1))
    cols = vcat(1,3*(1:nmax+1))
    X = BlockBandedMatrix(Zeros(sum(rows),sum(cols)), rows,cols, (l,u))

    pw = Normalized(SemiclassicalJacobi((b+1)/2, β₁, α₁, 0.0))
    Jw = jacobimatrix(pw)
    pϕw = Normalized(SemiclassicalJacobi((b+1)/2, β₁+1, α₁+1, 1))
    Jϕw = jacobimatrix(pϕw)

    k = 0

    pzk = Normalized(SemiclassicalJacobi((b+1)/2, 2k+1+β₂, α₂, 0.0))
    Jzk = jacobimatrix(pzk)
    pzk1 = Normalized(SemiclassicalJacobi((b+1)/2,2k+3+β₂, α₂, 0.0))
    Jzk1 = jacobimatrix(pzk1)
    Ck1 = pzk1\pzk

    for n = k:nmax
        row = nki2ind(n,k,1)
        kient = 2*Jw[k+1,k+1] - 1
        X[row,row] = Jzk[n+1,n+1]*kient
        col = nki2ind(n+1,k,1)
        X[row,col] = X[col,row] = Jzk[n+1,n+2]*kient

        kient = 2*Jw[k+1,k+2]
        col = nki2ind(n+1,k+1,1)
        X[row,col] = X[col,row] = Ck1[n+1,n+1]*kient
        if n > 0
            col = nki2ind(n,k+1,1)
            X[row,col] = X[col,row] = Ck1[n,n+1]*kient
        end
    end

    Jzk = Jzk1
    pzk = pzk1
    Ck = Ck1

    k = 1

    pzk1 = Normalized(SemiclassicalJacobi((b+1)/2,2k+3+β₂, α₂, 0.0))
    Jzk1 = jacobimatrix(pzk1)
    Ck1 = pzk1\pzk

    for n = k:nmax
        # (n,1,1)
        row = nki2ind(n,k,1)

        kient = 2*Jw[k,k+1]
        col = nki2ind(n+1,0,1)
        X[row,col] = X[col,row] = Ck[n,n+2]*kient

        kient = 2*Jw[k+1,k+1] - 1
        X[row,row] = Jzk[n,n]*kient
        col = nki2ind(n+1,k,1)
        X[row,col] = X[col,row] = Jzk[n,n+1]*kient

        kient = 2*Jw[k+1,k+2]
        col = nki2ind(n+1,k+1,2)
        X[row,col] = X[col,row] = Ck1[n,n]*kient
        if n > 1
            col = nki2ind(n,k+1,2)
            X[row,col] = X[col,row] = Ck1[n-1,n]*kient
        end

        # (n,1,2)
        row = nki2ind(n,k,2)

        kient = 2*Jϕw[k,k] - 1
        X[row,row] = Jzk[n,n]*kient
        col = nki2ind(n+1,k,2)
        X[row,col] = X[col,row] = Jzk[n,n+1]*kient

        kient = 2*Jϕw[k,k+1]
        col = nki2ind(n+1,k+1,3)
        X[row,col] = X[col,row] = Ck1[n,n]*kient
        if n > 1
            col = nki2ind(n,k+1,3)
            X[row,col] = X[col,row] = Ck1[n-1,n]*kient
        end

    end

    for k = 2:nmax
        Jzk = Jzk1
        pzk = pzk1
        Ck = Ck1

        pzk1 = Normalized(SemiclassicalJacobi((b+1)/2,2k+3+β₂, α₂, 0.0))
        Jzk1 = jacobimatrix(pzk1)
        Ck1 = pzk1\pzk

        for n = k:nmax

            if mod(k,2) == 0

                # i = 1
                d = deg(k,1)
                row = nki2ind(n,k,1)

                kient = 2*Jw[d+1,d+1]-1
                X[row,row] = Jzk[n-k+1,n-k+1]*kient
                col = nki2ind(n+1,k,1)
                X[row,col] = X[col,row] = Jzk[n-k+1,n-k+2]*kient

                kient = 2*Jw[d+1,d]
                col = nki2ind(n,k,2)
                X[row,col] = X[col,row] = Jzk[n-k+1,n-k+1]*kient
                col = nki2ind(n+1,k,2)
                X[row,col] = X[col,row] = Jzk[n-k+1,n-k+2]*kient

                kient = 2*Jw[d+1,d+2]
                col = nki2ind(n+1,k+1,1)
                X[row,col] = X[col,row] = Ck1[n-k+1,n-k+1]*kient
                if n > k
                    col = nki2ind(n,k+1,1)
                    X[row,col] = X[col,row] = Ck1[n-k,n-k+1]*kient
                end

                # i = 2
                d = deg(k,2)
                row = nki2ind(n,k,2)

                kient = 2*Jw[d+1,d]
                col = nki2ind(n+1,k-1,1)
                X[row,col] = X[col,row] = Ck[n-k+1,n-k+3]*kient
                col = nki2ind(n,k-1,1)
                X[row,col] = X[col,row] = Ck[n-k+1,n-k+2]*kient

                kient = 2*Jw[d+1,d+2]
                col = nki2ind(n+1,k,1)
                X[row,col] = X[col,row] = Jzk[n-k+1,n-k+2]*kient

                kient = 2*Jw[d+1,d+1]-1
                X[row,row] = Jzk[n-k+1,n-k+1]*kient
                col = nki2ind(n+1,k,2)
                X[row,col] = X[col,row] = Jzk[n-k+1,n-k+2]*kient

                # i = 3
                d = deg(k,3)
                row = nki2ind(n,k,3)

                kient = 2*Jϕw[d+1,d]
                col = nki2ind(n+1,k-1,2)
                X[row,col] = X[col,row] = Ck[n-k+1,n-k+3]*kient

                kient = 2*Jϕw[d+1,d+1]-1
                X[row,row] = Jzk[n-k+1,n-k+1]*kient
                col = nki2ind(n+1,k,3)
                X[row,col] = X[col,row] = Jzk[n-k+1,n-k+2]*kient

                kient = 2*Jϕw[d+1,d+2]
                col = nki2ind(n+1,k+1,3)
                X[row,col] = X[col,row] = Ck1[n-k+1,n-k+1]*kient
                if n > k
                    col = nki2ind(n,k+1,3)
                    X[row,col] = X[col,row] = Ck1[n-k,n-k+1]*kient
                end


            else
                # i = 1
                d = deg(k,1)
                row = nki2ind(n,k,1)

                kient = 2*Jw[d+1,d]
                col = nki2ind(n+1,k-1,1)
                X[row,col] = X[col,row] = Ck[n-k+1,n-k+3]*kient

                kient = 2*Jw[d+1,d+1]-1
                X[row,row] = Jzk[n-k+1,n-k+1]*kient
                col = nki2ind(n+1,k,1)
                X[row,col] = X[col,row] = Jzk[n-k+1,n-k+2]*kient

                kient = 2*Jw[d+1,d+2]
                col = nki2ind(n+1,k+1,2)
                X[row,col] = X[col,row] = Ck1[n-k+1,n-k+1]*kient
                if n > k
                    col = nki2ind(n,k+1,2)
                    X[row,col] = X[col,row] = Ck1[n-k,n-k+1]*kient
                end

                # i = 2
                d = deg(k,2)
                row = nki2ind(n,k,2)

                kient = 2*Jϕw[d+1,d+1]-1
                X[row,row] = Jzk[n-k+1,n-k+1]*kient
                col = nki2ind(n+1,k,2)
                X[row,col] = X[col,row] = Jzk[n-k+1,n-k+2]*kient

                kient = 2*Jϕw[d+1,d]
                col = nki2ind(n,k,3)
                X[row,col] = X[col,row] = Jzk[n-k+1,n-k+1]*kient
                col = nki2ind(n+1,k,3)
                X[row,col] = X[col,row] = Jzk[n-k+1,n-k+2]*kient

                kient = 2*Jϕw[d+1,d+2]
                col = nki2ind(n+1,k+1,3)
                X[row,col] = X[col,row] = Ck1[n-k+1,n-k+1]*kient
                if n > k
                    col = nki2ind(n,k+1,3)
                    X[row,col] = X[col,row] = Ck1[n-k,n-k+1]*kient
                end

                # i = 3
                d = deg(k,3)
                row = nki2ind(n,k,3)

                kient = 2*Jϕw[d+1,d]
                col = nki2ind(n+1,k-1,3)
                X[row,col] = X[col,row] = Ck[n-k+1,n-k+3]*kient

                kient = 2*Jϕw[d+1,d+2]
                col = nki2ind(n+1,k,2)
                X[row,col] = X[col,row] = Jzk[n-k+1,n-k+2]*kient

                kient = 2*Jϕw[d+1,d+1]-1
                X[row,row] = Jzk[n-k+1,n-k+1]*kient
                col = nki2ind(n+1,k,3)
                X[row,col] = X[col,row] = Jzk[n-k+1,n-k+2]*kient

            end

        end

    end

    X

end

function nki2ind(n,k,i)
    # convert the indices (n,k,i) of a polynomial to a linear index
    if n == 0
        ind = 1
    else
        prev = Int64(1 + 3*(n-1)*n/2) # number of polys of deg <= n-1
        if k == 0
            ind = prev + 1
        elseif k == 1
            ind = prev + 1 + i
        else
            ind = prev + 3*(k-1) + i
        end
    end
    ind
end

deg(k,i) = Int64(3*floor(k/2)) + mod(k,2) + 1-i
