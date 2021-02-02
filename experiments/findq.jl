
# Want Q = […,Q_1,Q_2,Q_1,Q_2…] so that
# Q*Y*Q' is 3x3
# This requires
# Q_2'*Be2 * Q_1' == Q_1*Bo2*Q_2


Q_p1 = qr(randn(3,3)).Q
Q_p2 = qr(randn(3,3)).Q

Be2,Bo2 = Q_p1*Be2*Q_p2', Q_p2*Bo2*Q_p1'

function eqns(p)
    p1,p2 = p[1:9],p[10:end]
    Q_1 = qr(reshape(p1,3,3)).Q
    Q_2 = qr(reshape(p2,3,3)).Q
    [vec(Q_2 * X̄_1 * Q_1' - Q_1*X̄_1 *Q_2');
     vec(Q_2 * Ȳ_1 * Q_1' - Q_1*Ỹ_1*Q_2');
     vec(Q_1' * X̄_0 * Q_1' - Q_2*X̄_0*Q_2');
     vec(Q_1' * Ỹ_0 * Q_1' - Q_2*Ȳ_0*Q_2');
     ]
end

p = randn(18)
for _=1:1000
    p = p - svd(jacobian(eqns,p)) \ eqns(p); norm(eqns(p))
    p = p/norm(p)
    p .+= 0.001*randn.()
end; norm(eqns(p))


## try 6x6 Q
function eqns(p)
     Q = qr(reshape(p,6,6)).Q
     QX_0 = Q'*X_0*Q
     QX_1 = Q'*X_1*Q
     QY_0 = Q'*Y_0*Q
     QY_1 = Q'*Y_1*Q
     [vec(QY_1[1:3,:]);
        vec(QY_1[4:end,4:end]);
        vec(QX_1[1:3,:]);
        vec(QX_1[4:end,4:end]);
        vec(QX_0[1:3,1:3]-QX_0[4:end,4:end]);
        vec(QY_0[1:3,1:3]-QY_0[4:end,4:end]);
        vec(QX_0[1:3,4:end]-QX_1[4:end,1:3]);
        vec(QY_0[1:3,4:end]-QY_1[4:end,1:3])]
end


mn = Inf
p = randn(36); for _=1:1000
    p = p - svd(jacobian(eqns,p)) \ eqns(p); norm(eqns(p))
    p = p/norm(p)
    mn = min(mn, norm(eqns(p)))
end; mn