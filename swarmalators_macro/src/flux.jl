using StaticArrays

"""
    eigenvalues_Roe(u,c1,c2,λ)

The eigenvalues of the Roe matrix with a 1D velocity `u`, density `ρ` and phase gradient `z1` are 

c₁*u + b*ρ*z1 (multiplicity 2) ; 
c₁*u + b*ρ*z1 ;
0.5*(2*c₂u + 3*b*ρ*z1 + √Δ) ;
0.5*(2*c₂u + 3*b*ρ*z1 - √Δ) ;

with the discriminant Δ = 4*(c₂^2 - c1*c2)*u^2 + 4*λ*c₁ + 4*b*ρ*u*z1*(c₁-c₂) + (b*ρ*z1)^2.
"""
function eigenvalues_Roe(ρ,u,z1,c1,c2,λ,b)
    eval_1 = c1*u + b*ρ*z1  # multiplicity 2
    eval_2 = c2*u + b*ρ*z1
    Δ = 4*(c2^2 - c1*c2) * u^2 + 4*λ*c1 + 4*b*ρ*u*z1*(c1-c2) + (b*ρ*z1)^2
    if Δ<0
        error("Nonpositive discriminant in the computation of the eigenvalues.")
    end
    eval_p = 0.5*(2*c2*u + 3*b*ρ*z1 + sqrt(Δ))
    eval_m = 0.5*(2*c2*u + 3*b*ρ*z1 - sqrt(Δ))
    return eval_p, eval_m, eval_1, eval_2
end


"""
    max_vap_x(ρl,ρr,ul,ur,vl,vr,cαl,cαr,sαl,sαr,c1,c2,λ,b,Δx)

Return the maximal absolute value of the eigenvalues associated to the right, left and Roe average data. 
"""
function max_vap_x(ρl,ρr,ul,ur,vl,vr,cαl,cαr,sαl,sαr,c1,c2,λ,b,Δx)
    zlr = angle(cαr*cαl + sαl*sαr + 1im*(sαr*cαl - sαl*cαr)) / Δx
    um = (sqrt(ρl)*ul + sqrt(ρr)*ur) / (sqrt(ρl) + sqrt(ρr))
    ρm = 0.5*(ρl + ρr)
    vp_l = eigenvalues_Roe(ρl,ul,zlr,c1,c2,λ,b)
    vp_r = eigenvalues_Roe(ρr,ur,zlr,c1,c2,λ,b)
    vp_Roe = eigenvalues_Roe(ρm,um,zlr,c1,c2,λ,b)
    max_vap = max(maximum(abs.(vp_l)),maximum(abs.(vp_r)),maximum(abs.(vp_Roe)))
    return max_vap
end

"""
    flux_x(ρl,ρr,ul,ur,vl,vr,cαl,cαr,sαl,sαr,c1,c2,λ,b,Δx;method="Roe")

Return the flux along the x-axis for the 1D Riemannian problem given by the respective left and right
densities `(ρl,ul,vl,cαl,sαl)` and `(ρr,ur,vr,cαr,sαr)`. The chosen method `method` can be either `"Roe"` or `"HLLE"`.
"""
function flux_x(ρl,ρr,ul,ur,vl,vr,cαl,cαr,sαl,sαr,c1,c2,λ,b,Δx;method="Roe")
    if ρl<1e-9 && ρr<1e-9       # The flux is set to zero if the left and right densities are too small
        return 0.,0.,0.,0.,0.
    end

    # Compute the phase gradient 
    zlr = angle(cαr*cαl + sαl*sαr + 1im*(sαr*cαl - sαl*cαr)) / Δx

    if method=="HLLE"
        um = (sqrt(ρl)*ul + sqrt(ρr)*ur) / (sqrt(ρl) + sqrt(ρr))
        ρm = 0.5*(ρl + ρr)
        vp_l = eigenvalues_Roe(ρl,ul,zlr,c1,c2,λ,b)
        vp_r = eigenvalues_Roe(ρr,ur,zlr,c1,c2,λ,b)
        vp_Roe = eigenvalues_Roe(ρm,um,zlr,c1,c2,λ,b)
        s_l = minimum(min(vp_l,vp_Roe))
        s_r = maximum(max(vp_r,vp_Roe))
        s_lm = min(s_l,0.)
        s_rp = max(s_r,0.)

        f_l = (c1*ρl*ul + b*ρl^2*zlr, c2*ρl*ul^2 + λ*ρl + b*ρl^2*ul*zlr, c2*ρl*ul*vl + b*ρl^2*vl*zlr, c1*ρl*cαl*ul + b*ρl^2*cαl*zlr, c1*ρl*sαl*ul + b*ρl^2*sαl*zlr)
        f_r = (c1*ρr*ur + b*ρr^2*zlr, c2*ρr*ur^2 + λ*ρr + b*ρr^2*ur*zlr, c2*ρr*ur*vr + b*ρr^2*vr*zlr, c1*ρr*cαr*ur + b*ρr^2*cαr*zlr, c1*ρr*sαr*ur + b*ρr^2*sαr*zlr)
        U_l = (ρl,ρl*ul,ρl*vl,ρl*cαl,ρl*sαl)
        U_r = (ρr,ρr*ur,ρr*vr,ρr*cαr,ρr*sαr)
        flux = ((s_rp.*f_l .- s_lm.*f_r) .+ (s_rp*s_lm) .* (U_r .- U_l)) ./ (s_rp - s_lm)
    elseif method == "Roe"
        um = (sqrt(ρl)*ul + sqrt(ρr)*ur) / (sqrt(ρl) + sqrt(ρr))
        ρm = 0.5*(ρl + ρr)
        vp_l = eigenvalues_Roe(ρl,ul,zlr,c1,c2,λ,b)
        vp_r = eigenvalues_Roe(ρr,ur,zlr,c1,c2,λ,b)
        vp_Roe = eigenvalues_Roe(ρm,um,zlr,c1,c2,λ,b)

        abs_A = Roe_matrix(ρl,ρr,ul,ur,vl,vr,cαl,cαr,sαl,sαr,zlr,c1,c2,λ,b)
        U_l = SA[ρl,ρl*ul,ρl*vl,ρl*cαl,ρl*sαl]
        U_r = SA[ρr,ρr*ur,ρr*vr,ρr*cαr,ρr*sαr]
        f_l = SA[c1*ρl*ul + b*ρl*zlr, c2*ρl*ul^2 + λ*ρl + b*ρl^2*ul*zlr, c2*ρl*ul*vl + b*ρl^2*vl*zlr, c1*ρl*cαl*ul + b*ρl^2*cαl*zlr, c1*ρl*sαl*ul + b*ρl^2*sαl*zlr]
        f_r = SA[c1*ρr*ur + b*ρr*zlr, c2*ρr*ur^2 + λ*ρr + b*ρr^2*ur*zlr, c2*ρr*ur*vr + b*ρr^2*vr*zlr, c1*ρr*cαr*ur + b*ρr^2*cαr*zlr, c1*ρr*sαr*ur + b*ρr^2*sαr*zlr]
        avg = 0.5*(f_l + f_r)
        Roe_term = 0.5*abs_A*(U_r-U_l)
        flux = avg .- Roe_term
    else
        error("Unknown method.") 

    end

    return flux

end


function Roe_matrix(ρl,ρr,ul,ur,vl,vr,cαl,cαr,sαl,sαr,z1,c1,c2,λ,b)
    u_Roe = (sqrt(ρl)*ul + sqrt(ρr)*ur) / (sqrt(ρl) + sqrt(ρr))
    v_Roe = (sqrt(ρl)*vl + sqrt(ρr)*vr) / (sqrt(ρl) + sqrt(ρr))
    cα_Roe = (sqrt(ρl)*cαl + sqrt(ρr)*cαr) / (sqrt(ρl) + sqrt(ρr))
    sα_Roe = (sqrt(ρl)*sαl + sqrt(ρr)*sαr) / (sqrt(ρl) + sqrt(ρr))
    ρ_Roe = 0.5 * (ρl + ρr)

    eval_p, eval_m, eval_1, eval_2 = eigenvalues_Roe(ρ_Roe,u_Roe,z1,c1,c2,λ,b)
    
    x1p = eval_p - 2*b*ρ_Roe*z1
    x1m = eval_m - 2*b*ρ_Roe*z1
    x2p = (c1*(c2*u_Roe*v_Roe - b*ρ_Roe*u_Roe*z1)-x1p*c2*v_Roe)/(c2*u_Roe + b*ρ_Roe - eval_p)
    x3p = (-c1*b*ρ_Roe*cα_Roe*z1 - x1p*c1*ρ_Roe*cα_Roe)/(c1*u_Roe + b*ρ_Roe*z1 - eval_p)
    x4p = (-c1*b*ρ_Roe*sα_Roe*z1 - x1p*c1*ρ_Roe*sα_Roe)/(c1*u_Roe + b*ρ_Roe*z1 - eval_p)
    x2m = (c1*(c2*u_Roe*v_Roe - b*ρ_Roe*u_Roe*z1)-x1m*c2*v_Roe)/(c2*u_Roe + b*ρ_Roe - eval_m)
    x3m = (-c1*b*ρ_Roe*cα_Roe*z1 - x1m*c1*ρ_Roe*cα_Roe)/(c1*u_Roe + b*ρ_Roe*z1 - eval_m)
    x4m = (-c1*b*ρ_Roe*sα_Roe*z1 - x1m*c1*ρ_Roe*sα_Roe)/(c1*u_Roe + b*ρ_Roe*z1 - eval_m)

    P = SA[c1       c1      0       0       0
           x1p      x1m     0       0       0
           x2p      x2m     1       0       0
           x3p      x3m     0       1       0
           x4p      x4m     0       0       1
        ]

    y0p = -x1m/(c1*(x1p-x1m))
    y1p = x1p/(c1*(x1p-x1m))
    y2p = -(y0p*x2p + y1p*x2m)
    y3p = -(y0p*x3p + y1p*x3m)
    y4p = -(y0p*x4p + y1p*x4m)

    y0m = 1/(x1p-x1m)
    y1m = -1/(x1p-x1m)
    y2m = -(y0m*x2p + y1m*x2m)
    y3m = -(y0m*x3p + y1m*x3m)
    y4m = -(y0m*x4p + y1m*x4m)

    invP = SA[y0p   y0m   0   0   0
              y1p   y1m   0   0   0
              y2p   y2m   1   0   0
              y3p   y3m   0   1   0
              y4p   y4m   0   0   1
            ]

    eval_l = eigenvalues_Roe(ρl,ul,z1,c1,c2,λ,b)
    eval_r = eigenvalues_Roe(ρr,ur,z1,c1,c2,λ,b)
    eval_p_fix = max(abs(eval_l[1]),abs(eval_r[1]))
    eval_m_fix = max(abs(eval_l[2]),abs(eval_r[2]))
    eval_1_fix = max(abs(eval_l[3]),abs(eval_r[3]))
    eval_2_fix = max(abs(eval_l[4]),abs(eval_r[4]))

    diag = SA[eval_p_fix           0                    0                  0                  0
              0                    eval_m_fix           0                  0                  0
              0                    0                    eval_2_fix         0                  0
              0                    0                    0                  eval_1_fix         0
              0                    0                    0                  0                  eval_1_fix
            ]

    return P*diag*invP
end

