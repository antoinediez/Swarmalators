"""
    scheme_iter!(
    ρ,u,v,cα,sα,
    Δx,Δy,Δt0,
    c1,c2,λ,λp,b,bp,
    bcond_x,bcond_y,
    Fx,Fy,method,
    flux_x_ρ, flux_x_u, flux_x_v,flux_x_cα,flux_x_sα,
    flux_y_ρ, flux_y_u, flux_y_v,flux_y_cα,flux_y_sα,
    ρ_x,u_x,v_x,cα_x,sα_x,max_vap_xy)

One iteration of the numerical scheme which updates the data `(ρ,u,v,cα,sα)`. The scheme
follows the methodology introduced by

S. Motsch, L. Navoret, *Numerical simulations of a non-conservative hyperbolic
system with geometric constraints describing swarming behavior*, Multiscale Model. Simul.,
Vol. 9, No. 3, pp. 1253-1275, 2011.

The finite volume scheme is based on the following three-step splitting with U=(u,v):

1. The conservative part

    ∂ₜρ + ∇ₓ⋅(ρ(c₁U + bρ∇ₓα)) = 0
    ∂ₜ(ρU) + ∇ₓ⋅(ρU⊗(c₂U + bρ∇ₓα)) + λ∇ₓρ = 0
    ∂ₜ(ρcos(α)) + ∇ₓ⋅(ρcos(α)(c₁U + bρ∇ₓα)) = 0
    ∂ₜ(ρsin(α)) + ∇ₓ⋅(ρsin(α)(c₁U + bρ∇ₓα)) = 0

It is solved using a dimensional splitting. At each time step, and for each dimension,
the phase gradient ∇ₓα is approximated by 

∂ₓα ≃ arg(cos(αr)*cos(αl) + sin(αr)sin(αl) + i(sin(αr)*cos(αl) - cos(αr)sin(αl)))/Δx

2. The relaxation part

    ε∂ₜU = (1-|U|^2)U

It reduces to a mere normalization U <- U/|U|

3. The source term

    ∂ₜρ = 0
    ∂ₜU = 0 
    ∂ₜcos(α) = -sin(α)[(b-b')ρ|∇ₓα|² + λ'(Δρ + |∇ₓ√ρ|²)]
    ∂ₜsin(α) = cos(α)[(b-b')ρ|∇ₓα|² + λ'(Δρ + |∇ₓ√ρ|²)]

The time step Δt is adaptative. 

"""
function scheme_iter!(
    ρ,u,v,cα,sα,
    Δx,Δy,Δt0,
    c1,c2,λ,λp,b,bp,
    bcond_x,bcond_y,
    Fx,Fy,method,
    flux_x_ρ, flux_x_u, flux_x_v,flux_x_cα,flux_x_sα,
    flux_y_ρ, flux_y_u, flux_y_v,flux_y_cα,flux_y_sα,
    ρ_x,u_x,v_x,cα_x,sα_x,max_vap_xy)

    ncellx = size(ρ)[1] - 2
    ncelly = size(ρ)[2] - 2

    Δt = Δt0

    #===============================================================================#
    #======================= Dimensional splitting: x-axis =========================#
    #===============================================================================#

    #----------------------- 1. Compute the flux -----------------------------------#

    @inbounds for j in 1:ncelly
        @simd for i in 1:(ncellx+1)
            F = flux_x(
                    ρ[i,j+1],ρ[i+1,j+1],
                    u[i,j+1],u[i+1,j+1],
                    v[i,j+1],v[i+1,j+1],
                    cα[i,j+1],cα[i+1,j+1],
                    sα[i,j+1],sα[i+1,j+1],
                    c1,c2,λ,b,Δx,method=method
            )
            flux_x_ρ[i,j] = F[1]
            flux_x_u[i,j] = F[2]
            flux_x_v[i,j] = F[3]
            flux_x_cα[i,j] = F[4]
            flux_x_sα[i,j] = F[5]
            max_vap_xy[i,j] = max_vap_x(
                    ρ[i,j+1],ρ[i+1,j+1],
                    u[i,j+1],u[i+1,j+1],
                    v[i,j+1],v[i+1,j+1],
                    cα[i,j+1],cα[i+1,j+1],
                    sα[i,j+1],sα[i+1,j+1],
                    c1,c2,λ,b,Δx
            )

        end
    end

    # Compute the maximal absolute value of all the eigenvalues 
    max_vap = maximum(max_vap_xy[1:(ncellx+1),1:ncelly])

    #----------------------- Update ------------------------------------------------#
    
    # The maximal allowed time-step Δt is such that Δt/Δx * max_vap = CFL_max
    CFL_max = 0.1
    min_step = min(Δx,Δy)
    Δt = min(Δt,min_step*CFL_max/max_vap)

    try_again = true

    while try_again

        @inbounds for j in 2:(ncelly+1)
            for i in 2:(ncellx+1)

                #------ 2. Conservative part -------------------------------------------#
                U = (ρ[i,j] - (Δt/Δx) * (flux_x_ρ[i, j-1] - flux_x_ρ[i-1, j-1]),
                    ρ[i,j]*u[i,j] - (Δt/Δx) * (flux_x_u[i, j-1] - flux_x_u[i-1, j-1]),
                    ρ[i,j]*v[i,j] - (Δt/Δx) * (flux_x_v[i, j-1] - flux_x_v[i-1, j-1]),
                    ρ[i,j]*cα[i,j] - (Δt/Δx) * (flux_x_cα[i, j-1] - flux_x_cα[i-1, j-1]),
                    ρ[i,j]*sα[i,j] - (Δt/Δx) * (flux_x_sα[i, j-1] - flux_x_sα[i-1, j-1]))
                #------- 3. Relaxation -------------------------------------------------#

                norm = sqrt(U[2]^2 + U[3]^2)

                if U[1]<0
                    ρ_x[i,j] = 1e-6   # Set ρ to a small value
                else
                    ρ_x[i,j] = U[1]
                end
                if norm>1e-12   # If the norm is too small, it is treated as zero
                    u_x[i,j] = U[2]/norm
                    v_x[i,j] = U[3]/norm
                else
                    normuv = sqrt(u_x[i,j]^2+v_x[i,j]^2)
                    u_x[i,j] = u_x[i,j]/normuv
                    v_x[i,j] = v_x[i,j]/normuv
                end
                
                normα = sqrt(U[4]^2 + U[5]^2)

                if normα>1e-12
                    cα_x[i,j] = U[4]/normα
                    sα_x[i,j] = U[5]/normα
                else
                    normα = sqrt(cα_x[i,j]^2+sα_x[i,j]^2)
                    cα_x[i,j] = cα_x[i,j]/normα
                    sα_x[i,j] = sα_x[i,j]/normα
                end
            end
        end


        #----------------------- 4. Boundary conditions --------------------------------#
        boundary_conditions_ρuv!(ρ_x,u_x,v_x,cα_x,sα_x,bcond_x,bcond_y)


        #===============================================================================#
        #======================= Dimensional splitting: y-axis =========================#
        #===============================================================================#

        #----------------------- 1. Compute the flux -----------------------------------#
        # Note: Change basis (u,v) -> (v,-u) in order to use the same function `flux_x`
        #-------------------------------------------------------------------------------#

        @inbounds for j in 1:(ncelly+1)
            @simd for i in 1:ncellx
                F = flux_x(
                        ρ_x[i+1,j],ρ_x[i+1,j+1],
                        v_x[i+1,j],v_x[i+1,j+1],
                        -u_x[i+1,j],-u_x[i+1,j+1],
                        cα_x[i+1,j],cα_x[i+1,j+1],
                        sα_x[i+1,j],sα_x[i+1,j+1],
                        c1,c2,λ,b,Δy,method=method
                )

                flux_y_ρ[i,j] = F[1]
                flux_y_u[i,j] = -F[3]
                flux_y_v[i,j] = F[2]
                flux_y_cα[i,j] = F[4]
                flux_y_sα[i,j] = F[5]
                max_vap_xy[i,j] = max_vap_x(
                    ρ_x[i+1,j],ρ_x[i+1,j+1],
                    v_x[i+1,j],v_x[i+1,j+1],
                    -u_x[i+1,j],-u_x[i+1,j+1],
                    cα_x[i+1,j],cα_x[i+1,j+1],
                    sα_x[i+1,j],sα_x[i+1,j+1],
                    c1,c2,λ,b,Δy
                )
            end
        end
        
        # Compute the new maximal absolute values of the eigenvalues. 
        # If the current time step exceed Δx*CFL_max/max_vap, restart from the beginning... 
        max_vap = maximum(max_vap_xy[1:ncellx,1:(ncelly+1)])
        Δt_max = min_step * CFL_max/max_vap 
        try_again = Δt >= 1.000001*Δt_max
        Δt = min(Δt,Δt_max)
        
    end


    #----------------------- Update ------------------------------------------------#
    @inbounds for j in 2:(ncelly+1)
        @simd for i in 2:(ncellx+1)

            #------ 2. Conservative part -------------------------------------------#
            U = (ρ_x[i,j] - (Δt/Δy) * (flux_y_ρ[i-1, j] - flux_y_ρ[i-1, j-1]),
                 ρ_x[i,j]*u_x[i,j] - (Δt/Δy) * (flux_y_u[i-1, j] - flux_y_u[i-1, j-1]),
                 ρ_x[i,j]*v_x[i,j] - (Δt/Δy) * (flux_y_v[i-1, j] - flux_y_v[i-1, j-1]),
                 ρ_x[i,j]*cα_x[i,j] - (Δt/Δy) * (flux_y_cα[i-1, j] - flux_y_cα[i-1, j-1]),
                 ρ_x[i,j]*sα_x[i,j] - (Δt/Δy) * (flux_y_sα[i-1, j] - flux_y_sα[i-1, j-1])
                 )
            #------- 3. Relaxation -------------------------------------------------#
            norm = sqrt(U[2]^2 + U[3]^2)

            if U[1]<0
                ρ[i,j] = 1e-6   # Set ρ to a small value
            else
                ρ[i,j] = U[1]
            end
            if norm>1e-12 # If the norm is too small, it is treated as zero
                u[i,j] = U[2]/norm
                v[i,j] = U[3]/norm
            else
                normuv = sqrt(u[i,j]^2+v[i,j]^2)
                u[i,j] = u[i,j]/normuv
                v[i,j] = v[i,j]/normuv
            end
            
            normα = sqrt(U[4]^2 + U[5]^2)

            if normα>1e-12
                cα[i,j] = U[4]/normα
                sα[i,j] = U[5]/normα
            else
                normα = sqrt(cα[i,j]^2+sα[i,j]^2)
                cα[i,j] = cα[i,j]/normα
                sα[i,j] = sα[i,j]/normα
            end
        end
    end

    #----------------------- 4. Boundary conditions --------------------------------#
    boundary_conditions_ρuv!(ρ,u,v,cα,sα,bcond_x,bcond_y)

    #===============================================================================#
    #======== Fractional splitting: exterior force and noise in the phase ==========#
    #===============================================================================#
    if !isnothing(Fx) && !isnothing(Fy)
        scheme_potential!(u,v,Fx,Fy,Δt,λ)
        boundary_conditions_ρuv!(ρ,u,v,cα,sα,bcond_x,bcond_y)
    end

    if (b!=bp) & (λp!=0.0)

        cα1 = deepcopy(cα)
        sα1 = deepcopy(sα)

        @inbounds for j in 2:(ncelly+1)
            @simd for i in 2:(ncellx+1)
                Δρ_ij = (ρ[i+1,j] + ρ[i-1,j] - 2*ρ[i,j])/Δx^2 + (ρ[i,j+1] + ρ[i,j-1] - 2*ρ[i,j])/Δy^2
                if abs(Δρ_ij) < 1e-6
                    Δρ_ij = 0.0
                end 
                ∇sqrtρ_squared_ij = (0.5*(sqrt(ρ[i+1,j]) - sqrt(ρ[i-1,j]))/Δx)^2 + (0.5*(sqrt(ρ[i,j+1]) - sqrt(ρ[i,j-1]))/Δy)^2
                if abs(∇sqrtρ_squared_ij) < 1e-6
                    ∇sqrtρ_squared_ij = 0.0
                end

                zx = 0.5 * angle(cα1[i+1,j]*cα1[i-1,j] + sα1[i-1,j]*sα1[i+1,j] + 1im*(sα1[i+1,j]*cα1[i-1,j] - sα1[i-1,j]*cα1[i+1,j])) / Δx
                zy = 0.5 * angle(cα1[i,j+1]*cα1[i,j-1] + sα1[i,j-1]*sα1[i,j+1] + 1im*(sα1[i,j+1]*cα1[i,j-1] - sα1[i,j-1]*cα1[i,j+1])) / Δy

                if (bp-b)*zx > 0
                    dx_α_ij = angle(cα1[i,j]*cα1[i-1,j] + sα1[i-1,j]*sα1[i,j] + 1im*(sα1[i-1,j]*cα1[i,j] - sα1[i,j]*cα1[i-1,j])) / Δx
                else
                    dx_α_ij = angle(cα1[i+1,j]*cα1[i,j] + sα1[i,j]*sα1[i+1,j] + 1im*(sα1[i+1,j]*cα1[i,j] - sα1[i,j]*cα1[i+1,j])) / Δx
                end

                if (bp-b)*zy > 0
                    dy_α_ij = angle(cα1[i,j]*cα1[i,j-1] + sα1[i,j-1]*sα1[i,j] + 1im*(sα1[i,j-1]*cα1[i,j] - sα1[i,j]*cα1[i,j-1])) / Δx
                else
                    dy_α_ij = angle(cα1[i,j+1]*cα1[i,j] + sα1[i,j]*sα1[i,j+1] + 1im*(sα1[i,j+1]*cα1[i,j] - sα1[i,j]*cα1[i,j+1])) / Δx
                end

                s_ij = sα1[i,j]
                c_ij = cα1[i,j]
                α_ij = atan(s_ij,c_ij)

                α_ij_h = α_ij
                α_ij_h -= (bp-b) * ρ[i,j] * dx_α_ij^2 * Δt
                α_ij_h -= (bp-b) * ρ[i,j] * dy_α_ij^2 * Δt
                α_ij_h += Δt * λp * (Δρ_ij + 4*∇sqrtρ_squared_ij)

                cα[i,j] = cos(α_ij_h)
                sα[i,j] = sin(α_ij_h)
            end
        end
        
        boundary_conditions_ρuv!(ρ,u,v,cα,sα,bcond_x,bcond_y)

    end



    return Δt
end

"""
    scheme_potential!(u,v,Fx,Fy,Δt,λ)

Solve explicitly the source term

∂ₜρ = 0
∂ₜΩ = λP(Ω)F, F=-∇ₓV.

In dimension two, the solution is

Ω(t) = (cos(θ),sin(θ))ᵀ where

θ(t) = ψ + 2atan(C_0 exp(-λ|F|t))

with C_0 = tan((θ(0)-ψ)/2) and F = |F|(cos(ψ),sin(ψ))ᵀ.
"""
function scheme_potential!(u,v,Fx,Fy,Δt,λ)
    ncellx = size(u)[1] - 2
    ncelly = size(u)[2] - 2
    @inbounds for j in 2:ncelly+1
        for i in 2:ncellx+1
            normF = sqrt(Fx[i,j]^2 + Fy[i,j]^2)
            if normF > 1e-9
                θ0 = atan(v[i,j],u[i,j])
                ψ = atan(Fy[i,j],Fx[i,j])
                C0 = tan((θ0-ψ)/2)
                θ = ψ + 2 * atan(C0*exp(-λ*normF*Δt))
                u[i,j] = cos(θ)
                v[i,j] = sin(θ)
            end
        end
    end
end
