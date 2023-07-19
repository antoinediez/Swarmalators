#-------- Add the package to the current path if not installed globally -------#
push!(LOAD_PATH,joinpath(@__DIR__,"src"))
#------------------------------------------------------------------------------#

using SOH   # import the package
using Plots
using JLD2
using NPZ
using IfElse

#-------- Model parameters ----------------------------------------------------#
# The coefficients are computed using the function `coefficients_Vicsek` in the 
# script `toolbox.jl` for the Fokker-Planck and the BGK models. 

# γ = 0.2     # phase attraction-repulsion force intensity
# γ = 0.03266999579051092
γ = 0.033
# κ_Vic = 5.0     # concentration parameter (velocity)
κ_Vic = 1.4    # concentration parameter (velocity)
# κp_phase = 3.0    # concentration parameter (phase)
κ_phase = 100.0    # concentration parameter (phase)
c1,c2,λ = coefficients_Vicsek(κ_Vic)
c1p,c2p,λpvic = coefficients_Vicsek(κ_phase)

#### SH case ####
b = -γ * c1p^2
bp = -γ * c1p * (1/κ_phase + c2p)
λp = -γ * c1p/κ_phase
#################

#### NSH case ####
# b = -γ
# bp = b  
# λp = 0.0
###################

#-------- Domain parameters ---------------------------------------------------#

# Rectangular domain of size Lx*Ly
Lx = 1.0   
Ly = 1.0

# Boundary conditions (possible choices "periodic", "Neumann", "reflecting")
# bcond_x = "periodic"
bcond_x = "Neumann"
bcond_y = "periodic"


#-------- Numerical parameters ------------------------------------------------#

# Numer of cells and spatial step
ncellx = 400
ncelly = 400
Δx = Lx / ncellx
Δy = Ly / ncelly

# Time step 
Δt = 0.0001
# Δt = 0.000005

# Final time
T = 40.0

# Method ("Roe" or "HLLE")
method = "HLLE"


#-------- Exterior force ------------------------------------------------------#

# If no exterior force:
# Fx = nothing
# Fy = nothing

# Otherwise define the x and y components as (ncellx+2)*(ncelly+2) matrices.

######################## EXAMPLE: STRIP GEOMETRY ###############################
Fx = zeros(ncellx+2,ncelly+2)
Fy = zeros(ncellx+2,ncelly+2)
V = zeros(ncellx+2,ncelly+2)
V0 = @view V[2:(ncellx+1),3]

κ = 1.1

q = λ/(λ+c1-c2)
κp = κ/(λ+c1-c2)

eps = 0.001
alph = 0.75
# alph = 0.4

for j in 2:(ncelly+1)
    for i in 2:(ncellx+1)
        x = (i-2)*Δx + Δx/2
        # y = (j-2)*Δy + Δy/2

        #-------------- log potential -----------------------#
        # V[i,j] = -κ/κp*log(x*(1-x)+eps) + κ/κp*log(0.25+eps)
        # Fx[i,j] = κ/κp*(1-2*x)/(x*(1-x)+eps)
        #----------------------------------------------------#

        #------------ inverse power potential ---------------#
        V[i,j] = κ*(x*(1-x)+eps)^(-alph) - κ*(0.25+eps)^(-alph)
        Fx[i,j] = κ*alph*(1-2*x)*(x*(1-x)+eps)^(-alph-1)
        #----------------------------------------------------#
    end
end
################################################################################

#-------- Saving parameters ---------------------------------------------------#

simu_name = "simu"
should_save = false
should_plot = true
save_video = true   

#-- Initial conditions: (perturbed) doubly periodic tavelling wave solution ---#

#################### DOUBLY-PERIODIC ###########################################

# ρ = ones(ncellx+2,ncelly+2) .+ 0.8*(rand(ncellx+2,ncelly+2).-0.5)
# θ = pi/2 .+ 0.8.*(rand(ncellx+2,ncelly+2).-0.5) 
# # θ = -pi/2 .+ 0.8.*(rand(ncellx+2,ncelly+2).-0.5)
# # θ = 0.0 .+ 0.8.*(rand(ncellx+2,ncelly+2).-0.5)
# u = cos.(θ)
# v = sin.(θ)
# α = zeros(ncellx+2,ncelly+2)
# for j in 2:(ncelly+1)
#     for i in 2:(ncellx+1)
#         yij = (j-2)*Δy + Δy/2 
#         α[i,j] = 2*pi*yij
#     end
# end
# α .+= pi .+ 2*pi.*(rand(ncellx+2,ncelly+2).-0.5)
# cα = cos.(α)
# sα = sin.(α)

################################################################################

####################### STRIP GEOMETRY #########################################

ρ = zeros(ncellx+2,ncelly+2)
u = zeros(ncellx+2,ncelly+2)
v = zeros(ncellx+2,ncelly+2)
α = zeros(ncellx+2,ncelly+2)

zx = zeros(ncellx+2,ncelly+2)
zy = 2*pi*ones(ncellx+2,ncelly+2)

I = sum(exp.(-κp.*V0))*Δx
cond = abs(b*2*pi)/(c1*I)

if cond>1
    error("cond>1")
else
    println("cond=$cond")
end

println("ℓ*="*string(lstar(c1/abs(2*pi*b);κp=κp,V0=V0,q=q,Δx=Δx)))

λspeed = -lstar(c1/abs(2*pi*b);κp=κp,V0=V0,q=q,Δx=Δx)*b*2*pi
el = -λspeed/(b*2*pi)
Ml = -el*q^q*(1-q)^(1-q)

rho = abs.(G.(Cl(el;κp=κp,V0=V0,q=q,Δx=Δx).*exp.(-κp.*V0);el=el,q=q))

for j in 2:ncelly+1
    for i in 2:ncellx+1
        x = Δx/2 + (i-2)*Δx
        yij = (j-2)*Δy + Δy/2 
        # α[i,j] = 2*pi*yij

        ρ[i,j] = rho[i-1]
        v[i,j] = -b*2*pi/c1*(ρ[i,j]+el)
        u[i,j] = sqrt(1-v[i,j]^2)
        u[i,j] = IfElse.ifelse(x<0.5,sqrt(1-v[i,j]^2),-sqrt(1-v[i,j]^2))
        zx[i,j] = -c1/b*u[i,j]/max(ρ[i,j],1e-9)
    end
end

for j in 2:(ncelly+1)
    yij = (j-2)*Δy + Δy/2 
    α[Int(ncellx/2),j] = 2*pi*yij
end
for j in 2:(ncelly+1)
    for i in (Int(ncellx/2)+1):(ncellx+1)
        α[i,j] += (α[i-1,j] + Δx*zx[i-1,j])
    end
    for i in (Int(ncellx/2)-1):-1:2
        α[i,j] += (α[i+1,j] - Δx*zx[i+1,j])
    end
end
α = mod.(α,2*pi)
cα = cos.(α)
sα = sin.(α)
################################################################################

boundary_conditions_ρuv!(ρ,u,v,cα,sα,bcond_x,bcond_y)

#-------- Functions to record -------------------------------------------------#

# function max_ρ(ρ,u,v,cα,sα)
#     return maximum(ρ)
# end

# function center_of_mass(ρ,u,v,cα,sα;Δx=Δx,Δy=Δy)
#     ncellx = size(cα)[1] - 2
#     ncelly = size(cα)[2] - 2
#     x_cos = 0.0
#     y_cos = 0.0
#     x_sin = 0.0
#     y_sin = 0.0
#     for j in 2:(ncelly+1)
#         for i in 2:(ncellx+1)
#             x_ij = (i-2)*Δx + Δx/2
#             y_ij = (j-2)*Δy + Δy/2
#             mass = 1.0 + cα[i,j]
#             x_cos += ρ[i,j] * mass * cos(2*pi*x_ij)
#             x_sin += ρ[i,j] * mass * sin(2*pi*x_ij)
#             y_cos += ρ[i,j] * mass * cos(2*pi*y_ij)
#             y_sin += ρ[i,j] * mass * sin(2*pi*y_ij)
#         end
#     end
#     mean_x = mod(angle(x_cos + 1im*x_sin),2*pi)/(2*pi) * ncellx * Δx
#     mean_y = mod(angle(y_cos + 1im*y_sin),2*pi)/(2*pi) * ncelly * Δy
#     return mean_x,mean_y
# end


#-------- Finally run the simulation ------------------------------------------#

output = run!(
    ρ,u,v,cα,sα;
    Lx=Lx,Ly=Ly,Δt=Δt,
    c1=c1, c2=c2, λ=λ, λp=λp, b=b, bp=bp,
    final_time=T,
    bcond_x=bcond_x,bcond_y=bcond_y,
    Fx=Fx,Fy=Fy,method=method,
    simu_name=simu_name,
    should_save=should_save,step_time_save=0.01,
    should_plot=should_plot,step_time_plot=0.01,
    save_video=save_video,
    range=(0.,4),resolution=(1200,600),theme="dark",fps=30,record=[]
);

save_object(simu_name * "/time.jld2",output[end-1])
save_object(simu_name * "/record.jld2",output[end])


