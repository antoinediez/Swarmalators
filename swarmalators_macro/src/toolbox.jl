using QuadGK, HCubature
using NLsolve
# using Images, FileIO, VideoIO

"""
Create a new directory `dirname` in the current path.
Append a number to the name if the directory already exists.
"""
function make_new_dir(dirname::String)
    if ispath(dirname)
        k=1
        while ispath(dirname*"_$k")
            k+=1
        end
        return mkdir(dirname*"_$k")
    else
        return mkdir(dirname)
    end
end

"""
    nice_number(x,K::Int)

Return a string which is the rounded value of `x` with `K` (decimal) digits and
if necessary, pad `"0"` to the right so that there are exactly `K` decimal places.
"""
function nice_float2string(x,K::Int)
    integer_part = trunc(Int,x)
    y = x - integer_part
    y10K = round(Int,y*10^K)
    if y10K >= 10
        decimal_part = rpad(y10K,K,"0")
    else
        decimal_part = lpad(y10K,K,"0")
    end
    return "$(integer_part).$decimal_part"
end

"""
Return the coefficients `c1,c2,λ` of the SOH model computed for the `model` `"Fokker-Planck"` (default)
or `"BGK"` with the concentration parameters `κ`.
These coefficients are given by

c₁ = ∫\\_[0,π] cos(θ)exp(κcos(θ))dθ / ∫\\_[0,π] exp(κcos(θ))dθ

c₂ = ∫\\_[0,π] cos(θ)sin(θ)g(θ)exp(κcos(θ))dθ / ∫\\_[0,π] sin(θ)g(θ)exp(κcos(θ))dθ

λ = 1/κ

where

g(θ) = θ/κ - π/κ * ∫\\_[0,θ] exp(-κcos(ψ))dψ / ∫\\_[0,π] exp(-κcos(ψ))dψ in the Fokker-Planck case

and

g(θ) = 1 in the BGK case.
"""
function coefficients_Vicsek(κ;model="Fokker-Planck")
    λ = 1/κ
    I1 = quadgk(θ->cos(θ)*exp(κ*cos(θ)),0,pi,rtol=1e-5)[1]
    Z1 = quadgk(θ->exp(κ*cos(θ)),0,pi,rtol=1e-5)[1]
    c1 = I1/Z1
    if model=="BGK"
        I2 = quadgk(θ->cos(θ)*sin(θ)^2*exp(κ*cos(θ)),0,pi,rtol=1e-5)[1]
        Z2 = quadgk(θ->sin(θ)^2*exp(κ*cos(θ)),0,pi,rtol=1e-5)[1]
        c2 = I2/Z2
        return c1,c2,λ
    elseif model=="Fokker-Planck"
        Zinv = quadgk(θ->exp(-κ*cos(θ)),0,pi,rtol=1e-5)[1]
        I21 = quadgk(θ->θ*cos(θ)*sin(θ)*exp(κ*cos(θ)),0,pi,rtol=1e-5)[1]/κ
        I22 = pi/(κ*Zinv) * hcubature(x->cos(x[1])*sin(x[1])*exp(κ*cos(x[1]))*exp(-κ*cos(x[2]))*(x[2]<x[1]),(0,0),(pi,pi),rtol=1e-3)[1]
        I2 = I21 - I22
        Z21 = quadgk(θ->θ*sin(θ)*exp(κ*cos(θ)),0,pi,rtol=1e-5)[1]/κ
        Z22 = pi/(κ*Zinv) * hcubature(x->sin(x[1])*exp(κ*cos(x[1]))*exp(-κ*cos(x[2]))*(x[2]<x[1]),(0,0),(pi,pi),rtol=1e-3)[1]
        Z2 = Z21 - Z22
        c2 = I2/Z2
        return c1,c2,λ
    else
        ArgumentError("Model not defined!")
    end
end


# function make_video(path;video_name="video.mp4",fps=30)
#     dir = joinpath(path,"plots") #path to directory holding images
#     imgnames = filter(x->occursin(".png",x), readdir(dir)) # Populate list of all .pngs
#     intstrings =  map(x->split(x,".")[1], imgnames) # Extract index from filenames
#     p = sortperm(parse.(Int, intstrings)) #sort files numerically
#     imgnames = imgnames[p]

#     encoder_options = (crf=23, preset="medium")

#     firstimg = load(joinpath(dir, imgnames[1]))
#     open_video_out(joinpath(path,video_name), firstimg, framerate=fps, encoder_options=encoder_options,target_pix_fmt=VideoIO.AV_PIX_FMT_YUV420P) do writer
#         @showprogress "Encoding video frames.." for i in eachindex(imgnames)
#             img = load(joinpath(dir, imgnames[i]))
#             write(writer, img)
#         end
#     end
# end


##################################################################################################
#################### EXPLICIT AND IMPLICIT FUNCTIONS FOR THE STRIP GEOMETRY ######################
##################################################################################################

function F(ρ;el,q)
    if ρ<0
        return ρ
    end
    if el>0
        return abs(ρ)^q*abs(ρ+el)^(1-q)
    else
        if ρ<-el
            return ρ^q*(-el-ρ)^(1-q)
        else
            return ρ^q*(ρ+el)^(1-q)
        end
    end
end


function F1(ρ;el,q)
    mq = q^q*(1-q)^(1-q)
    Ml = -mq*el
    if el>0
        error("F1 for ℓ<0 only.")
    end
    if ρ<0
        return 0.0
    elseif ρ>-q*el
        return Ml
    else
        return ρ^q*(-el-ρ)^(1-q)
    end
    # if ρ<-el
    #     return 
    # else
    #     return ρ^q*(ρ+el)^(1-q)
    # end
end

function F2(ρ;el,q)
    mq = q^q*(1-q)^(1-q)
    Ml = -mq*el
    if el>0
        error("F2 for ℓ<0 only.")
    end
    if ρ<-q*el
        return Ml
    elseif ρ<-el
        return ρ^q*(-el-ρ)^(1-q)
    else
        return -ρ-el
    end
end

function F3(ρ;el,q)
    mq = q^q*(1-q)^(1-q)
    Ml = -mq*el
    if el>0
        error("F3 for ℓ<0 only.")
    end
    if ρ<-el
        return ρ+el
    else
        return ρ^q*(ρ+el)^(1-q)
    end
end

function G(y;el,q)
    function fun!(f,ρ)
        f[1] = F(ρ[1];el=el,q=q) - y
    end
    sol = nlsolve(fun!,[1.0])
    if sol.f_converged
        return sol.zero[1]
    else
        error("Failed (G).")
    end
end

function G1(y;el,q)
    mq = q^q*(1-q)^(1-q)
    Ml = -mq*el
    if y>Ml
        # error("Argument y=$y out of the definition range of G1.")
        return -q*el + (y-Ml)*50000000
    end
    function fun!(f,ρ)
        f[1] = F1(ρ[1];el=el,q=q) - y + 100000*(ρ[1]>-q*el)
    end
    sol = nlsolve(fun!,[-0.5*q*el])
    if sol.f_converged
        return sol.zero[1]
    else
        error("Failed (G1).")
    end
end

function G2(y;el,q)
    mq = q^q*(1-q)^(1-q)
    Ml = -mq*el
    if y>Ml
        error("Argument y=$y out of the definition range of G1.")
    end
    function fun!(f,ρ)
        f[1] = F2(ρ[1];el=el,q=q) - y 
    end
    sol = nlsolve(fun!,[-0.5*(q*el+el)])
    if sol.f_converged
        return sol.zero[1]
    else
        error("Failed (G2).")
    end
end

function G3(y;el,q)
    function fun!(f,ρ)
        f[1] = F3(ρ[1];el=el,q=q) - y
    end
    sol = nlsolve(fun!,[-2*el],ftol=1e-3)
    if sol.f_converged
        return sol.zero[1]
    else
        error("Failed (G3).")
    end
end

function ICL(C;el,κp,V0,q,Δx=Δx,k=0)
    if k==0
        return sum(G.(C.*exp.(-κp.*V0);el=el,q=q))*Δx
    elseif k==1
        return sum(G1.(C.*exp.(-κp.*V0);el=el,q=q))*Δx
    elseif k==3
        return sum(G3.(C.*exp.(-κp.*V0);el=el,q=q))*Δx
    end
end

function Ik(k;q,κp,V0,Δx=Δx)
    mq = q^q*(1-q)^(1-q)
    if k==1
        return sum(G1.(mq.*exp.(-κp.*V0);el=-1.0,q=q))*Δx
    elseif k==2
        return sum(G2.(mq.*exp.(-κp.*V0);el=-1.0,q=q))*Δx
    elseif k==3
        return sum(G3.(mq.*exp.(-κp.*V0);el=-1.0,q=q))*Δx
    elseif k==12
        mid = floor(Int64,length(V0)/2.0)
        return sum(G1.(mq.*exp.(-κp.*V0[1:mid]);el=-1.0,q=q))*Δx + sum(G2.(mq.*exp.(-κp.*V0[mid+1:end]);el=-1.0,q=q))*Δx
    else
        error("I_k not defined for that k.")
    end
end

function Cl(el;κp,V0,q,Δx=Δx,k=0,start=1.0)
    function fun!(f,c)
        f[1] = ICL(c[1];el=el,κp=κp,V0=V0,Δx=Δx,q=q,k=k) - 1.0
    end
    sol = nlsolve(fun!,[start])
    if sol.f_converged
        return sol.zero[1]
    else
        error("Failed (Cl).")
    end
end

function Cl3(el;κp,V0,q,Δx=Δx)
    function fun!(f,c)
        f[1] = ICL(c[1];el=el,κp=κp,V0=V0,Δx=Δx,q=q) - 1.0
    end
    sol = nlsolve(fun!,[1.0])
    if sol.f_converged
        return sol.zero[1]
    else
        error("Failed (Cl).")
    end
end

function H(el;κp,V0,q,Δx=Δx)
    return G(Cl(el;κp=κp,V0=V0,Δx=Δx,q=q);el=el,q=q)+el
end

function lstar(cbz;κp,V0,q,Δx=Δx)
    function fun!(f,els)
        f[1] = H(els[1];κp=κp,V0=V0,Δx=Δx,q=q) - cbz
    end
    sol = nlsolve(fun!,[1.0])
    if sol.f_converged
        return sol.zero[1]
    else
        error("Failed (H).")
    end
end




