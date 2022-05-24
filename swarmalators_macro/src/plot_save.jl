using CairoMakie, ProgressMeter, Statistics

"""
    plot_rhoUV(
        ρ,u,v,α
        Δx,Δy,
        range,
        save_plot=true,save_video=false,dir_name=pwd(),file_name="0.png";
        resolution=(1200,600),arrow_step=20,fps=40,kwargs...
    )

Return the figure, axes, density heatmap, phase plot and arrows associated to the density `ρ`, velocity field
`(u,v)` and phases `α`. The plot is saved in the directory `dir_name` in the file `file_name`.
"""
function plot_rhoUV(
    ρ,u,v,α,
    Δx,Δy,
    range,
    save_plot=true,save_video=false,dir_name=pwd(),file_name="0.png";
    resolution=(1200,600),arrow_step=20,fps=40,kwargs...
)
    set_theme!(theme_black())
    arrowcolor = :black
    ncellx = size(ρ)[1] - 2
    ncelly = size(ρ)[2] - 2
    Lx = ncellx*Δx
    Ly = ncelly*Δy
    x = Δx .* collect(0:(ncellx-1)) .+ Δx/2
    y = Δy .* collect(0:(ncelly-1)) .+ Δy/2
    fig = Figure(resolution=resolution)

    axρ = Axis(fig[1, 1], title="Density \n time=0.00")
    rho = ρ[2:ncellx+1,2:ncelly+1]
    hmρ = heatmap!(axρ,x,y,rho,colorrange=range,colormap=:linear_kryw_5_100_c67_n256)
    axρ.aspect = AxisAspect(1)
    xlims!(axρ,0,Lx)
    ylims!(axρ,0,Ly)
    Colorbar(fig[1,2],hmρ)
    axα = Axis(fig[1, 3], title="Phase \n time=0.00")
    alpha = α[2:ncellx+1,2:ncelly+1]
    hmα = heatmap!(axα,x,y,mod.(alpha,2*pi),colorrange=(0,2*pi),colormap=:hsv)
    axα.aspect = AxisAspect(1)
    xlims!(axα,0,Lx)
    ylims!(axα,0,Ly)
    Colorbar(fig[1,4],hmα)

    
    extract_x = floor(Int,arrow_step/2):arrow_step:ncellx
    extract_y = floor(Int,arrow_step/2):arrow_step:ncelly
    xx = x[extract_x]
    yy = y[extract_y]
    u_grid = u[2:ncellx+1,2:ncelly+1]
    v_grid = v[2:ncellx+1,2:ncelly+1]
    uu = u_grid[extract_x,extract_y]
    vv = v_grid[extract_x,extract_y]
    arrowsρ = arrows!(axρ,xx,yy,uu,vv)
    arrowsρ.color = arrowcolor
    arrowsρ.lengthscale = 1/2*arrow_step*Δx
    arrowsρ.arrowsize = floor(Int,0.5/80*resolution[1])
    arrowsρ.linewidth = floor(Int,1/800*resolution[1])
    arrowsρ.origin = :center
    arrowsα = arrows!(axα,xx,yy,uu,vv)
    arrowsα.color = arrowcolor
    arrowsα.lengthscale = 1/2*arrow_step*Δx
    arrowsα.arrowsize = floor(Int,0.5/80*resolution[1])
    arrowsα.linewidth = floor(Int,1/800*resolution[1])
    arrowsα.origin = :center

    if save_plot
        save(joinpath(dir_name,file_name),fig)
    end
    if save_video
        stream = VideoStream(fig,framerate=fps)
        recordframe!(stream)
        return fig, axρ, axα, hmρ, hmα, arrowsρ, arrowsα, stream
    else
        return fig, axρ, axα, hmρ, hmα, arrowsρ, arrowsα
    end
end

"""
    update_plot!(fig, axρ, axα, hmρ, hmα, arrowsρ, arrowsα,ρ,u,v,α,title,dir_name,file_name,save_plot=true,stream=nothing)

Update a plot created with the function `plot_rhoUV` and save it.
"""
function update_plot!(fig, axρ, axα, hmρ, hmα, arrowsρ, arrowsα,ρ,u,v,α,title,dir_name,file_name,save_plot=true,stream=nothing)
    ncellx = size(ρ)[1] - 2
    ncelly = size(ρ)[2] - 2
    rho = ρ[2:ncellx+1,2:ncelly+1]
    hmρ[3] = rho
    alpha = mod.(α[2:ncellx+1,2:ncelly+1],2*pi)
    hmα[3] = alpha
    u_grid = u[2:ncellx+1,2:ncelly+1]
    v_grid = v[2:ncellx+1,2:ncelly+1]
    extract_x = 10:20:ncellx
    extract_y = 10:20:ncelly
    uu = u_grid[extract_x,extract_y]
    vv = v_grid[extract_x,extract_y]
    arrowsρ[:directions] = vec(Vec2f.(uu,vv))
    arrowsα[:directions] = vec(Vec2f.(uu,vv))
    axρ.title = "Density \n " * title
    axα.title = "Phase \n " * title
    if save_plot
        save(joinpath(dir_name,file_name),fig)
    end
    if !isnothing(stream)
        recordframe!(stream)
    end
end

"""
    save_data!(
        data,
        ρ,u,v,cα,sα,
        dir_name,file_name;key="data")

Update the dictionary `data` with the values `(ρ,u,v,cα,sα)` and save it.
"""
function save_data!(
    data,
    ρ,u,v,cα,sα,
    dir_name,file_name;key="data")
    data[:ρ] = ρ
    data[:u] = u
    data[:v] = v
    data[:cα] = cα
    data[:sα] = sα
    save(joinpath(dir_name,file_name),key,data)
end
