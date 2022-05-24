using JLD2, Dates, ProgressMeter
"""
    run!(
        ρ,u,v,cα,sα;
        Lx,Ly,Δt,
        c1, c2, λ, λp, b, bp,
        final_time,
        bcond_x="periodic",bcond_y="periodic",
        Fx=nothing,Fy=nothing,method="HLLE",
        simu_name="simu",
        should_save=false,step_time_save=1,
        should_plot=false,step_time_plot=1,
        save_video=false,fps=40,
        range=(0.,5.),record=[],
        kwargs...
    )

Run a simulation of the SOH system starting from the data `(ρ,u,v,cα,sα)`. The domain
grid has the following structure.

```
|------- |----------------|---------------|----------------------|--------|
| ghost  |    ghost       |       ...     |    ghost             | ghost  |
|--------║================|===============|======================║--------|
| ghost  ║  (2,ncelly+1)  |      ...      | (ncellx+1,ncelly+1)  ║ ghost  |
|--------║----------------|---------------|----------------------║--------|
| ghost  ║  (2,ncelly)    |      ...      | (ncellx+1,ncelly)    ║ ghost  |
|--------║----------------|---------------|----------------------║--------|
|        ║                |               |                      ║        |
|--------║----------------|---------------|----------------------║--------|
| ghost  ║    (2,3)       |      ...      |   (ncellx+1,3)       ║ ghost  |
|--------║----------------|---------------|----------------------║--------|
| ghost  ║    (2,2)       |      ...      |   (ncellx+1,2)       ║ ghost  |
---------║================|===============|======================║--------|
| ghost  |    ghost       |      ...      |    ghost             | ghost  |
|--------|----------------|---------------|----------------------|--------|
````

The ghost cells (first and last column and rows) are used for the boundary
conditions.

# Arguments
- `ρ` -- density (matrix of size `(ncellx+2)*(ncelly+2)`)
- `u` -- x-coordinate of the velocity (matrix of size `(ncellx+2)*(ncelly+2)`)
- `v` -- y-coordinate of the velocity (matrix of size `(ncellx+2)*(ncelly+2)`)
- `cα` -- cos of the phase (matrix of size `(ncellx+2)*(ncelly+2)`)
- `sα` -- sin of the phase (matrix of size `(ncellx+2)*(ncelly+2)`)
- `Lx` -- length of the domain along the x-axis
- `Ly` -- length of the domain along the y-axis
- `Δt` -- time-step
- `c1` -- coefficient c1
- `c2` -- coefficient c1
- `λ` -- coefficient λ
- `λp` -- coefficient λ'
- `b` -- coefficient b
- `bp` -- coefficient b'
- `final_time` -- final time of the simulation
- `bcond_x` -- (optional, default : `"periodic"`) boundary condition along the x-axis
- `bcond_y` -- (optional, default : `"periodic"`) boundary condition along the y-axis
- `Fx` -- (optional, default : `nothing`) if specified, x-coordinate of the exterior force
- `Fy` -- (optional, default : `nothing`) if specified, y-coordinate of the exterior force
- `method` -- (optional, default : `"HLLE`) can be either `"Roe"` or `"HLLE"`
- `simu` -- (optional, default : `"simu"`) name of the simulation directory
- `should_save` -- (optional, default : `false`) flag for saving the data (ρ,u,v,cα,sα)
- `step_time_save` -- (optional, default : `1`) time step between two saved data
- `should_plot` -- (optional, default : `false`) flag for plotting and saving the density heatmap and phases
- `step_time_plot` -- (optional, default : `1`) time step between two saved plots
- `save_video` -- (optional, default : `false`) flag for saving a video
- `fps` -- (optional, default : `40`) frame per seconds
- `range` -- (optional, default : `(0.0,4.0)`) range of the density on the heatmap
- `record`-- (optional, default : `[]`) list of functions to record
- `kwargs` -- Keyword arguments to be passed in the function `plot_rhoUV`
"""
function run!(
    ρ,u,v,cα,sα;
    Lx,Ly,Δt,
    c1, c2, λ, λp, b, bp,
    final_time,
    bcond_x="periodic",bcond_y="periodic",
    Fx=nothing,Fy=nothing,method="HLLE",
    simu_name="simu",
    should_save=false,step_time_save=1,
    should_plot=false,step_time_plot=1,
    save_video=false,fps=40,
    range=(0.,4.),record=[],
    kwargs...
)
    #=======================================================================================#
    #============ Collect and display the parameters =======================================#
    #=======================================================================================#

    ncellx = size(ρ)[1] - 2
    Δx = Lx/ncellx
    ncelly = size(ρ)[2] - 2
    Δy = Ly/ncelly

    parameters = Dict{Symbol,Any}(
        :Lx => Lx, :Ly => Ly, :ncellx => ncellx, :ncelly => ncelly,
        :Δx => Δx, :Δy => Δy, :Δt => Δt,
        :c1 => c1, :c2 => c2, :λ => λ, :λp => λp, :b => b, :bp => bp,
        :final_time => final_time,
        :bcond_x => bcond_x, :bcond_y => bcond_y,
        :Fx => Fx, :Fy => Fy,
        :date => string(Dates.now())
    )

    println("\n************* Model parameters **************\n")
    println("c1 = $(parameters[:c1])")
    println("c2 = $(parameters[:c2])")
    println("λ = $(parameters[:λ])")
    println("λ' = $(parameters[:λp])")
    println("b = $(parameters[:b])")
    println("b' = $(parameters[:bp])")
    println("\n*********************************************\n")

    println("************* Domain parameters *************\n")
    println("Lx = $(parameters[:Lx])")
    println("Ly = $(parameters[:Ly])")
    println("Boundary condition in x = $(parameters[:bcond_x])")
    println("Boundary condition in y = $(parameters[:bcond_y])")
    println("Exterior force : $(!isnothing(parameters[:Fx]) && !isnothing(parameters[:Fy]))")
    println("\n*********************************************\n")

    println("************* Numerical method **************\n")
    println("Δx = $(parameters[:Δx])")
    println("Δy = $(parameters[:Δy])")
    println("Δt = $(parameters[:Δt])")
    println("Final time : $(parameters[:final_time])")
    println("\n*********************************************\n")

    println("************* Saving parameters *************\n")
    if should_save
        println("Save data every $step_time_save units of time")
        estimated_size = round(8*ncellx*ncelly*5*floor(Int, final_time / step_time_save)*1e-9, digits=2)
        println("Estimated size : $estimated_size GB")
    else
        println("Only the initial and final data will be saved")
    end
    if should_plot
        println("A plot will be saved every $step_time_plot units of time")
    else
        println("No plot will be saved")
    end
    if save_video
        println("A video with $fps fps will be saved")
    else
        println("No video will be saved")
    end
    println("\n*********************************************\n")

    #=======================================================================================#
    #============ Create the simulation directory and save the initial data ================#
    #=======================================================================================#

    dir_name = make_new_dir(simu_name)
    data_dir = mkdir(joinpath(dir_name,"data"))
    data = Dict(:ρ => ρ, :u => u, :v => v, :cα => cα, :sα => sα)
    save(joinpath(data_dir,"data_0.jld2"),"time=0",data)

    if should_plot || save_video
        print("Initializing plot... ")
        plot_dir = make_new_dir(joinpath(dir_name,"plots"))
        α = mod.(angle.(cα .+ 1im.*sα),2*pi)
        if save_video
            fig, axρ, axα, hmρ, hmα, arrowsρ, arrowsα, stream = plot_rhoUV(ρ,u,v,α,Δx,Δy,range,should_plot,save_video,plot_dir,"0.png";fps=fps,kwargs...)
        else
            fig, axρ, axα, hmρ, hmα, arrowsρ, arrowsα = plot_rhoUV(ρ,u,v,α,Δx,Δy,range,should_plot,save_video,plot_dir,"0.png";kwargs...)
        end
        println("Done.\n")
    end

    ndata = length(record)
    to_record = [[] for i in 1:ndata]
    time = []

    #=======================================================================================#
    #============ The big loop =============================================================#
    #=======================================================================================#

    flux_x_ρ = zeros(ncellx + 1, ncelly)
    flux_x_u = zeros(ncellx + 1, ncelly)
    flux_x_v = zeros(ncellx + 1, ncelly)
    flux_x_cα = zeros(ncellx + 1, ncelly)
    flux_x_sα = zeros(ncellx + 1, ncelly)
    flux_y_ρ = zeros(ncellx, ncelly + 1)
    flux_y_u = zeros(ncellx, ncelly + 1)
    flux_y_v = zeros(ncellx, ncelly + 1)
    flux_y_cα = zeros(ncellx, ncelly + 1)
    flux_y_sα = zeros(ncellx, ncelly + 1)
    ρ_x = zeros(ncellx+2, ncelly+2)
    u_x = zeros(ncellx+2, ncelly+2)
    v_x = zeros(ncellx+2, ncelly+2)
    cα_x = zeros(ncellx+2, ncelly+2)
    sα_x = zeros(ncellx+2, ncelly+2)
    max_vap_xy = zeros(ncellx+1, ncelly+1)

    println("Run the simulation...\n")
    ntime = 1000
    step_time_progress = final_time/ntime
    p = Progress(ntime)
    itime_save = 1
    itime_plot = 1
    real_time = 0.0
    push!(time,real_time)
    if ndata >= 1
        for n in 1:ndata
            push!(to_record[n],record[n](ρ,u,v,cα,sα))
        end
    end

    while real_time < final_time

        time_step = scheme_iter!(
            ρ,u,v,cα,sα,
            Δx,Δy,Δt,
            c1,c2,λ,λp,b,bp,
            bcond_x,bcond_y,Fx,Fy,method,
            flux_x_ρ,flux_x_u,flux_x_v,flux_x_cα,flux_x_sα,
            flux_y_ρ,flux_y_u,flux_y_v,flux_y_cα,flux_y_sα,
            ρ_x,u_x,v_x,cα_x,sα_x,max_vap_xy)
        
        real_time += time_step
        
        if should_save
            save_time = real_time>itime_save*step_time_save
            if save_time
                save_data!(data,ρ,u,v,cα,sα,data_dir,"data_$itime_save.jld2",key="time=$real_time")
                itime_save += 1
            end
        end
        if should_plot || save_video
            plot_time = real_time>itime_plot*step_time_plot
            if plot_time
                round_time = nice_float2string(real_time,2)
                α = mod.(angle.(cα .+ 1im.*sα),2*pi)
                if save_video
                    update_plot!(fig, axρ, axα, hmρ, hmα, arrowsρ, arrowsα,ρ,u,v,α,"time=$round_time",plot_dir,"$itime_plot.png",should_plot,stream)
                else
                    update_plot!(fig, axρ, axα, hmρ, hmα, arrowsρ, arrowsα,ρ,u,v,α,"time=$round_time",plot_dir,"$itime_plot.png")
                end
                itime_plot += 1
            end
        end

        update!(p,floor(Int,real_time/step_time_progress))

        push!(time,real_time)

        if ndata>=1
            for n in 1:ndata
                push!(to_record[n],record[n](ρ,u,v,cα,sα))
            end
        end

    end

    #=======================================================================================#
    #============ Collect the simulation time and save the final data ======================#
    #=======================================================================================#

    parameters[:duration] = p.tlast-p.tinit
    save(joinpath(dir_name,"parameters.jld2"), "parameters", parameters)
    if !should_save
        save_data!(data,ρ,u,v,cα,sα,data_dir,"data_$itime_save.jld2",key="time=$real_time")
    end

    if should_plot || save_video
        if save_video
            save(joinpath(dir_name,"video.mp4"), stream)
            return fig, axρ, axα, hmρ, hmα, arrowsρ, arrowsα, stream, time, to_record
        else
            return fig, axρ, axα, hmρ, hmα, arrowsρ, arrowsα, time, to_record
        end
    else
        return time, to_record
    end


end
