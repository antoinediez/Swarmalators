module SOH
include("boundary_conditions.jl")
include("flux.jl")
include("plot_save.jl")
include("run.jl")
include("scheme.jl")
include("toolbox.jl")

export boundary_conditions_xy!, boundary_conditions_ρuv!
export flux_x,max_CFL
export plot_rhoUV, update_plot!, save_data!, radial_density
export run!
export scheme_iter!
export make_new_dir, coefficients_Vicsek, nice_float2string



end     #module
