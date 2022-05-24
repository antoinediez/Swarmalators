"""
    boundary_conditions_x!(density,bcond_x)

Apply the boundary conditions `bcond_x` on the x-axis of the matrix `density`.
Currently, only the boundary conditions `"periodic"`, `"Neumann"` and `"reflecting"` are implemented.

"""
function boundary_conditions_x!(density,bcond_x)
    if bcond_x == "periodic"
        density[1,:] = density[end-1,:]
        density[end,:] = density[2,:]
    elseif bcond_x == "Neumann"
        density[1,:] = density[2,:]
        density[end,:] = density[end-1,:]
    elseif bcond_x == "reflecting"
        density[1,:] = -density[2,:]
        density[end,:] = -density[end-1,:]
    else
        ArgumentError("Boundary condition not defined!")
    end
end

"""
    boundary_conditions_y!(density,bcond_y)

Apply the boundary conditions `bcond_y` on the y-axis of the matrix `density`.
Currently, only the boundary conditions `"periodic"`, `"Neumann"` and `"reflecting"` are implemented.
"""
function boundary_conditions_y!(density,bcond_y)
    if bcond_y == "periodic"
        density[:,1] = density[:,end-1]
        density[:,end] = density[:,2]
    elseif bcond_y == "Neumann"
        density[:,1] = density[:,2]
        density[:,end] = density[:,end-1]
    elseif bcond_y == "reflecting"
        density[:,1] = -density[:,2]
        density[:,end] = -density[:,end-1]
    else
        ArgumentError("Boundary condition not defined!")
    end
end

"""
    boundary_conditions_xy!(density,bcond_x,bcond_y)

Apply successively the functions `boundary_conditions_x!` and `boundary_conditions_x!` to
the matrix `density`.
"""
function boundary_conditions_xy!(density,bcond_x,bcond_y)
    boundary_conditions_x!(density,bcond_x)
    boundary_conditions_y!(density,bcond_y)
end


function boundary_conditions_ρuv!(ρ,u,v,cα,sα,bcond_x,bcond_y)
    if bcond_x == "periodic" || bcond_x == "Neumann"
        boundary_conditions_x!(ρ,bcond_x)
        boundary_conditions_x!(u,bcond_x)
        boundary_conditions_x!(v,bcond_x)
        boundary_conditions_x!(cα,bcond_x)
        boundary_conditions_x!(sα,bcond_x)
    elseif bcond_x == "reflecting"
        boundary_conditions_x!(ρ,"Neumann")
        boundary_conditions_x!(u,"reflecting")
        boundary_conditions_x!(v,"Neumann")
        boundary_conditions_x!(cα,"Neumann")
        boundary_conditions_x!(sα,"Neumann")
    else
        ArgumentError("Boundary condition not defined!")
    end

    if bcond_y == "periodic" || bcond_y == "Neumann"
        boundary_conditions_y!(ρ,bcond_y)
        boundary_conditions_y!(u,bcond_y)
        boundary_conditions_y!(v,bcond_y)
        boundary_conditions_y!(cα,bcond_y)
        boundary_conditions_y!(sα,bcond_y)
    elseif bcond_y == "reflecting"
        boundary_conditions_y!(ρ,"Neumann")
        boundary_conditions_y!(u,"Neumann")
        boundary_conditions_y!(v,"reflecting")
        boundary_conditions_y!(cα,"Neumann")
        boundary_conditions_y!(sα,"Neumann")
    else
        ArgumentError("Boundary condition not defined!")
    end

    if bcond_x == "periodic" && bcond_y == "periodic"
        boundary_corners!(ρ)
        boundary_corners!(u)
        boundary_corners!(v)
        boundary_corners!(cα)
        boundary_corners!(sα)
    end
end

function boundary_corners!(density)
    density[1,1] = density[end-1,end-1]
    density[1,end] = density[end-1,2]
    density[end,end] = density[2,2]
    density[end,1] = density[2,end-1]
end
