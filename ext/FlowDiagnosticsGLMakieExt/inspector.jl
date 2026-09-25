# Flow-diagnostics interactive 3-D inspector built on GLMakie.
#
# Entry points
#   flow_diagnostics_inspector(result, model, forces; kwarg...)
#   flow_diagnostics_inspector(result, case::JutulCase; kwarg...)

# Seconds in one Julian year (365.25 days)
const SECONDS_PER_YEAR = 365.25 * 86400.0

# -------------------------------------------------------------------------
# Public entry points
# -------------------------------------------------------------------------

"""
    flow_diagnostics_inspector(result, model, forces; kwarg...)
    flow_diagnostics_inspector(result, case::JutulCase; kwarg...)

Open an interactive GLMakie window for inspecting flow diagnostics.

# User interface
- **Step slider** – selects the simulation step. Flow diagnostics are
  recomputed whenever the step changes. Time is shown in years.
- **Quantity menu** – choose what to display:
  - `Forward TOF (years)` / `Backward TOF (years)` / `Residence time (years)`.
  - Per-injector and per-producer tracer concentrations.
  - Dynamic state variables (Pressure, Saturations, …) for the current step.
  - Static domain properties (Permeability, Porosity, …).
- **Colormap menu** – select from a list of common Makie colormaps.
- **TOF range slider** (`IntervalSlider`) – show only cells whose forward TOF
  falls within the selected [lo, hi] interval (in years). Cells with
  non-finite (unreachable) forward or backward TOF are always hidden.
- **Well markers** – injector cells are marked in red, producer cells in blue.
- The colorbar label and range update automatically.

# Arguments
- `result::ReservoirSimResult` – output of `simulate_reservoir`.
- `model` – the `MultiModel` or `SimulationModel` used for the simulation.
- `forces` – driving forces (wells, source terms) used during the simulation.
- `case::JutulCase` – alternative to passing `model` and `forces` separately.

# Keyword arguments
- `resolution` – window size in pixels, default `(1600, 1000)`.
- `z_is_depth::Bool` – if `true` the z-axis is inverted (depth convention).
  Inferred automatically from the mesh when not provided.
- `new_window::Bool` – if `true` (default outside CI) the figure is displayed
  in an independent window.
"""
function FlowDiagnostics.flow_diagnostics_inspector(
        result::JutulDarcy.ReservoirSimResult,
        model,
        forces;
        resolution::Tuple{Int,Int} = (1600, 1000),
        z_is_depth::Union{Missing,Bool} = missing,
        new_window::Bool = get(ENV, "CI", "false") == "false",
        max_tof::Float64 = 20.0
    )
    launch_inspector(result, model, forces;
        resolution  = resolution,
        z_is_depth  = z_is_depth,
        new_window  = new_window,
        max_tof     = max_tof
    )
end

# Convenience overload accepting a JutulCase
function FlowDiagnostics.flow_diagnostics_inspector(
        result::JutulDarcy.ReservoirSimResult,
        case::JutulCase;
        kwarg...
    )
    return FlowDiagnostics.flow_diagnostics_inspector(
        result, case.model, case.forces; kwarg...)
end

# -------------------------------------------------------------------------
# Internal implementation
# -------------------------------------------------------------------------

function launch_inspector(result, model, forces;
        resolution, z_is_depth, new_window, max_tof = 20.0)

    # ---- Mesh and geometry ------------------------------------------------
    rmodel  = JutulDarcy.reservoir_model(model)
    domain  = JutulDarcy.reservoir_domain(rmodel)
    mesh    = Jutul.physical_representation(domain)
    nc      = Jutul.number_of_cells(domain)
    nsteps  = length(result.states)

    if ismissing(z_is_depth)
        z_is_depth = Jutul.mesh_z_is_depth(mesh)
    end

    # Triangulate once – the expensive geometry step
    pts_raw, tri_raw, mapper = triangulate_mesh(mesh)
    pts_c = Makie.to_vertices(pts_raw)
    tri_c = Makie.to_triangles(tri_raw)

    # Cell centroids for well markers
    geo            = Jutul.tpfv_geometry(mesh)
    cell_centroids = geo.cell_centroids   # D × nc

    # ---- Initial diagnostics (last step) ----------------------------------
    step_init  = nsteps
    setup_init = fd_build_setup(result, model, forces, step_init)
    diag_init  = solve_flow_diagnostics(setup_init; compute_tracers = true)

    # ---- Static and initial dynamic quantities ----------------------------
    static_qty = fd_static_quantities(domain, nc)
    dyn_init   = fd_dynamic_quantities(
        fd_reservoir_state(result.states[step_init]), nc)

    # ---- Build quantity list ----------------------------------------------
    inj_keys  = sort(collect(keys(diag_init.injector_tracers)),  by = string)
    prod_keys = sort(collect(keys(diag_init.producer_tracers)), by = string)

    diag_labels = ["Forward TOF (years)", "Backward TOF (years)",
                   "Residence time (years)"]
    inj_labels  = ["Inj tracer: $k"  for k in inj_keys]
    prod_labels = ["Prod tracer: $k" for k in prod_keys]
    dyn_labels  = sort(collect(keys(dyn_init)))
    stat_labels = ["Static: $k" for k in sort(collect(keys(static_qty)))]
    all_quantities = vcat(diag_labels, inj_labels, prod_labels,
                          dyn_labels, stat_labels)

    # ---- Observables -------------------------------------------------------
    diag_obs = Observable{FlowDiagnosticsResult}(diag_init)
    dyn_obs  = Observable{Dict{String,Vector{Float64}}}(dyn_init)
    qty_obs  = Observable{String}(diag_labels[1])
    cmap_obs = Observable{Symbol}(:viridis)

    # ---- Figure layout ----------------------------------------------------
    fig = Figure(size = resolution)

    # Row 1: Quantity + Colormap menus
    fig[1, 1:4] = ctrl_grid = GridLayout(tellwidth = false)
    ctrl_grid[1, 1] = Label(fig, "Quantity:", font = :bold, tellwidth = false)
    menu_qty  = Menu(ctrl_grid[1, 2], options = all_quantities,
                     default = all_quantities[1])
    ctrl_grid[1, 3] = Label(fig, "Colormap:", font = :bold, tellwidth = false)
    available_cmaps = ["viridis", "turbo", "jet", "hot", "cool", "plasma",
                       "inferno", "RdBu", "seismic", "bwr", "gnuplot2"]
    menu_cmap = Menu(ctrl_grid[1, 4], options = available_cmaps,
                     default = "viridis")

    # Row 2: Step slider
    fig[2, 1:4] = step_grid = GridLayout(tellwidth = false)
    step_grid[1, 1] = Label(fig, "Step:", font = :bold, tellwidth = false)
    sl_step    = Slider(step_grid[1, 2:3], range = 1:nsteps,
                        startvalue = step_init)
    step_index = sl_step.value   # Observable{Int} – no extra binding needed
    step_grid[1, 4] = Label(fig,
        @lift(begin
            yr = result.time[$step_index] / SECONDS_PER_YEAR
            "$($step_index)/$nsteps  (t = $(round(yr; digits=3)) yr)"
        end),
        tellwidth = false
    )

    # Row 3: TOF interval slider (years) – replaces two broken single sliders
    fwd_max_yr  = finite_max(diag_init.forward_tof)  / SECONDS_PER_YEAR
    bwd_max_yr  = finite_max(diag_init.backward_tof) / SECONDS_PER_YEAR
    tof_hi_init = min(max(fwd_max_yr, bwd_max_yr, 1.0), max_tof)
    tof_range   = LinRange(0.0, tof_hi_init, 500)

    fig[3, 1:4] = tof_grid = GridLayout(tellwidth = false)
    tof_grid[1, 1] = Label(fig, "TOF range (yr):", font = :bold,
                            tellwidth = false)
    sl_tof = IntervalSlider(tof_grid[1, 2:3], range = tof_range)
    tof_grid[1, 4] = Label(fig,
        @lift(begin
            lo, hi = $(sl_tof.interval)
            "[$(round(lo; digits=2)), $(round(hi; digits=2))] yr"
        end),
        tellwidth = false
    )

    # Row 4: 3-D axis
    ax = Axis3(fig[4, 1:4],
        title     = @lift("Step $($step_index)/$nsteps  |  $($qty_obs)"),
        aspect    = (1.0, 1.0, 1/3),
        zreversed = z_is_depth
    )

    # ---- Observable for cell colours ---------------------------------------
    cell_colors = @lift begin
        diag = $diag_obs
        dyn  = $dyn_obs
        qty  = $qty_obs
        tof_interval = $(sl_tof.interval)
        tof_lo_s = tof_interval[1] * SECONDS_PER_YEAR
        tof_hi_s = tof_interval[2] * SECONDS_PER_YEAR

        raw = fd_extract_quantity(diag, dyn, static_qty, qty,
                                   inj_keys, prod_keys)

        fwd_tof = diag.forward_tof
        bwd_tof = diag.backward_tof
        out = copy(raw)
        for i in eachindex(out)
            ft = fwd_tof[i]
            bt = bwd_tof[i]
            if !isfinite(ft) || ft < tof_lo_s || ft > tof_hi_s
                out[i] = NaN
            elseif !isfinite(bt)
                out[i] = NaN
            end
        end
        out
    end

    # Map per-cell colours to per-triangle-vertex colours via the mapper
    vertex_colors = @lift(mapper.Cells($cell_colors))

    # Colour range (finite values only)
    crange = @lift begin
        vals = filter(isfinite, $cell_colors)
        if isempty(vals)
            (0.0, 1.0)
        else
            lo, hi = extrema(vals)
            if lo ≈ hi
                (lo, lo + 1.0)
            else
                (lo, hi)
            end
        end
    end

    # Lifted colormap (Symbol → RGBA vector so it can be swapped reactively)
    colormap_vec = @lift(Makie.to_colormap($cmap_obs))

    # ---- Mesh plot ---------------------------------------------------------
    scat = mesh!(ax, pts_c, tri_c;
        color        = vertex_colors,
        colorrange   = crange,
        colormap     = colormap_vec,
        backlight    = 1,
        nan_color    = :transparent,
        transparency = false
    )

    # Row 5: Colorbar
    Colorbar(fig[5, 1:4], scat, vertical = false, label = qty_obs)

    # ---- Well markers ------------------------------------------------------
    fd_plot_wells!(ax, cell_centroids, setup_init)

    # ---- Reactive updates --------------------------------------------------

    # Step slider → recompute diagnostics and dynamic quantities.
    # Uses sl_step.value (= step_index) directly; NO write-back to the slider
    # observable, so there is no circular dependency.
    on(step_index) do idx
        setup    = fd_build_setup(result, model, forces, idx)
        diag_obs[] = solve_flow_diagnostics(setup; compute_tracers = true)
        dyn_obs[]  = fd_dynamic_quantities(
            fd_reservoir_state(result.states[idx]), nc)
    end

    on(menu_qty.selection) do selection
        if !isnothing(selection)
            qty_obs[] = selection
        end
    end
    on(menu_cmap.selection) do selection
        if !isnothing(selection)
            cmap_obs[] = Symbol(selection)
        end
    end

    # ---- Display -----------------------------------------------------------
    if new_window
        display(GLMakie.Screen(), fig)
    else
        display(fig)
    end
    return fig
end

# -------------------------------------------------------------------------
# Helper: build FlowDiagnosticsSetup for a given step index
# -------------------------------------------------------------------------

function fd_build_setup(result, model, forces, step_index)
    f = forces
    if f isa AbstractVector
        f = f[min(step_index, lastindex(f))]
    end
    return setup_flow_diagnostics(result, model, f; step_index = step_index)
end

# -------------------------------------------------------------------------
# Helper: extract the reservoir sub-state from a (possibly multi-model) state
# -------------------------------------------------------------------------

function fd_reservoir_state(state)
    if isa(state, AbstractDict) && haskey(state, :Reservoir)
        return state[:Reservoir]
    end
    return state
end

# -------------------------------------------------------------------------
# Helper: static cell-level quantities from the domain
# -------------------------------------------------------------------------

function fd_static_quantities(domain, nc)
    d = Dict{String, Vector{Float64}}()
    for key in (:Permeability, :Porosity, :FluidVolume, :volumes)
        try
            val = domain[key]
            if val isa AbstractVector && length(val) == nc
                d[string(key)] = Float64.(val)
            elseif val isa AbstractMatrix && size(val, 2) == nc
                for i in 1:size(val, 1)
                    d["$(key)[$i]"] = Float64.(val[i, :])
                end
            end
        catch
        end
    end
    return d
end

# -------------------------------------------------------------------------
# Helper: dynamic cell-level quantities from a reservoir state snapshot
# -------------------------------------------------------------------------

function fd_dynamic_quantities(res_state, nc)
    d = Dict{String, Vector{Float64}}()
    for k in keys(res_state)
        try
            val = res_state[k]
            if val isa AbstractVector && length(val) == nc &&
                    eltype(val) <: Real
                d[string(k)] = Float64.(val)
            elseif val isa AbstractMatrix && size(val, 2) == nc &&
                    eltype(val) <: Real
                for i in 1:size(val, 1)
                    d["$(k)[$i]"] = Float64.(val[i, :])
                end
            end
        catch
        end
    end
    return d
end

# -------------------------------------------------------------------------
# Helper: extract the requested quantity as a Float64 vector (length = nc)
#
# TOF/residence-time values are returned in *years*. Non-finite (unreachable)
# cells are mapped to NaN so that they render as transparent.
# -------------------------------------------------------------------------

function fd_extract_quantity(
        diag::FlowDiagnosticsResult,
        dyn_quantities::Dict{String,Vector{Float64}},
        static_quantities::Dict{String,Vector{Float64}},
        qty::String,
        inj_keys::Vector{Symbol},
        prod_keys::Vector{Symbol}
    )
    nc = length(diag.forward_tof)

    if qty == "Forward TOF (years)"
        return map(x -> ifelse(isfinite(x), x / SECONDS_PER_YEAR, NaN),
                   diag.forward_tof)
    elseif qty == "Backward TOF (years)"
        return map(x -> ifelse(isfinite(x), x / SECONDS_PER_YEAR, NaN),
                   diag.backward_tof)
    elseif qty == "Residence time (years)"
        return map(x -> ifelse(isfinite(x), x / SECONDS_PER_YEAR, NaN),
                   diag.residence_time)
    elseif startswith(qty, "Inj tracer: ")
        k = Symbol(qty[length("Inj tracer: ")+1:end])
        return Float64.(get(diag.injector_tracers, k, zeros(nc)))
    elseif startswith(qty, "Prod tracer: ")
        k = Symbol(qty[length("Prod tracer: ")+1:end])
        return Float64.(get(diag.producer_tracers, k, zeros(nc)))
    elseif haskey(dyn_quantities, qty)
        return Float64.(dyn_quantities[qty])
    elseif startswith(qty, "Static: ")
        k = qty[length("Static: ")+1:end]
        return Float64.(get(static_quantities, k, zeros(nc)))
    end
    return zeros(nc)
end

# -------------------------------------------------------------------------
# Helper: finite maximum (returns 1.0 if all values are non-finite)
# -------------------------------------------------------------------------

function finite_max(v::AbstractVector)
    m = -Inf
    for x in v
        isfinite(x) && (m = max(m, x))
    end
    return ifelse(isfinite(m), m, 1.0)
end

# -------------------------------------------------------------------------
# Helper: plot well locations as scatter markers
# -------------------------------------------------------------------------

function fd_plot_wells!(ax, cell_centroids, setup::FlowDiagnosticsSetup)
    D = size(cell_centroids, 1)
    function point_for_cell(c)
        x = Float32(cell_centroids[1, c])
        y = 0f0
        z = 0f0
        if D >= 2
            y = Float32(cell_centroids[2, c])
        end
        if D >= 3
            z = Float32(cell_centroids[3, c])
        end
        return (x, y, z)
    end
    for (_, cells) in setup.injector_cells
        isempty(cells) && continue
        points = map(point_for_cell, cells)
        xs = first.(points)
        ys = getindex.(points, 2)
        zs = last.(points)
        scatter!(ax, xs, ys, zs; color = :red,  markersize = 20, overdraw = true)
    end
    for (_, cells) in setup.producer_cells
        isempty(cells) && continue
        points = map(point_for_cell, cells)
        xs = first.(points)
        ys = getindex.(points, 2)
        zs = last.(points)
        scatter!(ax, xs, ys, zs; color = :blue, markersize = 20, overdraw = true)
    end
end
