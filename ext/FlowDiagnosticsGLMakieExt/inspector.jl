# Flow-diagnostics interactive 3-D inspector built on GLMakie.
#
# Entry points
#   flow_diagnostics_inspector(result, model, forces; kwarg...)
#   flow_diagnostics_inspector(result, case::JutulCase; kwarg...)

# -------------------------------------------------------------------------
# Public entry points
# -------------------------------------------------------------------------

"""
    flow_diagnostics_inspector(result, model, forces; kwarg...)
    flow_diagnostics_inspector(result, case::JutulCase; kwarg...)

Open an interactive GLMakie window for inspecting flow diagnostics.

# User interface
- **Step slider** – selects the simulation step. Flow diagnostics are
  recomputed whenever the step changes.
- **Quantity menu** – choose what to display:
  - `Forward TOF` – time of flight from injectors (seconds).
  - `Backward TOF` – time of flight to producers (seconds).
  - `Residence time` – sum of forward and backward TOF.
  - Injector/producer tracer concentrations (one entry per well / source).
- **Forward TOF threshold** and **Backward TOF threshold** sliders – cells
  with a TOF value *above* the chosen threshold are hidden from the plot.
  Moving a slider to its maximum value disables that filter.
- The colorbar is updated automatically when the displayed quantity or step
  changes.

# Arguments
- `result::ReservoirSimResult` – output of `simulate_reservoir`.
- `model` – the `MultiModel` or `SimulationModel` used for the simulation.
- `forces` – driving forces (wells, source terms) used during the simulation.
- `case::JutulCase` – alternative to passing `model` and `forces` separately.

# Keyword arguments
- `resolution` – window size in pixels, default `(1400, 900)`.
- `colormap` – Makie colormap symbol, default `:viridis`.
- `z_is_depth::Bool` – if `true` the z-axis is inverted (depth convention).
  Inferred automatically from the mesh when not provided.
- `new_window::Bool` – if `true` (default outside CI) the figure is displayed
  in an independent window.
"""
function FlowDiagnostics.flow_diagnostics_inspector(
        result::JutulDarcy.ReservoirSimResult,
        model,
        forces;
        resolution::Tuple{Int,Int} = (1400, 900),
        colormap = :viridis,
        z_is_depth::Union{Missing,Bool} = missing,
        new_window::Bool = get(ENV, "CI", "false") == "false"
    )
    _launch_inspector(result, model, forces;
        resolution = resolution,
        colormap = colormap,
        z_is_depth = z_is_depth,
        new_window = new_window
    )
end

# Convenience overload accepting a JutulCase
function FlowDiagnostics.flow_diagnostics_inspector(
        result::JutulDarcy.ReservoirSimResult,
        case::JutulCase;
        kwarg...
    )
    # Pass case.forces directly so that _compute_diag can pick the right per-step
    # forces when forces is a vector.
    return FlowDiagnostics.flow_diagnostics_inspector(result, case.model, case.forces; kwarg...)
end

# -------------------------------------------------------------------------
# Internal implementation
# -------------------------------------------------------------------------

function _launch_inspector(result, model, forces;
        resolution, colormap, z_is_depth, new_window)

    # ---- Mesh and geometry ------------------------------------------------
    rmodel  = JutulDarcy.reservoir_model(model)
    domain  = JutulDarcy.reservoir_domain(rmodel)
    mesh    = Jutul.physical_representation(domain)
    nc      = Jutul.number_of_cells(domain)
    nsteps  = length(result.states)

    if ismissing(z_is_depth)
        z_is_depth = Jutul.mesh_z_is_depth(mesh)
    end

    # Triangulate once – this is the expensive geometry step
    pts_raw, tri_raw, mapper = triangulate_mesh(mesh)
    pts_c   = Makie.to_vertices(pts_raw)
    tri_c   = Makie.to_triangles(tri_raw)

    # ---- Initial diagnostics (last step) ----------------------------------
    step_init = nsteps
    diag_init = _compute_diag(result, model, forces, step_init)

    # ---- Build quantity list ----------------------------------------------
    # Fixed quantities always present
    base_quantities = ["Forward TOF", "Backward TOF", "Residence time"]
    # Per-injector tracers
    inj_keys  = sort(collect(keys(diag_init.injector_tracers)),  by = string)
    prod_keys = sort(collect(keys(diag_init.producer_tracers)), by = string)
    inj_labels  = ["Inj tracer: $k"  for k in inj_keys]
    prod_labels = ["Prod tracer: $k" for k in prod_keys]
    all_quantities = vcat(base_quantities, inj_labels, prod_labels)

    # ---- Observables -------------------------------------------------------
    step_obs    = Observable{Int}(step_init)
    qty_obs     = Observable{String}(base_quantities[1])
    diag_obs    = Observable{FlowDiagnosticsResult}(diag_init)
    fwd_thresh  = Observable{Float64}(1.0)   # fraction 0–1 of max finite fwd TOF
    bwd_thresh  = Observable{Float64}(1.0)   # fraction 0–1 of max finite bwd TOF

    # ---- Figure layout ----------------------------------------------------
    fig = Figure(size = resolution)

    # Top controls row
    fig[1, 1:3] = ctrl_grid = GridLayout(tellwidth = false)

    # Step slider (row 2)
    fig[2, 1:3] = step_grid = GridLayout(tellwidth = false)
    step_grid[1, 1] = Label(fig, "Step:", font = :bold, tellwidth = false)
    sl_step = Slider(step_grid[1, 2], range = 1:nsteps, value = step_obs, snap = true)
    step_grid[1, 3] = Label(fig, @lift("$($step_obs) / $nsteps"), tellwidth = false)

    # Forward TOF threshold slider (row 3)
    fig[3, 1:3] = fwd_grid = GridLayout(tellwidth = false)
    fwd_grid[1, 1] = Label(fig, "Fwd TOF threshold:", font = :bold, tellwidth = false)
    sl_fwd = Slider(fwd_grid[1, 2], range = LinRange(0.0, 1.0, 500), value = fwd_thresh, snap = false)
    fwd_grid[1, 3] = Label(fig, @lift(string(round($fwd_thresh * 100; digits=1)) * "%"), tellwidth = false)

    # Backward TOF threshold slider (row 4)
    fig[4, 1:3] = bwd_grid = GridLayout(tellwidth = false)
    bwd_grid[1, 1] = Label(fig, "Bwd TOF threshold:", font = :bold, tellwidth = false)
    sl_bwd = Slider(bwd_grid[1, 2], range = LinRange(0.0, 1.0, 500), value = bwd_thresh, snap = false)
    bwd_grid[1, 3] = Label(fig, @lift(string(round($bwd_thresh * 100; digits=1)) * "%"), tellwidth = false)

    # Quantity selector (row 1 of ctrl_grid)
    ctrl_grid[1, 1] = Label(fig, "Quantity:", font = :bold, tellwidth = false)
    menu_qty = Menu(ctrl_grid[1, 2], options = all_quantities, default = all_quantities[1])

    # 3-D axis (row 5)
    ax = Axis3(fig[5, 1:3],
        title  = @lift("Step $($step_obs)/$nsteps – $($qty_obs)"),
        aspect = (1.0, 1.0, 1/3),
        zreversed = z_is_depth
    )

    # Colorbar (row 6)
    # Placeholder – will be rebuilt when the plot updates

    # ---- Observable for cell colours ---------------------------------------
    cell_colors = @lift begin
        diag  = $diag_obs
        qty   = $qty_obs
        raw   = _extract_quantity(diag, qty, inj_keys, prod_keys)

        # Apply TOF thresholds: cells above either threshold → NaN (hidden)
        fwd_tof = diag.forward_tof
        bwd_tof = diag.backward_tof

        fwd_max = _finite_max(fwd_tof)
        bwd_max = _finite_max(bwd_tof)

        fwd_cut = $fwd_thresh * fwd_max
        bwd_cut = $bwd_thresh * bwd_max

        out = copy(raw)
        for i in eachindex(out)
            if !isfinite(fwd_tof[i]) || fwd_tof[i] > fwd_cut
                out[i] = NaN
            elseif !isfinite(bwd_tof[i]) || bwd_tof[i] > bwd_cut
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
            lo ≈ hi ? (lo, lo + 1e-12) : (lo, hi)
        end
    end

    # ---- Mesh plot ---------------------------------------------------------
    scat = mesh!(ax, pts_c, tri_c;
        color      = vertex_colors,
        colorrange = crange,
        colormap   = colormap,
        backlight  = 1,
        nan_color  = :transparent,
        transparency = false
    )

    Colorbar(fig[6, 1:3], scat, vertical = false, label = qty_obs)

    # ---- Reactive updates --------------------------------------------------

    # Step slider → recompute diagnostics
    on(sl_step.selected_index) do idx
        step_obs[] = idx
        diag_obs[] = _compute_diag(result, model, forces, idx)
    end

    # Forward TOF threshold slider
    on(sl_fwd.value) do v
        fwd_thresh[] = v
    end

    # Backward TOF threshold slider
    on(sl_bwd.value) do v
        bwd_thresh[] = v
    end

    # Quantity menu
    on(menu_qty.selection) do s
        if !isnothing(s)
            qty_obs[] = s
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
# Helper: compute diagnostics for one step
# -------------------------------------------------------------------------

function _compute_diag(result, model, forces, step_index)
    # If forces are per-step, pick the matching entry
    f = forces
    if f isa AbstractVector
        f = f[min(step_index, lastindex(f))]
    end
    setup = setup_flow_diagnostics(result, model, f; step_index = step_index)
    return solve_flow_diagnostics(setup; compute_tracers = true)
end

# -------------------------------------------------------------------------
# Helper: extract the requested quantity as a Float64 vector (length = nc)
# -------------------------------------------------------------------------

function _extract_quantity(
        diag::FlowDiagnosticsResult,
        qty::String,
        inj_keys::Vector{Symbol},
        prod_keys::Vector{Symbol}
    )
    nc = length(diag.forward_tof)
    if qty == "Forward TOF"
        return float.(diag.forward_tof)
    elseif qty == "Backward TOF"
        return float.(diag.backward_tof)
    elseif qty == "Residence time"
        return float.(diag.residence_time)
    elseif startswith(qty, "Inj tracer: ")
        k = Symbol(qty[length("Inj tracer: ")+1:end])
        return float.(get(diag.injector_tracers, k, zeros(nc)))
    elseif startswith(qty, "Prod tracer: ")
        k = Symbol(qty[length("Prod tracer: ")+1:end])
        return float.(get(diag.producer_tracers, k, zeros(nc)))
    end
    return zeros(nc)
end

# -------------------------------------------------------------------------
# Helper: finite maximum (returns 1.0 if all Inf)
# -------------------------------------------------------------------------

function _finite_max(v::AbstractVector)
    m = -Inf
    for x in v
        isfinite(x) && (m = max(m, x))
    end
    return isfinite(m) ? m : 1.0
end
