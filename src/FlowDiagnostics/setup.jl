"""
    setup_flow_diagnostics(result, model, forces; step_index = lastindex(result.states))
    setup_flow_diagnostics(result, case::JutulCase; step_index = lastindex(result.states))

Set up the data structures needed for flow diagnostics from a reservoir
simulation result.

Picks the reservoir state, the corresponding time-step size (dt), and the
provided forces at `step_index`, then uses JutulDarcy's two-point flux
approximation (TPFA) routines to compute the total volumetric Darcy flux at
every internal face.  Injection and production cells are identified from the
forces (wells or `SourceTerm`s) and their reservoir volumetric rates are
computed.

The returned `FlowDiagnosticsSetup` can be passed to
[`solve_flow_diagnostics`](@ref) one or more times without repeating this
potentially expensive setup.

# Arguments
- `result::ReservoirSimResult`: Output from `simulate_reservoir`.
- `model`: The `MultiModel` or `SimulationModel` used to produce `result`.
- `forces`: The driving forces used at the selected time step. For a
  `JutulCase` overload these are read from `case.forces`.
- `case::JutulCase`: Alternative to passing `model` and `forces` separately.

# Keyword arguments
- `step_index`: Index into `result.states` / `result.time` to use. Defaults
  to the last reported step.
"""
function setup_flow_diagnostics(
        result::ReservoirSimResult,
        model,
        forces;
        step_index::Int = lastindex(result.states)
    )
    rmodel = reservoir_model(model)
    domain = reservoir_domain(rmodel)
    nc = number_of_cells(domain)
    N = domain[:neighbors]  # 2 × nf

    # --- Pick the state at the requested step ---
    state = result.states[step_index]

    # dt at this step (used to characterise the flow snapshot)
    t = result.time
    if step_index == 1
        dt = t[1]
    else
        dt = t[step_index] - t[step_index - 1]
    end

    # --- Compute parameters (transmissibilities, gravity) ---
    parameters = setup_parameters(rmodel)

    # --- Compute total volumetric face fluxes ---
    q = _compute_total_flux(rmodel, state, parameters, N)

    # --- Pore volumes ---
    pv = pore_volume(domain)

    # --- Identify injectors and producers ---
    injector_cells  = Dict{Symbol, Vector{Int}}()
    injector_rates  = Dict{Symbol, Float64}()
    producer_cells  = Dict{Symbol, Vector{Int}}()
    producer_rates  = Dict{Symbol, Float64}()

    _collect_sources!(
        injector_cells, injector_rates,
        producer_cells, producer_rates,
        model, forces, q, N, nc
    )

    return FlowDiagnosticsSetup(
        model, N, q, pv,
        injector_cells, injector_rates,
        producer_cells, producer_rates
    )
end

# Convenience overload accepting a JutulCase directly
function setup_flow_diagnostics(
        result::ReservoirSimResult,
        case::JutulCase;
        step_index::Int = lastindex(result.states)
    )
    forces = case.forces
    if forces isa AbstractVector
        # Forces are given per timestep; pick the closest one
        forces = forces[min(step_index, lastindex(forces))]
    end
    return setup_flow_diagnostics(result, case.model, forces; step_index = step_index)
end

# -------------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------------

"""
    _compute_total_flux(rmodel, state, parameters, N)

Compute the total volumetric Darcy flux (m³/s) at every internal face using
the TPFA formula with upwinding.  Returns a vector of length n_faces where a
positive value means net flow from the left cell (`N[1,f]`) to the right cell
(`N[2,f]`).

Uses `Jutul.evaluate_all_secondary_variables` to compute the phase mobilities,
densities, and other quantities from the primary state variables, and then
calls `JutulDarcy.darcy_phase_volume_fluxes` for per-phase flux computation —
the same routine used during the forward simulation.
"""
function _compute_total_flux(rmodel, state, parameters, N)
    sys = rmodel.system
    phases = eachphase(sys)
    nf = size(N, 2)

    # Evaluate all secondary variables (PhaseMobilities, PhaseMassDensities,
    # etc.) from the primary state using the same JutulDarcy routines that are
    # called during the forward simulation.  Parameters (Transmissibilities,
    # TwoPointGravityDifference) are also merged into the resulting state so
    # that the TPFA flux routines can find everything they need.
    full_state_data = Jutul.evaluate_all_secondary_variables(rmodel, state, parameters)

    # Wrap in JutulStorage so that dot-access (state.Transmissibilities etc.)
    # works, as expected by the JutulDarcy TPFA internal functions.
    full_state = Jutul.JutulStorage(full_state_data)

    q = zeros(nf)
    for f in 1:nf
        l = N[1, f]
        r = N[2, f]
        kgrad = Jutul.TPFA(l, r, 1)
        upw   = Jutul.SPU(l, r)
        # darcy_phase_volume_fluxes sums pressure + gravity terms for each
        # phase and applies an upwind mobility; we sum all phases for the
        # total volumetric flux.
        vol_fluxes = darcy_phase_volume_fluxes(f, full_state, rmodel, nothing, kgrad, upw, phases)
        for vf in vol_fluxes
            q[f] += vf
        end
    end
    return q
end

# -------------------------------------------------------------------------
# Well / source identification
# -------------------------------------------------------------------------

"""
    _collect_sources!(inj_cells, inj_rates, prod_cells, prod_rates,
                      model, forces, q, N, nc)

Populate injector / producer dictionaries from the model topology and forces.

Supports two source types:
1. **Explicit `SourceTerm`s** in `forces[:Reservoir][:sources]` (or
   `forces[:sources]` for a single-model setup).
2. **Well perforations** for `MultiModel` setups.  The well type
   (injector / producer) is determined from the `InjectorControl` /
   `ProducerControl` stored in the forces.  The volumetric perforation rate
   is estimated from the flux divergence at each perforated cell.
"""
function _collect_sources!(
        inj_cells, inj_rates,
        prod_cells, prod_rates,
        model, forces, q, N, nc
    )
    # --- SourceTerm sources ---
    res_forces = _get_reservoir_forces(forces)
    if !isnothing(res_forces)
        srcs = res_forces isa NamedTuple ? get(res_forces, :sources, nothing) : nothing
        if !isnothing(srcs)
            for (idx, src) in enumerate(srcs)
                c    = src.cell
                rate = src.value   # positive = injection
                name = Symbol("Source_$idx")
                if rate > 0
                    inj_cells[name]  = [c]
                    inj_rates[name]  = Float64(rate)
                elseif rate < 0
                    prod_cells[name] = [c]
                    prod_rates[name] = Float64(-rate)
                end
            end
        end
    end

    # --- Well perforations ---
    if model isa MultiModel
        # Pre-compute the volumetric divergence at each cell (positive = net outflow).
        div_q = _flux_divergence(q, N, nc)

        for (wname, wmodel) in pairs(model.models)
            model_or_domain_is_well(wmodel) || continue
            g = physical_representation(wmodel.data_domain)
            reservoir_cells = vec(g.perforations.reservoir)
            isempty(reservoir_cells) && continue

            # Determine whether this well is injecting or producing
            ctrl = _get_well_control(forces, wname)
            isnothing(ctrl) && continue

            # Volumetric rate: sum of |divergence| at the perforated cells.
            # The sign is determined by the control type.
            perf_div = sum(div_q[c] for c in reservoir_cells)

            if ctrl isa InjectorControl
                inj_cells[wname]  = reservoir_cells
                inj_rates[wname]  = max(0.0, -perf_div)  # injection = net inflow
            elseif ctrl isa ProducerControl
                prod_cells[wname] = reservoir_cells
                prod_rates[wname] = max(0.0, perf_div)   # production = net outflow
            end
        end
    end
end

"""
    _get_reservoir_forces(forces)

Extract the reservoir sub-forces from the (potentially multi-model) forces
tuple, returning `nothing` if no reservoir forces are found.
"""
function _get_reservoir_forces(forces)
    isnothing(forces) && return nothing
    if forces isa NamedTuple || forces isa AbstractDict
        if haskey(forces, :Reservoir)
            return forces[:Reservoir]
        elseif haskey(forces, :sources) || haskey(forces, :bc)
            # Single-model forces passed directly
            return forces
        end
    end
    return nothing
end

"""
    _get_well_control(forces, well_name)

Return the well control object for `well_name` from the forces, or `nothing`
if the well is not found or is disabled.
"""
function _get_well_control(forces, well_name::Symbol)
    isnothing(forces) && return nothing
    # Try unified Facility
    if forces isa Vector
        forces = forces[end]
    end
    if haskey(forces, :Facility)
        fac = forces[:Facility]
        if fac isa NamedTuple && haskey(fac, :control)
            ctrl_dict = fac[:control]
            haskey(ctrl_dict, well_name) || return nothing
            ctrl = ctrl_dict[well_name]
            ctrl isa DisabledControl && return nothing
            return ctrl
        end
    end
    # Try individual well controller (split_wells = true)
    ctrl_key = Symbol("$(well_name)_ctrl")
    if haskey(forces, ctrl_key)
        fc = forces[ctrl_key]
        if fc isa NamedTuple && haskey(fc, :control)
            ctrl_dict = fc[:control]
            haskey(ctrl_dict, well_name) || return nothing
            ctrl = ctrl_dict[well_name]
            ctrl isa DisabledControl && return nothing
            return ctrl
        end
    end
    return nothing
end

"""
    _flux_divergence(q, N, nc)

Compute the net volumetric outflow (m³/s) at each cell from the face fluxes.
A positive value means the cell has more outflow than inflow (producer / sink).
"""
function _flux_divergence(q, N, nc)
    div_q = zeros(nc)
    for f in axes(N, 2)
        l = N[1, f]
        r = N[2, f]
        div_q[l] += q[f]   # positive q leaves cell l
        div_q[r] -= q[f]   # positive q enters cell r
    end
    return div_q
end
