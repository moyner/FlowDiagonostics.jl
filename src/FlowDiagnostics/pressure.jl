"""
    flow_diagnostics_all_states(case::JutulCase, result::ReservoirSimResult; kwarg...)

Compute diagnostics for every reported state. Forces are selected at each step,
and the case parameters are reused when evaluating face fluxes. Returns a vector
of `FlowDiagnosticsResult` in the same order as `result.states`.
"""
function flow_diagnostics_all_states(case::JutulCase, result::ReservoirSimResult;
        compute_tracers::Bool = true, compute_well_tof::Bool = false,
        perforation_tracers = nothing,
        max_tof::Real = DEFAULT_MAX_TOF, solver::Symbol = :direct)
    diagnostics = Vector{FlowDiagnosticsResult}(undef, length(result.states))
    for step in eachindex(result.states)
        setup = setup_flow_diagnostics(result, case; step_index = step)
        diagnostics[step] = solve_flow_diagnostics(setup;
            compute_tracers, compute_well_tof, perforation_tracers, max_tof, solver)
    end
    return diagnostics
end

function merged_pressure_initialization(model, state, parameters)
    if model isa MultiModel
        merged = Dict{Symbol, Any}()
        for name in keys(model.models)
            merged[name] = merged_pressure_initialization(model[name], state[name], parameters[name])
        end
        return merged
    end
    merged = Dict{Symbol, Any}()
    for (name, value) in pairs(parameters)
        merged[name] = value
    end
    for (name, value) in pairs(state)
        merged[name] = value
    end
    return merged
end

function immiscible_pressure_model(model)
    original = reservoir_model(model)
    if original.system isa ImmiscibleSystem
        return model
    end
    phases = get_phases(original.system)
    system = ImmiscibleSystem(phases;
        reference_densities = JutulDarcy.reference_densities(original.system))
    wells = Any[]
    if model isa MultiModel
        for submodel in values(model.models)
            if model_or_domain_is_well(submodel)
                push!(wells, submodel.data_domain)
            end
        end
    end
    surrogate = setup_reservoir_model(reservoir_domain(original), system;
        wells = wells, block_backend = false)
    target = reservoir_model(surrogate)
    for name in (:RelativePermeabilities, :CapillaryPressure,
            :PhaseMassDensities, :PhaseViscosities)
        if haskey(original.secondary_variables, name)
            target.secondary_variables[name] = original.secondary_variables[name]
        end
    end
    return surrogate
end

function ensure_pressure_saturations!(initial, case)
    reservoir_initial = initial
    reservoir_state = case.state0
    reservoir_parameters = case.parameters
    if case.model isa MultiModel
        reservoir_initial = initial[:Reservoir]
        reservoir_state = case.state0[:Reservoir]
        reservoir_parameters = case.parameters[:Reservoir]
    end
    if !haskey(reservoir_initial, :Saturations)
        original = reservoir_model(case.model)
        evaluated = Jutul.evaluate_all_secondary_variables(original,
            reservoir_state, reservoir_parameters)
        reservoir_initial[:Saturations] = evaluated[:Saturations]
    end
    return initial
end

"""
    solve_pressure_flow_diagnostics(case::JutulCase; dt=case.dt[step_index],
        step_index=1, compute_tracers=true, solver=:direct, kwarg...)

Solve one pressure step with JutulDarcy's sequential pressure formulation,
without requiring simulation results. The pressure model keeps the number of
phases and the reservoir's relative permeability, capillary, density and
viscosity definitions. Returns `(setup, diagnostics, pressure_state)`; the
velocity field is `setup.q` in m³/s, oriented by `setup.N`.
"""
function solve_pressure_flow_diagnostics(case::JutulCase;
        step_index::Int = 1, dt::Real = case.dt[step_index],
        compute_tracers::Bool = true, compute_well_tof::Bool = false,
        perforation_tracers = nothing,
        max_tof::Real = DEFAULT_MAX_TOF, solver::Symbol = :direct,
        info_level::Int = -1)
    model = immiscible_pressure_model(case.model)
    pressure_model = JutulDarcy.Sequential.convert_to_sequential(model;
        pressure = true)
    initial = merged_pressure_initialization(model, case.state0, case.parameters)
    if model !== case.model
        ensure_pressure_saturations!(initial, case)
    end
    pressure_state0, pressure_parameters = Jutul.setup_state_and_parameters(
        pressure_model, initial)
    forces = forces_at_step(case.forces, step_index)
    simulation = Jutul.simulate(pressure_state0, pressure_model, [Float64(dt)];
        parameters = pressure_parameters, forces = forces, info_level = info_level)
    states = simulation.states
    isempty(states) && error("Pressure solve did not return a state")
    pressure_state = states[end]
    reservoir = reservoir_model(pressure_model)
    reservoir_state = pressure_state
    reservoir_parameters = pressure_parameters
    if pressure_model isa MultiModel
        reservoir_state = pressure_state[:Reservoir]
        reservoir_parameters = pressure_parameters[:Reservoir]
    end
    neighbors = reservoir_domain(reservoir)[:neighbors]
    flux = compute_total_flux(reservoir, reservoir_state,
        reservoir_parameters, neighbors)
    wells = collect_well_cells(pressure_model)
    directions = collect_well_directions(wells, forces)
    sources = collect_source_rates(forces)
    setup = FlowDiagnosticsSetup(pressure_model, neighbors, flux,
        pore_volume(reservoir_domain(reservoir)), wells, directions, sources)
    diagnostics = solve_flow_diagnostics(setup;
        compute_tracers, compute_well_tof, perforation_tracers, max_tof, solver)
    return (setup = setup, diagnostics = diagnostics, pressure_state = pressure_state)
end
