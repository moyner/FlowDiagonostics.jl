"""
    setup_flow_diagnostics(result, model, forces; step_index=lastindex(result.states), parameters=setup_parameters(reservoir_model(model)))
    setup_flow_diagnostics(result, case::JutulCase; step_index=lastindex(result.states))

Extract a Darcy velocity snapshot and well topology. All well perforations are
recorded, even when their controls are disabled. Active tracer direction comes
from `InjectorControl` or `ProducerControl`, never flux divergence.
"""
function setup_flow_diagnostics(result::ReservoirSimResult, model, forces;
        step_index::Int = lastindex(result.states),
        parameters = setup_parameters(reservoir_model(model)))
    reservoir = reservoir_model(model)
    domain = reservoir_domain(reservoir)
    neighbors = domain[:neighbors]
    state = result.states[step_index]
    selected_forces = forces_at_step(forces, step_index)
    flux = compute_total_flux(reservoir, state, parameters, neighbors)
    wells = collect_well_cells(model)
    directions = collect_well_directions(wells, selected_forces)
    sources = collect_source_rates(selected_forces)
    return FlowDiagnosticsSetup(model, neighbors, flux, pore_volume(domain),
        wells, directions, sources)
end

function setup_flow_diagnostics(result::ReservoirSimResult, case::JutulCase;
        step_index::Int = lastindex(result.states))
    parameters = case.parameters
    if case.model isa MultiModel
        parameters = parameters[:Reservoir]
    end
    return setup_flow_diagnostics(result, case.model, case.forces;
        step_index = step_index, parameters = parameters)
end

forces_at_step(forces::AbstractVector, step_index) = forces[min(step_index, lastindex(forces))]
forces_at_step(forces, step_index) = forces

function compute_total_flux(model, state, parameters, neighbors)
    phases = eachphase(model.system)
    full_data = Jutul.evaluate_all_secondary_variables(model, state, parameters)
    full_state = Jutul.JutulStorage(full_data)
    flux = zeros(size(neighbors, 2))
    for face in eachindex(flux)
        left = neighbors[1, face]
        right = neighbors[2, face]
        gradient = Jutul.TPFA(left, right, 1)
        upstream = Jutul.SPU(left, right)
        phase_flux = darcy_phase_volume_fluxes(face, full_state, model,
            nothing, gradient, upstream, phases)
        total = 0.0
        for value in phase_flux
            total += value
        end
        flux[face] = total
    end
    return flux
end

function collect_well_cells(model)
    wells = Dict{Symbol, Vector{Int}}()
    if model isa MultiModel
        for (name, well_model) in pairs(model.models)
            model_or_domain_is_well(well_model) || continue
            geometry = physical_representation(well_model.data_domain)
            wells[name] = collect(Int, vec(geometry.perforations.reservoir))
        end
    end
    return wells
end

function collect_well_directions(wells, forces)
    directions = Dict{Symbol, Symbol}()
    for name in keys(wells)
        control = get_well_control(forces, name)
        if control isa InjectorControl
            directions[name] = :injector
        elseif control isa ProducerControl
            directions[name] = :producer
        end
    end
    return directions
end

function collect_source_rates(forces)
    sources = Dict{Symbol, Pair{Int, Float64}}()
    reservoir_forces = get_reservoir_forces(forces)
    if !isnothing(reservoir_forces)
        terms = get(reservoir_forces, :sources, nothing)
        if !isnothing(terms)
            for (index, source) in enumerate(terms)
                rate = Float64(source.value)
                iszero(rate) && continue
                sources[Symbol("Source_", index)] = Int(source.cell) => rate
            end
        end
    end
    return sources
end

function get_reservoir_forces(forces)
    if forces isa NamedTuple || forces isa AbstractDict
        if haskey(forces, :Reservoir)
            return forces[:Reservoir]
        elseif haskey(forces, :sources) || haskey(forces, :bc)
            return forces
        end
    end
    return nothing
end

function get_well_control(forces, name::Symbol)
    if !(forces isa NamedTuple || forces isa AbstractDict)
        return nothing
    end
    if haskey(forces, :Facility)
        facility = forces[:Facility]
        if haskey(facility, :control)
            return get(facility[:control], name, nothing)
        end
    end
    controller_name = Symbol(name, :_ctrl)
    if haskey(forces, controller_name)
        controller = forces[controller_name]
        if haskey(controller, :control)
            return get(controller[:control], name, nothing)
        end
    end
    return nothing
end
