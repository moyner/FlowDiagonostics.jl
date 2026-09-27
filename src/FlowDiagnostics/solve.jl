const DEFAULT_MAX_TOF = 100 * 365.25 * 86400.0

struct PreparedDirection{F}
    matrix::SparseMatrixCSC{Float64, Int}
    boundary::BitVector
    solvable::Vector{Int}
    local_index::Vector{Int}
    factorization::F
end

"""
    prepare_flow_diagnostics(setup; forces=nothing, solver=:direct)

Assemble the upwind systems once and factorize each direction once. Pass the
returned context to `solve_flow_diagnostics` for repeated solves. `forces` can
override the well controls captured at setup time. Use `solver=:reordered` for
the optional graph ordered solver with strongly connected components.
"""
function prepare_flow_diagnostics(setup::FlowDiagnosticsSetup;
        forces = nothing, solver::Symbol = :direct)
    injectors, producers = active_cells(setup, forces)
    forward_matrix = upwind_matrix(setup, false, producers, setup.producer_rates)
    backward_matrix = upwind_matrix(setup, true, injectors, setup.injector_rates)
    forward = prepare_direction(forward_matrix, injectors, solver)
    backward = prepare_direction(backward_matrix, producers, solver)
    return PreparedFlowDiagnostics(setup, forward, backward, injectors, producers)
end

function active_cells(setup, forces)
    injectors = Dict{Symbol, Vector{Int}}()
    producers = Dict{Symbol, Vector{Int}}()
    for (name, cells) in setup.well_cells
        direction = get(setup.well_directions, name, :disabled)
        if !isnothing(forces)
            control = get_well_control(forces, name)
            if control isa InjectorControl
                direction = :injector
            elseif control isa ProducerControl
                direction = :producer
            else
                direction = :disabled
            end
        end
        if direction == :injector
            injectors[name] = cells
        elseif direction == :producer
            producers[name] = cells
        end
    end
    for (name, source) in setup.source_rates
        cell, rate = source
        if rate > 0
            injectors[name] = [cell]
        else
            producers[name] = [cell]
        end
    end
    # The legacy positional constructor has no source topology.
    if isempty(setup.well_cells) && isempty(setup.source_rates)
        merge!(injectors, setup.injector_cells)
        merge!(producers, setup.producer_cells)
    end
    return injectors, producers
end

function upwind_matrix(setup, reverse::Bool, sinks, legacy_rates)
    neighbors = setup.N
    nc = length(setup.pore_volume)
    nf = length(setup.q)
    sink_entries = sum(length, values(sinks); init = 0)
    rows = Vector{Int}(undef, 2nf + sink_entries)
    cols = similar(rows)
    coefficients = Vector{Float64}(undef, length(rows))
    divergence = zeros(nc)
    entries = 0
    for face in 1:nf
        left = neighbors[1, face]
        right = neighbors[2, face]
        flux = setup.q[face]
        if reverse
            flux = -flux
        end
        if flux == 0
            continue
        end
        upstream = left
        downstream = right
        if flux < 0
            upstream = right
            downstream = left
            flux = -flux
        end
        entries += 1
        rows[entries] = upstream
        cols[entries] = upstream
        coefficients[entries] = flux
        entries += 1
        rows[entries] = downstream
        cols[entries] = upstream
        coefficients[entries] = -flux
        divergence[upstream] += flux
        divergence[downstream] -= flux
    end
    # Source terms may be mass rates, and well controls may specify surface
    # rates or BHP. Use the local volume-flux imbalance to close sink rows;
    # it never determines whether a source or well is an injector or producer.
    for (name, cells) in sinks
        for cell in unique(cells)
            rate = 0.0
            if haskey(setup.source_rates, name)
                rate = max(-divergence[cell], 0.0)
            elseif haskey(legacy_rates, name)
                rate = legacy_rates[name] / length(cells)
            else
                rate = max(-divergence[cell], 0.0)
            end
            if rate > 0
                entries += 1
                rows[entries] = cell
                cols[entries] = cell
                coefficients[entries] = rate
            end
        end
    end
    resize!(rows, entries)
    resize!(cols, entries)
    resize!(coefficients, entries)
    return sparse(rows, cols, coefficients, nc, nc)
end

function boundary_mask(cells, nc)
    mask = falses(nc)
    for perforations in values(cells)
        for cell in perforations
            checkbounds(mask, cell)
            mask[cell] = true
        end
    end
    return mask
end

function reachable_cells(matrix, boundary)
    nc = size(matrix, 1)
    reachable = copy(boundary)
    queue = findall(boundary)
    first_item = 1
    rows = rowvals(matrix)
    values = nonzeros(matrix)
    while first_item <= length(queue)
        cell = queue[first_item]
        first_item += 1
        for index in nzrange(matrix, cell)
            neighbor = rows[index]
            if neighbor != cell && values[index] < 0 && !reachable[neighbor]
                reachable[neighbor] = true
                push!(queue, neighbor)
            end
        end
    end
    return reachable
end

function prepare_direction(matrix, cells, solver)
    nc = size(matrix, 1)
    boundary = boundary_mask(cells, nc)
    reachable = reachable_cells(matrix, boundary)
    local_index = zeros(Int, nc)
    solvable = Int[]
    for cell in 1:nc
        if reachable[cell] && !boundary[cell] && matrix[cell, cell] > 0
            push!(solvable, cell)
            local_index[cell] = length(solvable)
        end
    end
    if solver == :direct
        if isempty(solvable)
            factor = nothing
        else
            factor = lu(matrix[solvable, solvable])
        end
    elseif solver == :reordered
        factor = prepare_reordered(matrix, solvable, local_index)
    else
        throw(ArgumentError("solver must be :direct or :reordered"))
    end
    return PreparedDirection(matrix, boundary, solvable, local_index, factor)
end

boundary_value(::Nothing, cell) = 0.0
boundary_value(values::AbstractVector, cell) = values[cell]

function solve_direction(direction::PreparedDirection, rhs::AbstractVector,
        boundary_values, fallback::Float64; tracer::Bool = false)
    nc = size(direction.matrix, 1)
    solution = fill(fallback, nc)
    for cell in 1:nc
        if direction.boundary[cell]
            solution[cell] = boundary_value(boundary_values, cell)
        end
    end
    isempty(direction.solvable) && return solution
    reduced_rhs = Vector{Float64}(undef, length(direction.solvable))
    for (position, cell) in enumerate(direction.solvable)
        reduced_rhs[position] = rhs[cell]
    end
    rows = rowvals(direction.matrix)
    values = nonzeros(direction.matrix)
    for cell in 1:nc
        value = boundary_value(boundary_values, cell)
        if direction.boundary[cell] && value != 0
            for index in nzrange(direction.matrix, cell)
                position = direction.local_index[rows[index]]
                if position != 0
                    reduced_rhs[position] -= values[index] * value
                end
            end
        end
    end
    solved = solve_reduced(direction, reduced_rhs)
    for (position, cell) in enumerate(direction.solvable)
        if tracer
            solution[cell] = clamp(solved[position], 0.0, 1.0)
        else
            solution[cell] = max(solved[position], 0.0)
        end
    end
    return solution
end

solve_reduced(direction::PreparedDirection, rhs) = direction.factorization \ rhs

"""
    solve_flow_diagnostics(setup_or_prepared; compute_tracers=true,
        compute_well_tof=false,
        perforation_tracers=nothing, max_tof=DEFAULT_MAX_TOF,
        solver=:direct, forces=nothing)

Compute TOF and one tracer per active well. With `compute_well_tof=true`, also
compute conditional forward TOF for each injector and backward TOF for each
producer. These are keyed by well name in `forward_tof_by_well` and
`backward_tof_by_well`. Cells not reached by a well receive `max_tof`.
Set `perforation_tracers` to a well name, a vector of well names, or `:all`
to add a separate tracer for every perforation of those wells. Perforation
keys are `:W_perf_1`, `:W_perf_2`, etc.
"""
function solve_flow_diagnostics(setup::FlowDiagnosticsSetup;
        compute_tracers::Bool = true, compute_well_tof::Bool = false,
        perforation_tracers = nothing,
        max_tof::Real = DEFAULT_MAX_TOF, solver::Symbol = :direct, forces = nothing)
    prepared = prepare_flow_diagnostics(setup; forces = forces, solver = solver)
    return solve_flow_diagnostics(prepared;
        compute_tracers, compute_well_tof, perforation_tracers, max_tof)
end

function solve_flow_diagnostics(prepared::PreparedFlowDiagnostics;
        compute_tracers::Bool = true, compute_well_tof::Bool = false,
        perforation_tracers = nothing,
        max_tof::Real = DEFAULT_MAX_TOF)
    validate_perforation_selection(perforation_tracers, prepared.setup.well_cells)
    pv = prepared.setup.pore_volume
    forward = solve_direction(prepared.forward, pv, nothing, Float64(max_tof))
    backward = solve_direction(prepared.backward, pv, nothing, Float64(max_tof))
    forward_by_well = Dict{Symbol, Vector{Float64}}()
    backward_by_well = Dict{Symbol, Vector{Float64}}()
    injector_tracers = Dict{Symbol, Vector{Float64}}()
    producer_tracers = Dict{Symbol, Vector{Float64}}()
    if compute_tracers || compute_well_tof
        compute_direction_diagnostics!(injector_tracers, forward_by_well,
            prepared.forward, pv,
            prepared.injector_cells, prepared.setup.well_cells,
            perforation_tracers, compute_tracers, compute_well_tof,
            Float64(max_tof))
        compute_direction_diagnostics!(producer_tracers, backward_by_well,
            prepared.backward, pv,
            prepared.producer_cells, prepared.setup.well_cells,
            perforation_tracers, compute_tracers, compute_well_tof,
            Float64(max_tof))
    end
    return FlowDiagnosticsResult(forward, backward, forward_by_well,
        backward_by_well, injector_tracers, producer_tracers)
end

function validate_perforation_selection(option, wells)
    if isnothing(option) || option == :all
        return nothing
    end
    names = option
    if option isa Symbol
        names = (option,)
    elseif !(option isa AbstractVector || option isa Tuple)
        throw(ArgumentError("perforation_tracers must be a well name, collection, :all, or nothing"))
    end
    for name in names
        haskey(wells, name) || throw(ArgumentError("Unknown well $name"))
    end
    return nothing
end

function wants_perforation_tracers(option, name)
    if isnothing(option)
        return false
    elseif option == :all || option == name
        return true
    elseif option isa Symbol
        return false
    elseif option isa AbstractVector || option isa Tuple
        return name in option
    end
    throw(ArgumentError("perforation_tracers must be a well name, collection, :all, or nothing"))
end

function compute_direction_diagnostics!(tracers, well_tof, direction, pv,
        cells, wells, option, compute_tracers, compute_well_tof, max_tof)
    nc = length(pv)
    rhs = zeros(nc)
    boundary = zeros(nc)
    for (name, perforations) in cells
        fill!(boundary, 0.0)
        for cell in perforations
            boundary[cell] = 1.0
        end
        concentration = solve_direction(direction, rhs, boundary, 0.0; tracer = true)
        if compute_tracers
            tracers[name] = concentration
        end
        if compute_well_tof
            moment = solve_direction(direction, pv .* concentration, nothing, 0.0)
            times = fill(max_tof, nc)
            for cell in eachindex(times)
                if concentration[cell] > 0
                    times[cell] = moment[cell] / concentration[cell]
                end
            end
            well_tof[name] = times
        end
        if compute_tracers && haskey(wells, name) && wants_perforation_tracers(option, name)
            for (index, cell) in enumerate(perforations)
                fill!(boundary, 0.0)
                boundary[cell] = 1.0
                key = Symbol(name, :_perf_, index)
                tracers[key] = solve_direction(direction, rhs, boundary, 0.0; tracer = true)
            end
        end
    end
    return tracers
end
