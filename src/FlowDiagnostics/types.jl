"""Flow field, pore volumes, and well/source topology for one reservoir state.

`well_cells` contains every well, including disabled wells. `well_directions`
contains only active controls (`:injector` or `:producer`). Source terms are
recorded separately, with their signed native JutulDarcy values.
"""
struct FlowDiagnosticsSetup
    model
    N::AbstractMatrix{Int}
    q::Vector{Float64}
    pore_volume::Vector{Float64}
    well_cells::Dict{Symbol, Vector{Int}}
    well_directions::Dict{Symbol, Symbol}
    source_rates::Dict{Symbol, Pair{Int, Float64}}
    injector_cells::Dict{Symbol, Vector{Int}}
    injector_rates::Dict{Symbol, Float64}
    producer_cells::Dict{Symbol, Vector{Int}}
    producer_rates::Dict{Symbol, Float64}
end

function FlowDiagnosticsSetup(model, N, q, pv, wells, directions, sources)
    injectors = Dict{Symbol, Vector{Int}}()
    producers = Dict{Symbol, Vector{Int}}()
    injector_rates = Dict{Symbol, Float64}()
    producer_rates = Dict{Symbol, Float64}()
    for (name, cells) in wells
        direction = get(directions, name, :disabled)
        if direction == :injector
            injectors[name] = cells
        elseif direction == :producer
            producers[name] = cells
        end
    end
    for (name, source) in sources
        cell, rate = source
        if rate > 0
            injectors[name] = [cell]
            injector_rates[name] = rate
        elseif rate < 0
            producers[name] = [cell]
            producer_rates[name] = -rate
        end
    end
    return FlowDiagnosticsSetup(model, N, float_vector(q), float_vector(pv),
        wells, directions, sources, injectors, injector_rates, producers, producer_rates)
end

float_vector(values::Vector{Float64}) = values
float_vector(values) = Float64.(values)

# Keep the original positional constructor for existing callers.
function FlowDiagnosticsSetup(model, N, q, pv, injectors::Dict{Symbol, Vector{Int}},
        injector_rates::Dict{Symbol, Float64}, producers::Dict{Symbol, Vector{Int}},
        producer_rates::Dict{Symbol, Float64})
    wells = merge(copy(injectors), producers)
    directions = Dict{Symbol, Symbol}()
    for name in keys(injectors)
        directions[name] = :injector
    end
    for name in keys(producers)
        directions[name] = :producer
    end
    sources = Dict{Symbol, Pair{Int, Float64}}()
    return FlowDiagnosticsSetup(model, N, float_vector(q), float_vector(pv),
        wells, directions, sources, injectors, injector_rates, producers, producer_rates)
end

function Base.show(io::IO, setup::FlowDiagnosticsSetup)
    print(io, "FlowDiagnosticsSetup ($(length(setup.pore_volume)) cells, $(length(setup.q)) faces, $(length(setup.well_cells)) wells)")
end

"""Forward/backward time of flight and well tracer concentrations."""
struct FlowDiagnosticsResult
    forward_tof::Vector{Float64}
    backward_tof::Vector{Float64}
    residence_time::Vector{Float64}
    injector_tracers::Dict{Symbol, Vector{Float64}}
    producer_tracers::Dict{Symbol, Vector{Float64}}
end

function Base.show(io::IO, result::FlowDiagnosticsResult)
    print(io, "FlowDiagnosticsResult ($(length(result.forward_tof)) cells, $(length(result.injector_tracers)) injector tracers, $(length(result.producer_tracers)) producer tracers)")
end

"""Prepared forward and backward systems. Reuse this for repeated tracer/TOF solves."""
struct PreparedFlowDiagnostics{F, B}
    setup::FlowDiagnosticsSetup
    forward::F
    backward::B
    injector_cells::Dict{Symbol, Vector{Int}}
    producer_cells::Dict{Symbol, Vector{Int}}
end
