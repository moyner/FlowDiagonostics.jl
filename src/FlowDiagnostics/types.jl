"""
    FlowDiagnosticsSetup

Pre-computed data for flow diagnostics. Created once from a
`ReservoirSimResult` and reused across multiple solve calls.

# Fields
- `model`: The full reservoir `MultiModel` (or single `SimulationModel`).
- `N`: Face neighborship matrix (2 × n_faces); `N[1,f]` is the left cell and
  `N[2,f]` is the right cell for face `f`.
- `q`: Total volumetric flux at each internal face (m³/s). A positive value
  means net flow from the left cell to the right cell.
- `pore_volume`: Pore volume (m³) for every reservoir cell.
- `injector_cells`: Dict mapping each injector name to the reservoir cell
  indices that it perforates (or injects into).
- `injector_rates`: Dict mapping each injector name to the total volumetric
  injection rate (m³/s, positive).
- `producer_cells`: Dict mapping each producer name to the reservoir cell
  indices that it perforates (or produces from).
- `producer_rates`: Dict mapping each producer name to the total volumetric
  production rate (m³/s, positive).
"""
struct FlowDiagnosticsSetup
    model
    N::AbstractMatrix{Int}
    q::Vector{Float64}
    pore_volume::Vector{Float64}
    injector_cells::Dict{Symbol, Vector{Int}}
    injector_rates::Dict{Symbol, Float64}
    producer_cells::Dict{Symbol, Vector{Int}}
    producer_rates::Dict{Symbol, Float64}
end

function Base.show(io::IO, fd::FlowDiagnosticsSetup)
    nc = length(fd.pore_volume)
    nf = length(fd.q)
    ni = length(fd.injector_cells)
    np = length(fd.producer_cells)
    print(io, "FlowDiagnosticsSetup ($nc cells, $nf faces, $ni injectors, $np producers)")
end

"""
    FlowDiagnosticsResult

Results from a flow diagnostics computation.

# Fields
- `forward_tof`: Forward time-of-flight (s) in each cell. This is the travel
  time from the nearest injector to the cell along streamlines.
- `backward_tof`: Backward time-of-flight (s) in each cell. This is the
  travel time from the cell to the nearest producer along streamlines.
- `residence_time`: Total residence time (s) per cell, equal to the sum of
  forward and backward TOF. Cells not reachable from any injector and producer
  simultaneously have residence time equal to `2 * max_tof` (or `Inf` when
  `max_tof = Inf`).
- `injector_tracers`: Dict mapping each injector name to a vector of
  steady-state tracer concentrations (0–1) in every cell. A value of 1
  means the cell is entirely swept by that injector.
- `producer_tracers`: Dict mapping each producer name to a vector of backward
  tracer concentrations (0–1) in every cell. A value of 1 means the cell
  drains entirely into that producer.
"""
struct FlowDiagnosticsResult
    forward_tof::Vector{Float64}
    backward_tof::Vector{Float64}
    residence_time::Vector{Float64}
    injector_tracers::Dict{Symbol, Vector{Float64}}
    producer_tracers::Dict{Symbol, Vector{Float64}}
end

function Base.show(io::IO, r::FlowDiagnosticsResult)
    nc = length(r.forward_tof)
    ni = length(r.injector_tracers)
    np = length(r.producer_tracers)
    print(io, "FlowDiagnosticsResult ($nc cells, $ni injector tracers, $np producer tracers)")
end
