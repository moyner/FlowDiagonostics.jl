"""
    FlowDiagnostics

Flow diagnostics for reservoir simulation results based on JutulDarcy.jl.
Computes forward and backward time-of-flight (TOF) as well as steady-state
tracer concentrations from a `ReservoirSimResult` using finite-volume
discretization with an upwind scheme.

The workflow is split into two phases:
1. **Setup** (`setup_flow_diagnostics`): extract a single state, dt and forces
   from the simulation result, compute inter-cell fluxes with JutulDarcy
   routines, and assemble the upwind connectivity structure. The resulting
   `FlowDiagnosticsSetup` can be reused for multiple solve calls.
2. **Solve** (`solve_flow_diagnostics`): build and solve the sparse linear
   systems for forward/backward TOF and for tracer concentrations.
"""
module FlowDiagnostics
    using JutulDarcy
    using Jutul
    using SparseArrays
    using LinearAlgebra

    import JutulDarcy:
        reservoir_model,
        reservoir_domain,
        number_of_phases,
        get_phases,
        setup_parameters,
        pore_volume,
        model_or_domain_is_well,
        ReservoirSimResult,
        darcy_phase_volume_fluxes

    import Jutul:
        number_of_cells,
        number_of_faces,
        physical_representation

    include("FlowDiagnostics/types.jl")
    include("FlowDiagnostics/setup.jl")
    include("FlowDiagnostics/solve.jl")

    export FlowDiagnosticsSetup, FlowDiagnosticsResult
    export setup_flow_diagnostics, solve_flow_diagnostics
end
