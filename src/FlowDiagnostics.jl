"""
    FlowDiagnostics

Flow diagnostics for reservoir simulation results based on JutulDarcy.jl.
Computes forward and backward time-of-flight (TOF) as well as steady-state
tracer concentrations from a `ReservoirSimResult` using finite-volume
discretization with an upwind scheme.

`setup_flow_diagnostics` extracts a velocity field from a reported state.
`prepare_flow_diagnostics` assembles and factorizes the directional systems
once for repeated solves. `solve_flow_diagnostics` computes TOF and tracers.
`solve_pressure_flow_diagnostics` obtains a velocity field from a pressure
step when no simulation result is available.
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
        darcy_phase_volume_fluxes,
        eachphase

    import Jutul:
        number_of_cells,
        number_of_faces,
        physical_representation

    include("FlowDiagnostics/types.jl")
    include("FlowDiagnostics/setup.jl")
    include("FlowDiagnostics/solve.jl")
    include("FlowDiagnostics/reordered.jl")
    include("FlowDiagnostics/pressure.jl")

    export FlowDiagnosticsSetup, FlowDiagnosticsResult, PreparedFlowDiagnostics
    export setup_flow_diagnostics, prepare_flow_diagnostics, solve_flow_diagnostics
    export flow_diagnostics_all_states, solve_pressure_flow_diagnostics

end
