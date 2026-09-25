# FlowDiagnostics.jl

Flow diagnostics for JutulDarcy reservoir models. The package computes forward
and backward time of flight and steady state well tracers from reservoir volume
fluxes.

```julia
using FlowDiagnostics

setup = setup_flow_diagnostics(result, case; step_index = 1)
prepared = prepare_flow_diagnostics(setup)
diagnostics = solve_flow_diagnostics(prepared;
    perforation_tracers = :INJ1)

# One result per reported state, using that step's forces:
series = flow_diagnostics_all_states(case, result)

# Obtain a velocity field from a pressure solve without simulation results:
pressure = solve_pressure_flow_diagnostics(case; dt = case.dt[1])
velocity = pressure.setup.q
```

`setup.well_cells` maps each modeled well to all of its reservoir perforation
cells. `setup.well_directions` records the active injector or producer controls
for the selected step. `prepare_flow_diagnostics(setup; forces=...)` can apply
new well controls to the same velocity field. It factorizes each directional
system once, so repeated tracer and time of flight solves can reuse the
prepared context.

`perforation_tracers` accepts a well name, a collection of names, or `:all`.
Individual perforation tracers use keys such as `:INJ1_perf_1` in the
corresponding injector or producer tracer dictionary. The well tracer remains
available under the well name.

The default solver uses sparse direct factorization. Pass `solver=:reordered`
to `prepare_flow_diagnostics` or `solve_flow_diagnostics` to use the optional
graph ordered solver in `src/FlowDiagnostics/reordered.jl`. It solves acyclic
cells in flow order and groups recirculating cells with Tarjan's algorithm.

The pressure entry point uses JutulDarcy's sequential pressure formulation. It
constructs an immiscible system with the same phases as the input system when
needed and copies reservoir relative permeability, capillary pressure, density,
and viscosity definitions. It returns the flow setup, diagnostics, and pressure
state as a named tuple.
