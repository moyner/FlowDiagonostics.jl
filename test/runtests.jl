using Test
using Jutul
using JutulDarcy
using FlowDiagnostics

# Simple mean function for the test
_mean(x) = sum(x) / length(x)

@testset "FlowDiagnostics" begin

    # -----------------------------------------------------------------------
    # Build a minimal 1-D single-phase reservoir with one injector and one
    # producer modelled as explicit SourceTerm forcing.
    # Cells: [inj=1] -- [2] -- [3] -- [4] -- [prod=5]
    # -----------------------------------------------------------------------
    nx = 5
    g  = CartesianMesh((nx, 1, 1), (100.0, 10.0, 10.0))  # 100 m × 10 m × 10 m
    domain = reservoir_domain(g,
        permeability = convert_to_si(100.0, :millidarcy),
        porosity = 0.2
    )

    # PVT: slightly compressible water (defined first so reference_density can be passed to sys)
    bar   = 1e5
    pRef  = 100*bar
    rhoLS = 1000.0
    cl    = 1e-5 / bar

    phase = LiquidPhase()
    sys   = SinglePhaseSystem(phase; reference_density = rhoLS)

    # Use setup_reservoir_model to get a proper JutulDarcy MultiModel.
    # block_backend = false avoids the BlockMajorLayout linear solver which
    # requires matrix-valued Jacobian entries (incompatible with single-phase).
    model = setup_reservoir_model(domain, sys; block_backend = false)
    rmodel = reservoir_model(model)

    set_secondary_variables!(rmodel,
        PhaseMassDensities = ConstantCompressibilityDensities(sys, pRef, rhoLS, cl))

    # Injection rate that fills the pore volume in ~ 1 year
    pv       = pore_volume(domain)
    tot_time = 365.25 * 86400.0   # 1 year in seconds
    irate    = sum(pv) / tot_time  # m³/s

    inj_src  = SourceTerm(1,   irate)
    prod_src = SourceTerm(nx, -irate)
    # Forces: wrap reservoir forces in the MultiModel format
    res_forces = setup_forces(rmodel, sources = [inj_src, prod_src])
    forces     = setup_forces(model, Reservoir = res_forces)

    p0     = 100*bar
    state0 = setup_state(model, Dict(:Reservoir => Dict(:Pressure => p0)))

    dt = [tot_time / 20 for _ in 1:20]   # 20 equal steps of ~ 18 days

    result = simulate_reservoir(state0, model, dt;
        forces     = forces,
        info_level = -1
    )

    @testset "setup_flow_diagnostics" begin
        setup = setup_flow_diagnostics(result, model, forces)

        @test setup isa FlowDiagnosticsSetup
        @test length(setup.q) == number_of_faces(domain)
        @test length(setup.pore_volume) == number_of_cells(domain)

        # There should be one injector and one producer
        @test length(setup.injector_cells)  == 1
        @test length(setup.producer_cells)  == 1
        @test length(setup.injector_rates)  == 1
        @test length(setup.producer_rates)  == 1

        # Injector rate should be positive
        @test first(values(setup.injector_rates)) > 0.0
        @test first(values(setup.producer_rates)) > 0.0
    end

    @testset "solve_flow_diagnostics" begin
        setup = setup_flow_diagnostics(result, model, forces)
        diag  = solve_flow_diagnostics(setup)

        @test diag isa FlowDiagnosticsResult
        nc = number_of_cells(domain)
        @test length(diag.forward_tof)   == nc
        @test length(diag.backward_tof)  == nc
        @test !hasproperty(diag, :residence_time)
        @test isempty(diag.forward_tof_by_well)
        @test isempty(diag.backward_tof_by_well)

        # Forward TOF: injector cell = 0, increases towards producer
        @test diag.forward_tof[1] ≈ 0.0 atol=1e-10
        @test diag.forward_tof[end] > diag.forward_tof[1]

        # Backward TOF: producer cell = 0, increases towards injector
        @test diag.backward_tof[end] ≈ 0.0 atol=1e-10
        @test diag.backward_tof[1] > diag.backward_tof[end]

        # Total pore volume = integral of flux × TOF (Lorenz coefficient check)
        # For a perfectly uniform 1-D system, mean TOF ≈ pore_volume / injection_rate
        mean_fwd = _mean(diag.forward_tof[isfinite.(diag.forward_tof)])
        pv_total = sum(setup.pore_volume)
        inj_rate = first(values(setup.injector_rates))
        expected_mean_tof = pv_total / inj_rate
        @test abs(mean_fwd - expected_mean_tof) / expected_mean_tof < 0.8
    end

    @testset "tracer concentrations" begin
        setup = setup_flow_diagnostics(result, model, forces)
        diag  = solve_flow_diagnostics(setup, compute_tracers = true)

        @test length(diag.injector_tracers) == 1
        @test length(diag.producer_tracers) == 1

        inj_name  = only(keys(setup.injector_cells))
        prod_name = only(keys(setup.producer_cells))

        C_inj  = diag.injector_tracers[inj_name]
        C_prod = diag.producer_tracers[prod_name]

        # All concentrations should be in [0, 1]
        @test all(c -> 0.0 <= c <= 1.0 + 1e-10, C_inj)
        @test all(c -> 0.0 <= c <= 1.0 + 1e-10, C_prod)

        # Injector cell: tracer = 1 (source)
        @test C_inj[1] ≈ 1.0 atol=1e-10
        # Producer cell: backward tracer = 1 (source)
        @test C_prod[end] ≈ 1.0 atol=1e-10

        # In a 1-D system a single injector sweeps everything → C_inj ~ 1 everywhere
        @test minimum(C_inj) > 0.5
    end

    @testset "step_index keyword" begin
        # Using an earlier step should still produce valid results
        setup = setup_flow_diagnostics(result, model, forces; step_index = 5)
        @test setup isa FlowDiagnosticsSetup
        diag  = solve_flow_diagnostics(setup)
        @test diag isa FlowDiagnosticsResult
    end

    @testset "case workflow and pressure velocity" begin
        case = JutulCase(model, dt, forces; state0 = state0)
        diagnostics = flow_diagnostics_all_states(case, result;
            compute_tracers = false, compute_well_tof = true)
        @test length(diagnostics) == length(result.states)
        @test length(diagnostics[end].forward_tof_by_well) == 1
        @test length(diagnostics[end].backward_tof_by_well) == 1
        @test diagnostics[end].forward_tof ≈ solve_flow_diagnostics(
            setup_flow_diagnostics(result, case); compute_tracers = false).forward_tof

        pressure = solve_pressure_flow_diagnostics(case; dt = dt[1],
            compute_tracers = false, compute_well_tof = true)
        @test length(pressure.setup.q) == number_of_faces(domain)
        @test all(isfinite, pressure.setup.q)
        @test pressure.diagnostics isa FlowDiagnosticsResult
        @test length(pressure.diagnostics.forward_tof_by_well) == 1

        two_system = ImmiscibleSystem((AqueousPhase(), LiquidPhase());
            reference_densities = (1000.0, 800.0))
        two_model = setup_reservoir_model(domain, two_system;
            block_backend = false)
        two_reservoir = reservoir_model(two_model)
        two_forces = setup_forces(two_model;
            Reservoir = setup_forces(two_reservoir;
                sources = [
                    SourceTerm(1, irate; fractional_flow = [1.0, 0.0]),
                    SourceTerm(nx, -irate; fractional_flow = [1.0, 0.0])
                ]))
        two_state = setup_state(two_model, Dict(:Reservoir => Dict(
            :Pressure => p0, :Saturations => fill(0.5, 2, nx))))
        two_case = JutulCase(two_model, [dt[1]], two_forces;
            state0 = two_state)
        two_pressure = solve_pressure_flow_diagnostics(two_case;
            compute_tracers = false)
        @test length(two_pressure.setup.q) == number_of_faces(domain)
        @test all(isfinite, two_pressure.setup.q)
    end

    @testset "max_tof for disconnected cells" begin
        # Construct a minimal FlowDiagnosticsSetup directly: 3 cells, one face
        # connecting cell 1 (injector) to cell 2 (producer).  Cell 3 has no
        # face connections and is therefore disconnected from both wells.
        #
        # Expected behaviour:
        #   - Cell 3 gets Inf TOF when max_tof = Inf
        #   - Cell 3 gets max_tof when a finite max_tof is supplied
        N3   = [1 ; 2][:, :]              # 2 × 1 face matrix
        q3   = [1e-4]                      # positive flux: cell 1 → cell 2
        pv3  = [10.0, 10.0, 10.0]         # pore volumes
        inj3  = Dict{Symbol, Vector{Int}}(:I => [1])
        irat3 = Dict{Symbol, Float64}(:I => 1e-4)
        pro3  = Dict{Symbol, Vector{Int}}(:P => [2])
        prat3 = Dict{Symbol, Float64}(:P => 1e-4)

        setup3 = FlowDiagnosticsSetup(nothing, N3, q3, pv3, inj3, irat3, pro3, prat3)

        # Without max_tof: disconnected cell 3 should be Inf
        diag_inf = solve_flow_diagnostics(setup3; compute_tracers = false, max_tof = Inf)
        @test isinf(diag_inf.forward_tof[3])
        @test isinf(diag_inf.backward_tof[3])

        # With a finite max_tof: cell 3 should be capped
        max_tof_val = 5_000 * 365.25 * 86400.0   # 5 000 years in seconds
        diag_capped = solve_flow_diagnostics(setup3; compute_tracers = false, max_tof = max_tof_val)
        @test diag_capped.forward_tof[3]  ≈ max_tof_val
        @test diag_capped.backward_tof[3] ≈ max_tof_val
        @test all(isfinite, diag_capped.forward_tof)
        @test all(isfinite, diag_capped.backward_tof)

        # Default max_tof: connected cells and cell 3 all finite
        diag_default = solve_flow_diagnostics(setup3; compute_tracers = false)
        @test all(isfinite, diag_default.forward_tof)
        @test all(isfinite, diag_default.backward_tof)
    end

end

@testset "Per-well time of flight" begin
    # Two injectors meet at cell 4 after different path lengths; flow then
    # splits between two producers. Conditional TOFs distinguish the paths.
    neighbors = [1 3 2 4 4; 3 4 4 5 6]
    flux = [1.0, 1.0, 1.0, 1.0, 1.0]
    wells = Dict(:I1 => [1], :I2 => [2], :P1 => [5], :P2 => [6])
    directions = Dict(:I1 => :injector, :I2 => :injector,
        :P1 => :producer, :P2 => :producer)
    setup = FlowDiagnosticsSetup(nothing, neighbors, flux,
        [1.0, 1.0, 1.0, 2.0, 1.0, 1.0], wells, directions,
        Dict{Symbol, Pair{Int, Float64}}())
    result = solve_flow_diagnostics(setup; compute_tracers = false,
        compute_well_tof = true, max_tof = Inf)
    @test isempty(result.injector_tracers)
    @test isempty(result.producer_tracers)
    @test Set(keys(result.forward_tof_by_well)) == Set((:I1, :I2))
    @test Set(keys(result.backward_tof_by_well)) == Set((:P1, :P2))
    @test result.forward_tof_by_well[:I1][4] ≈ 2.0
    @test result.forward_tof_by_well[:I2][4] ≈ 1.0
    @test result.backward_tof_by_well[:P1][4] ≈ 1.0
    @test result.backward_tof_by_well[:P2][4] ≈ 1.0
    @test isinf(result.forward_tof_by_well[:I1][2])
    @test isinf(result.backward_tof_by_well[:P1][6])

    ordered = solve_flow_diagnostics(setup; solver = :reordered,
        compute_well_tof = true, max_tof = 100.0)
    @test ordered.forward_tof_by_well[:I1][4] ≈ 2.0
    @test ordered.backward_tof_by_well[:P1][6] == 100.0
end

@testset "Prepared and reordered solves" begin
    # 1 -> 2 -> 3 -> 1 is a cycle. Cell 1 has an injector boundary;
    # cell 3 connects to a producer at cell 4.
    neighbors = [1 2 3 3; 2 3 1 4]
    flux = [1.0, 1.0, 0.25, 0.75]
    wells = Dict(:I => [1, 2], :P => [4])
    directions = Dict(:I => :injector, :P => :producer)
    setup = FlowDiagnosticsSetup(nothing, neighbors, flux, ones(4),
        wells, directions, Dict{Symbol, Pair{Int, Float64}}())
    direct = solve_flow_diagnostics(prepare_flow_diagnostics(setup);
        perforation_tracers = :I)
    ordered = solve_flow_diagnostics(prepare_flow_diagnostics(setup;
        solver = :reordered); perforation_tracers = :I)
    @test direct.forward_tof ≈ ordered.forward_tof
    @test direct.backward_tof ≈ ordered.backward_tof
    @test direct.injector_tracers[:I] ≈ ordered.injector_tracers[:I]
    @test haskey(direct.injector_tracers, :I_perf_1)
    @test haskey(direct.injector_tracers, :I_perf_2)
    @test direct.injector_tracers[:I_perf_1][1] == 1.0
    @test direct.injector_tracers[:I_perf_1][2] == 0.0
    @test direct.injector_tracers[:I_perf_2][2] == 1.0

    switched_forces = (
        Facility = (control = Dict(
            :I => ProducerControl(TotalRateTarget(-1.0)),
            :P => InjectorControl(TotalRateTarget(1.0), [1.0])
        ),),
    )
    switched = prepare_flow_diagnostics(setup; forces = switched_forces)
    @test haskey(switched.producer_cells, :I)
    @test haskey(switched.injector_cells, :P)
    @test !haskey(switched.injector_cells, :I)

    cycle_neighbors = [1 2 3 4 4; 2 3 4 2 5]
    cycle_flux = [0.75, 1.0, 1.0, 0.25, 0.75]
    cycle_setup = FlowDiagnosticsSetup(nothing, cycle_neighbors, cycle_flux,
        ones(5), Dict(:I => [1], :P => [5]), directions,
        Dict{Symbol, Pair{Int, Float64}}())
    cycle_direct = solve_flow_diagnostics(cycle_setup)
    cycle_ordered = solve_flow_diagnostics(cycle_setup; solver = :reordered)
    @test cycle_direct.forward_tof ≈ cycle_ordered.forward_tof
    @test cycle_direct.backward_tof ≈ cycle_ordered.backward_tof
    @test cycle_direct.injector_tracers[:I] ≈ cycle_ordered.injector_tracers[:I]

    closed_setup = FlowDiagnosticsSetup(nothing, [1 2 3; 2 3 1],
        ones(3), ones(3), Dict{Symbol, Vector{Int}}(),
        Dict{Symbol, Symbol}(), Dict{Symbol, Pair{Int, Float64}}())
    closed = solve_flow_diagnostics(closed_setup; solver = :reordered,
        compute_tracers = false, max_tof = Inf)
    @test all(isinf, closed.forward_tof)
    @test all(isinf, closed.backward_tof)
end
