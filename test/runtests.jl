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
        @test length(diag.residence_time) == nc

        # Forward TOF: injector cell = 0, increases towards producer
        @test diag.forward_tof[1] ≈ 0.0 atol=1e-10
        @test diag.forward_tof[end] > diag.forward_tof[1]

        # Backward TOF: producer cell = 0, increases towards injector
        @test diag.backward_tof[end] ≈ 0.0 atol=1e-10
        @test diag.backward_tof[1] > diag.backward_tof[end]

        # Residence time should be the sum
        @test diag.residence_time ≈ diag.forward_tof .+ diag.backward_tof

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

    @testset "max_tof for disconnected cells" begin
        # Construct a minimal FlowDiagnosticsSetup directly: 3 cells, one face
        # connecting cell 1 (injector) to cell 2 (producer).  Cell 3 has no
        # face connections and is therefore disconnected from both wells.
        #
        # Expected behaviour:
        #   - Cell 3 gets Inf TOF when max_tof = Inf (default _solve_tof)
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
        @test all(isfinite, diag_capped.residence_time)

        # Default max_tof (10 000 years): connected cells and cell 3 all finite
        diag_default = solve_flow_diagnostics(setup3; compute_tracers = false)
        @test all(isfinite, diag_default.forward_tof)
        @test all(isfinite, diag_default.backward_tof)
    end

end

