"""
    solve_flow_diagnostics(setup; compute_tracers = true, max_tof = 10_000 * 365.25 * 86400.0)

Compute forward/backward time-of-flight and, optionally, steady-state tracer
concentrations from a pre-assembled `FlowDiagnosticsSetup`.

Solves the steady-state upwind finite-volume equation

    A τ = b

where `A` is the upwind flux matrix and `b[c] = pore_volume[c]` for all cells
that are not fixed by a boundary condition (injectors / producers).

- **Forward TOF** (`τ_fwd`): flow direction is the natural direction of the
  face fluxes.  Injector cells are fixed at `τ = 0`.
- **Backward TOF** (`τ_bwd`): flow direction is reversed.  Producer cells are
  fixed at `τ = 0`.
- **Injector tracers** (`C_inj`): one tracer per injector.  The tracer
  concentration is set to 1 at the injector cells and advected forward.
- **Producer tracers** (`C_prod`): one tracer per producer.  The tracer
  concentration is set to 1 at the producer cells and advected backward
  (reversed flux).

# Arguments
- `setup::FlowDiagnosticsSetup`: output from [`setup_flow_diagnostics`](@ref).

# Keyword arguments
- `compute_tracers::Bool = true`: whether to solve for tracer concentrations
  in addition to time-of-flight.
- `max_tof::Float64`: upper bound (in seconds) assigned to cells that are
  disconnected from all wells (i.e. have no flow path to any injector or
  producer).  Defaults to 10 000 years.  Set to `Inf` to retain the previous
  behaviour of returning infinite values for such cells.

# Returns
A `FlowDiagnosticsResult` with the computed diagnostics.
"""
function solve_flow_diagnostics(setup::FlowDiagnosticsSetup; compute_tracers::Bool = true, max_tof::Float64 = 10_000 * 365.25 * 86400.0)
    (; N, q, pore_volume, injector_cells, injector_rates,
       producer_cells, producer_rates) = setup

    nc = length(pore_volume)

    # -----------------------------------------------------------------
    # Forward TOF: fixed cells = injectors (τ = 0)
    # -----------------------------------------------------------------
    fwd_bc = Dict{Int, Float64}()
    for cells in values(injector_cells)
        for c in cells
            fwd_bc[c] = 0.0
        end
    end
    # Add explicit outflow-rate contributions from producers to the diagonal
    fwd_src = Dict{Int, Float64}()
    for (wname, cells) in pairs(producer_cells)
        rate = producer_rates[wname]
        n    = length(cells)
        for c in cells
            fwd_src[c] = get(fwd_src, c, 0.0) + rate / n
        end
    end

    fwd_A = _build_upwind_matrix(q, N, nc, fwd_src)
    fwd_tof = _solve_tof(fwd_A, pore_volume, fwd_bc, nc; max_tof = max_tof)

    # -----------------------------------------------------------------
    # Backward TOF: reverse flux, fixed cells = producers (τ = 0)
    # -----------------------------------------------------------------
    bwd_bc = Dict{Int, Float64}()
    for cells in values(producer_cells)
        for c in cells
            bwd_bc[c] = 0.0
        end
    end
    bwd_src = Dict{Int, Float64}()
    for (wname, cells) in pairs(injector_cells)
        rate = injector_rates[wname]
        n    = length(cells)
        for c in cells
            bwd_src[c] = get(bwd_src, c, 0.0) + rate / n
        end
    end

    bwd_A = _build_upwind_matrix(-q, N, nc, bwd_src)
    bwd_tof = _solve_tof(bwd_A, pore_volume, bwd_bc, nc; max_tof = max_tof)

    # Residence time
    res_time = fwd_tof .+ bwd_tof

    # -----------------------------------------------------------------
    # Tracers
    # -----------------------------------------------------------------
    inj_tracers  = Dict{Symbol, Vector{Float64}}()
    prod_tracers = Dict{Symbol, Vector{Float64}}()

    if compute_tracers
        for (wname, cells) in pairs(injector_cells)
            bc = copy(fwd_bc)
            # This injector: C = 1; all other injectors: C = 0
            for (oname, ocells) in pairs(injector_cells)
                conc = (oname == wname) ? 1.0 : 0.0
                for c in ocells
                    bc[c] = conc
                end
            end
            inj_tracers[wname] = _solve_tracer(fwd_A, bc, nc)
        end

        for (wname, cells) in pairs(producer_cells)
            bc = copy(bwd_bc)
            for (oname, ocells) in pairs(producer_cells)
                conc = (oname == wname) ? 1.0 : 0.0
                for c in ocells
                    bc[c] = conc
                end
            end
            prod_tracers[wname] = _solve_tracer(bwd_A, bc, nc)
        end
    end

    return FlowDiagnosticsResult(fwd_tof, bwd_tof, res_time, inj_tracers, prod_tracers)
end

# -------------------------------------------------------------------------
# Linear system assembly
# -------------------------------------------------------------------------

"""
    _build_upwind_matrix(q, N, nc, src_rates)

Assemble the sparse upwind finite-volume flux matrix `A` of size `nc × nc`.

For each internal face `f` with left cell `l` and right cell `r` and flux
`q[f]` (positive = flow from `l` to `r`), the upstream cell contributes:

    A[l, l] += q[f]   (outflow from l)
    A[r, l] -= q[f]   (inflow to r from upstream l)

if `q[f] > 0`, and symmetrically if `q[f] < 0`.

Additional diagonal entries from `src_rates` (e.g. production wells) are
added to account for the outflow through wells.

BC cells are handled outside this function to keep the matrix structure clean
and avoid dense row/column operations.
"""
function _build_upwind_matrix(
        q::AbstractVector,
        N::AbstractMatrix{Int},
        nc::Int,
        src_rates::Dict{Int, Float64}
    )
    I_idx = Int[]
    J_idx = Int[]
    V_val = Float64[]

    for f in axes(N, 2)
        l   = N[1, f]
        r   = N[2, f]
        qf  = q[f]
        if qf > 0.0
            push!(I_idx, l); push!(J_idx, l); push!(V_val,  qf)
            push!(I_idx, r); push!(J_idx, l); push!(V_val, -qf)
        elseif qf < 0.0
            qf_abs = -qf
            push!(I_idx, r); push!(J_idx, r); push!(V_val,  qf_abs)
            push!(I_idx, l); push!(J_idx, r); push!(V_val, -qf_abs)
        end
    end

    for (c, rate) in src_rates
        if rate > 0.0
            push!(I_idx, c); push!(J_idx, c); push!(V_val, rate)
        end
    end

    return sparse(I_idx, J_idx, V_val, nc, nc)
end

# -------------------------------------------------------------------------
# Linear system solvers
# -------------------------------------------------------------------------

"""
    _solve_tof(A, pv, bc_cells, nc; max_tof = Inf)

Solve the TOF linear system for interior cells.  BC cells (injectors or
producers, depending on whether this is a forward or backward solve) are set
to zero.  Cells that have no flux outflow are unreachable and receive `max_tof`.

The system solved for the solvable (non-BC) cells is:

    A[solvable, solvable] · τ[solvable] = pv[solvable]
        - A[solvable, bc] · bc_vals

Returns a vector of length `nc`.
"""
function _solve_tof(
        A::SparseMatrixCSC,
        pv::AbstractVector,
        bc_cells::Dict{Int, Float64},
        nc::Int;
        max_tof::Float64 = Inf
    )
    τ = fill(max_tof, nc)

    solvable = _find_solvable_cells(A, bc_cells, nc)
    isempty(solvable) && return τ

    # RHS: pore volume minus contributions from BC cells
    b_solvable = Vector{Float64}(pv[solvable])
    _subtract_bc_contributions!(b_solvable, solvable, A, bc_cells)

    A_sub = A[solvable, solvable]
    τ_sub = A_sub \ b_solvable
    τ[solvable] .= max.(0.0, τ_sub)

    # BC cells get their fixed value (0 for injectors/producers)
    for (c, val) in bc_cells
        τ[c] = val
    end

    return τ
end

"""
    _solve_tracer(A, bc_cells, nc)

Solve a steady-state tracer equation reusing the structure of the upwind
matrix.  Boundary concentrations are in `bc_cells` (0 or 1).

The system solved for interior cells is:

    A[solvable, solvable] · C[solvable] = -A[solvable, bc] · bc_vals

Returns a concentration vector in [0, 1] for every cell.
"""
function _solve_tracer(
        A::SparseMatrixCSC,
        bc_cells::Dict{Int, Float64},
        nc::Int
    )
    C = fill(0.0, nc)

    solvable = _find_solvable_cells(A, bc_cells, nc)
    isempty(solvable) && return C

    # RHS: contributions from BC cells (the "known" concentrations)
    b_solvable = zeros(length(solvable))
    _subtract_bc_contributions!(b_solvable, solvable, A, bc_cells)

    A_sub = A[solvable, solvable]
    C_sub = A_sub \ b_solvable
    C[solvable] .= clamp.(C_sub, 0.0, 1.0)

    for (c, val) in bc_cells
        C[c] = clamp(val, 0.0, 1.0)
    end

    return C
end

"""
    _subtract_bc_contributions!(b, solvable_idx, A, bc_cells)

Subtract from `b` the contributions of BC columns of `A` to the solvable rows.
Exploits the CSC storage of `A` for efficient column access.

For each BC cell `c` with value `val`, updates
    b[local_row] -= A[solvable_idx[local_row], c] * val
"""
function _subtract_bc_contributions!(
        b::AbstractVector,
        solvable_idx::AbstractVector{Int},
        A::SparseMatrixCSC,
        bc_cells::Dict{Int, Float64}
    )
    # Build a reverse lookup: global row index → position in solvable_idx
    row_to_local = Dict{Int, Int}()
    for (local_i, global_row) in enumerate(solvable_idx)
        row_to_local[global_row] = local_i
    end

    rv = rowvals(A)
    nz = nonzeros(A)

    for (bc_col, val) in bc_cells
        abs(val) < 1e-14 && continue
        for k in nzrange(A, bc_col)
            global_row = rv[k]
            local_i    = get(row_to_local, global_row, 0)
            local_i == 0 && continue          # row not in solvable set
            b[local_i] -= nz[k] * val
        end
    end
end

# -------------------------------------------------------------------------
# Utility
# -------------------------------------------------------------------------

"""
    _find_solvable_cells(A, bc_cells, nc)

Return the sorted vector of cell indices that are neither BC cells nor
isolated (i.e. have a strictly positive diagonal entry in `A`).
"""
function _find_solvable_cells(
        A::SparseMatrixCSC,
        bc_cells::Dict{Int, Float64},
        nc::Int
    )
    solvable = Int[]
    for c in 1:nc
        haskey(bc_cells, c) && continue
        abs(A[c, c]) < 1e-14 && continue
        push!(solvable, c)
    end
    return solvable
end
