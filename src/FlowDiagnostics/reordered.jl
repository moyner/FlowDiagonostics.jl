"""
Optional graph ordered solve. Edges follow the upwind matrix from an upstream
column to a downstream row. Tarjan's algorithm groups recirculating cells;
acyclic cells use scalar division, while each cyclic component is factorized
once. Components are then solved in flow order.
"""
struct ReorderedFactor
    blocks::Vector{Vector{Int}}
    factors::Vector{Any}
    outgoing::Vector{Vector{Pair{Int, Float64}}}
    block_of::Vector{Int}
end

function strongly_connected_components(adjacency)
    count = length(adjacency)
    discovery = zeros(Int, count)
    lowlink = zeros(Int, count)
    on_stack = falses(count)
    stack = Int[]
    components = Vector{Vector{Int}}()
    frame_vertex = Int[]
    frame_next = Int[]
    index = 0
    for root in 1:count
        discovery[root] != 0 && continue
        index += 1
        discovery[root] = index
        lowlink[root] = index
        push!(stack, root)
        on_stack[root] = true
        push!(frame_vertex, root)
        push!(frame_next, 1)
        while !isempty(frame_vertex)
            vertex = last(frame_vertex)
            next_edge = last(frame_next)
            if next_edge <= length(adjacency[vertex])
                neighbor = adjacency[vertex][next_edge]
                frame_next[end] += 1
                if discovery[neighbor] == 0
                    index += 1
                    discovery[neighbor] = index
                    lowlink[neighbor] = index
                    push!(stack, neighbor)
                    on_stack[neighbor] = true
                    push!(frame_vertex, neighbor)
                    push!(frame_next, 1)
                elseif on_stack[neighbor]
                    lowlink[vertex] = min(lowlink[vertex], discovery[neighbor])
                end
            else
                pop!(frame_vertex)
                pop!(frame_next)
                if lowlink[vertex] == discovery[vertex]
                    block = Int[]
                    while true
                        member = pop!(stack)
                        on_stack[member] = false
                        push!(block, member)
                        member == vertex && break
                    end
                    push!(components, block)
                end
                if !isempty(frame_vertex)
                    parent = last(frame_vertex)
                    lowlink[parent] = min(lowlink[parent], lowlink[vertex])
                end
            end
        end
    end
    reverse!(components)
    return components
end

function prepare_reordered(matrix, solvable, local_index)
    count = length(solvable)
    adjacency = [Int[] for _ in 1:count]
    outgoing = [Pair{Int, Float64}[] for _ in 1:count]
    rows = rowvals(matrix)
    values = nonzeros(matrix)
    for (column_local, column) in enumerate(solvable)
        for entry in nzrange(matrix, column)
            row_local = local_index[rows[entry]]
            if row_local != 0 && row_local != column_local && values[entry] != 0
                push!(adjacency[column_local], row_local)
                push!(outgoing[column_local], row_local => values[entry])
            end
        end
    end
    blocks = strongly_connected_components(adjacency)
    factors = Any[]
    block_of = zeros(Int, count)
    for (block_index, block) in enumerate(blocks)
        for member in block
            block_of[member] = block_index
        end
        if length(block) == 1
            cell = solvable[only(block)]
            push!(factors, matrix[cell, cell])
        else
            cells = solvable[block]
            push!(factors, lu(matrix[cells, cells]))
        end
    end
    return ReorderedFactor(blocks, factors, outgoing, block_of)
end

function solve_reduced(direction::PreparedDirection{ReorderedFactor}, rhs)
    factor = direction.factorization
    values = zeros(length(rhs))
    work = copy(rhs)
    for (block_index, block) in enumerate(factor.blocks)
        if length(block) == 1
            member = only(block)
            values[member] = work[member] / factor.factors[block_index]
        else
            values[block] = factor.factors[block_index] \ work[block]
        end
        for member in block
            for (downstream, coefficient) in factor.outgoing[member]
                if factor.block_of[downstream] != block_index
                    work[downstream] -= coefficient * values[member]
                end
            end
        end
    end
    return values
end
