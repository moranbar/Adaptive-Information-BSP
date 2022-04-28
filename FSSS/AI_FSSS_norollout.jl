
Tree(B,A) = Tree{B,A}(B, A, [TreeNode{B,A}(0, Vector{Int}(), 0, Vector{}(), Vector{Float64}(), 0, NaN, NaN, Vector{}(), Vector{Int}(), Vector{Float64}(), Any, 0, NaN, NaN, Any, false, Vector{Float64}(), Vector{Int64}())])

function addchild(tree::Tree, id::Int)
    1 <= id <= length(tree.nodes) || throw(BoundsError(tree, id))
    B = tree.typeofB
    A = tree.typeofA
    push!(tree.nodes, TreeNode{B,A}(0, Vector{Int}(), 0, Vector{}(), Vector{Float64}(), 0, NaN, NaN, Vector{}(), Vector{Int}(), Vector{Float64}(), Any, 0, NaN, NaN, Any, false, Vector{Float64}(), Vector{Int64}()))
    child = length(tree.nodes)
    push!(tree.nodes[id].children, child)
    return child
end

function rewardState(p, b)
    x_g = p.pomdp.goal
    x_o = p.pomdp.obstacles[1,:]
    r_b_x, rewardObs, rewardGoal = 0, 0, 0
    b = particles(b)
    for x in b
        r_b_x += norm(x - x_g,2) / length(b)
        for i in 1:length(p.pomdp.obstacles[:,1])
            x_o = p.pomdp.obstacles[i,:]
            if norm(x - x_o,2) < p.pomdp.obsRadii
                rewardObs += p.pomdp.rewardObs / length(b)
            end
        end
        if norm(x - x_g,2) < p.pomdp.goalRadii
            rewardGoal += p.pomdp.rewardGoal / length(b)
        end
    end
    return -r_b_x + rewardObs + rewardGoal
end

function rewardBelief(p, b, ba, a, weights)
    return expected_entropy(ba, weights, a, b) 
end

function estReward(p, b, ba, p_z_x, a, binSize)
    r_belief, eta = 0, 0
    for bin in 1:Int(p.solver.C / binSize)
        (unnorm_r, normalizer) = rewardBelief(p, b, ba, a, p_z_x[bin*binSize,:])
        r_belief += unnorm_r
        eta += normalizer
    end
    r_belief = r_belief / eta
    r = rewardState(p, b) + p.pomdp.λ*r_belief 
    return (r, r_belief)
end

function ReuseReward(p, b, r_belief)
    return rewardState(p, b) + p.pomdp.λ*r_belief
end


function abstractObservations(p, ba_id, binSize)
    # calculate abstract observation likelihood values
    p_z_x_sum, p̄_z̄_x_sum = zeros(p.solver.C), zeros(p.solver.C)
    n_bins = Int(p.solver.C / binSize)
    p.tree.nodes[ba_id].abs_likelihood = zeros(p.solver.C, length(particles(p.tree.nodes[ba_id].belief[1]))) # initialize
    for b in 0:(n_bins-1)
        p̄_z̄_x = zeros(length(particles(p.tree.nodes[ba_id].belief[1])))
        for k in 1:binSize
            bao_id = p.tree.nodes[ba_id].children[b*binSize + k]
            p.tree.nodes[bao_id].likelihood = []
            # make abstract likelihood
            for (i, x) in enumerate(particles(p.tree.nodes[ba_id].belief[1]))
                p_z_x = likelihood(x, p.tree.nodes[bao_id].o)
                append!(p.tree.nodes[bao_id].likelihood, p_z_x)
                p̄_z̄_x[i] += p_z_x / binSize; 
                p_z_x_sum[b*binSize + k] += p_z_x
            end
        end
        # equal likelihood for each bin
        p̄_z̄_x_sum[b*binSize + 1: (b + 1)*binSize] = sum(p̄_z̄_x)*ones((b + 1)*binSize - b*binSize)
        p.tree.nodes[ba_id].abs_likelihood[b*binSize + 1: (b + 1)*binSize, :] = p̄_z̄_x'.*ones((b + 1)*binSize - b*binSize)
    end
    return p.tree.nodes[ba_id].abs_likelihood
end


function selectAction(p, b_id)
    # least visited subtree
    Nmin = p.tree.nodes[p.tree.nodes[b_id].children[1]].n
    ba_id_best = p.tree.nodes[b_id].children[1]
    a_best = p.tree.nodes[ba_id_best].a[2] # p.pomdp.a_space[1,:]
    for (i, ba_id) in enumerate(p.tree.nodes[b_id].children)
        if p.tree.nodes[ba_id].n < Nmin
            ba_id_best = ba_id
            a_best = p.tree.nodes[ba_id].a[2] # p.pomdp.a_space[i,:]
        end
    end 
    return (a_best, ba_id_best)
end

function maxUB(p, b_id)
    ub, a_best, ba_id_best = -Inf, [0.], 0
    for (a_id, ba_id) in enumerate(p.tree.nodes[b_id].children)
        if ub < p.tree.nodes[ba_id].q_UB
            ub = p.tree.nodes[ba_id].q_UB
            ba_id_best = ba_id
            a_best = p.tree.nodes[ba_id].a[2] 
        end
    end
    return a_best, ba_id_best
end

function simplify(p::Planner, b_id::Int64, ba_id::Int64, a::Vector{Float64}, d::Int64)
    if d == 0 #|| ba_id == 0
        return 0, 0
    elseif ba_id == 0 # leaf node due to heuristic mid-tree
        return p.tree.nodes[b_id].q, p.tree.nodes[b_id].q_UB
    end
    binSize = 1

    if p.tree.nodes[ba_id].simplified == true
        p_z_x = p.tree.nodes[ba_id].abs_likelihood
        r = p.tree.nodes[ba_id].r
    else
        p_z_x = abstractObservations(p, ba_id, binSize)
        (r, r_belief) = estReward( p, p.tree.nodes[b_id].belief[1],
                                p.tree.nodes[ba_id].belief[1],
                                p_z_x, a, binSize)
        p.tree.nodes[ba_id].r_reuse = r_belief  # to be reused across obs siblings
        p.tree.nodes[ba_id].simplified = true   # only true when binSize=1
    end
    r_UB = r + p.pomdp.λ * log(binSize)

    # only traverse existing ba / bao nodes
    # fast way to get all visited nodes
    bao_visited = p.tree.nodes[ba_id].children[length(p.tree.nodes[ba_id].open)+1:end]


    # choose max gap observation, otherwise choose randomly
    max_gap, bao_id = -1., rand(bao_visited)
    for bao in bao_visited
        gap = p.tree.nodes[bao].q_UB - p.tree.nodes[bao].q
        if gap > max_gap
            max_gap = gap
            bao_id = bao
        end
    end

    # choose best action 
    if d != 1
        # (baoa, baoa_id) = p.tree.nodes[bao_id].a
        (baoa, baoa_id) = maxUB(p, bao_id)
    else # leaf node
        baoa, baoa_id = [0.], 0
    end

    # update Q(b,a) values
    r_old = p.tree.nodes[ba_id].r
    r_UB_old = r_old + p.pomdp.λ * p.tree.nodes[ba_id].ub
    n_bao = p.solver.C - length(p.tree.nodes[ba_id].open)


    # delete old reward and the observation subtree (both updated)
    if p.tree.nodes[bao_id].q == -Inf # whenever bao is leaf node, w/o heuristic
        bao_values = n_bao*(p.tree.nodes[ba_id].q - r_old)
        bao_UB = n_bao*(p.tree.nodes[ba_id].q_UB - r_UB_old)
    else
        bao_values = n_bao*(p.tree.nodes[ba_id].q - r_old) - pomdp.γ*p.tree.nodes[bao_id].q
        bao_UB = n_bao*(p.tree.nodes[ba_id].q_UB - r_UB_old) - pomdp.γ*p.tree.nodes[bao_id].q_UB    
    end

    # add new reward and subtree
    v_next, v_UB = simplify(p, bao_id, baoa_id, baoa, d-1)
    p.tree.nodes[ba_id].q = r + (pomdp.γ*v_next + bao_values) / n_bao
    p.tree.nodes[ba_id].q_UB = r_UB + (pomdp.γ*v_UB + bao_UB) / n_bao
    p.tree.nodes[ba_id].r = r
    p.tree.nodes[ba_id].ub = log(binSize)

    if p.tree.nodes[b_id].a[2] == ba_id
        p.tree.nodes[b_id].q = p.tree.nodes[ba_id].q
        p.tree.nodes[b_id].q_UB = p.tree.nodes[ba_id].q_UB
        p.tree.nodes[b_id].a = (a, ba_id)
    end
    # V(b) = argmax Q(b,a)
    for ba in p.tree.nodes[b_id].children
        if p.tree.nodes[ba].q_UB > p.tree.nodes[b_id].q_UB
            #p.tree.nodes[b_id].q = p.tree.nodes[ba].q
            p.tree.nodes[b_id].q_UB = p.tree.nodes[ba].q_UB
            p.tree.nodes[b_id].a = (a, ba)
        end
        if p.tree.nodes[ba].q > p.tree.nodes[b_id].q
            p.tree.nodes[b_id].q = p.tree.nodes[ba].q
        end
    end

    p.tree.nodes[b_id].n  += 0
    p.tree.nodes[ba_id].n += 0


    return p.tree.nodes[b_id].q, p.tree.nodes[b_id].q_UB
end

function optimalAction(p::Planner)
    epsilon = 10^(-8)
    max_simplification, first_iter = false, true
    ub, lb = -Inf, Inf
    ub⁻, lb⁻ = -Inf, Inf
    a_best, ba_id_best, ba_id⁻, a⁻, a_id_best = nothing, nothing, nothing, nothing, nothing
    
    while (lb < ub⁻ && !max_simplification) || first_iter
        if !first_iter && p.solver.InitBinSize != 1
            simplify(p, 1, ba_id_best, a_best, p.solver.Dmax)
            simplify(p, 1, ba_id⁻, a⁻, p.solver.Dmax)
            
            ub = p.tree.nodes[ba_id_best].q_UB
            lb = p.tree.nodes[ba_id_best].q
            ub⁻ = p.tree.nodes[ba_id⁻].q_UB
        end
        a_best, ba_id_best, ba_id⁻, a⁻, a_id_best = nothing, nothing, nothing, nothing, nothing
        ub, lb = -Inf, Inf
        ub⁻, lb⁻ = -Inf, Inf
        # can simplify by doing bookkeeping in planning/refine instead of another for-loop 
        for (a_id, ba_id) in enumerate(p.tree.nodes[1].children)
            if p.tree.nodes[ba_id].q_UB > ub
                ub = p.tree.nodes[ba_id].q_UB
                lb = p.tree.nodes[ba_id].q
                a_best = p.tree.nodes[ba_id].a[2]
                ba_id_best = ba_id
                a_id_best = p.tree.nodes[ba_id].a[1]
            end
        end
        for (a_id, ba_id) in enumerate(p.tree.nodes[1].children)
            if p.tree.nodes[ba_id].q_UB > ub⁻ && ba_id != ba_id_best
                ub⁻ = p.tree.nodes[ba_id].q_UB
                lb⁻ = p.tree.nodes[ba_id].q
                a⁻ = p.tree.nodes[ba_id].a[2]
                ba_id⁻ = ba_id
            end
        end
        current = peektimer()
        if (abs(lb - ub) < epsilon && abs(lb⁻ - ub⁻) < epsilon) || p.solver.InitBinSize == 1 || current > Inf
            max_simplification = true
        end
        first_iter = false
    end
    return (a_best, a_id_best, ba_id_best)
end

function createMyopicTree(p::Planner, b_id::Int64)
    rng = p.pomdp.rng

    ba_id = addchild(p.tree, b_id)
    a_id = pop!(p.tree.nodes[b_id].openActions)
    a = p.pomdp.a_space[a_id,:]
    p.tree.nodes[ba_id].a = (a_id, a)
    bp = predict(p.solver.PF, p.tree.nodes[b_id].belief[1], a, rng)
    p.tree.nodes[ba_id].belief = [ParticleCollection(bp)]

    # create all observations
    for c in 1:p.solver.C
        bao_id = addchild(p.tree, ba_id)
        p.tree.nodes[bao_id].openActions = shuffle(rng, 1:length(p.pomdp.a_space[:,1]))
        p.tree.nodes[bao_id].q = -Inf
        append!(p.tree.nodes[ba_id].open, bao_id)
        x = rand(rng, bp)
        o = SampleObservation(p, x)
        append!(p.tree.nodes[bao_id].o, o); 
    end
    return (a, ba_id)
end

function simulate(p::Planner,b_id::Int,d::Int)
    if d == 0
        #p.tree.nodes[b_id].q = 0
        #p.tree.nodes[b_id].q_UB = 0
        return 0, 0
    end

    a = nothing
    # create hollow subtree
    if length(p.tree.nodes[b_id].children) < length(p.pomdp.a_space[:,1])
        (a, ba_id) = createMyopicTree(p, b_id)
    else
        (a, ba_id) = selectAction(p, b_id)
    end

    # update Q(b,a) values
    if !isempty(p.tree.nodes[ba_id].open)
        bao_id = pop!(p.tree.nodes[ba_id].open)
        n_bao = p.solver.C - length(p.tree.nodes[ba_id].open)
        if isnan(p.tree.nodes[ba_id].r_reuse)
            p_z_x = abstractObservations(p, ba_id, p.solver.InitBinSize)
            (r, r_belief) = estReward( p, p.tree.nodes[b_id].belief[1],
                        p.tree.nodes[ba_id].belief[1],
                        p_z_x, a, p.solver.InitBinSize)
            p.tree.nodes[ba_id].r_reuse = r_belief  # to be reused across obs siblings
            p.tree.nodes[ba_id].ub = log(p.solver.InitBinSize)
        else 
            r = ReuseReward(p, p.tree.nodes[b_id].belief[1], p.tree.nodes[ba_id].r_reuse) 
        end

        p.tree.nodes[ba_id].r = r
        r_UB = r + p.pomdp.λ * p.tree.nodes[ba_id].ub
        p.tree.nodes[bao_id].belief = [posterior(p, 
        p.tree.nodes[b_id].belief[1], a, 
        p.tree.nodes[ba_id].belief[1],
        p.tree.nodes[bao_id].o)]
        bao_values, bao_UB = 0, 0
        if n_bao-1 > 0
            bao_values = (n_bao - 1)*(p.tree.nodes[ba_id].q - r)
            bao_UB = (n_bao - 1)*(p.tree.nodes[ba_id].q_UB - r_UB)
        end 
        v_next, v_UB = simulate(p, bao_id, d-1)
        p.tree.nodes[ba_id].q = r + (p.pomdp.γ*v_next + bao_values) / n_bao
        p.tree.nodes[ba_id].q_UB = r_UB + (p.pomdp.γ*v_UB + bao_UB) / n_bao

    else
        r = p.tree.nodes[ba_id].r
        r_UB = r + p.pomdp.λ * p.tree.nodes[ba_id].ub
        bao_id = rand(p.pomdp.rng, p.tree.nodes[ba_id].children)
        bao_values = p.solver.C*(p.tree.nodes[ba_id].q - r) - pomdp.γ*p.tree.nodes[bao_id].q
        bao_UB = p.solver.C*(p.tree.nodes[ba_id].q_UB - r_UB) - pomdp.γ*p.tree.nodes[bao_id].q_UB
        v_next, v_UB = simulate(p, bao_id, d-1)
        p.tree.nodes[ba_id].q = r + (pomdp.γ*v_next + bao_values) / p.solver.C
        p.tree.nodes[ba_id].q_UB = r_UB + (pomdp.γ*v_UB + bao_UB) / p.solver.C
    end

    if p.tree.nodes[b_id].a[2] == ba_id || p.tree.nodes[b_id].n <= 1
        p.tree.nodes[b_id].q = p.tree.nodes[ba_id].q
        p.tree.nodes[b_id].q_UB = p.tree.nodes[ba_id].q_UB
        p.tree.nodes[b_id].a = (a, ba_id)
    end
    # V(b) = argmax Q(b,a)
    for ba in p.tree.nodes[b_id].children
        if p.tree.nodes[ba].q > p.tree.nodes[b_id].q
            #p.tree.nodes[b_id].q = p.tree.nodes[ba].q
            p.tree.nodes[b_id].q_UB = p.tree.nodes[ba].q_UB
            p.tree.nodes[b_id].a = (a, ba)
        end
        if p.tree.nodes[ba].q > p.tree.nodes[b_id].q
            p.tree.nodes[b_id].q = p.tree.nodes[ba].q
        end
    end

    p.tree.nodes[b_id].n  += 1
    p.tree.nodes[ba_id].n += 1

    return p.tree.nodes[b_id].q, p.tree.nodes[b_id].q_UB
end

function Solver(B, A, pf::Any)
    solver = Adaptive(c = 1, C = 4, InitBinSize = 4, n = 50000, n_refine = 10, kₒ = 1, αₒ = 1, 
                        kₐ = 100, αₐ = 1, ϵ=0.01, Dmax = 10, PF = pf)
    
    # define tree and tree types
    tree = Tree(B,A)
    return (solver, tree)
end

function Solve(planner::Planner, B, A, b_PF)

        # initialize tree
        planner.tree = Tree(B,A)
        root_id = 1
        planner.tree.nodes[root_id].q = -Inf
        planner.tree.nodes[root_id].q_UB = -Inf
        planner.tree.nodes[root_id].belief = [b_PF[end]]
        planner.tree.nodes[root_id].openActions = shuffle(planner.pomdp.rng, 1:length(planner.pomdp.a_space[:,1]))

        tick()
        for i in 1:planner.solver.n
            simulate(planner, root_id, planner.solver.Dmax)
            current = peektimer()
            if current > 1.
                printstyled("completed $i iterations\n", color= :green)
                break
            end
        end
        tock()

        # refine solution
        println("refine . . . . . . . . . . . .")
        tick()
        (a_best, a_id_best, ba_id_best) = optimalAction(planner)
        tock()
    return (a_best, a_id_best, ba_id_best)
end