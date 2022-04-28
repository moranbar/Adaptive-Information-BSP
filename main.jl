using Base: Int64, debug_color
using Revise
using Distributions
using Random
using LinearAlgebra
using StatsPlots
using Parameters
using TickTock
using ParticleFilters

sol = "AI_FSSS" # AI_FSSS, AI_FSSS_norollout

function main()
    rng = MersenneTwister(1)


    Nstatistics = 50  # number of final solutions for statistics
    V_approx, cumulatives, iter_time = 0, [], []
    tick()
    for i in 1:Nstatistics

    pomdp = BeaconsWorld2D(rng)

    # create particle filter
    model = ParticleFilterModel{Vector{Float64}}(dynamics, pdfObservationModel)
    N = 20 # number of particles
    pf = BootstrapFilter(model, N, rng)

    # init trajectory
    x_gt  = [initState()]
    b_gt  = [initBelief()]
    b_PF = [ParticleCollection([rand(pomdp.rng, b_gt[1]) for i in 1:N])]
    r_gt, o_gt, policy = [], [], []

    # define solver hyper params
    B = typeof(b_PF[end])
    A = typeof(pomdp.a_space[1,:])
    (solver, tree) = Solver(B, A, pf)

    # plan
    planner = Planner(solver, pomdp, tree)
    a_best, ba_id_best = nothing, nothing

    tick()
    for t in 1:(planner.pomdp.Dmax-1)
        # solve using planner
        (a_best, a_id_best, ba_id_best) = Solve(planner, B, A, b_PF)

        # perform action, get ground truth
        push!(policy, planner.pomdp.a_space[a_id_best,:])
        (b_step, r_step, x_step, o_step) = oneStepSim(planner, b_gt[end], x_gt[end], policy[end])
        push!(b_gt, b_step)
        push!(x_gt, x_step)
        push!(r_gt, r_step)
        push!(o_gt , o_step)

        b_post = ParticleBeliefMDP(b_PF[end], planner.pomdp, policy[end], planner.solver.PF, o_gt[end])
        push!(b_PF, b_post)
    end
    current = peektimer()
    append!(iter_time, current)
    tock()

    summ = sum(r_gt)
    append!(cumulatives, summ)
    V_approx  += sum(r_gt) / Nstatistics
    printstyled("∑rₜ of abstract, optimized tree: $summ\n, iter: $i", color = :yellow)
    runtime = peektimer()
    printstyled("(Current V approx - selfcheck: $V_approx\n, running time: $runtime)", color = :yellow)


    visualizeTrajectory(planner, x_gt, b_gt, b_PF)
    end
    tock()
    sigma2 = sqrt(sum(V_approx.^2) - (V_approx^2/Nstatistics)) / Nstatistics
    sigma = sqrt(sum((cumulatives .- V_approx).^2) / (Nstatistics*(Nstatistics-1)) )
    printstyled("Eᵢ[∑rₜ] of abstract, optimized tree:     $V_approx,  std: +- $sigma, std_old: +- $sigma2\n"; color = :blue)
    println("time per iteration: $iter_time")
end


include("SharedStructs.jl")
include("BeaconsWorld2D.jl")
include("BeaconsWorld2DnoObstacles.jl")
include("utils.jl")
include("visualize.jl")
if sol == "AI_FSSS_norollout"
    include("FSSS/structs.jl")
    include("FSSS/AI_FSSS_norollout.jl")
elseif sol == "AI_FSSS"
    include("AI_FSSS/structs.jl") 
    include("AI_FSSS/AI_FSSS.jl")
end

main()