#include("structs.jl")

@with_kw mutable struct POMDPscenario
    F::Array{Float64, 2}   
    Σw::Array{Float64, 2}
    Σv::Array{Float64, 2}
    Dmax::Int64
    λ::Float64
    rng::MersenneTwister
    a_space::Array{Float64, 2}
    beacons::Array{Float64, 2}
    obstacles::Array{Float64, 2}
    d::Float64
    rmin::Float64
    obsRadii::Float64
    goalRadii::Float64
    goal::Array{Float64, 1}
    rewardGoal::Float64
    rewardObs::Float64
    γ::Float64
end


function update_observation_cov(pomdp, x)
    mindist = Inf
    for i in 1:length(pomdp.beacons[:,1])
        distance = norm(x - pomdp.beacons[i,:])
        if distance <= pomdp.d
            pomdp.Σv = Matrix(Diagonal([1, 1]))*0.01^2
            return pomdp.Σv
        elseif distance < mindist
            mindist = distance
        end
    end
    # if no beacon is near by, get noise meas.
    pomdp.Σv = Matrix(Diagonal([1, 1]))*0.1*mindist
    return pomdp.Σv
end

function dynamics(x::Array{Float64, 1}, a::Array{Float64, 1}, rng)
    global pomdp
    return SampleMotionModel(pomdp, a, x)
end

function pdfObservationModel(x_prev::Vector{Float64}, a::Vector{Float64}, x::Array{Float64, 1}, obs::Array{Float64, 1})
    global pomdp
    pomdp.Σv = update_observation_cov(pomdp, x)
    Nv = MvNormal([0, 0], pomdp.Σv)
    noise = obs - x
    return pdf(Nv, noise)
end

# input: belief at k, b(x_k), action a_k
# output: predicted gaussian belief b(xp)~N((μp,Σp)
function PropagateBelief(b::FullNormal, pomdp::POMDPscenario, a::Array{Float64, 1})::FullNormal
    μb, Σb = b.μ, b.Σ
    F  = pomdp.F
    Σw, Σv = pomdp.Σw, pomdp.Σv
    # calculations
    A = [ Σb^(-0.5) zeros(2,2); -Σw^(-0.5) Σw^(-0.5)]
    b = [ Σb^(-0.5)*μb; Σw^(-0.5)*a]
    # predict
    μp = inv(transpose(A)*A)*(transpose(A)*b)
    Σp = inv(transpose(A)*A) 
    μp = μp[3:4] # add your code here 
    Σp = Σp[3:4, 3:4] # add your code here 
    return MvNormal(μp, Σp)
end 

# input: belief at k, b(x_k), action a_k and observation z_k+1
# output: updated posterior gaussian belief b(x')~N(μb′, Σb′)
function PropagateUpdateBelief(b::FullNormal, pomdp::POMDPscenario, a::Array{Float64, 1}, o::Array{Float64, 1})::FullNormal
    μb, Σb = b.μ, b.Σ
    F  = pomdp.F
    Σw, Σv = pomdp.Σw, pomdp.Σv
    # calculations
    A = [ Σb^(-0.5) zeros(2,2); -Σw^(-0.5) Σw^(-0.5); zeros(2,2) Σv^(-0.5)]
    b = [ Σb^(-0.5)*μb; Σw^(-0.5)*a; Σv^(-0.5)*o ]
    # predict
    μp = inv(transpose(A)*A)*(transpose(A)*b)
    Σp = inv(transpose(A)*A)
    # update
    # marginalize
    μb′ = μp[3:4]
    Σb′ = Σp[3:4, 3:4]
    return MvNormal(μb′, Σb′)
end

# input: state x and action a
# output: next state x'
function SampleMotionModel(pomdp::POMDPscenario, a::Array{Float64, 1}, x::Array{Float64, 1})
    Nw = MvNormal([0, 0], pomdp.Σw) # multivariate gaussian
    w = rand(pomdp.rng, Nw)
    return x + a + w
end 

function pdfMotionModel(pomdp::POMDPscenario, a::Array{Float64, 1}, x::Array{Float64, 1}, x_prev::Array{Float64, 1})
    Nw = MvNormal([0, 0], pomdp.Σw)
    w = x - x_prev - a
    return pdf(Nw,w)
end 

# input: state x
# output: available observation z_rel and index, null otherwise
function GenerateObservationFromBeacons(pomdp::POMDPscenario, x::Array{Float64, 1}, fixed_cov::Bool)::Union{NamedTuple, Nothing}
    distances = zeros(length(pomdp.beacons[:,1]))
    for index in 1:length(pomdp.beacons[:,1])
        distances[index] = norm(x - pomdp.beacons[index, :]) # calculate distances from x to all beacons
    end
    index = argmin(distances) # get observation only from nearest beacon
    pomdp.Σv = update_observation_cov(pomdp, x)
    Nv = MvNormal([0, 0], pomdp.Σv)
    v = rand(pomdp.rng, Nv)
    dX = x - pomdp.beacons[index, :]
    obs = dX + v 
    return (obs=obs, index=index) 
end

function SampleObservation(p::Planner, x_propagated::Array{Float64, 1})
    o = GenerateObservationFromBeacons(p.pomdp, x_propagated, false)
    return o[1] + p.pomdp.beacons[o[2], :]
end

function likelihood(x::Vector{Float64},o::Vector{Float64})
    return pdfObservationModel([0.], [0.], x, o)
end

function initBelief()
    μ0 = [0.0,0.0]
    Σ0 = [1.0 0.0; 0.0 1.0]
    return MvNormal(μ0, Σ0)
end

initState() = [-0.5, -0.2]

function reward(p::Planner, b::FullNormal, x::Vector{Float64})
    x_g, x_o = p.pomdp.goal, p.pomdp.obstacles[1,:]
    rewardObs, rewardGoal = 0, 0
    n = length(b.μ) # dimension of state vector
    if norm(x - x_o, 2) < pomdp.obsRadii
        rewardObs = p.pomdp.rewardObs
    end
    if norm(x - x_g, 2) < pomdp.goalRadii
        rewardGoal = p.pomdp.rewardGoal
    end
    return -(norm(b.μ-x_g,2)+p.pomdp.λ*0.5*log((2* π *exp(1))^n * det(b.Σ))) + rewardObs + rewardGoal
end


function exactReward(p, b, ba, a, ba_id)
    # for debug porpuses only, w/o obstacles
    r_belief, eta = 0, 0
    for bao_id in p.tree.nodes[ba_id].children
        p_z_x = p.tree.nodes[bao_id].likelihood
        (unnorm_r, normalizer) = rewardBelief(p, b, ba, a, p_z_x)
        r_belief += unnorm_r
        eta += normalizer
    end
    r_belief = r_belief / eta
    r = rewardState(p, b) + p.pomdp.λ*r_belief
    return r
end


function oneStepSim(p::Planner, b::FullNormal, x_prev::Array{Float64, 1}, a::Array{Float64, 1})
    # create GT Trajectory, update horizon
    p.pomdp.Dmax -= 1

    b_prop = PropagateBelief(b, p.pomdp, a)
    x = SampleMotionModel(p.pomdp, a, x_prev)
    o_rel = GenerateObservationFromBeacons(p.pomdp, x, false)
    if o_rel === nothing
        b_post = b_prop
    else
        o = o_rel[1] + p.pomdp.beacons[o_rel[2], :]
        # update Cov. according to distance from beacon
        update_observation_cov(p.pomdp, x)
        b_post = PropagateUpdateBelief(b_prop, p.pomdp, a, o)
        r = reward(p, b_post, x)
    end

    return b_post, r, x, o
end

function BeaconsWorld2D(rng)
    d = 1.0 
    rmin = 0.1
    # set beacons locations 
    beacons = [0.0 0.0; 
               #2.0 0.0; 
               4.0 0.0;
               #6.0 0.0;
               8.0 0.0;
               #10.0 0.0;
               10.0 2.0;
               #10.0 4.0;
               10.0 6.0;
               #10.0 8.0;
               10.0 10.0;]
    obstacles = [10.0 * rand(rng,1) 5.0;
                 10.0 * rand(rng,1) 3.0;
                 10.0 * rand(rng,1) 9.0;]
    goal = [10, 10]
    a_space = [1.0  0.0;
              -1.0  0.0;
               0.0  1.0;
               0.0 -1.0;
               1/sqrt(2)  1/sqrt(2);
              -1/sqrt(2)  1/sqrt(2);
               1/sqrt(2) -1/sqrt(2);
              -1/sqrt(2) -1/sqrt(2);
               0.0  0.0             ]
    pomdp = POMDPscenario(F=[1.0 0.0; 0.0 1.0],
                        Σw=0.1*[1.0 0.0; 0.0 1.0],
                        Σv=0.01*[1.0 0.0; 0.0 1.0], 
                        γ=0.99,
                        Dmax = 25,
                        λ = 1,
                        obsRadii = 1.5,
                        goalRadii = 1.,
                        goal = goal,
                        rewardGoal = 10,
                        rewardObs = -10,
                        rng = rng, a_space=a_space, beacons=beacons, 
                        obstacles=obstacles, d=d, rmin=rmin) 
    global pomdp

    return pomdp
end