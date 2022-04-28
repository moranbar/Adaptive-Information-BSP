
# propegate belief
function ParticleBeliefMDP(b::ParticleCollection, pomdp::POMDPscenario, a::Array{Float64, 1}, pf, o::Array{Float64, 1})
    bp = predict(pf, b, a, pomdp.rng)
    b_post = update(pf, b, a, o)
    return b_post
end

function bores_entropy(ba::ParticleCollection, likelihood::Array{Float64}, 
    a::Array{Float64, 1}, b::ParticleCollection)
    summ = sum(likelihood)
    if summ != 0.
        bao_weights = likelihood ./ sum(likelihood)
    else
        bao_weights = likelihood 
    end
    normalizer, nominator = 0, 0
    N = length(particles(ba))
    M = length(particles(b))
    x_prev = [0.]
    for i in 1:N
        x = particles(ba)[i]
        pdf_prop = 0.
        for j in 1:M
            x_prev = particles(b)[j]
            pdf_prop += pdfMotionModel(pomdp, a, x, x_prev) / M
        end
        if likelihood[i] > eps(10^-100) # avoid rounding errors within log
            nominator += log(likelihood[i] * pdf_prop) * bao_weights[i]
            normalizer += likelihood[i] / N
        end
    end
    normalizer = log(normalizer)
    return normalizer - nominator
end

function expected_entropy(ba::ParticleCollection, likelihood::Array{Float64}, 
    a::Array{Float64, 1}, b::ParticleCollection)
    N = length(particles(ba))
    M = length(particles(b))
    pdf_prop = 0
    bao_weights = likelihood./N # Divide by N to get posterior weight
    denominator, nominator = 0, 0

    x_prev = [0.]
    for i in 1:N
        x = particles(ba)[i]
        pdf_prop = 0.
        for j in 1:M
            x_prev = particles(b)[j]
            pdf_prop += pdfMotionModel(pomdp, a, x, x_prev) / M
        end
        if likelihood[i] > eps(10^-100) # avoid rounding errors within log
            nominator += log(likelihood[i] * pdf_prop) * bao_weights[i]
            denominator += likelihood[i] / N
        end
    end
    denominator = sum(bao_weights)*log(denominator) 
    unnormalized_entropy = denominator - nominator
    normalizer = sum(bao_weights)
    return -unnormalized_entropy, normalizer
end

function posterior(p, b, a, bp, o)
    weights = reweight(p.solver.PF, b, a, particles(bp), o)
    bw = WeightedParticleBelief(particles(bp), weights, sum(weights), nothing)
    posterior = resample(LowVarianceResampler(
        length(particles(b))), bw, p.solver.PF.predict_model.f,
        p.solver.PF.reweight_model.g, 
        b, a, o, p.pomdp.rng) 
    return posterior
end

function posterior(p, b, a)
    bp = predict(p.solver.PF, b, a, p.pomdp.rng)
    x = rand(p.pomdp.rng, bp)
    o = SampleObservation(p, x)
    weights = reweight(p.solver.PF, b, a, bp, o)
    bw = WeightedParticleBelief(bp, weights, sum(weights), nothing)
    posterior = resample(LowVarianceResampler(
        length(particles(b))), bw, p.solver.PF.predict_model.f,
        p.solver.PF.reweight_model.g, 
        b, a, o, p.pomdp.rng)
    return (posterior, bp, o)
end




