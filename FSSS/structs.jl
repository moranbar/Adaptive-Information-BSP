
mutable struct TreeNode{B,A}
    parent::Int
    children::Vector{Int}

    total_n::Int
    belief::Vector{B}
    o::Vector{Float64}

    n::Int
    r::Float64
    q::Float64
    a_labels::Vector{A}
    open::Vector{Int64}
    likelihood::Vector{Float64}
    abs_likelihood::Any
    ub::Float64
    q_UB::Float64
    r_reuse::Float64
    a::Any
    simplified::Bool
    pdf_prop::Vector{Float64}
    openActions::Vector{Int64}
end
struct Tree{B,A}
    typeofB::DataType
    typeofA::DataType
    nodes::Vector{TreeNode}
end


@with_kw struct Adaptive
    c::Int64    # UCB exploration constant
    C::Int64    # number of observations
    InitBinSize::Int64   # initial bin size
    n::Int64    # number of iterations
    n_refine::Int64 # number of refine iterations
    kₒ::Int64   # observation widening hyper-param
    αₒ::Int64   # observation widening hyper-param
    kₐ::Int64   # action widening hyper-param
    αₐ::Int64   # action widening hyper-param
    ϵ::Float64  # long horizon cutoff
    Dmax::Int64 # maximum depth of solver
    PF::Any     # particle filter
end
