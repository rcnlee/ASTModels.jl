module Walk1Ds

using POMDPs, Distributions, AdaptiveStressTesting2

export 
    Walk1DParams,
    Walk1D,
    Walk1DState,
    Walk1DAction

struct Walk1DParams
    startx::Float64
    threshx::Float64
    t_max::Int
end
Walk1DParams() = Walk1DParams(1.0, 10.0, 20)

struct Walk1DState
    t::Int
    x::Float64
end
mutable struct Walk1D <: MDP{Walk1DState,Void}
    p::Walk1DParams
    distr::Distribution
    t::Int
    x::Float64
end
Walk1D(p::Walk1DParams, distr::Distribution) = Walk1D(p, distr, 0, 0.0)
Walk1D() = Walk1D(Walk1DParams(), Normal(0.0, 1.0))

POMDPs.actions(mdp::Walk1D) = [nothing]
function POMDPs.initial_state(mdp::Walk1D, rng::AbstractRNG)
    mdp.t, mdp.x = 0, mdp.p.startx
    Walk1DState(mdp.t, mdp.x)
end
function POMDPs.generate_s(mdp::Walk1D, s::Walk1DState, a::Void, rng::AbstractRNG)
    mdp.t += 1
    mdp.x += rand(rng, mdp.distr)
    Walk1DState(mdp.t, mdp.x)
end
POMDPs.isterminal(mdp::Walk1D, s::Walk1DState) = isevent(mdp, s) || mdp.t >= mdp.p.t_max
POMDPs.reward(mdp::Walk1D, s::Walk1DState, a::Void, sp::Walk1DState) = 0.0

AdaptiveStressTesting2.isevent(mdp::Walk1D, s::Walk1DState) = mdp.x > mdp.p.threshx
AdaptiveStressTesting2.state_distance(mdp::Walk1D, s1::Walk1DState, s2::Walk1DState) = abs(s1.x-s2.x)
function AdaptiveStressTesting2.miss_distance(mdp::Walk1D, s::Walk1DState)
    max(mdp.p.threshx-abs(mdp.x), 0.0)
end

Base.hash(mdp::Walk1D) = hash(mdp.t, hash(mdp.x))

function Distributions.pdf(mdp::Walk1D, s::Walk1DState, a::Void, sp::Walk1DState)
    pdf(mdp.distr, sp.x-s.x)
end


end #module
