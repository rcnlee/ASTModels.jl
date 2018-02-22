module GridWorlds

using POMDPs, POMDPModels, POMDPToolbox, Distributions, DiscreteValueIteration, AdaptiveStressTesting2

export 
    GWParams,
    GWSim,
    GWState

const RS = [GridWorldState(10,3),
            GridWorldState(3,10), 
            GridWorldState(3,9), 
            GridWorldState(3,8),
            GridWorldState(1,3), 
            GridWorldState(10,6), 
            GridWorldState(9,6),
            GridWorldState(6,5),
            GridWorldState(6,4),
            GridWorldState(6,3),
            GridWorldState(5,3)]
const RV = [10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0]

const EVENTSTATES = RS[2:end]
const EVENTSTATES_VEC = [[s.x,s.y] for s in EVENTSTATES]
struct GWState
    t::Int
    x::GridWorldState
end
struct GWParams
    t_max::Int
end
GWParams() = GWParams(10)
mutable struct GWSim <: MDP{GWState,Void}
    p::GWParams
    x0::GridWorldState
    mdp::MDP
    policy::Policy
    t::Int
    x::GridWorldState
    md::Float64
end
GWSim(p::GWParams, mdp::MDP, policy::Policy, x0::GridWorldState) = GWSim(p, x0, mdp, policy, 0, x0, Inf)
function GWSim(x0=GridWorldState(3,1))
    mdp = GridWorld(rs=RS, rv=RV, terminals=Set(RS), discount_factor=0.95, tp=0.91)
    solver = ValueIterationSolver(max_iterations=100, belres=1e-3)
    policy = solve(solver, mdp)
    return GWSim(GWParams(), mdp, policy, x0)
end

POMDPs.actions(gw::GWSim) = [nothing]
function POMDPs.initial_state(sim::GWSim, rng::AbstractRNG)
    sim.t, sim.x, sim.md = 0, sim.x0, Inf
    GWState(sim.t, sim.x)
end
function POMDPs.generate_s(sim::GWSim, s::GWState, a::Void, rng::AbstractRNG)
    sim.t += 1
    sim.x = generate_s(sim.mdp, sim.x, action(sim.policy, sim.x), rng)
    ss = [sim.x.x, sim.x.y]
    md = Float64(minimum(manhattan(ss,e) for e in EVENTSTATES_VEC))
    sim.md = min(sim.md, md)
    GWState(sim.t, sim.x)
end
POMDPs.isterminal(sim::GWSim, s::GWState) = sim.x in RS || sim.t >= sim.p.t_max
POMDPs.reward(sim::GWSim, s::GWState, a::Void, sp::GWState) = 0.0

function Distributions.pdf(sim::GWSim, s::GWState, a::Void, sp::GWState)
    if s.x == sp.x && sp.done
        return 1.0
    end
    sp_dist = transition(sim.mdp, s.x, action(sim.policy,s.x))
    prob = pdf(sp_dist, sp.x)
    prob
end

Base.hash(sim::GWSim) = hash(sim.t, hash(sim.x))

AdaptiveStressTesting2.isevent(sim::GWSim, s::GWState) = s.x in EVENTSTATES
AdaptiveStressTesting2.miss_distance(sim::GWSim, s::GWState) = sim.md

manhattan(x,y) = sum(abs, x - y)



end #module
