using ResumableFunctions, ConcurrentSim, Distributions, DataFrames, Random

function increment!(a::Array{Int64}); push!(a, a[end] + 1); end
function decrement!(a::Array{Int64}); push!(a, a[end] - 1); end
function carryover!(a::Array{Int64}); push!(a, a[end]); end

mutable struct SIRPerson
    id::Int64
    status::Symbol   # :S, :I, :R, :D
end

mutable struct SIRModel
    sim::ConcurrentSim.Simulation
    β::Float64
    c::Float64
    γ::Float64
    μ::Float64
    ν::Float64
    deterministic_recovery::Bool
    ta::Array{Float64}
    Sa::Array{Int64}
    Ia::Array{Int64}
    Ra::Array{Int64}
    allIndividuals::Array{SIRPerson}
end

function infection_update!(sim, m)
    push!(m.ta, ConcurrentSim.now(sim))
    decrement!(m.Sa); increment!(m.Ia); carryover!(m.Ra)
end

function recovery_update!(sim, m)
    push!(m.ta, ConcurrentSim.now(sim))
    carryover!(m.Sa); decrement!(m.Ia); increment!(m.Ra)
end

function death_update!(sim, m, st::Symbol)
    push!(m.ta, ConcurrentSim.now(sim))
    st === :S && (decrement!(m.Sa); carryover!(m.Ia); carryover!(m.Ra))
    st === :I && (carryover!(m.Sa); decrement!(m.Ia); carryover!(m.Ra))
    st === :R && (carryover!(m.Sa); carryover!(m.Ia); decrement!(m.Ra))
end

function birth_update!(sim, m)
    push!(m.ta, ConcurrentSim.now(sim))
    increment!(m.Sa); carryover!(m.Ia); carryover!(m.Ra)
end

# --- ВАКЦИНАЦИЯ ---
@resumable function vaccinate(env, m::SIRModel, t_vac::Float64, fraction::Float64)
    @yield timeout(env, t_vac)
    target = floor(Int, fraction * length(m.allIndividuals))
    vacc = 0
    for ind in m.allIndividuals
        ind.status === :S || continue
        ind.status = :R
        vacc += 1
        vacc >= target && break
    end
    # Одно событие -> одна новая точка в каждом массиве статистики
    push!(m.ta, ConcurrentSim.now(env))
    push!(m.Sa, m.Sa[end] - vacc)
    push!(m.Ia, m.Ia[end])
    push!(m.Ra, m.Ra[end] + vacc)
end

@resumable function live(env, individual::SIRPerson, m::SIRModel)
    while individual.status == :S
        @yield timeout(env, rand(Exponential(1/m.c)))
        individual.status === :D && return
        alter = individual
        while alter === individual
            N = length(m.allIndividuals)
            alter = m.allIndividuals[rand(DiscreteUniform(1, N))]
        end
        if alter.status == :I && rand(Uniform(0, 1)) < m.β
            individual.status = :I
            infection_update!(env, m)
        end
    end
    if individual.status == :I
        dt = m.deterministic_recovery ? 1/m.γ : rand(Exponential(1/m.γ))
        @yield timeout(env, dt)
        individual.status === :D && return
        individual.status = :R
        recovery_update!(env, m)
    end
end

@resumable function lifespan(env, individual::SIRPerson, m::SIRModel)
    @yield timeout(env, rand(Exponential(1/m.μ)))
    individual.status === :D && return
    st = individual.status
    individual.status = :D
    death_update!(env, m, st)
end

@resumable function birth_process(env, m::SIRModel)
    while true
        @yield timeout(env, rand(Exponential(1/m.ν)))
        new_id = length(m.allIndividuals) + 1
        newp = SIRPerson(new_id, :S)
        push!(m.allIndividuals, newp)
        birth_update!(env, m)
        @process live(env, newp, m)
        @process lifespan(env, newp, m)
    end
end

function MakeSIRModel(u0, p; μ::Float64=0.0, ν::Float64=0.0,
                       deterministic_recovery::Bool=false)
    (S, I, R) = u0; N = S + I + R; (β, c, γ) = p
    sim = ConcurrentSim.Simulation()
    allIndividuals = SIRPerson[]
    for i in 1:S;         push!(allIndividuals, SIRPerson(i, :S)); end
    for i in (S+1):(S+I); push!(allIndividuals, SIRPerson(i, :I)); end
    for i in (S+I+1):N;   push!(allIndividuals, SIRPerson(i, :R)); end
    SIRModel(sim, β, c, γ, μ, ν, deterministic_recovery,
             Float64[0.0], Int64[S], Int64[I], Int64[R], allIndividuals)
end

function activate(m::SIRModel; t_vac::Float64=Inf, fraction::Float64=0.0)
    for ind in m.allIndividuals
        @process live(m.sim, ind, m)
        m.μ > 0 && @process lifespan(m.sim, ind, m)
    end
    m.ν > 0 && @process birth_process(m.sim, m)
    isfinite(t_vac) && fraction > 0 && @process vaccinate(m.sim, m, t_vac, fraction)
end

function sir_run(m::SIRModel, tf::Float64)
    ConcurrentSim.run(m.sim, tf)
end

function out(m::SIRModel)
    n = length(m.ta)
    if length(m.Sa) != n || length(m.Ia) != n || length(m.Ra) != n
        error("Рассинхронизация массивов статистики: ta=$n, " *
              "Sa=$(length(m.Sa)), Ia=$(length(m.Ia)), Ra=$(length(m.Ra))")
    end
    r = DataFrame()
    r[!, :t] = m.ta; r[!, :S] = m.Sa
    r[!, :I] = m.Ia; r[!, :R] = m.Ra
    return r
end
