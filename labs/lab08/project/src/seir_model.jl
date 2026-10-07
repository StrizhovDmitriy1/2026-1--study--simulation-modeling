using ResumableFunctions, ConcurrentSim, Distributions, DataFrames, Random

# --- Вспомогательные функции ---
function increment!(a::Array{Int64}); push!(a, a[end] + 1); end
function decrement!(a::Array{Int64}); push!(a, a[end] - 1); end
function carryover!(a::Array{Int64}); push!(a, a[end]); end

# --- Структуры данных ---
mutable struct SEIRPerson
    id::Int64
    status::Symbol   # :S, :E, :I, :R
end

mutable struct SEIRModel
    sim::ConcurrentSim.Simulation
    β::Float64
    c::Float64
    σ::Float64       # интенсивность перехода E -> I
    γ::Float64       # интенсивность выздоровления I -> R
    ta::Array{Float64}
    Sa::Array{Int64}
    Ea::Array{Int64}
    Ia::Array{Int64}
    Ra::Array{Int64}
    allIndividuals::Array{SEIRPerson}
end

# --- Функции обновления статистики ---

# Заражение: S -> E
function exposure_update!(sim, m::SEIRModel)
    push!(m.ta, ConcurrentSim.now(sim))
    decrement!(m.Sa)
    increment!(m.Ea)
    carryover!(m.Ia)
    carryover!(m.Ra)
end

# Конец латентного периода: E -> I
function infectious_update!(sim, m::SEIRModel)
    push!(m.ta, ConcurrentSim.now(sim))
    carryover!(m.Sa)
    decrement!(m.Ea)
    increment!(m.Ia)
    carryover!(m.Ra)
end

# Выздоровление: I -> R
function recovery_update!(sim, m::SEIRModel)
    push!(m.ta, ConcurrentSim.now(sim))
    carryover!(m.Sa)
    carryover!(m.Ea)
    decrement!(m.Ia)
    increment!(m.Ra)
end

# --- Основная логика жизни индивида ---
@resumable function live(env::ConcurrentSim.Simulation,
                         individual::SEIRPerson, m::SEIRModel)
    # Фаза восприимчивости: ждём контакта, пока не заразимся
    while individual.status == :S
        @yield timeout(env, rand(Exponential(1/m.c)))
        # выбираем случайного собеседника, отличного от себя
        alter = individual
        while alter === individual
            N = length(m.allIndividuals)
            alter = m.allIndividuals[rand(DiscreteUniform(1, N))]
        end
        # заражаемся, если собеседник инфекционный
        if alter.status == :I && rand(Uniform(0, 1)) < m.β
            individual.status = :E
            exposure_update!(env, m)
        end
    end

    # Латентный период: E -> I
    if individual.status == :E
        @yield timeout(env, rand(Exponential(1/m.σ)))
        individual.status = :I
        infectious_update!(env, m)
    end

    # Инфекционный период: I -> R
    if individual.status == :I
        @yield timeout(env, rand(Exponential(1/m.γ)))
        individual.status = :R
        recovery_update!(env, m)
    end
end

# --- Создание и запуск модели ---
function MakeSEIRModel(u0, p)
    (S, E, I, R) = u0
    N = S + E + I + R
    (β, c, σ, γ) = p

    sim = ConcurrentSim.Simulation()
    allIndividuals = SEIRPerson[]

    for i in 1:S;             push!(allIndividuals, SEIRPerson(i, :S)); end
    for i in (S+1):(S+E);     push!(allIndividuals, SEIRPerson(i, :E)); end
    for i in (S+E+1):(S+E+I); push!(allIndividuals, SEIRPerson(i, :I)); end
    for i in (S+E+I+1):N;     push!(allIndividuals, SEIRPerson(i, :R)); end

    ta = Float64[0.0]
    Sa = Int64[S]
    Ea = Int64[E]
    Ia = Int64[I]
    Ra = Int64[R]

    SEIRModel(sim, β, c, σ, γ, ta, Sa, Ea, Ia, Ra, allIndividuals)
end

function activate(m::SEIRModel)
    [@process live(m.sim, ind, m) for ind in m.allIndividuals]
end

function seir_run(m::SEIRModel, tf::Float64)
    ConcurrentSim.run(m.sim, tf)
end

function out(m::SEIRModel)
    n = length(m.ta)
    if length(m.Sa) != n || length(m.Ea) != n ||
       length(m.Ia) != n || length(m.Ra) != n
        error("Рассинхронизация массивов: ta=$n, Sa=$(length(m.Sa)), " *
              "Ea=$(length(m.Ea)), Ia=$(length(m.Ia)), Ra=$(length(m.Ra))")
    end
    r = DataFrame()
    r[!, :t] = m.ta
    r[!, :S] = m.Sa
    r[!, :E] = m.Ea
    r[!, :I] = m.Ia
    r[!, :R] = m.Ra
    return r
end
