using DrWatson
@quickactivate "project"
include(srcdir("seir_model.jl"))
using Random, StatsPlots, DataFrames, CSV, Dates, BenchmarkTools

# --- Параметры модели ---
tmax = 60.0
u0   = [990, 0, 10, 0]         # S, E, I, R
p    = [0.05, 10.0, 0.2, 0.25] # β, c, σ, γ
stamp = Dates.format(now(), "yyyymmdd_HHMMSS")

Random.seed!(1234)

# --- Запуск модели ---
seir = MakeSEIRModel(u0, p)
activate(seir)
seir_run(seir, tmax)
data = out(seir)

# --- Сохранение результатов ---
CSV.write(datadir("sims", "seir_$(stamp).csv"), data)

# --- Визуализация ---
@df data plot(:t, [:S :E :I :R],
    labels = ["S" "E" "I" "R"],
    xlab   = "Время",
    ylab   = "Численность",
    title  = "Дискретно-событийная SEIR модель (σ=$(p[3]))")
savefig(plotsdir("seir_des.png"))

# --- Ключевые метрики ---
ipk = argmax(data.I)
println("Пик I = $(data.I[ipk]) в момент t = $(round(data.t[ipk], digits=2))")
println("Итоговое R = $(data.R[end]) из $(sum(u0))")
println("Пик E = $(maximum(data.E)) в момент t = $(round(data.t[argmax(data.E)], digits=2))")
