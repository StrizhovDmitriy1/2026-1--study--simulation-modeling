using DrWatson
@quickactivate "project"
include(srcdir("sir_model.jl"))
using Random, StatsPlots, DataFrames, CSV, Dates, BenchmarkTools

tmax = 40.0; u0 = [990, 10, 0]; p = [0.05, 10.0, 0.25]
stamp = Dates.format(now(), "yyyymmdd_HHMMSS")

Random.seed!(1234)
m0 = MakeSIRModel(u0, p)
activate(m0); sir_run(m0, tmax)
data0 = out(m0)
CSV.write(datadir("sims", "sir_base_$(stamp).csv"), data0)
@df data0 plot(:t, [:S :I :R], labels=["S" "I" "R"], title="Базовый прогон SIR")
savefig(plotsdir("sir_des.png"))

betas  = [0.03, 0.05, 0.07]; cs = [10.0]; gammas = [0.15, 0.25, 0.40]
metrics = DataFrame(β=Float64[], c=Float64[], γ=Float64[],
                    peak_I=Int64[], peak_t=Float64[], final_R=Int64[])
for β in betas, c in cs, γ in gammas
    Random.seed!(1234)
    m = MakeSIRModel(u0, [β, c, γ])
    activate(m); sir_run(m, tmax)
    d = out(m); ipk = argmax(d.I)
    push!(metrics, (β, c, γ, d.I[ipk], d.t[ipk], d.R[end]))
    fname = "sir_$(u0[1])_$(u0[2])_$(u0[3])_$(β)_$(c)_$(γ)_$(stamp).csv"
    CSV.write(datadir("sims", fname), d)
    @df d plot(:t, [:S :I :R], labels=["S" "I" "R"],
        title="SIR: β=$(β), c=$(c), γ=$(γ)")
    savefig(plotsdir("sir_beta$(β)_c$(c)_g$(γ).png"))
end
CSV.write(datadir("sims", "sensitivity_$(stamp).csv"), metrics)

Random.seed!(1234)
m_stoch = MakeSIRModel(u0, p; deterministic_recovery=false)
activate(m_stoch); sir_run(m_stoch, tmax)
data_stoch = out(m_stoch)
CSV.write(datadir("sims", "sir_stoch_$(stamp).csv"), data_stoch)

Random.seed!(1234)
m_det = MakeSIRModel(u0, p; deterministic_recovery=true)
activate(m_det); sir_run(m_det, tmax)
data_det = out(m_det)
CSV.write(datadir("sims", "sir_det_$(stamp).csv"), data_det)

p1 = @df data_stoch plot(:t, [:S :I :R], labels=["S" "I" "R"], title="Стохастическое")
p2 = @df data_det   plot(:t, [:S :I :R], labels=["S" "I" "R"], title="Детерминированное 1/γ")
plot(p1, p2, layout=(1,2)); savefig(plotsdir("sir_recovery_compare.png"))

μ = 0.01; ν = 0.01
Random.seed!(1234)
m_demo = MakeSIRModel(u0, p; μ=μ, ν=ν)
activate(m_demo); sir_run(m_demo, tmax)
data_demo = out(m_demo)
CSV.write(datadir("sims", "sir_demo_mu$(μ)_nu$(ν)_$(stamp).csv"), data_demo)

p3 = @df data0     plot(:t, [:S :I :R], labels=["S" "I" "R"], title="Без демографии")
p4 = @df data_demo plot(:t, [:S :I :R], labels=["S" "I" "R"], title="С демографией μ=$(μ), ν=$(ν)")
plot(p3, p4, layout=(1,2)); savefig(plotsdir("sir_demography.png"))

t_vac    = 5.0
fraction = 0.3
Random.seed!(1234)
m_vac = MakeSIRModel(u0, p)
activate(m_vac; t_vac=t_vac, fraction=fraction)
sir_run(m_vac, tmax)
data_vac = out(m_vac)
CSV.write(datadir("sims", "sir_vac_$(t_vac)_$(fraction)_$(stamp).csv"), data_vac)

p5 = @df data0 plot(:t, [:S :I :R], labels=["S" "I" "R"], title="Без вакцинации")
p6 = @df data_vac plot(:t, [:S :I :R], labels=["S" "I" "R"],
        title="Вакцинация $(Int(fraction*100))% при t=$(t_vac)")
plot(p5, p6, layout=(1,2)); savefig(plotsdir("sir_vaccination.png"))

u0_big = [9900, 100, 0]
Random.seed!(1234)
big_model = MakeSIRModel(u0_big, p)
activate(big_model)
report = @benchmark sir_run($big_model, $tmax)
show(stdout, MIME("text/plain"), report)
