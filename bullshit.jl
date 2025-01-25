# Debugging scratchpad

data_sd = 1.0
dt = 1.0
T = 1000.0
numTrials = 1000

function loss_function(u, p)
    ou = generate_ou_process(u[1], data_sd, dt, T, numTrials;
                            backend="sciml")
    acf = comp_ac_fft(ou)
    acf_mean = mean(acf, dims=1)[:]
    lags = collect((0:length(acf_mean)-1) * dt)
    tau = fit_expdecay(lags, acf_mean)
    d = abs2(10.0 - tau)
    println(d)
    return d
end

u0 = [1.0]
p = [0.0]
optf = OptimizationFunction(loss_function)
prob = OptimizationProblem(optf, u0)
sol = solve(prob, NelderMead())
