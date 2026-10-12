using IntrinsicTimescales

# TODO: decide on NonlinearSolve vs CurveFit.jl

x = randn(3, 10, 10) # 3 trials, 10 time points, 10 channels
result = calcInt(x) # by default, calculate all methods, assume time dimension is dimension 1 and trial dimension is the last dimension
result = calcInt(x, methods=[:acw50, :auc, :knee], time_dim=2, trial_dim=1) # dimension of trials, 1 result per trial. Default is average_ints=false, average_acf_and_psd=true
result = calcInt(x, methods=[:acw50, :auc, :knee], time_dim=2, trial_dim=1, average_ints=false, average_acf_and_psd=false) # dimension of trials, 1 result per trial. 
result = calcInt(x, methods=[:acw50, :auc, :knee], time_dim=2, trial_dim=1, average_ints=false, average_acf_and_psd=true) # dimension of trials, average the acf/psd before calculating int
result = calcInt(x, methods=[:acw50, :auc, :knee], fit_solver=LevenbergMarquardt()) # the user is able to specify fit method 
result = calcInt(x, methods=[:acw50, :auc, :knee], fit_solver=LevenbergMarquardt(), fit_kwargs=(abstol=1e-6, )) # NonlinearSolve.jl / Optimization.jl kwargs
result = calcInt(x, methods=[:acw50, :auc, :knee], keep_acf_and_psd=true) # keeps individual ACF/PSDs if the user wants to keep them (for plotting etc.). If false, the algorithm will keep SE nonetheless.

# extracting values
acw50 = result.values[:acw50] # array with original dimensions (the dimensions that were averaged over become singleton)
result.dims.trial 
result.dims.time 

plot(result) 
# should plot ACF/PSD if there was only one signal (time X 1). Should plot a single plot pane with individual ACF/PSDs as faint colors and mean thick color if keep_acf_and_psd=true. 
# otherwise it should plot mean +- SE

plot(result, rowdims=1) # Now it will do a column of figures where each row is comes from one element of dimension 1
plot(result, coldims=1) # Same for columns
plot(result, rowdims=2, coldims=1) # rows from dimension 2, cols from dimensions 1

result.diagnostics # should hold fitting information (e.g. coming from NonlinearSolve vs Optimization)

# TODO: parallel support: via OhMyThreads?

# TODO: I want to get rid of approximate Bayesian computation and keep only Turing. Turing has a lot of way to go in terms of optimization
