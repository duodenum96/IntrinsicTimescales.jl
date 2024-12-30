# examples/two_timescale_example.jl
using BayesianINT
using Distributions

# Set up parameters
dt = 1.0
T = 1000.0
num_trials = 500
bin_size = 1.0

# Create prior distributions
prior = [
    Uniform(0.0, 60.0),  # tau1
    Uniform(20.0, 140.0), # tau2
    Uniform(0.0, 1.0)     # mixing coefficient
]

# Generate some synthetic data for testing
true_params = [20.0, 80.0, 0.4]
model = TwoTimescaleModel(
    randn(num_trials, Int(T/dt)), # placeholder data
    prior,
    Float64[], # will be computed
    1.0,      # epsilon
    dt,
    T,
    num_trials,
    1.0       # data_var
)

# Run ABC
results = parallel_basic_abc(
    model,
    4;  # number of processes
    samples_per_proc=100,
    epsilon=1.0,
    max_iter=10000
)

# Process results
samples, distances = results