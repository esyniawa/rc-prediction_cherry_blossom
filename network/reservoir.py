import numpy as np
import ANNarchy as ann


# Variable N ESN population
N = 1000
con_prob = 0.2  # connection probability in reservoir

########################################################## ANNarchy/ESN ########################################################
ann.clear()
# step size 1 millisecond
ann.setup(dt=1.0, num_threads=6)

# The neuron has three parameters and two variables
ESN_Neuron = ann.Neuron(
    parameters = """
        tau = 10.0 : population
        g = 1.2 : population
        noise = 0.01 : population
    """,
    equations="""
        tau * dx/dt + x = sum(in) + g * sum(exc) + noise * Uniform(-1, 1)

        r = tanh(x)
    """
)

InputNeuron = ann.Neuron(
    parameters="""
    baseline = 0.0
    phi = 0.0
    """,
    equations="""
    r = baseline + phi * Uniform(-1.0,1.0)
    """
)

reservoir_pop = ann.Population(geometry=N, neuron=ESN_Neuron)

# Inputs => Temp, Latitude, Longitude
input_size: int = 3
input_pop = ann.Population(geometry=input_size, neuron=InputNeuron)

# Distribute the input weights uniformly between -1 and 1
Wi = ann.Projection(pre=input_pop, post=reservoir_pop, target='in')
Wi.connect_all_to_all(weights=ann.Uniform(-1.0, 1.0))

# Recurrent weights from normal distribution
Wrec = ann.Projection(reservoir_pop, reservoir_pop, 'exc')
Wrec.connect_fixed_probability(probability=con_prob, weights=ann.Normal(mu=0, sigma=np.sqrt(1 / (con_prob * N))))
