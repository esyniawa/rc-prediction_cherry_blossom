import numpy as np
import ANNarchy as ann

from .definitions import ESN_Neuron, InputNeuron, OutputNeuron

ann.setup(num_threads=4)

# Variable N ESN population
N = 1500
con_prob = 0.2  # connection probability in reservoir

########################################################## ANNarchy/ESN ########################################################
# step size 1 millisecond
ann.setup(dt=1.0, num_threads=4)

reservoir_pop = ann.Population(geometry=N, neuron=ESN_Neuron, name='reservoir')

# Inputs => Temp, Latitude, Longitude
input_size: int = 3
input_pop = ann.Population(geometry=input_size, neuron=InputNeuron, name='input')

# Output dummy
output_pop = ann.Population(geometry=1, neuron=OutputNeuron, name='output')

# Distribute the input weights uniformly between -1 and 1
Wi = ann.Projection(pre=input_pop, post=reservoir_pop, target='in')
Wi.connect_all_to_all(weights=ann.Uniform(-1.0, 1.0))

# Recurrent weights from normal distribution
Wrec = ann.Projection(reservoir_pop, reservoir_pop, 'exc')
Wrec.connect_fixed_probability(probability=con_prob, weights=ann.Normal(mu=0, sigma=np.sqrt(1 / (con_prob * N))))

# Output weights
Wo = ann.Projection(reservoir_pop, output_pop, 'out')
Wo.connect_all_to_all(weights=0.0, force_multiple_weights=True)
