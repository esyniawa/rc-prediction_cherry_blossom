import ANNarchy as ann

# The neuron has three parameters and two variables
ESN_Neuron = ann.Neuron(
    parameters = """
        tau = 5.0 : population
        g = 1.5 : population
        noise = 0.0 : population
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

OutputNeuron = ann.Neuron(
    parameters="""
    baseline = 0.0 : population
    """,
    equations="""
    r = sum(out) + baseline
    """
)
