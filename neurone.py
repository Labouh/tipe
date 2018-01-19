import numpy as np
neuronsperlayer = []


class neurone(): #arguments nblayers, neuronsperlayer  (, inputnb, outputnb included in neuronsperlayer) 
    def __init__(self):
        Weights = inputs
        for k in range(nblayers):
            Weights.append(np.random.randn(neuronsperlayer[k], neuronsperlayer[k+1])) # append the list of randomized weights between the different layers.
    
    sigmoid = lambda x: np.exp(-np.logaddexp(0, -x)) # to avoid overflow / même si l'overflow était pas vraiment dérangeant 
    
    def forward(self):
        result = []
        for k in range(nblayers):
            result = self.sigmoid(np.dot(self.Weights[k], self.Weights[k+1]))
        return(result)
    
        
    
    
    