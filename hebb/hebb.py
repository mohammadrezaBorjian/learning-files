import numpy as np

def hebb(inputs , targets):
    if(not isTargetsEqualsInputs(inputs , targets)):
        return
    
    weights = createDefaultWeight(inputs)
    return findWeights(weights , inputs , targets)


def isTargetsEqualsInputs(inputs , targets):

    return inputs.shape[0] == targets.shape[0]

def findWeights(defaultWeights,inputs,targets):
    finalWeights = defaultWeights
    for i in range(np.size(inputs,0)):

        multiplaction = inputs[i] * targets[i]

        finalWeights = finalWeights + multiplaction

    return finalWeights

def createDefaultWeight(inputs:np):
    return np.zeros(inputs.shape[1])
