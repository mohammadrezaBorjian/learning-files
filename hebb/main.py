import hebb
import numpy as np
import matplotlib.pyplot as plt 


def andAlgorithm():
    input = [[-1,-1,1],[-1,1,1],[1,-1,1],[1,1,1]]
    targets = [-1,-1,-1,1]
    testAlgorithm(input , targets)
    


def orAlgorithm():
    input = np.array([[-1,-1,1],[-1,1,1],[1,-1,1],[1,1,1]])
    targets = np.array([-1,1,1,1])
    testAlgorithm(input , targets)

def xorAlgorithm():
    input = [[-1,-1,1],[1,-1,1],[-1,1,1],[1,1,1]]
    targets = [-1,1,1,-1]
    testAlgorithm(input , targets)


def testAlgorithm(inputs,targets):
    weights = hebb.hebb(inputs , targets)
    predict = np.array([])
    for i in range(np.size(inputs , 0)):
        multiplaction = np.dot(inputs[i],weights)
        sum = np.sum(multiplaction)
        predict = np.append(predict , activationFunction(sum))
    if (predict == targets).all:
        print(f"neural network learned with hebb. weights are {weights}")
        showChart(inputs , targets,weights)
    else:
        print("sorry,this neural network couldn't learn with hebb")

def activationFunction(input):
    if input > 0 :
        return 1
    else:
        return -1
    
def showChart(inputs,targets, weights):

    positive_points = inputs[targets == 1]
    negative_points = inputs[targets != 1]


    xlist = np.linspace(-2, 2,2)
    ylist = np.linspace(-2, 2,2) 
    X,Y = np.meshgrid(xlist , ylist)
    
    F = weights[0] * X + weights[1] * Y + weights[2]

    plt.contour(X, Y, F, [0], colors = 'k', linestyles = 'solid')

    plt.scatter(positive_points[:, 0], positive_points[:, 1], color="red", marker="*", s=30, label="Positive")
    plt.scatter(negative_points[:, 0], negative_points[:, 1], color="black", marker="*", s=30, label="Negative")


    plt.axhline(y = 0, color = 'r', linestyle = '-') 
    plt.axvline(x=0,color = "r" , linestyle = '-')
    
    plt.legend() 
    plt.show()
    
print("which algohritm would you like run? \n 1-OR 2-AND 3-XOR \n")
selectedItem = input()

if selectedItem == "1":
    orAlgorithm()
elif selectedItem == "2":
    andAlgorithm()
elif selectedItem == "3":
    xorAlgorithm()
else:
    print("number is wrong!!!")