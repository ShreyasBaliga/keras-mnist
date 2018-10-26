import numpy as np
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
X = X/np.amax(X,axis=0) 
y = y/100

def sigmoid (x):
    return 1/(1 + np.exp(-x))
	
def derivatives_sigmoid(x):
    return x * (1 - x)
	
epoch=7000 
learning_rate=0.1 
inputlayer_neurons = 2
hiddenlayer_neurons = 3 
output_neurons = 1 

weight_hidden_layer=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bias_hidden_layer=np.random.uniform(size=(1,hiddenlayer_neurons))
weight_output_layer=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bias_output_layer=np.random.uniform(size=(1,output_neurons))

for i in range(epoch):
#Forward Propogation
    hidden_layer_net_input=np.dot(X,weight_hidden_layer) + bias_hidden_layer
    hidden_layer_activation = sigmoid(hidden_layer_net_input)
    output_layer_net_input= np.dot(hidden_layer_activation,weight_output_layer)+ bias_output_layer
    output = sigmoid(output_layer_net_input)
#Backpropagation
    deltaK = (y-output)* derivatives_sigmoid(output) ##calculate deltak
    deltaH =  deltaK.dot(weight_output_layer.T) * derivatives_sigmoid(hidden_layer_activation) #deltaH
    weight_output_layer = weight_output_layer + hidden_layer_activation.T.dot(deltaK) *learning_rate
    weight_hidden_layer = weight_hidden_layer + X.T.dot(deltaH) *learning_rate
	
print("Input: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Predicted Output: \n" ,output)