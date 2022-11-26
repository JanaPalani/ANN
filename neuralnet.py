import numpy as np
import random



class Network:

    def __init__(self):
        self.is_input = False 
        self.hidden_num = 0
        self.hidden_layers= {}
        self.weights_len = 0
        self.weights ={}
        self.is_output = False
        self.is_completed = False 


    def add_weights(self,layer_1 , layer_2):
        return np.random.randn(len(layer_1),len(layer_2))
        
    def generate_x(self,data_x):
        # data_x = np.c_[np.ones((len(data_x),1)),data_x]
        return data_x
    
    def predict(self,data_to):
        self.data_to = data_to
        for i in self.weights.keys():
            self.data_to = self.generate_x(self.data_to)
            self.data_to = np.dot(self.data_to,self.weights[i])
        return self.data_to
    
    def epoch(self,data_x,data_y):
        each_out = {}
        num = 0
        alpha = 0.1
        self.data_x = data_x
        for i in self.weights.keys():
            self.data_x = self.generate_x(self.data_x)
            each_out[num] = self.data_x
            num += 1
            self.data_x = np.dot(self.data_x,self.weights[i])
            print(self.data_x) 
        # print(self.data_x)
        
        n = len(self.data_x)
        error = self.data_x - data_y
        reversed_keys = list(self.weights.keys())[::-1]
        for i in reversed_keys:
            weight_updater = each_out[i].T.dot(error) / n
            self.weights[i] -= (alpha * weight_updater) 
            if i != 0 :
                error = error.dot(weight_updater.T)
        

        


        


    def train(self,data_x,data_y):
        # print(self.predict(data_x))
        # for i in range(1000):
            self.epoch(data_x,data_y)
            # print(self.weights)
        # print(self.predict(data_x))


    def add_layers(self,type, nodes_in_layer:int):
        if type == 'input':
            if not self.is_input:
                self.input_layer =np.ones((nodes_in_layer,1))
                self.is_input = True
                self.present = self.input_layer
            else:
                print('there can be only one input layer')

        if type == 'hidden':
            if self.is_input:
                self.hidden_layers[self.hidden_num] = np.random.rand(nodes_in_layer,1)
                self.weights[self.weights_len] = self.add_weights(self.present,self.hidden_layers[self.hidden_num])
                self.weights_len += 1
                self.present = self.hidden_layers[self.hidden_num]
                self.hidden_num += 1
            else:
                print('you have not added the input layer')
        
        if type == 'output':
            if self.is_input:
                if not self.is_output:
                    self.output_layer= np.ones((nodes_in_layer,1))
                    self.hidden_layers[self.hidden_num] = np.random.rand(1,nodes_in_layer)
                    self.is_output = True
                    self.is_completed = True
                    self.weights[self.weights_len] = self.add_weights(self.present,self.output_layer)
                else:
                    print('already added output layer')
            else:
                print('you have not added the input layer ')

# [testing]     
# data_x =[]
# data_y = []
# for i in range(1):
#     j = [random.random() for _ in range(4)]
#     k = 0.5* j[0] + 0.2*j[1] - 0.3*j[2] +0.3*j[3] 
#     data_x.append(j)
#     data_y.append(k)
# data_x = np.array(data_x)
# data_y = np.array(data_y)
# data_y = np.reshape(data_y,(len(data_y),1))
    

# data_x = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
# data_y = np.array([[0], [0], [1], [1]])

# network = Network()
# network.add_layers('input', 2)
# network.add_layers('hidden',3)
# network.add_layers('hidden',4)
# network.add_layers('output',1)
# network.train(data_x,data_y)

# c = network.predict(data_x)
# print(c)
# print(data_y)
# print(network.weights)
# print(network.hidden_layers)

