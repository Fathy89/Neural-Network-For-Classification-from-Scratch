import numpy as np 
from sklearn.base import BaseEstimator

EPS = 1e-15

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))


def soft_max(arr) :
    
    arr = arr - np.max(arr, axis=-1, keepdims=True)
    exp_vals = np.exp(arr)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


def dl_dz(arr,labels) :  #dE/dnet
    # Combined derivative of (cross entropy + softmax)
    # dL/dz = a - y

    return arr-labels



class nueral_network(BaseEstimator)  : 
    def __init__(self,neurone_in_each_hidden_layer,output_neurone=1,type='multiclass',lr=.01,hidden_activation='relu',activation_function='identity'):
        self.activation_function=activation_function
        self.hidden_activation=hidden_activation
        self.type=type
        if self.type=='multiclass' : 
            self.activation_function='softmax'
        elif self.type=='multilabel' :
            self.activation_function='sigmoid'
        else :
                raise(TypeError,'Wrong Type it should be (multiclass) or (multilabel)')

        self.neurone_in_each_layer=neurone_in_each_hidden_layer
        self.nn_layers=[]
        self.output_neurone=output_neurone
        self.lr=lr
    def feed_forward_over_nn(self,input): 
        outputs = []
        outputs.append(input)
        for i in self.nn_layers :
            output=i.feed_forward(input)
            outputs.append(output)
            input=output
        return outputs
    
    def Back_propagation(self,target,outputs): 
        
        derv_for_each_node_layer = []
        prev_derv = outputs[-1] - target  # dE/dA (output layer)
        
        ##( comments was used in the debuging)
        # print(outputs[-1])
        # print(target)
        # print(prev_derv)
        # print(f"The Target shape is {target.shape}")
# loop over layers in reverse, but also track the next layer
        for idx in reversed(range(len(self.nn_layers))):
            layer = self.nn_layers[idx]
            # print(idx, layer.activation_function)  # debug
                
    #Compute dZ = dA * f'(Z)
            if  idx==len(self.nn_layers)-1:
                dZ=prev_derv
            else  : 
                dZ = prev_derv * layer.activation_derivative()
            derv_for_each_node_layer.append(dZ)

    #  Compute dA for previous layer (unless this is the first)
            if idx != 0:
                prev_derv = np.dot(dZ, layer.weights_matrix.T)

        derv_for_each_node_layer.reverse()
        return derv_for_each_node_layer
        #updating the wieghts  derv 
        
    def  update_wiegth(self,derv,outputs)  :
        # outputs=self.feed_forward_over_nn(input)
    
        for i,j,out_prev in zip(self.nn_layers,derv,outputs[:-1]):    # j= dZ 
            partail_w= (np.dot(out_prev.T,j))/j.shape[0]  ## alway i multply by j or d/dz to relative erorr to thr cost function  dw/dE
            partail_b=np.sum(j,axis=0,keepdims=True)/j.shape[0]  ## / (Batch size (no of rows)) acg for batch 
            
            i.weights_matrix-=self.lr*partail_w
            i.bias-=self.lr*partail_b #d/dbias =1  * d/dz to relative the errorr
            

        
    def train_steps(self,input,target) :
        self.outputs= self.feed_forward_over_nn(input) 
        derv_for_each_node_layer=self.Back_propagation(target,self.outputs)
        self.update_wiegth(derv_for_each_node_layer,self.outputs)
            
    def fit(self,x,y,epochs=100 ,batch_size=64) :
        n_samples,n_feat = x.shape

        ## add the  input layer 
        self.nn_layers.append(
        neurone_layer(n_feat, self.neurone_in_each_layer[0], activation_function=self.hidden_activation)
    )
        
        ## add hidden Layers
        for idx in range (len(self.neurone_in_each_layer)-1)  : 
            self.nn_layers.append(neurone_layer(self.neurone_in_each_layer[idx],self.neurone_in_each_layer[idx+1],activation_function=self.hidden_activation))
        
        ## add the output layer
        self.nn_layers.append(neurone_layer(self.neurone_in_each_layer[-1],self.output_neurone,activation_function=self.activation_function))
        
        if self.type=='multiclass'  : 
            n_classes = np.max(y) + 1
            y = np.eye(n_classes)[y.flatten()] ## one Hot Encoding
            print(y[:2])
        for i in range(epochs):
            samples = np.random.permutation(n_samples)
            x_shuffled = x[samples]
            y_shuffled = y[samples]
            for start in range(0,n_samples,batch_size):
                end=start+batch_size
                batch_x = x_shuffled[start:end]
                batch_y = y_shuffled[start:end]
            
                self.train_steps(batch_x,batch_y)
        return self
    
    def predict(self,x)   :   
        out=self.feed_forward_over_nn(x)[-1]  ### كده كده داله التفعيل اتطبقت ينجم هنا 
        if self.type=='multiclass':
            predicted = np.argmax(out,axis=1)
        
        elif  self.type=='multilabel':
            predicted = (out>=.5).astype(int)
        return predicted
    
class neurone_layer():
    def __init__(self, input_neurone, output_neurons, activation_function):
        self.activation_function = activation_function  
        

        self.weights_matrix = np.random.randn(input_neurone, output_neurons) * np.sqrt(2 / input_neurone)
        
        self.bias = np.zeros((1, output_neurons))
    
    def apply_activation(self, x):  # Renamed to avoid conflict with stored activation_function
        if self.activation_function == 'poly':
            return x**2
        elif self.activation_function == 'sigmoid':
            return  1 / (1 + np.exp(-np.clip(x, -250, 250)))
        elif self.activation_function == 'identity':
            return x
        elif self.activation_function == 'relu':
            # print('relu')
            return np.maximum(0, x)
        elif self.activation_function=='tanh'  :
            return np.tanh(x)
        elif self.activation_function=='softmax' : 
            # print('softmax')
            return soft_max(x)
        
    def activation_derivative(self):
            
        if self.activation_function == 'poly':
            return 2*self.net
        elif self.activation_function == 'sigmoid':
            sig =  1 / (1 + np.exp(-np.clip(self.net, -250, 250)))
            return sig*(1-sig)
        elif self.activation_function == 'identity':
            return np.ones_like(self.net)
        elif self.activation_function == 'relu':
            # print('relu derv')
            return (self.net > 0).astype(float)
        elif self.activation_function=='tanh'  :
            return 1 - np.tanh(self.net)**2
        elif self.activation_function=='softmax' : 
            return np.ones_like(self.net)   # Never used   (back propagation will handel it)

        
    
        
    
    def feed_forward(self, inputs):   
        self.net = np.dot(inputs, self.weights_matrix) + self.bias  # Removed .T and added bias  (victorized version ya nemg)
        output = self.apply_activation(self.net)
        # print("Layer output mean:", np.mean(output))

        return output
    
    
### if you have any comments dont hesitate to contact me 


