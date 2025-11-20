from nueral_network_for_classification   import nueral_network
from sklearn.datasets import make_classification,make_multilabel_classification
if __name__ == "__main__":

    ##  make a smaples of multi label
    x,y=make_multilabel_classification(n_features=10,n_labels=2,n_classes=2,n_samples=500,random_state=42)

#  make a smaples of multiclass
    # x,y=make_classification(n_features=10,n_informative=6,n_classes=2,n_samples=500,random_state=42)
    # y=y.reshape(-1,1)



    nn = nueral_network(neurone_in_each_hidden_layer=[32,16,8], output_neurone=2,lr=.01,activation_function='sigmoid',type='multilabel')


    nn.fit(x,y,epochs=100,batch_size=64)
    predictions = nn.predict(x)
    

        
    print(f"The Pridiction is :{predictions[:20]}")
    print(f"The Truth is :     {y[:20]} ")
