import sys
import numpy
import sklearn.datasets
from sklearn import preprocessing
import random;
import kmeans_for_bcancerint as s;
import matplotlib.pyplot as plt
import pygame
from superwires import games,color



error_list = []
epoch_list = []
numpy.seterr(all='ignore')

def sigmoid(x):
    #return numpy.tanh(x);
    return 1. / (1 + numpy.exp(-x))



class dA(object):
    def __init__(self, input=None, n_visible=9, n_hidden=4, \
        W=None, hbias=2, vbias=2, numpy_rng=None):

        self.n_visible = n_visible  # num of units in visible (input) layer
        self.n_hidden = n_hidden    # num of units in hidden layer

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(1234)
            
        if W is None:
            a = 1. / n_visible
            initial_W = numpy.array(numpy_rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(n_visible, n_hidden)))

            W = initial_W
            
            W = numpy.array(W)
         

        if hbias is None:
            hbias = numpy.ones(n_hidden)  # initialize h bias 0

        if vbias is None:
            vbias = numpy.ones(n_visible)  # initialize v bias 0

        self.numpy_rng = numpy_rng
        self.x = input
        self.W = W
        self.W_prime = self.W.T
        self.hbias = hbias
        self.vbias = vbias

        # self.params = [self.W, self.hbias, self.vbias]


        
    def get_corrupted_input(self, input, corruption_level):
        assert corruption_level < 1

        return self.numpy_rng.binomial(size=input.shape,
                                       n=1,
                                       p=1-corruption_level) * input

    # Encode
    def get_hidden_values(self, input):
        return sigmoid(numpy.dot(input, self.W) + self.hbias)

    # Decode
    def get_reconstructed_input(self, hidden):
        return sigmoid(numpy.dot(hidden, self.W_prime) + self.vbias)


    def train(self,lr=0.1, corruption_level=0.0, input=None):
        if input is not None:
            self.x = input

        x = self.x
        tilde_x = self.get_corrupted_input(x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        print("Error"+ str(numpy.sum((tilde_x - z)**2)))
        error_list.append(numpy.sum((tilde_x - z)**2))
         

        L_h2 = x - z
        L_h1 = numpy.dot(L_h2, self.W) * y * (1 - y)

        L_vbias = L_h2
        L_hbias = L_h1
        L_W =  numpy.dot(tilde_x.T, L_h1) + numpy.dot(L_h2.T, y)


        self.W += lr * L_W
        self.hbias += lr * numpy.mean(L_hbias, axis=0)
        self.vbias += lr * numpy.mean(L_vbias, axis=0)



    def negative_log_likelihood(self, corruption_level=0.07):
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        cross_entropy = - numpy.mean(
            numpy.sum(self.x * numpy.log(z) +
            (1 - self.x) * numpy.log(1 - z),
                      axis=1))
        #print cross_entropy;

        return cross_entropy


    def reconstruct(self, x):
        y = self.get_hidden_values(x)
        i=0;
        print "Trained Weights"
        print self.W;
        #print "Hidden Layer Activation";
        i=0;
        #for a in y :
            #print(str(i)+" "+str(a));
            #i=i+1
        
        #print numpy.ndarray.tolist(y);
        s.k_means(y,2);
       
        z = self.get_reconstructed_input(y)
        #print "Reconstructed data"
        #print z
        return z



def test_dA(learning_rate=0.005, corruption_level=0.0, training_epochs=10000):
    
    
    input_array = numpy.genfromtxt("C:\\Users\\SUMANTH C\\Desktop\\Deep Learning\\Datasets\\bcancerint_sort1.csv",delimiter=',');
   
        
    input_array = input_array[:,:9];
    print (input_array.shape);
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.1,0.9))
    
    data= min_max_scaler.fit_transform(input_array);
 
    data = numpy.array(data);
    
    print"------------"
    print"------------"
    print"------------"
    
    
    rng = numpy.random.RandomState(123)

    # construct dA
    da = dA(input=data, n_visible=9, n_hidden=4, numpy_rng=rng)

    # train
    for epoch in xrange(training_epochs):
        da.train(lr=learning_rate, corruption_level=corruption_level);

    for i in range(0,training_epochs):
        epoch_list.append(i)
        
    plt.plot(epoch_list, error_list)
    plt.title("Error vs No of epochs")
    plt.xlabel("No of Epochs")
    plt.ylabel("Error")
    plt.show()
    

    print("Completed")
    print("-------------------------------")
    print("\n")

    da.reconstruct(data)



if __name__ == "__main__":
    games.init(screen_width = 1000, screen_height = 800, fps = 50)
    back_image = games.load_image("white_back.jpg",transparent = False)
    games.screen.background = back_image
    auto_image = games.load_image("auto_arch.jpg")
    the_auto = games.Sprite(image = auto_image,x = games.screen.width/2,y = games.screen.height/2)
    games.screen.add(the_auto)
    name = games.Text(value = "Autoencoders Architecture",size = 40,color = color.black,x =games.screen.width/2-40 ,y = 60)
    games.screen.add(name)
    games.screen.mainloop()

    
    test_dA()
    

