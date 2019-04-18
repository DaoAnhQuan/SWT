from network import *
from utils import *

from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import pickle

#parser = argparse.ArgumentParser(description='Train a convolutional neural network.')
#parser.add_argument('save_path', metavar = 'Save Path', help='name of file to save parameters in.')

if __name__ == '__main__':
    
    #args = parser.parse_args()
    #save_path = args.save_path
    save_path = 'param.pkl'
    num = 1
    
    
    # Plot cost 
    """plt.plot(cost, 'r')
    plt.xlabel('# Iterations')
    plt.ylabel('Cost')
    plt.legend('Loss', loc='upper right')
    plt.show()"""

    # Get test data
    """m =10000
    X = extract_data('t10k-images-idx3-ubyte.gz', m, 28)
    y_dash = extract_labels('t10k-labels-idx1-ubyte.gz', m).reshape(m,1)"""

    X,y_dash = extract_data('C:\\Users\\daoan\\Desktop\\Python\\SWT\\test',28)
    # Normalize the data
    X-= int(np.mean(X)) # subtract mean
    X/= int(np.std(X)) # divide by standard deviation
    test_data = np.hstack((X,y_dash))
    
    X = test_data[:,0:-1]
    X = X.reshape(len(test_data), 1, 28, 28)
    y = test_data[:,-1]

    acc = 0
    """corr = 0
    digit_count = [0 for i in range(28)]
    digit_correct = [0 for i in range(28)]
   
    print()
    print("Computing accuracy over test set:")

    t = tqdm(range(len(X)), leave=True)

    for i in t:
        x = X[i]
        pred, prob = predict(x, f1, f2, w3, w4, b1, b2, b3, b4)
        digit_count[int(y[i])]+=1
        if pred==y[i]:
            corr+=1
            digit_correct[pred]+=1

        t.set_description("Acc:%0.2f%%" % (float(corr/(i+1))*100))
        
    print("Overall Accuracy: %.2f" % (float(corr/len(test_data)*100)))
    """
    while (acc < 92):
        print(num)
        cost = train(num_filt1 = 12, num_filt2 = 12, num_epochs = num, save_path = save_path)

        params, cost = pickle.load(open(save_path, 'rb'))
        [f1, f2, w3, w4, b1, b2, b3, b4] = params
        corr = 0
        length = len(X)
        for i in range(length):
            x = X[i]
            pred, prob = predict(x, f1, f2, w3, w4, b1, b2, b3, b4)
            if pred==y[i]:
                corr+=1
        acc = float(corr/length*100)
        print("Overall Accuracy: %.2f" % acc)
        num +=1
    
    
