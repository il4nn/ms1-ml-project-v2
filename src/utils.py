import numpy as np 
from src.methods.knn import KNN
from src.methods.linear_regression import LinearRegression
from matplotlib import pyplot as plt 

# Generaly utilies
##################

def label_to_onehot(labels, C=None):
    """
    Transform the labels into one-hot representations.

    Arguments:
        labels (array): labels as class indices, of shape (N,)
        C (int): total number of classes. Optional, if not given
                 it will be inferred from labels.
    Returns:
        one_hot_labels (array): one-hot encoding of the labels, of shape (N,C)
    """
    N = labels.shape[0]
    if C is None:
        C = get_n_classes(labels)
    one_hot_labels = np.zeros([N, C])
    one_hot_labels[np.arange(N), labels.astype(int)] = 1
    return one_hot_labels

def onehot_to_label(onehot):
    """
    Transform the labels from one-hot to class index.

    Arguments:
        onehot (array): one-hot encoding of the labels, of shape (N,C)
    Returns:
        (array): labels as class indices, of shape (N,)
    """
    return np.argmax(onehot, axis=1)

def append_bias_term(data):
    """
    Append to the data a bias term equal to 1.

    Arguments:
        data (array): of shape (N,D)
    Returns:
        (array): shape (N,D+1)
    """
    N = data.shape[0]
    data = np.concatenate([np.ones([N, 1]),data], axis=1)
    return data

def normalize_fn(data, means, stds):
    """
    Return the normalized data, based on precomputed means and stds.
    
    Arguments:
        data (array): of shape (N,D)
        means (array): of shape (1,D)
        stds (array): of shape (1,D)
    Returns:
        (array): shape (N,D)
    """
    # return the normalized features
    return (data - means) / stds

def get_n_classes(labels):
    """
    Return the number of classes present in the data labels.
    
    This is approximated by taking the maximum label + 1 (as we count from 0).
    """
    return int(np.max(labels) + 1)

def compute_mean(data):
    """"
    Arguments:
        data (array): An array of shape (N,D) where N is the number of samples and D the number of features 
    Returns:
        array: An array of shape (D,) containing the mean of each sample
    """
    return np.mean(data,axis=0)

def compute_std(data):
    """"
    Arguments:
        data (array): An array of shape (N,D) where N is the number of samples and D the number of features 
    Returns:
        array: An array of shape (D,) containing the standard deviation of each sample
    """
    return np.std(data,axis=0)

def create_validation_set(data,labels,split_ratio):
    """"
    Arguments:
        data (array): An array of shape (N,D) where N is the number of samples and D the number of features 
        labels (array): An array of shape (N,) where N is the number of samples 
        split_ratio: The percentage of the data set that will act as a test set
    Returns 
        X_new_train: The feature matrix for the training set
        Y_new_train: The label vector for the training set
        X_new_test: The feature matrix for the validation set
        Y_new_test: The label vector for the validation set 
    """
    N = data.shape[0] 
    all_indices = np.arange(0,N)
    np.random.shuffle(all_indices)

    validation_set_nb_elements = int(split_ratio * N)
    
    validation_set_indices = all_indices[:validation_set_nb_elements]
    train_set_indices = all_indices[validation_set_nb_elements:]

    X_new_train = data[train_set_indices]
    Y_new_train = labels[train_set_indices]
    X_new_test = data[validation_set_indices]
    Y_new_test = labels[validation_set_indices]

    return X_new_train, Y_new_train, X_new_test, Y_new_test

# Metrics
#########

def accuracy_fn(pred_labels, gt_labels):
    """
    Return the accuracy of the predicted labels.
    """
    return np.mean(pred_labels == gt_labels) * 100.

def macrof1_fn(pred_labels, gt_labels):
    """Return the macro F1-score."""
    class_ids = np.unique(gt_labels)
    macrof1 = 0
    for val in class_ids:
        predpos = (pred_labels == val)
        gtpos = (gt_labels==val)
        
        tp = sum(predpos*gtpos)
        fp = sum(predpos*~gtpos)
        fn = sum(~predpos*gtpos)
        if tp == 0:
            continue
        else:
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)

        macrof1 += 2*(precision*recall)/(precision+recall)

    return macrof1/len(class_ids)

def mse_fn(pred,gt):
    '''
        Mean Squared Error
        Arguments:
            pred: NxD prediction matrix
            gt: NxD groundtruth values for each predictions
        Returns:
            returns the computed loss

    '''
    loss = (pred-gt)**2
    loss = np.mean(loss)
    return loss


### Methods for K-fold cross validation 

def KFold_cross_validation_KNN(X, Y, K, k,M,T):

    '''
    K-Fold Cross validation function for K-NN
    Inputs:
        X : training data, shape (NxD)
        Y: training labels, shape (N,)
        K: number of folds (K in K-fold)
        k: number of neighbors for kNN algorithm (the hyperparameter)
        M: method 
        T: task kind 
    Returns:
        Average validation accuracy for the selected k.
    '''

    N = X.shape[0]
    accuracies = []  # list of accuracies
    all_indices = np.arange(N)
    np.random.shuffle(all_indices)
    for fold_ind in range(K):
        #Split the data into training and validation folds:
        split_size = N // K

        # Indices of the validation and training examples
        val_ind = all_indices[fold_ind * split_size : (fold_ind + 1) * split_size]

        train_ind = np.setdiff1d(all_indices,val_ind)

        X_train_fold = X[train_ind,:]
        Y_train_fold = Y[train_ind]
        X_val_fold = X[val_ind,:]
        Y_val_fold = Y[val_ind]

        if M == "knn":
            if T == "breed_identifying":
                model = KNN(k,task_kind="classification")  # Instantiate the KNN model with the appropriate 'k'
            else:
                model = KNN(k,task_kind="regression")
    
        model.fit(X_train_fold, Y_train_fold)  # Train the model

        Y_val_fold_pred = model.predict(X_val_fold)  # Make predictions

        if T == "breed_identifying":
            acc = accuracy_fn(Y_val_fold_pred,Y_val_fold)
        else:
            acc = mse_fn(Y_val_fold_pred,Y_val_fold)
        accuracies.append(acc)
        #Find the average validation accuracy over K:
    print(accuracies)
    average_accuracy = np.sum(accuracies)/len(accuracies)
    return average_accuracy


def run_cv_for_hyperparam(X, Y, K, k_list,M,T):

    '''
    K-Fold Cross validation function for K-NN
    Inputs:
        X : training data, shape (NxD)
        Y: training labels, shape (N,)
        K: number of folds (K in K-fold)
        k: a list of k values for kNN 

    Returns:
        model_performance: a list of validation accuracies corresponding to the k-values     
    '''
    model_performance = [] 
    for k in k_list:
        model_performance.append(KFold_cross_validation_KNN(X,Y,K,k,M,T))

    # Pick hyperparameter value that yields the best performance
    max = np.max(model_performance)
    max_occurences_indices = np.where(model_performance==max)[0]
    best_k=max_occurences_indices[-1]

    # print(model_performance)
    print(f"Best number of nearest neighbors on validation set is k={best_k+1}")
    return model_performance


### Functions for plotting 

def plot_k_vs_accuracy(k_values, accuracies):

    '''
    Plot K values vs. validation accuracies.
    Parameters:
        k_values (list of int): A list of different k values used in KNN.
        accuracies (list of float): A list of validation accuracies corresponding to each k value.
    '''

    plt.figure(figsize=(10, 6))  # Set figure size

    plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')  # Plot k values vs accuracies

    plt.xlabel('Number of Neighbors (k)')  # X-axis label

    plt.ylabel('Validation Accuracy')  # Y-axis label

    plt.title('KNN Performance: Number of Neighbors vs. Validation Accuracy')  # Title

    plt.grid(True)  # Show grid

    plt.xticks(k_values)  # Set x-ticks to be exact k values for better readability

    # Highlight the best k value

    max_accuracy = max(accuracies)  # Find the maximum accuracy

    best_k = k_values[accuracies.index(max_accuracy)]  # Find the k value that corresponds to the maximum accuracy

    plt.annotate(f'Best k={best_k}', xy=(best_k, max_accuracy),xytext=(best_k, max_accuracy),arrowprops=dict(facecolor='red', shrink=0.05),horizontalalignment='center')

    plt.show()

