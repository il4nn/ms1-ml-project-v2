import numpy as np


class KNN(object):
    """
        kNN classifier object.
    """

    ##Added by me 

    def find_k_nearest_neighbors(self,k, distances):
        """ Find the indices of the k smallest distances from a list of distances.
            Tip: use np.argsort()

        Inputs:
            k: integer
            distances: shape (N,) 
        Outputs:
            indices of the k nearest neighbors: shape (k,)
        """
        indices = np.argsort(distances)
        return indices[:k]

    def euclidean_dist(self,example, training_examples):
        """Compute the Euclidean distance between a single example
        vector and all training_examples.

        Inputs:
            example: shape (D,)
            training_examples: shape (NxD) 
        Outputs:
            euclidean distances: shape (N,)
        """
        return np.sqrt(np.sum(np.square(example - training_examples),axis = 1))


    def chi_square_dist(self,example, training_examples, epsilon=1e-10):
        """Compute the Chi square distance between a single example
        vector and all training_examples.

        Inputs:
            example: shape (D,)
            training_examples: shape (NxD) 
        Outputs:
            euclidean distances: shape (N,)
        """   
        return np.sum(np.square(example - training_examples) / (example + training_examples), axis=1)
 
    def l3_norm(example, training_examples,epsilon=1e-10):
        return (np.sum((np.abs(example-training_examples))**3,axis=1))**(1/3)

    def l4_norm(example, training_examples):
        return (np.sum((example-training_examples)**4,axis=1))**(1/4)



    def cosine_distances_to_all( example,training_examples):

        # Normaliser chaque vecteur d'entraînement

        normalized_training = training_examples / np.linalg.norm(training_examples, axis=1, keepdims=True)

        # Normaliser l'exemple

        normalized_example = example / np.linalg.norm(example)

        # Calculer la similarité cosinus pour tous les exemples

        cosine_similarities = normalized_training @ normalized_example

        # Convertir les similarités en distances

        return 1 - cosine_similarities
    

    def predict_label_aux(self,neighbor_labels):
        """Return the most frequent label in the neighbors'.

        Inputs:
            neighbor_labels: shape (N,) 
        Outputs:
            most frequent label
        """
        return np.argmax(np.bincount(neighbor_labels))

    def kNN_one_example(self,unlabeled_example, training_features, training_labels, k, task):
        """Returns the label of a single unlabelled example.

        Inputs:
            unlabeled_example: shape (D,) 
            training_features: shape (NxD)
            training_labels: shape (N,) 
            k: integer
        Outputs:
            predicted label
        """    
        # Compute distances
        distances = self.euclidean_dist(unlabeled_example,training_features)
        
        # Find neighbors
        nn_indices = self.find_k_nearest_neighbors(k,distances)
        
        # Get neighbors' labels
        neighbor_labels = training_labels[nn_indices]

        # Pick the most common
        if task == "classification":
            return self.predict_label_aux(neighbor_labels)
        else:
            return np.mean(neighbor_labels,axis=0)

    ### Provided code
    def __init__(self, k=1, task_kind = "classification"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """
        self.training_data = training_data
        self.training_labels = training_labels
        return self.predict(training_data)

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        return np.apply_along_axis(self.kNN_one_example,1,test_data,self.training_data,self.training_labels,self.k,self.task_kind)

    

