from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import sys
import os
import re
import glob
import copy
import pickle
from datetime import datetime
dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

from matplotlib.cm import viridis
from matplotlib.ticker import MaxNLocator

from pycm import ConfusionMatrix

import sys

import collections

from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, f1_score, r2_score, \
                            accuracy_score, cohen_kappa_score, balanced_accuracy_score, \
                            precision_score, recall_score, precision_recall_curve

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier

from concurrent.futures import ProcessPoolExecutor

from rdkit.Chem import MolFromSmiles

def get_morgan_fingerprint(smiles, radius=3, nBits=2048):
    """Generate Morgan fingerprint for a molecule given a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)

def compute_drfp(reaction_smiles, radius=3, nBits=2048):
    """
    Compute DRFP vector for a chemical reaction given as a SMILES string.
    
    Args:
        reaction_smiles (str): Reaction SMILES in the form "reactants >> products".
        radius (int): Radius of the Morgan fingerprint.
        nBits (int): Size of the fingerprint bit vector.
    
    Returns:
        numpy.array: DRFP vector representing the binary difference between product and reactant fingerprints.
    """
    # Split the reaction SMILES into reactants and products
    try:
        reactants_smiles, products_smiles = reaction_smiles.split(">>")
    except ValueError:
        raise ValueError("Invalid reaction SMILES format. It should be 'reactants>>products'.")

    # Initialize fingerprint bit vectors for reactants and products
    reactants_fps = np.zeros((nBits,), dtype=int)
    products_fps = np.zeros((nBits,), dtype=int)
    
    # Generate fingerprint for each reactant and combine using OR (since we want a binary presence)
    for smiles in reactants_smiles.split('.'):
        reactants_fps |= np.array(get_morgan_fingerprint(smiles, radius, nBits))
    
    # Generate fingerprint for each product and combine using OR
    for smiles in products_smiles.split('.'):
        products_fps |= np.array(get_morgan_fingerprint(smiles, radius, nBits))
    
    # Compute the binary difference (XOR): DRFP = products_fps XOR reactants_fps
    drfp_vector = np.bitwise_xor(products_fps, reactants_fps)
    
    return drfp_vector
    
def add_clusters_and_labels(df, n_clusters=20, n_clusters_in_test=7, num_iterations=20):
    """
    For each iteration, adds cluster assignments and train/test labels to the DataFrame.

    Parameters:
    - df: The DataFrame with 'DRFP' column (list of 0s and 1s representing reactions).
    - n_clusters: The number of clusters to form. Default is 20.
    - n_clusters_in_test: Number of clusters to include in each test set. Default is 7.
    - num_iterations: Number of iterations with different random states. Default is 20.

    Returns:
    - df: The DataFrame with additional cluster assignment and train/test label columns for each iteration.
    """
    df = df.copy()  # Make a copy to avoid modifying the original DataFrame

    for iteration in tqdm(range(num_iterations), total = num_iterations):
        random_state = iteration

        # Cluster assignment column name
        cluster_col_name = f'iteration_{iteration} cluster'

        # Extract features
        X = np.array(df['DRFP'].tolist())  # Convert DRFP lists to a 2D array

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        df[cluster_col_name] = kmeans.fit_predict(X)

        # Get unique clusters and shuffle them
        clusters = df[cluster_col_name].unique()

        # Set the random seed for shuffling clusters
        np.random.seed(random_state)
        np.random.shuffle(clusters)  # Shuffle clusters to randomize test sets

        # Split clusters into groups of n_clusters_in_test
        test_sets = [clusters[i:i + n_clusters_in_test] for i in range(0, len(clusters), n_clusters_in_test)]

        # Initialize list to store train/test labels for each reaction
        train_test_labels = []

        # For each reaction, determine 'train'/'test' labels per test set grouping
        for idx, row in df.iterrows():
            cluster = row[cluster_col_name]
            labels_per_grouping = []

            for test_set in test_sets:
                if cluster in test_set:
                    labels_per_grouping.append('test')
                else:
                    labels_per_grouping.append('train')

            train_test_labels.append(labels_per_grouping)

        # Add the labels as a column
        labels_col_name = f'iteration_{iteration} train_test'
        df[labels_col_name] = train_test_labels

    return df

def compute_drfp_wrapper(reaction_smiles):
    try:
        return compute_drfp(reaction_smiles)
    except Exception as e:
        print(f"Error processing {reaction_smiles}: {e}")
        return None

def parallel_apply(df, func, workers=4, drfp_col = "DRFP", rxn_col = "Reaction SMILES"):
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(executor.map(func, df[rxn_col]), total=len(df)))
    df[drfp_col] = results
    return df

