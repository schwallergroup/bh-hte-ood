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
import collections
from datetime import datetime
dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

from concurrent.futures import ProcessPoolExecutor

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
from sklearn.calibration import CalibratedClassifierCV


def train_models(df, model_names = ["rf_model", "xgb_model", "lr_model"], feats_col = "DRFP+QM"):
    models = {}
    X = np.array(df[feats_col].tolist())  # DRFP is a list of 2048 elements
    y = df['labels']
    
    # Standardization for Logistic Regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if "rf_model" in model_names:        
        rf_model = RandomForestClassifier(n_estimators = 300, max_depth = None, min_samples_split = 4, # 20 and 2
                                          min_samples_leaf = 3, criterion = "entropy", #1 and “log_loss”
                                          max_features = "log2", random_state=42, n_jobs = 7)
        rf_model = CalibratedClassifierCV(rf_model, method='isotonic')
        rf_model.fit(X, y)
        models["rf_model"] = rf_model
    
    if "xgb_model" in model_names:
        # XGBoost
        xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
        xgb_model.fit(X, y)
        models["xgb_model"] = xgb_model
    
    if "lr_model" in model_names:
        # Logistic Regression
        lr_model = LogisticRegression(random_state=42, max_iter=400) # LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, class_weight='balanced') # 
        lr_model.fit(X_scaled, y)
        models["lr_model"] = lr_model
    
    if "svc_model" in model_names:
        #Support Vector Machines
        svc_model = SVC(kernel='linear', class_weight='balanced', probability=True)
        svc_model.fit(X_scaled, y)
        models["svc_model"] = svc_model
        
    if "gpc_model" in model_names:   
        
        C = 10
        kernel = 1.0 * RBF(length_scale=np.ones(X_scaled.shape[1]))       
        gpc_model = GaussianProcessClassifier(kernel)
        gpc_model.fit(X_scaled, y)
        models["gpc_model"] = gpc_model
        
    if "gaussianNB_model" in model_names:

        gaussianNB_model = GaussianNB()
        gaussianNB_model.fit(X_scaled, y)
        models["gaussianNB_model"] = gaussianNB_model
    
    if "mlp_model" in model_names:
        mlp_model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 20), max_iter=200, random_state=42)
        mlp_model.fit(X_scaled, y)
        models["mlp_model"] = mlp_model
        
    if "gradient_boosting_model" in model_names:
        gradient_boosting_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gradient_boosting_model.fit(X_scaled, y)
        models["gradient_boosting_model"] = gradient_boosting_model
        
    if "knn_model" in model_names:
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(X_scaled, y)
        models["knn_model"] = knn_model
    
    if "ada_boost_model" in model_names:
        ada_boost_model = AdaBoostClassifier(n_estimators=100, random_state=42)
        ada_boost_model.fit(X_scaled, y)
        models["ada_boost_model"] = ada_boost_model

    # Return trained models
    return models, scaler

def train_test_ood_cluster_static(df_HTE_BH, model_names, metrics_names, rand_state,  
                                  no_jjhte_train=True, reverse_clusters=False, features="DRFP", hyperparams=None,
                                  sample_train = 1):
    
    results_per_cluster_model = {}
    per_bucket_results = {}  # To store per-bucket metrics across splits

    # Initialize results dictionaries
    for model_name in model_names:
        results_per_cluster_model[model_name] = {}
        per_bucket_results[model_name] = {}
        for metric_name in metrics_names:
            results_per_cluster_model[model_name][metric_name] = []

    cluster_col_name = f'iteration_{rand_state} cluster' #
    test_sets = get_test_sets_per_iteration(df_HTE_BH)
    test_sets = test_sets[rand_state]

    # Iterate over each test cluster
    for clustergroup, test_clusters in enumerate(test_sets):
        if not reverse_clusters:
            df_train = df_HTE_BH[~df_HTE_BH[cluster_col_name].isin(test_clusters)]
            df_test = df_HTE_BH[df_HTE_BH[cluster_col_name].isin(test_clusters)]
        else:
            df_train = df_HTE_BH[df_HTE_BH[cluster_col_name].isin(test_clusters)]
            df_test = df_HTE_BH[~df_HTE_BH[cluster_col_name].isin(test_clusters)]

        # Remove specific source from the training set if required
        if no_jjhte_train:
            df_train = df_train[df_train["Source"] != "JNJ HTE 2024"]

        df_train = df_train.sample(frac=sample_train, random_state=0)
        
        #ensure aryls and amines combinations in test are novel to the model        
        #df_test = df_test[~df_test['Aryl_Amine'].isin(list(df_train["Aryl_Amine"].unique()))]
        df_test = df_test[~df_test['Aryl SMILES'].isin(list(df_train["Aryl SMILES"].unique()))]
        print(f"Test Size is: {len(df_test)}.")

        models, scaler = train_models(df_train, model_names, features)
        metrics, conf_bucket_metrics = calculate_metrics(models, scaler, df_test, features)

        # Append overall metrics for each model and metric name
        for model_name in model_names:
            for metric_name in metrics_names:
                results_per_cluster_model[model_name][metric_name].append(metrics[model_name][metric_name])

            # Collect per-bucket metrics
            for bucket_name, bucket_metrics in conf_bucket_metrics[model_name].items():
                if bucket_name not in per_bucket_results[model_name]:
                    per_bucket_results[model_name][bucket_name] = {}
                for metric_name, metric_value in bucket_metrics.items():
                    if metric_name not in per_bucket_results[model_name][bucket_name]:
                        per_bucket_results[model_name][bucket_name][metric_name] = []
                    # Convert 'None' strings or None to np.nan
                    if metric_value is None or metric_value == 'None':
                        metric_value = np.nan
                    per_bucket_results[model_name][bucket_name][metric_name].append(metric_value)

    # Calculate the average and standard deviation for each model and metric
    for model in results_per_cluster_model.keys():
        for metric in metrics_names:
            metric_values = results_per_cluster_model[model][metric]
            results_per_cluster_model[model][metric + "_avg"] = np.mean(metric_values)
            results_per_cluster_model[model][metric + "_std"] = np.std(metric_values)

    # Calculate the average and standard deviation per bucket
    per_bucket_avg_results = {}
    for model_name, buckets in per_bucket_results.items():
        per_bucket_avg_results[model_name] = {}
        for bucket_name, metrics_dict in buckets.items():
            per_bucket_avg_results[model_name][bucket_name] = {}
            for metric_name, values in metrics_dict.items():
                values_array = np.array(values, dtype=np.float64)
                per_bucket_avg_results[model_name][bucket_name][metric_name + '_avg'] = np.nanmean(values_array)
                per_bucket_avg_results[model_name][bucket_name][metric_name + '_std'] = np.nanstd(values_array)

    # Convert results_per_cluster_model to DataFrame for easier handling
    overall_metrics_df = pd.DataFrame(results_per_cluster_model)

    return overall_metrics_df, test_sets, per_bucket_avg_results

def get_test_sets_per_iteration(df):
    """
    Processes the DataFrame to extract clusters tagged as 'test' in each split of each iteration,
    ensuring that each cluster ID is only included once per split.

    Parameters:
    df (pd.DataFrame): The DataFrame containing 'iteration_{iteration} cluster' and
                       'iteration_{iteration} train_test' columns.

    Returns:
    list: A list of lists where each outer list corresponds to an iteration and contains a list of 
          NumPy arrays. Each inner NumPy array represents the unique cluster IDs tagged as 'test' in a split.
    """
    test_sets_per_iteration = []

    # Identify all iterations based on the column names
    iterations = set()
    for col in df.columns:
        if col.startswith('iteration_') and 'cluster' in col:
            iteration_num = int(col[len('iteration_'):col.find(' cluster')])
            iterations.add(iteration_num)
    iterations = sorted(iterations)

    # Process each iteration
    for iteration in iterations:
        cluster_col = f'iteration_{iteration} cluster'
        train_test_col = f'iteration_{iteration} train_test'

        clusters = df[cluster_col]
        train_test_lists = df[train_test_col]

        # Determine the number of splits
        num_splits = max(len(tt_list) for tt_list in train_test_lists if isinstance(tt_list, list))

        # Initialize lists to collect 'test' clusters for each split in this iteration
        split_test_clusters = [set() for _ in range(num_splits)]

        # Group the data by cluster_id to ensure each cluster is processed only once
        grouped = df.groupby(cluster_col)

        for cluster_id, group in grouped:
            # Assuming that train_test_list is the same for all entries with the same cluster_id
            tt_list = group[train_test_col].iloc[0]

            for i in range(num_splits):
                if tt_list[i] == 'test':
                    split_test_clusters[i].add(cluster_id)

        # Convert sets to sorted NumPy arrays and append to the result list
        split_test_clusters = [np.array(sorted(cluster_ids), dtype=np.int32) for cluster_ids in split_test_clusters]
        test_sets_per_iteration.append(split_test_clusters)

    return test_sets_per_iteration