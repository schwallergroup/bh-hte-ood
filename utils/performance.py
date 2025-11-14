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

from pycm import ConfusionMatrix

from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, f1_score, r2_score, \
                            accuracy_score, cohen_kappa_score, balanced_accuracy_score, \
                            precision_score, recall_score, precision_recall_curve

def get_roc_auc(dft):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(1):
        fpr[i], tpr[i], _ = roc_curve(dft['labels'].values, dft['prediction'].values, pos_label=1)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    #y_test = dft['reactivity'].values

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(dft['labels'].values.ravel(), dft['prediction'].values.ravel(), pos_label=1)# pred_reactivity_int ,prediction
    auc_res = metrics.auc(fpr["micro"], tpr["micro"])
    return auc_res


def calculate_model_performance(dataframes, metrics, selected_metrics):
    # Filter each dataframe to include only the rows that match the performance metrics
    filtered_dataframes = []
    for df in dataframes:
        # Select only rows that match the provided metrics
        if type(df) == dict: 
            df = pd.DataFrame.from_dict(df)
        
        filtered_df = df.loc[df.index.intersection(metrics)]
        filtered_dataframes.append(filtered_df)
    
    # Concatenate all filtered dataframes along a new axis
    combined_df = pd.concat(filtered_dataframes, axis=0, keys=range(len(filtered_dataframes)))
    
    # Ensure that all the data is numeric (remove non-numeric columns or rows)
    combined_df = combined_df.apply(pd.to_numeric, errors='coerce')

    # Calculate the mean and std across the dataframes for each performance metric
    means_df = combined_df.groupby(level=1).mean()  # Mean across the dataframes for each model
    stds_df = combined_df.groupby(level=1).std()    # Std deviation across the dataframes for each model

    # Filter the dataframes based on the provided list of performance metrics
    means_filtered = means_df.loc[metrics]
    stds_filtered = stds_df.loc[metrics]

    # Create a new dataframe that combines the mean and std for each model
    summary_df = pd.DataFrame()
    for metric in metrics:
        summary_df[f'{metric} Avg'] = means_filtered.loc[metric]
        summary_df[f'{metric} Std'] = stds_filtered.loc[metric]

    # Calculate the average performance based on the selected metrics to find the best model
    selected_means = means_filtered.loc[selected_metrics].mean()
    
    # Find the best model (which has the highest average performance in the selected metrics)
    best_model = selected_means.idxmax()

    return summary_df, best_model

def calculate_metrics(models, scaler, test_data, feats_col):
    """
    Calculate performance metrics for given models and test data.

    Args:
    models (dict): Dictionary of trained models {'rf_model': rf_model, 'xgb_model': xgb_model, 'lr_model': lr_model}.
    scaler (scaler object): Scaler that was used for the feature scaling.
    test_data (pd.DataFrame): DataFrame containing the test data with features in the 'DRFP' or 'QM' column and labels in the 'labels' column.
    with_QM (bool): Flag to indicate if 'QM' features are to be included.

    Returns:
    metrics (dict): Dictionary with overall metrics for each model.
    bucket_metrics (dict): Dictionary with metrics for each confidence bucket.
    """
    metrics = {}
    bucket_metrics = {}

    # Extract features and labels
    X_test = np.vstack(test_data[feats_col])
    y_test = test_data['labels'].values

    # Scale the features
    X_test_scaled = scaler.transform(X_test)

    # Define confidence buckets
    buckets = np.arange(0.5, 1.05, 0.1)  # Buckets from 0.5 to 1.0 in steps of 0.1

    for model_name, model in models.items():
        # Get the model predictions and predicted probabilities
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probability for the positive class (1)
        
        # Apply confidence transformation
        confidence_scores = abs(y_proba - 0.5) + 0.5

        # Calculate overall metrics
        if len(set(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, y_proba)
            f1 = f1_score(y_test, y_pred)
            f0 = f1_score(y_test, y_pred, pos_label=0) 
        elif len(set(y_test)) == 1:
            roc_auc = accuracy_score(y_test, y_pred)
            if y_test[0] == 0:
                f1 = accuracy_score(y_test, y_pred)
                f0 = f1_score(y_test, y_pred, pos_label=0) 
            elif y_test[0] == 1:
                f0 = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

        bal_acc = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
        auprc = auc(recall_curve, precision_curve)

        # Calculate additional metrics using pycm
        cmat = ConfusionMatrix(actual_vector=y_test, predict_vector=y_pred)

        # Store overall metrics
        metrics[model_name] = {
            'ROC AUC': roc_auc,
            'Balanced Accuracy': bal_acc,
            'F1 Score': f1,
            'F0 Score': f0,
            'AU-PR-C': auprc,
            'Precision 1': precision,
            'Recall 1': recall,
        }

        # Initialize bucket metrics dictionary for this model
        bucket_metrics[model_name] = {}

        # Calculate metrics per confidence bucket
        for i in range(len(buckets) - 1):
            bucket_min = buckets[i]
            bucket_max = buckets[i + 1]
           
            # Select samples in the current confidence bucket
            if i == len(buckets) - 2:
                bucket_mask = (confidence_scores >= bucket_min) & (confidence_scores <= bucket_max)
            else:
                bucket_mask = (confidence_scores >= bucket_min) & (confidence_scores < bucket_max)
                
            y_test_bucket = y_test[bucket_mask]
            y_pred_bucket = y_pred[bucket_mask]
            y_proba_bucket = y_proba[bucket_mask]
            
            data_point_count = len(y_test_bucket)

            if len(y_test_bucket) > 0:
                # Calculate bucket-specific metrics
                if len(set(y_test_bucket)) == 2:
                    roc_auc_bucket = roc_auc_score(y_test_bucket, y_proba_bucket)
                    f1_bucket = f1_score(y_test_bucket, y_pred_bucket)
                    f0_bucket = f1_score(y_test_bucket, y_pred_bucket, pos_label=0)
                else:
                    roc_auc_bucket = accuracy_score(y_test_bucket, y_pred_bucket)
                    if y_test_bucket[0] == 0:
                        f1_bucket = accuracy_score(y_test_bucket, y_pred_bucket)
                        f0_bucket = f1_score(y_test_bucket, y_pred_bucket, pos_label=0)
                    elif y_test_bucket[0] == 1:
                        f0_bucket = accuracy_score(y_test_bucket, y_pred_bucket)
                        f1_bucket = f1_score(y_test_bucket, y_pred_bucket)

                bal_acc_bucket = balanced_accuracy_score(y_test_bucket, y_pred_bucket)
                precision_bucket = precision_score(y_test_bucket, y_pred_bucket, zero_division=0)
                recall_bucket = recall_score(y_test_bucket, y_pred_bucket)

                # Ensure all metrics are numeric or np.nan
                metrics_dict = {
                    'ROC AUC': roc_auc_bucket,
                    'Balanced Accuracy': bal_acc_bucket,
                    'F1 Score': f1_bucket,
                    'F0 Score': f0_bucket,
                    'Precision 1': precision_bucket,
                    'Recall 1': recall_bucket,
                    # You can include AU-PR-C for buckets if needed
                    # 'AU-PR-C': auprc_bucket,  # Uncomment and calculate if desired
                }

                # Convert None to np.nan
                for key, value in metrics_dict.items():
                    if value is None or value == 'None':
                        metrics_dict[key] = np.nan
                    else:
                        metrics_dict[key] = float(value)

                # Store metrics for this bucket
                bucket_metrics[model_name][f'{bucket_min:.1f}-{bucket_max:.1f}'] = {
                    **metrics_dict,
                    'Data Points': data_point_count
                }
            else:
                # If the bucket is empty, fill with np.nan
                bucket_metrics[model_name][f'{bucket_min:.1f}-{bucket_max:.1f}'] = {
                    'ROC AUC': np.nan,
                    'Balanced Accuracy': np.nan,
                    'F1 Score': np.nan,
                    'F0 Score': np.nan,
                    'Precision 1': np.nan,
                    'Recall 1': np.nan,
                    'Data Points': 0
                }

    return metrics, bucket_metrics

def combine_per_bucket_results(conf_bucket_all_results):
    # Initialize a dictionary to aggregate results across all iterations
    combined_per_bucket_results = {}

    # Aggregate results from each iteration
    for per_bucket_avg_results in conf_bucket_all_results:
        for model_name, buckets in per_bucket_avg_results.items():
            if model_name not in combined_per_bucket_results:
                combined_per_bucket_results[model_name] = {}
            for bucket_name, metrics_dict in buckets.items():
                if bucket_name not in combined_per_bucket_results[model_name]:
                    combined_per_bucket_results[model_name][bucket_name] = {'Data Points': []}

                for metric_key, value in metrics_dict.items():
                    # If average, split the key to remove '_avg' for compatibility
                    if metric_key.endswith('_avg'):
                        metric_name = metric_key[:-4]  # Remove '_avg'
                        if metric_name not in combined_per_bucket_results[model_name][bucket_name]:
                            combined_per_bucket_results[model_name][bucket_name][metric_name] = []
                        combined_per_bucket_results[model_name][bucket_name][metric_name].append(value)
                        
                    # Collect data point counts
                    if metric_key == 'Data Points':
                        combined_per_bucket_results[model_name][bucket_name]['Data Points'].append(value)

    # Calculate final average and standard deviation for each model, bucket, and metric
    final_per_bucket_results = {}
    for model_name, buckets in combined_per_bucket_results.items():
        final_per_bucket_results[model_name] = {}
        for bucket_name, metrics_dict in buckets.items():
            final_per_bucket_results[model_name][bucket_name] = {}
            for metric_name, values_list in metrics_dict.items():
                if metric_name != 'Data Points':
                    # Compute avg/standard deviation for each metric
                    values_array = np.array(values_list, dtype=np.float64)
                    mean_value = np.nanmean(values_array)
                    std_value = np.nanstd(values_array)
                    final_per_bucket_results[model_name][bucket_name][metric_name + '_avg'] = mean_value
                    final_per_bucket_results[model_name][bucket_name][metric_name + '_std'] = std_value
                else:
                    # Average the data points, convert sum to average across iterations
                    mean_data_points = np.mean(values_list) if values_list else 0
                    final_per_bucket_results[model_name][bucket_name]['Data Points_avg'] = mean_data_points

    return final_per_bucket_results