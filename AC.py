import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import time
import os
import pickle as pickle
from re import X # Kept as in user's original code
from datetime import datetime
# Generate random repairs and replace with edge repairs if available
import warnings
warnings.filterwarnings("ignore")
import argparse
from torchvision import datasets, transforms
import torch
from sklearn.decomposition import PCA
import utility_function as ut
from active_clean_drive import active_clean_driver 
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--itter_method_loop", type=str, default="gt", help="Imputation method to use in iteration loop")
# parser.add_argument("--dataset", type=str, default="tuadromd", help="Dataset choice")
parser.add_argument("--specific_data", type=str, default="./MI/SVM/synthetic/ICLR_2025/Data/default_MNAR_train_60.csv", help="Have specific data path")
# parser.add_argument("--specific_data", type=str, default="", help="Have specific data path")
# parser.add_argument("--batch", type=int, default=10, help="batch size")
parser.add_argument("--top", type=float, default=0.3, help="top MR percentage")
args = parser.parse_args()
itter_method_loop = args.itter_method_loop

SPECIFIC_DATA = args.specific_data
print(SPECIFIC_DATA)
top_k = args.top
SEEDS_TO_TRY = [42,43,45]
# SAVE_DIR = './Data MR/'
K=8
type_miss =  Path(SPECIFIC_DATA).stem.split("_")[1]
dataset = Path(SPECIFIC_DATA).stem.split("_")[0]
miss = int(Path(args.specific_data).stem.rsplit("_", 1)[1])
DATASET_CHOICE = dataset


batch_map = {
    "tuadromd": 10,
    "malware": 10,
    "mnist": 100,
    "default": 100,
    "winnipegs1": 500,
    "winnipegs2": 500,
    "susy": 10000,
    "fraud": 10000,
}

# override args.batch if name is in the map
if DATASET_CHOICE in batch_map:
    BATCH_SIZE = batch_map[DATASET_CHOICE]

print(f"Dataset: {DATASET_CHOICE}, Batch size: {BATCH_SIZE}")


#############################################################################################
#                                                                                           #
#                                   MAIN STARTS HERE                                        #
#                                                                                           #
#############################################################################################

# Main execution block exactly as it was in the Canvas artifact "optimized_imputation_code_minimal_changes"
# WITH CORRECTION FOR 'name' vs 'name_main'
if __name__ == '__main__':
    print("Run started at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    if SPECIFIC_DATA != "":
        df_1 = pd.read_csv(SPECIFIC_DATA)
        X_train_np, Y_train_np = ut.get_Xy(df_1)

        # scaler = MinMaxScaler()
        # X_temp = np.copy(X_train_np)

        # # Replace NaNs with column means temporarily
        # col_mean = np.nanmean(X_temp, axis=0)
        # inds = np.where(np.isnan(X_temp))
        # X_temp[inds] = np.take(col_mean, inds[1])

        # # Scale
        # X_scaled = scaler.fit_transform(X_temp)

        # # Restore NaNs
        # X_scaled[inds] = np.nan

        # X_train_np = np.copy(X_scaled)

        # print(save_name)
        # exit()
        # print(df.shape)
        print(X_train_np.shape)
        # print(Y_train_np.shape)


        # #DELETE THIS AFTER FIXING
        # file_path = './MI/SVM/synthetic/ICLR_2025/Data/tuadromd.csv' # CORRECTED
        # name = "tuadromd" # CORRECTED
        # col_num = 48       # CORRECTED
        # df = pd.read_csv(file_path) # CORRECTED
        # last_column_index = df.columns[-1] # CORRECTED
        # df[last_column_index] = df[last_column_index].replace({'malware': 1, 'goodware': -1}).astype(int) # CORRECTED
        # OG_train_data_df, OG_test_data_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True) # CORRECTED (uses df)
        # # _,Y_train_np = ut.get_Xy(OG_train_data_df)
        # X_test_np, Y_test_np = ut.get_Xy(OG_test_data_df)
        # missing_level = [1]

        

    print("DATASET:", DATASET_CHOICE)
    print("TOP_K: ", top_k)
    missing_level = [0.2] 

    file_path = None
    name = None # CORRECTED: This will be the 'name' variable used later
    col_num = None
    df = None   # CORRECTED: This will be the 'df' variable used later

    if DATASET_CHOICE == "tuadromd":
        file_path = 'MI/SVM/synthetic/ICLR_2025/Data/tuadromd.csv' # CORRECTED
        name = "tuadromd" # CORRECTED
        col_num = 48       # CORRECTED
        df = pd.read_csv(file_path) # CORRECTED
        last_column_index = df.columns[-1] # CORRECTED
        df[last_column_index] = df[last_column_index].replace({'malware': 1, 'goodware': -1}).astype(int) # CORRECTED

    elif DATASET_CHOICE == "malware":
        data_file = './Minimal-Imputation/Synthetic-Datasets/REJAFADA.data' # Kept original 'data_file' for this block
        name = "malware"  # CORRECTED
        df = pd.read_csv(data_file) # CORRECTED
        df = df.drop(df.columns[0], axis=1)
        first_column = df.pop(df.columns[0])
        df[df.columns[-1]] = first_column 
        df[df.columns[-1]] = df[df.columns[-1]].replace({'M': 1, 'B': -1})
        col_num = 48  # CORRECTED

    elif DATASET_CHOICE == "default":
        data_file = './MI/SVM/synthetic/data/original/default.csv'
        name = "default"
        df = pd.read_csv(data_file, header=None).iloc[2:, 1:]
        df = df.astype(float)
        col_num = 10

    elif DATASET_CHOICE == "mnist":
        transform = transforms.Compose([   
        transforms.ToTensor()
        ])

        # 2. Load the full MNIST training and test sets and extract 0 and 1 only, you could change this to whatever number
        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        X_train_OG_np, Y_train_OG_np = ut.filter_digits(train_dataset, digits=(0, 1))

        print(X_train_OG_np.shape)
        X_test_OG_np, Y_test_OG_np = ut.filter_digits(test_dataset, digits=(0, 1))
        print(X_test_OG_np.shape)

        print("Any NaN:", np.isnan(X_train_OG_np).any())
        print("Any inf:", np.isinf(X_train_OG_np).any())
        print("Max:", np.nanmax(X_train_OG_np))
        print("Min:", np.nanmin(X_train_OG_np))
        print("dtype:", X_train_OG_np.dtype)

        col_num = 200
    elif DATASET_CHOICE == "winnipegs1":
        data_file = './MI/SVM/synthetic/data/original/filtered_1_5_normalized_winipegs.csv'
        name = "winipegs"
        df = pd.read_csv(data_file)

        col_num = 50

    elif DATASET_CHOICE == "winnipegs2":
        data_file = './MI/SVM/synthetic/data/original/filtered_3_6_normalized_winipegs.csv'
        name = "winipegs"
        df = pd.read_csv(data_file)

        col_num = 50
    elif DATASET_CHOICE == "susy":
        OG_test_data_df = pd.read_csv("MI/SVM/synthetic/data/og_splits/susy_OG_test.csv", header=0)
        OG_train_data_df = pd.read_csv("MI/SVM/synthetic/data/og_splits/susy_OG_train.csv", header=0)
        name = "susy"
    elif DATASET_CHOICE == "fraud":
        OG_test_data_df = pd.read_csv("MI/SVM/synthetic/data/og_splits/fraud_OG_test.csv", header=0)
        OG_train_data_df = pd.read_csv("MI/SVM/synthetic/data/og_splits/fraud_OG_train.csv", header=0)
        name = "fraud"
    else:
        raise ValueError(f"Invalid DATASET_CHOICE: {DATASET_CHOICE}")
            

    for missingness in missing_level:

        test_data = OG_test_data_df.copy()
        X_test_np, Y_test_np = ut.get_Xy(test_data) 



        print("Any NaN:", np.isnan(X_train_np).any())
        print("Any inf:", np.isinf(X_train_np).any())
        print("Max:", np.nanmax(X_train_np))
        print("Min:", np.nanmin(X_train_np))
        print("dtype:", X_train_np.dtype)


        ##TEST
        # df = pd.read_csv("./Data/bank_MAR_20.csv")
        # X_train_np, _ = ut.get_Xy(df)
        total_examples = len(X_train_np)
        rows_with_missing_values = np.isnan(X_train_np).any(axis=1).sum() 
        # rows_with_missing_values = pd.DataFrame(X_train_np).isna().any(axis=1).sum()
        missing_factor = rows_with_missing_values / total_examples if total_examples > 0 else 0
        
        print("Number of rows with missing values:", rows_with_missing_values)
        print(f"Total example {X_train_np.shape}, MISSING FACTOR: {missing_factor}") 
        min_time = float('inf') 
        seed_main_loop = None # This was 'seed' in original user code, but 'seed' is also a param in SGD_class.
                            # Kept 'seed_main_loop' to avoid conflict if it was intended for something else.
                            # If it was meant to be the same seed for SGD_class, this logic would need adjustment.
        number_of_example_dropped = None     

        accuracy_seed = [] 
        MI_time_seed = [] 

        train_data_scaled = df_1.copy()

        test_data_scaled = OG_test_data_df.copy()



        # X_train, Y_train = ut.get_Xy(train_data) # This will give a numpy array on X_train and Y_train
        # X_test, Y_test = ut.get_Xy(test_data) # This will give a numpy array on X_test and Y_test

        indi = []
        #-----------------------------Active Clean------------------------------------------
        for i in range(1):
            examples_cleaned_AC, _ ,training_time_AC, indi = active_clean_driver(train_data_scaled, test_data_scaled)
            unique = np.unique(indi)
            print(f"Examples Cleaned AC: {examples_cleaned_AC}")
            print(f"Training_Time_AC: {training_time_AC}\n")
            # print(unique)

            train_mask_complete = ~np.isnan(X_train_np).any(axis=1)
            # # test_mask_complete  = ~np.isnan(X_test_np).any(axis=1)

            # # Split training data
            X_tr_drop = X_train_np[train_mask_complete]
            Y_tr_drop = Y_train_np[train_mask_complete]
            mask_keep = train_mask_complete | np.isin(np.arange(len(X_train_np)), unique)
            X_tr_missing = X_train_np[mask_keep]
            Y_tr_missing = Y_train_np[mask_keep]

            combined = np.hstack((X_tr_missing, Y_tr_missing.reshape(-1, 1)))
            n_features = X_tr_missing.shape[1]
            columns = [f'x{i}' for i in range(n_features)] + ['y']
            df = pd.DataFrame(combined, columns=columns)
            print(df.shape)

        
            df.to_csv(f"MI/SVM/synthetic/ICLR_2025/Data_AC3/{DATASET_CHOICE}_{type_miss}_{miss}_AC_{i}.csv", index=False, header=False)

            X_train_OG, Y_train_OG = ut.get_Xy(OG_train_data_df)
            X_tr = X_train_OG[mask_keep]
            Y_tr = Y_train_OG[mask_keep]


            train_accuracies = []
            test_accuracies = []
            durations = []

            for trial in range(10):
                train_acc, test_acc, duration = ut.SGD_class(X_tr, Y_tr, X_test_np, Y_test_np)
                train_accuracies.append(train_acc)
                test_accuracies.append(test_acc)
                durations.append(duration)
            top_k_accs_test = sorted(test_accuracies, reverse=True)[:K]
            top_k_accs_train = sorted(train_accuracies, reverse=True)[:K]

            # Final average
            avg_test_acc = np.mean(top_k_accs_test)
            std_test_acc = np.std(top_k_accs_test)
            avg_train_acc = np.mean(train_accuracies)
            avg_duration = np.mean(durations)
            std_duration = np.std(durations)

            print(f"Average train accuracy: {avg_train_acc:.4f}")
            print(f"Average test accuracy: {avg_test_acc:.4f}")
            print(f"Test accuracy standard deviation: {std_test_acc:.4f}")
            print(f"Average duration: {avg_duration:.4f}")
            print(f"Duration standard deviation: {std_duration:.4f}")
            print("total time: ", avg_duration)
            print("")