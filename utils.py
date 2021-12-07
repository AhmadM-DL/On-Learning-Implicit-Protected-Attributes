import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os 

def plot_confusion_matrix(data, labels, title):
    fig, ax = plt.subplots()
    pos = ax.matshow(data, cmap='seismic')
    for (i, j), z in np.ndenumerate(data):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    fig.colorbar(pos)
    ax.set_ylabel("Ground Truth")
    ax.set_yticklabels([""]+labels)
    ax.set_xlabel("Predictions")
    ax.xaxis.set_label_position('top') 
    ax.set_xticklabels([""]+labels)
    ax.set_title(title, position=(0.5, 1.2))
    return ax

def data_split(seed, output_dir):
    TRAIN_PERCENT = 0.8
    VALID_PERCENT= 0.1
    TEST_PERCENT = 0.1
    ROOTDIR='/Datasets/Chexpert/csv/'
    output_filename = f"chexpert_single_img_per_patient_{TRAIN_PERCENT}_{VALID_PERCENT}_{ROOTDIR}_{seed}.csv"
    
    #read data
    data_df = pd.read_csv( os.path.join(ROOTDIR, 'train.csv') )
    demo_df = pd.DataFrame(pd.read_excel( os.path.join(ROOTDIR, "demographics.xlsx"), engine='openpyxl'))
    data_df["patient_id"] =  data_df.Path.str.split("/", expand = True)[2]
    demo_df = demo_df.rename(columns={'PATIENT': 'patient_id'})
    
    #combine demographics and train data 
    combine_df = data_df.merge(demo_df, on="patient_id", how="left")
    
    # Remove hispanic and latino and take only Frontal images
    combine_df.insert(3, "race", "")
    combine_df.loc[(combine_df.PRIMARY_RACE.str.contains("Black", na=False)), "race"] = "BLACK/AFRICAN AMERICAN"
    combine_df.loc[(combine_df.PRIMARY_RACE.str.contains("White", na=False)), "race"] = "WHITE"
    combine_df.loc[(combine_df.PRIMARY_RACE.str.contains("Asian", na=False)), "race"] = "ASIAN"
    combine_df = combine_df[combine_df.race.isin(['ASIAN','BLACK/AFRICAN AMERICAN','WHITE'])]
    combine_df = combine_df[combine_df.ETHNICITY.isin(["Non-Hispanic/Non-Latino","Not Hispanic"])]
    combine_df = combine_df[combine_df["Frontal/Lateral"]=="Frontal"]
    
    # Keep one image per patient id 
    combine_df = combine_df.drop_duplicates('patient_id')
    
    # Return the min value of patients per race
    n_patients = combine_df.race.value_counts().min()

    combine_df = shuffle(combine_df, random_state=seed)
    asian_df = combine_df[combine_df.race=="ASIAN"][:n_patients]
    black_df = combine_df[combine_df.race=="BLACK/AFRICAN AMERICAN"][:n_patients]
    white_df = combine_df[combine_df.race=="WHITE"][:n_patients]
    asian_df=_split(asian_df, TRAIN_PERCENT, VALID_PERCENT, TEST_PERCENT)
    white_df=_split(white_df, TRAIN_PERCENT, VALID_PERCENT, TEST_PERCENT)
    black_df=_split(black_df, TRAIN_PERCENT, VALID_PERCENT, TEST_PERCENT)
    
    # Combine the splits
    frames=[asian_df, black_df, white_df]
    all_df = pd.concat(frames)
    
    # Save only index and split columns
    split_df = all_df.reset_index()[["index", "split"]]
    split_df.to_csv(os.path.join(output_dir, output_filename), index= False)
    return split_df 

def _split(df,train_percent,valid_percent,test_percent):
    # Use data of patients of black, white, and asian race and only frontal images
    # Split based on patient id
    unique_sub_id = df.patient_id.unique()
    value1 = (round(len(unique_sub_id)*(train_percent*100)))
    value2 = (round(len(unique_sub_id)*(valid_percent*100)))
    value3 = (round(len(unique_sub_id)*(test_percent*100)))

    train_sub_id = unique_sub_id[:value1]
    validate_sub_id = unique_sub_id[value1:value1+value2]
    test_sub_id = unique_sub_id[value1+value2:]

    # Populate split column
    df.loc[df.patient_id.isin(train_sub_id), "split"]="train"
    df.loc[df.patient_id.isin(validate_sub_id), "split"]="validate"
    df.loc[df.patient_id.isin(test_sub_id), "split"]="test"
    return df

def plot_aur_roc_curves():
    return


