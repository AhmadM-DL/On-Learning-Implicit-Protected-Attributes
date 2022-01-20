import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os 

def plot_confusion_matrix(data, labels, title,output_dir):
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
    fig.savefig(output_dir)
    return ax

def validate_split(split_filename, check_stratified_on_race=False):
  ROOTDIR='./Datasets/Chexpert/csv/'
  
  data_df = pd.read_csv( os.path.join(ROOTDIR, 'train.csv') )
  split_df = pd.read_csv(split_filename)
  demo_df = pd.DataFrame(pd.read_excel( os.path.join(ROOTDIR, "demographics.xlsx"), engine='openpyxl'))

  demo_df = demo_df.rename(columns={'PATIENT': 'patient_id'})
  data_df["patient_id"] = data_df.Path.str.split("/", expand = True)[2]
  data_df = pd.merge(split_df, data_df.reset_index(), on="index", how="left")
  data_df = data_df[~data_df.split.isna()]

  combine_df = data_df.merge(demo_df, on="patient_id", how="left")
  combine_df.insert(3, "race", "")
  combine_df.loc[(combine_df.PRIMARY_RACE.str.contains("Black", na=False)), "race"] = "BLACK/AFRICAN AMERICAN"
  combine_df.loc[(combine_df.PRIMARY_RACE.str.contains("White", na=False)), "race"] = "WHITE"
  combine_df.loc[(combine_df.PRIMARY_RACE.str.contains("Asian", na=False)), "race"] = "ASIAN"

  train_df = combine_df[combine_df.split=="train"]
  validation_df = combine_df[combine_df.split=="validate"]
  test_df = combine_df[combine_df.split=="test"]

  # Assert we are not using data from one patient to another
  assert train_df.patient_id.isin(validation_df.patient_id).sum() ==  0
  assert train_df.patient_id.isin(test_df.patient_id).sum() ==  0
  assert validation_df.patient_id.isin(test_df.patient_id).sum() ==  0

  # Assert is balanced over race
  assert len(combine_df.race.value_counts().unique()) == 1

  # Test Stratified | Should be equal size in each split / race
  if check_stratified_on_race:
    assert len(train_df.race.value_counts().unique()) == 1
    assert len(test_df.race.value_counts().unique()) == 1
    assert len(validation_df.race.value_counts().unique()) == 1

def race_balanced_split(seed, output_dir, max_images_per_patient, splits_ratio = (.8, .1, .1)):
    ROOTDIR='./Datasets/Chexpert/csv/'
    output_filename = f"chexpert_{splits_ratio[0]}_{splits_ratio[1]}_{splits_ratio[2]}_{seed}_{max_images_per_patient}.csv"

    # Read data
    data_df = pd.read_csv( os.path.join(ROOTDIR, 'train.csv') )
    demo_df = pd.DataFrame(pd.read_excel( os.path.join(ROOTDIR, "demographics.xlsx"), engine='openpyxl'))
    data_df["patient_id"] =  data_df.Path.str.split("/", expand = True)[2]
    demo_df = demo_df.rename(columns={'PATIENT': 'patient_id'})

    # Combine demographics and train data 
    combine_df = data_df.merge(demo_df, on="patient_id", how="left")

    # Remove hispanic and latino and take only Frontal images
    combine_df.insert(3, "race", "")
    combine_df.loc[(combine_df.PRIMARY_RACE.str.contains("Black", na=False)), "race"] = "BLACK/AFRICAN AMERICAN"
    combine_df.loc[(combine_df.PRIMARY_RACE.str.contains("White", na=False)), "race"] = "WHITE"
    combine_df.loc[(combine_df.PRIMARY_RACE.str.contains("Asian", na=False)), "race"] = "ASIAN"
    combine_df = combine_df[combine_df.race.isin(['ASIAN','BLACK/AFRICAN AMERICAN','WHITE'])]
    combine_df = combine_df[combine_df.ETHNICITY.isin(["Non-Hispanic/Non-Latino","Not Hispanic"])]
    combine_df = combine_df[combine_df["Frontal/Lateral"]=="Frontal"]

    # Keep up to max_images_per_patient per patient id 
    combine_df = combine_df.groupby("patient_id").head(max_images_per_patient)

    # Return the min value of images per race
    min_imgs_per_race = combine_df.race.value_counts().min()

    combine_df = shuffle(combine_df, random_state=seed)
    asian_df = combine_df[combine_df.race=="ASIAN"][:min_imgs_per_race]
    black_df = combine_df[combine_df.race=="BLACK/AFRICAN AMERICAN"][:min_imgs_per_race]
    white_df = combine_df[combine_df.race=="WHITE"][:min_imgs_per_race]
    asian_df=_split(asian_df, splits_ratio[0], splits_ratio[1], splits_ratio[2])
    white_df=_split(white_df, splits_ratio[0], splits_ratio[1], splits_ratio[2])
    black_df=_split(black_df, splits_ratio[0], splits_ratio[1], splits_ratio[2])

    # Combine the splits
    frames=[asian_df, black_df, white_df]
    all_df = pd.concat(frames)

    # Save only index and split columns
    split_df = all_df.reset_index()[["index", "split"]]
    split_df.to_csv(os.path.join(output_dir, output_filename), index= False)    

    validate_split(os.path.join(output_dir, output_filename))
    return split_df

def _split(df, train_ratio, valid_ratio, test_ratio):
    # Use data of patients of black, white, and asian race and only frontal images
    # Split based on patient id
    unique_sub_id = df.patient_id.unique()
    value1 = (round(len(unique_sub_id)*(train_ratio)))
    value2 = (round(len(unique_sub_id)*(valid_ratio)))
    value3 = (round(len(unique_sub_id)*(test_ratio)))

    train_sub_id = unique_sub_id[:value1]
    validate_sub_id = unique_sub_id[value1:value1+value2]
    test_sub_id = unique_sub_id[value1+value2:]

    # Populate split column
    df.loc[df.patient_id.isin(train_sub_id), "split"]="train"
    df.loc[df.patient_id.isin(validate_sub_id), "split"]="validate"
    df.loc[df.patient_id.isin(test_sub_id), "split"]="test"
    return df

def plot_aur_roc_curves(auc_roc_dictionary, auc_roc_scores, title):
    _, ax = plt.subplots(figsize=(10, 5))
    for label, auc_roc in auc_roc_dictionary.items():
        fpr = auc_roc["fpr"]
        tpr = auc_roc["tpr"]
        ax.plot(fpr, tpr, label=label+f"({auc_roc_scores[label]:0.2f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc=(1.02, 0))
    return ax