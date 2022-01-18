# %%
import pandas as pd
import numpy as np

data_df = pd.read_csv("./Datasets/Chexpert/csv/train.csv")
demo_df = pd.read_excel("./Datasets/Chexpert/csv/demographics.xlsx", engine='openpyxl')

demo_df = demo_df.rename(columns={'PATIENT': 'patient_id'})
data_df["patient_id"] = data_df.Path.str.split("/", expand = True)[2]

combine_df = data_df.merge(demo_df, on="patient_id", how="left")
combine_df.insert(24, "Race", "")
combine_df.loc[(combine_df.PRIMARY_RACE.str.contains("Black", na=False)), "Race"] = "BLACK"
combine_df.loc[(combine_df.PRIMARY_RACE.str.contains("White", na=False)), "Race"] = "WHITE"
combine_df.loc[(combine_df.PRIMARY_RACE.str.contains("Asian", na=False)), "Race"] = "ASIAN"
combine_df = combine_df[combine_df.Race.isin(['ASIAN','BLACK','WHITE'])]
combine_df = combine_df[combine_df.ETHNICITY.isin(["Non-Hispanic/Non-Latino","Not Hispanic"])]
combine_df = combine_df[combine_df["Frontal/Lateral"]=="Frontal"]
race_info = combine_df.Race.str.get_dummies()
combine_df.drop(["Race", "Sex", "Age", "patient_id", 'Frontal/Lateral', 'AP/PA', "GENDER",	"AGE_AT_CXR",	"PRIMARY_RACE",	"ETHNICITY"], axis=1, inplace=True)
combine_df["ASIAN"] = race_info["ASIAN"]
combine_df["BLACK"] = race_info["BLACK"]
combine_df["WHITE"] = race_info["WHITE"]
combine_df.replace(-1, 0, inplace=True)
combine_df.fillna(0, inplace=True)
combine_df.to_csv("./Datasets/Chexpert/csv/pathology_race_train.csv")
# %%
from torchvision import models
