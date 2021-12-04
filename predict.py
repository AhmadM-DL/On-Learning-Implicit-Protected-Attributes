import numpy as np
import pandas as pd
import random, math, argparse, os
import json

from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from classification_models.tfkeras import Classifiers
from tensorflow.keras.models import load_model

def use_mixed_precision():
  policy = mixed_precision.Policy('mixed_float16')
  mixed_precision.set_policy(policy)

def prepare_split_dataset(dataset_path, split_path):
  data_df = pd.read_csv(dataset_path, index_col=0).reset_index()
  split_df = pd.read_csv(split_path, index_col=0).reset_index()
  data_df = pd.merge(data_df, split_df, on="index")
  data_df = data_df[~data_df.split.isna()].drop("index", axis=1)

  train_df = data_df[data_df.split=="train"].loc[:, data_df.columns != "split"]
  validation_df = data_df[data_df.split=="validate"].loc[:, data_df.columns != "split"]
  test_df = data_df[data_df.split=="test"].loc[:, data_df.columns != "split"]

  return train_df, validation_df, test_df

def predict(dataset, split_path, model_name, pretrain_model_path, tag="test_predict_probability",
         output_dir="./", batch_size= 32, class_mode= "raw", height=320, width=320):
  # Preparing Datasets
  if "chexpert" in dataset.lower():
    img_root_dir = "./Datasets/Chexpert/"
    if "pathology" in dataset.lower():
      dataset_path = "./Datasets/Chexpert/csv/pathology_train.csv"
    elif "race" in dataset.lower():  
      dataset_path = "./Datasets/Chexpert/csv/race_train.csv"
    else:
      raise Exception("For chexpert dataset have to be 'chexpert_pathology' or 'chexpert_race'")
  else:
    raise Exception("Not supported dataset")
  _, _, test_df = prepare_split_dataset(dataset_path, split_path)
  
  # Load model
  _, preprocess_input = Classifiers.get(model_name)
  # Load and Set model
  model = load_model(pretrain_model_path)
  # Data Loaders
  test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
  test_batches= test_gen.flow_from_dataframe(dataframe= test_df,
                                                    directory= img_root_dir,
                                                    x_col= test_df.columns[0],
                                                    y_col= test_df.columns[1:],
                                                    class_mode= class_mode,
                                                    target_size= (height, width),
                                                    shuffle= False,
                                                    batch_size= batch_size)
  # test_epoch = math.ceil(len(test_df) / batch_size)
  predictions = model.predict(test_batches)
  results_groundtruth = pd.DataFrame(
    np.concatenate([predictions, test_df.iloc[:, 1:].values], axis=1),
    columns= test_df.columns[1:].tolist() + [c+"_gt" for c in test_df.columns[1:]]
  )
  results_groundtruth.to_csv(os.path.join(output_dir, tag+".csv"))
  return predictions

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='A module to infere data from models')

  parser.add_argument('--dataset', type=str, help='')
  parser.add_argument('--split_file', type=str, help='')
  parser.add_argument('--model_name', type=str, choices= Classifiers.models_names())
  parser.add_argument('--pretrain_model_path', type=str)
  parser.add_argument('--tag', type= str)
  parser.add_argument("--output_dir", type=str)
  parser.add_argument('--batch_size', type=int, default= 16, help='')
  parser.add_argument('--class_mode', type=str, default="raw", choices = ["categorical", "raw"], help='')
  parser.add_argument("--height")
  parser.add_argument("--width")

  args = parser.parse_args()

  predict(args.dataset, args.split_file, args.model_name, args.pretrain_model_path,
          args.tag, args.output_dir, args.batch_size, args.class_mode, args.height, args.width)