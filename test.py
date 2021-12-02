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

def test(dataset, split_path, model_name, pretrain_model_path,
         batch_size= 32, class_mode= "raw", height=320, width=320):
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
  results = model.evaluate(test_batches)
  return results

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='A module to train models')

  parser.add_argument('--dataset_path', type=str, help='')
  parser.add_argument('--split_path', type=str, help='')
  parser.add_argument("--output_dir", type=str, help='')
  parser.add_argument('--img_root_dir', type=str, help='')
  parser.add_argument('--tag', type= str, default = "v1",  help='')

  parser.add_argument('--height', type=int, default=320, help='')
  parser.add_argument('--width', type=int, default=320, help='')
  parser.add_argument('--model', type=str, choices= Classifiers.models_names())
  parser.add_argument('--seed', type=int, default= 0, help='')
  parser.add_argument('--weights', type=str, default= None, help='')
  parser.add_argument('--n_labels', type=int, help='')
  parser.add_argument('--multilabel', type=bool, help='')

  # Training Configuration
  parser.add_argument('--learning_rate', type=float, default= 1e-3,  help='')
  parser.add_argument('--momentum_val', type=float, default=0.9, help='')
  parser.add_argument('--decay_val', type=float, default=0.0, help='')
  parser.add_argument('--batch_size', type=int, default= 16, help='')

  # Transformations
  parser.add_argument('--rotation_range', type=int, default=15,  help='')
  parser.add_argument('--fill_mode', type=str, default="constant", help='')
  parser.add_argument('--horizontal_flip', type=bool, default=True, help='')
  parser.add_argument('--crop_to_aspect_ratio', type=bool, default=True, help='')
  parser.add_argument('--zoom_range', type=float, default=0.1, help='')

  parser.add_argument('--class_mode', type=str, default="raw", choices = ["categorical", "raw"], help='')
  parser.add_argument('--freeze', type=int, help='')
  parser.add_argument('--resume', action= "store_true",  help='')

  args = parser.parse_args()

  train(args.dataset_path, args.split_path, args.img_root_dir, args.tag,
            args.height, args.width, args.model, args.seed, args.weights, args.n_labels,
            args.freeze, args.resume, args.output_dir, args.multi_label,
            learning_rate= args.learning_rate, momentum_val=args.momentum_val, decay_val=args.decay_val,
            batch_size= args.batch_size, rotation_range= args.rotation_range, fill_mode= args.fill_mode,
            horizontal_flip= args.horizontal_flip, crop_to_aspect_ratio= args.crop_to_aspect_ratio,
            zoom_range= args.zoom_range, class_mode= args.class_mode)