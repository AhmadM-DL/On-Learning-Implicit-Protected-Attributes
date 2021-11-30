import numpy as np
import pandas as pd
import random, math, argparse, os
from datetime import datetime
import json

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from classification_models.tfkeras import Classifiers


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

def test(dataset_path, split_path, img_root_dir, tag,
          height, width, model_name, pretrain_model_path, n_labels,
          output_dir, multi_label, batch_size, class_mode= "raw"):

  # Preparing Datasets
  train_df, validation_df, test_df = prepare_split_dataset(dataset_path, split_path)

  #TODO
  arc_name = f"{tag}-{height}x{width}_{get_split_percent_as_str(train_df, validation_df, test_df)}_{model}"

  # Load model
  _, preprocess_input = Classifiers.get(model_name)

  # Adjust head
  input_shape = (height, width, 3)

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

  test_epoch = math.ceil(len(test_df) / batch_size)

  # Setup Checkpoints #TODO

  checkpoint_filename =  str(arc_name) + str(args.learning_rate) + "_" + var_date+"_epoch:{epoch:03d}_val_loss:{val_loss:.5f}.hdf5"
  log_dir = os.path.join(args.output_dir, "logs", var_date)
  if not os.path.exists(os.path.join(args.output_dir, "params")):
    os.mkdir(os.path.join(args.output_dir, "params"))
  arguments_file = open(os.path.join(args.output_dir, "params", f"params_{var_date}.json"), "w")
  json.dump(args.__dict__, arguments_file, indent=2) # TODO this shuld be uncommented in script

  # Train Model
  adjusted_model.fit(train_batches, validation_data=validate_batches,
            steps_per_epoch=int(train_epoch), validation_steps=int(val_epoch),
            epochs = 50, initial_epoch=start_epoch, workers=32, max_queue_size=50,
            callbacks=[checkloss, reduce_lr, ES, tensorboard_callback])              

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