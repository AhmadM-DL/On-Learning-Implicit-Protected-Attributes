import numpy as np
import pandas as pd
import random, math, argparse, os, json, time
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from classification_models.tfkeras import Classifiers
from tensorflow.keras.models import load_model
import keras
from tensorflow.python.training.tracking.util import Checkpoint

CHECKPOINTS_DIR = "checkpoints"

def set_seed(seed):
  """ A function to fix teh seed of different random generators used so far"""
  # Todo set pytoch seeds
  np.random.seed(seed)
  random.seed(seed)
  tf.random.set_seed(seed)

def use_mixed_precision():
  """ Set tensorflow mixed precision policy; it speeds up training"""
  policy = mixed_precision.Policy('mixed_float16')
  mixed_precision.set_policy(policy)

def get_values_percent_as_str(series):
  """ Return the percent of values in a given pandas series"""
  value_ratios = series.value_counts(normalize=True).round(1)
  value_percents = (100*value_ratios).astype("int")
  value_percents_dict = value_percents.to_dict()
  value_percents_str = "_".join([ f"{k}:{v}" for k,v in value_percents_dict.items()])
  return value_percents_str

def get_split_percent_as_str(train_df, valid_df, test_df):
  """ Return the split ratio of train/valid/test datasets"""
  total_size = train_df.size + valid_df.size + test_df.size
  train_ratio = train_df.size/total_size*100
  valid_ratio = valid_df.size/total_size*100
  test_ratio = test_df.size/total_size*100
  return f"train({train_ratio:.3})_valid({valid_ratio:.3})_test({test_ratio:.3})"

def prepare_split_dataset(dataset_path, split_path):
  """ Read the csv dataset referenced in dataset_path and cross it with a split csv file that defines
      the location (train/test/valid/ignore) of each record in the referenced datset. It returns
      the splitted dataset.
  """
  data_df = pd.read_csv(dataset_path, index_col=0).reset_index()
  split_df = pd.read_csv(split_path, index_col=0).reset_index()
  data_df = pd.merge(data_df, split_df, on="index")
  data_df = data_df[~data_df.split.isna()].drop("index", axis=1)

  train_df = data_df[data_df.split=="train"].loc[:, data_df.columns != "split"]
  validation_df = data_df[data_df.split=="validate"].loc[:, data_df.columns != "split"]
  test_df = data_df[data_df.split=="test"].loc[:, data_df.columns != "split"]

  return train_df, validation_df, test_df

class MyCallback(keras.callbacks.Callback):
  """ A class that defines a set of callbacks to call at different events during training."""
  def __init__(self, output_dir, checkpoint_filename, resume_filename="resume_info.json", verbose=0):
    self.output_dir = output_dir
    self.resume_filename= resume_filename
    self.checkpoint_filename = checkpoint_filename
    self.verbose = verbose

    if os.path.isfile(os.path.join(self.output_dir, self.resume_filename)):
      self.lastInfo = json.load( open(os.path.join(self.output_dir, self.resume_filename), "r"))
    else:
      self.lastInfo = {"last_epoch": None, "best_epoch": None, "best_val_loss": math.inf}

    if not os.path.isdir(os.path.join(self.output_dir, CHECKPOINTS_DIR)):
      os.mkdir(os.path.join(self.output_dir, CHECKPOINTS_DIR))

  def on_epoch_end(self, epoch, logs=None):
    """ When a training epoch ends do teh followg:
        1- If the new loss is better than the lates loss defined in resume_info.json
           set the current loss/ epoch as the new best and save in resume_info.json
           then save the model as best model overwriting the previous one
        2- Save the current epoch as last epoch in resume_info.json
    """
    if self.verbose:
      print(f"SaveInfoForResume epoch ({epoch}) : loss ({logs['val_loss']}) : is best {logs['val_loss'] < self.best_valid_loss}")
    
    if logs["val_loss"] < self.lastInfo["best_val_loss"]:
      self.lastInfo["best_val_loss"] = logs["val_loss"]
      self.lastInfo["best_epoch"] = epoch
      self.model.save(os.path.join(self.output_dir, CHECKPOINTS_DIR, self.checkpoint_filename))

    self.lastInfo["last_epoch"] = epoch
    resume_file =  open(os.path.join(self.output_dir, self.resume_filename), "w")
    json.dump(self.lastInfo, resume_file)
    resume_file.close()


class LRTensorBoard(tf.keras.callbacks.TensorBoard):
    """ Onther class that defines a set of callbacks to call at different events during training.
        Specific for tensorboard.
    """
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        """ Add learning rate to tensorboard logs"""
        logs = logs or {}
        logs.update({'lr': tf.keras.backend.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


def train(dataset, split_file, tag, model_name, seed, weights, n_labels,
          freeze, resume, output_dir, multi_label, n_epochs, batch_size= 32,
          height=320, width=320, learning_rate= 1e-3, decay_val=0.0,
          rotation_range=15, fill_mode="constant", horizontal_flip= True,
          crop_to_aspect_ratio= True, zoom_range=0.1,
          class_mode= "raw", reduce_lr_on_plateau=True, verbose=0):
  """ A function that abstracts training procedure using both multi-class and multi-label approaches 
      Parameters:
      dataset: a string that specify the dataset to train on ['chexpert_pathology','chexpert_race']
      model_name: a string that specify the model to use from classification_models.tfkeras.Classifiers modeule
  """

  # save arguments
  arguments_dict = {
    "dataset": dataset, "split_file": split_file, "tag": tag,
    "model_name": model_name, "seed": seed, "weights": weights,
    "n_labels": n_labels, "freeze": freeze, "resume": resume,
    "output_dir": output_dir, "multi_label": multi_label,
    "batch_size": batch_size, "height": height, "width": width,
    "learning_rate": learning_rate, "decay_val": decay_val,
    "rotation_range": rotation_range, "fill_mode": fill_mode,
    "horizontal_flip": horizontal_flip, "crop_to_aspect_ratio": crop_to_aspect_ratio,
    "zoom_range": zoom_range, "class_mode": class_mode,
    "reduce_lr_on_plateau": reduce_lr_on_plateau,
    "n_epochs" : n_epochs,
  }
  arguments_file = open(os.path.join(output_dir, "params.json"), "w")
  json.dump(arguments_dict, arguments_file)
  arguments_file.close()

  set_seed(seed)

  # In case of multilabel the final activation function is sigmoid so that all teh outputs have a value between 0 and 1
  if multi_label:
      activation = "sigmoid"
  else:
      activation = "softmax"

  # Load datasets files from hard-coded paths 
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

  # Preparing Datasets
  # Split the loaded dataset based on the passed split_file path. the split file is a one column csv file
  # that specifies the location of each record [train/test/valid/ignore]
  train_df, validation_df, test_df = prepare_split_dataset(dataset_path, split_file)
  
  # # Remove in production
  # train_df = train_df.iloc[:100, :]
  # validation_df = validation_df.iloc[:100, :]

  # Compose the experiment name to be used for logs
  arc_name = f"{tag}_{height}x{width}_{get_split_percent_as_str(train_df, validation_df, test_df)}_{model_name}"

  # Load model
  model, preprocess_input = Classifiers.get(model_name)

  # Adjust input shape
  # So far we have rgb images only
  input_shape = (height, width, 3)

  # Resume
  start_epoch = 0
  if resume and os.path.isfile(os.path.join(output_dir, "resume_info.json")):
      # Load the resume_info.json file and continue from the latest epoch
      start_epoch = json.load(open(os.path.join(output_dir, "resume_info.json")))["last_epoch"]
      adjusted_model = load_model(os.path.join(output_dir, CHECKPOINTS_DIR, arc_name + ".hdf5"))
      print(f"Resuming from epoch {start_epoch}")
  else:
    # Load and Set model
    if weights!=None and os.path.isfile(weights):
      # In case of weights are specified (hdf5 file path) load from them
      # used to start training from a pretrained model
      model_transfer = keras.models.load_model(weights)
      x = GlobalAveragePooling2D()(model_transfer.layers[-4].output)
      x = tf.keras.layers.Dense(n_labels, name='dense_logits')(x)
      predictions = Activation(activation, dtype='float32', name='predictions')(x)
      adjusted_model = Model(inputs=[model_transfer.input], outputs=[predictions])
    else:
      # Load base model and add classification head based on parameters
      base_model = model(input_tensor = Input(input_shape), include_top = False, 
                        input_shape = input_shape, weights = weights)
      x = GlobalAveragePooling2D()(base_model.output)
      x = Dense(n_labels, name='dense_logits')(x)
      predictions = Activation(activation, dtype='float32', name='predictions')(x)
      adjusted_model = Model(inputs=base_model.input, outputs=predictions)   

  # Freeze
  if freeze != None:
    # Freeze the model layers upto the layer specified by "freeze"
    # e.g. -2 will freeze all layers except for the last two layers.
    for layers in adjusted_model.layers[:freeze]:
      layers.trainable = False
  else:
    for layers in adjusted_model.layers:
      layers.trainable = True 

  # Learning Configuration
  adam_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=decay_val)
  adam_opt = tf.keras.mixed_precision.LossScaleOptimizer(adam_opt)
  
  # Compile
  adjusted_model.compile(optimizer=adam_opt, loss='binary_crossentropy',
                metrics=[ tf.keras.metrics.AUC(curve='ROC', name='ROC-AUC', multi_label = multi_label),
                          tf.keras.metrics.AUC(curve='PR', name='PR-AUC', multi_label = multi_label),
                            'accuracy'])
  
  # Data Loaders
  train_gen = ImageDataGenerator(rotation_range= rotation_range,
                                 fill_mode= fill_mode,
                                 horizontal_flip= horizontal_flip,
                                 zoom_range= zoom_range,
                                 preprocessing_function=preprocess_input)
  validate_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

  # Training Batches
  train_batches= train_gen.flow_from_dataframe(dataframe= train_df,
                                               directory= img_root_dir,
                                               x_col= train_df.columns[0],
                                               y_col= train_df.columns[1:],
                                               class_mode= class_mode,
                                               target_size= (height, width),
                                               shuffle= True,
                                               seed= seed,
                                               crop_to_aspect_ratio= crop_to_aspect_ratio,
                                               batch_size= batch_size)
  validate_batches= validate_gen.flow_from_dataframe(dataframe= validation_df,
                                                    directory= img_root_dir,
                                                    x_col= validation_df.columns[0],
                                                    y_col= validation_df.columns[1:],
                                                    class_mode= class_mode,
                                                    target_size= (height, width),
                                                    shuffle= False,
                                                    batch_size= batch_size)
  train_epoch = math.ceil(len(train_df) / batch_size)
  val_epoch = math.ceil(len(validation_df) / batch_size)
  
  # Callbacks
  save_last_model = ModelCheckpoint(
    os.path.join(output_dir, CHECKPOINTS_DIR, arc_name + ".hdf5"),
    verbose=1, save_weights_only=False, save_freq='epoch')
  
  custome_callback = MyCallback(output_dir, checkpoint_filename= arc_name + "_best.hdf5")
  
  log_dir = os.path.join(output_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
  
  tensorboard_callback = LRTensorBoard(log_dir=log_dir, histogram_freq=1)
  
  if reduce_lr_on_plateau:
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=2, min_lr=1e-5, verbose=verbose)
  else:
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0, patience=2, min_lr=1e-5, verbose=verbose)
  
  # Train Model
  adjusted_model.fit(train_batches, validation_data=validate_batches,
            steps_per_epoch=int(train_epoch), validation_steps=int(val_epoch),
            epochs = n_epochs, initial_epoch=start_epoch, workers=32, max_queue_size=50,
            callbacks=[save_last_model, custome_callback,
                       tensorboard_callback, reduce_lr]) 

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='A module to train models')

  parser.add_argument('--dataset', choices=["chexpert_race", "chexpert_pathology"])
  parser.add_argument('--split_path')
  parser.add_argument("--output_dir")
  parser.add_argument('--tag', default = "v1")

  parser.add_argument('--height', type=int, default=320)
  parser.add_argument('--width', type=int, default=320)
  parser.add_argument('--model_name', choices= Classifiers.models_names())
  parser.add_argument('--seed', type=int, default= 0)
  parser.add_argument('--weights')
  parser.add_argument('--n_labels', type=int)
  parser.add_argument('--multi_label', choices = ["True", "False"])

  # Training Configuration
  parser.add_argument('--learning_rate', type=float, default= 1e-3)
  parser.add_argument('--momentum_val', type=float, default=0.9)
  parser.add_argument('--decay_val', type=float, default=0.0)
  parser.add_argument('--batch_size', type=int, default= 32)
  parser.add_argument('--n_epochs', type=int, default= 50)

  # Transformations
  parser.add_argument('--rotation_range', type=int, default=15)
  parser.add_argument('--fill_mode', default="constant")
  parser.add_argument('--horizontal_flip', choices = ["True", "False"])
  parser.add_argument('--crop_to_aspect_ratio', choices = ["True", "False"])
  parser.add_argument('--zoom_range', type=float, default=0.1)

  parser.add_argument('--class_mode', default="raw", choices = ["categorical", "raw"])
  parser.add_argument('--freeze')
  parser.add_argument('--resume', choices = ["True", "False"])
  parser.add_argument('--reduce_lr_on_plateau', default='True', choices = ["True", "False"])

  args = parser.parse_args()
  
  args.multi_label = args.multi_label.title() == "True"
  args.horizontal_flip = args.horizontal_flip.title() == "True"
  args.crop_to_aspect_ratio = args.crop_to_aspect_ratio.title() == "True"
  args.resume = args.resume.title() == "True"
  args.reduce_lr_on_plateau=args.reduce_lr_on_plateau.title()=="True"
  if args.freeze == "None":
    args.freeze= None
  else:
    args.freeze = int(args.freeze)
  if args.weights == "None":
    args.weights= None
  
  train(args.dataset, args.split_path, args.tag, 
        args.model_name, args.seed, args.weights,
        args.n_labels, args.freeze, args.resume,
        args.output_dir, args.multi_label,args.reduce_lr_on_plateau,
        n_epochs = args.n_epochs, batch_size= args.batch_size,
        height = args.height, width = args.width, learning_rate= args.learning_rate,
        momentum_val=args.momentum_val, decay_val=args.decay_val,
        rotation_range= args.rotation_range, fill_mode= args.fill_mode,
        horizontal_flip= args.horizontal_flip, crop_to_aspect_ratio= args.crop_to_aspect_ratio,
        zoom_range= args.zoom_range, class_mode= args.class_mode)