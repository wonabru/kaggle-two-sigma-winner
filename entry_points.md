1. `python prepare_data.py`, which would
  
  Read training data from RAW_DATA_DIR (specified in SETTINGS.json)
  
  Run any preprocessing steps
  
  Save the cleaned data to TRAIN_DATA_CLEAN_PATH (specified in SETTINGS.json)
  
2. `python train.py`, which would
  
  Read training data from TRAIN_DATA_CLEAN_PATH (specified in SETTINGS.json)
  
  Train your model.
  
  Save your model to MODEL_DIR (specified in SETTINGS.json)

3. `python predict.py`, which would

  Load your model from MODEL_DIR (specified in SETTINGS.json)
  
  Use your model to make predictions on new samples
  
  Save your predictions to SUBMISSION_DIR (specified in SETTINGS.json)
