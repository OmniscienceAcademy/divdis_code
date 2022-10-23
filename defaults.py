import os

default_user = "cuda:2"
project_name = "ConceptExtrapolation"
experiment_presets_file = "experiment_presets.json"
num_wb_images = 8
num_workers = 0
image_log_frequency = 100
PARENT_DIR = "tmp"
DATA_PATH = os.path.join(PARENT_DIR, "data")
PARAM_PATH = os.path.join(PARENT_DIR, "params")
LOG_PATH = os.path.join(PARENT_DIR, "logs")

import os

for directory in [DATA_PATH, PARAM_PATH, LOG_PATH]:
    if not os.path.exists(directory):
        os.makedirs(directory)
