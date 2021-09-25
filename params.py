import os

curr_path = os.path.dirname(os.path.abspath(__file__))

class Params:
    OUTPUT = os.path.join(curr_path, 'results')
    DATASET_PATH = {
        'content': "/media/totolia/datos_4/research/train2017",
        'style': "/media/totolia/datos_4/research/wikiart"
    }
    WEIGHTS = {
        'content': 1.0,
        'style': 10.0
    }

    BATCH_SIZE = 12
    EPOCHS = 400
    CHECKPOINT_SAVE_INTERVAL = 1
    MODEL_PATTERN = "checkpoint_{epoch:d}_{loss:f}.pkl"
    LR = 1e-4
    LOGGER_PATTERN = {
        'TRAIN': "[TRAIN] Epoch {epoch:d} | {losses:s} | {metrics:s}",
        'VAL': "[VAL] Epoch {epoch:d} | {losses:s} | {metrics:s}"
    }
    VAL_SPLIT = 0.2