"""
Hyperparameter and augmentation configurations for each model.

Usage in notebook:
    from utils.config import FFNN_CONFIG, RNN_CONFIG, CNN_CONFIG, TRANSFER_CONFIG
    lr = FFNN_CONFIG['lr']
"""

FFNN_CONFIG = {
    'image_size': 64,
    'batch_size': 128,
    'epochs': 150,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'dropout': 0.3,
    'patience': 10,
}

RNN_CONFIG = {
    'image_size': 64,       # sequence_len=64, input_size=64
    'batch_size': 64,
    'epochs': 150,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'hidden_size': 256,
    'num_layers': 2,
    'dropout': 0.3,
    'patience': 10,
}

CNN_CONFIG = {
    'image_size': 128,      # CNNs handle larger inputs well
    'batch_size': 64,
    'epochs': 150,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'dropout': 0.4,
    'patience': 10,
}

TRANSFER_CONFIG = {
    'image_size': 224,      # ImageNet-pretrained ResNet18 expects 224x224
    'batch_size': 64,
    'epochs': 30,           # fine-tuning converges faster than from-scratch
    'lr': 1e-4,             # smaller than from-scratch — pretrained weights need gentler updates
    'weight_decay': 1e-4,
    'dropout': 0.3,
    'patience': 5,
}

# ---------------------------------------------------------------------------
# Optuna search spaces — (type, low, high) or (type, choices)
#   'log_float'   → trial.suggest_float(..., log=True)
#   'float'       → trial.suggest_float(...)
#   'int'         → trial.suggest_int(...)
#   'categorical' → trial.suggest_categorical(...)
# ---------------------------------------------------------------------------
FFNN_SEARCH_SPACE = {
    'lr':           ('log_float',   1e-5,  1e-2),
    'weight_decay': ('log_float',   1e-6,  1e-2),
    'dropout':      ('float',       0.1,   0.5),
    'batch_size':   ('categorical', [64, 128, 256]),
    'num_layers':   ('int',         2,     4),
    'optimizer':    ('categorical', ['adam', 'sgd', 'rmsprop']),
    'scheduler':    ('categorical', ['plateau', 'cosine', 'step']),
}

RNN_SEARCH_SPACE = {
    'lr':           ('log_float',   1e-5,  1e-2),
    'weight_decay': ('log_float',   1e-6,  1e-2),
    'dropout':      ('float',       0.1,   0.5),
    'batch_size':   ('categorical', [32, 64, 128]),
    'hidden_size':  ('categorical', [128, 256, 512]),
    'num_layers':   ('int',         1,     3),
    'optimizer':    ('categorical', ['adam', 'sgd', 'rmsprop']),
    'scheduler':    ('categorical', ['plateau', 'cosine', 'step']),
}

CNN_SEARCH_SPACE = {
    'lr':           ('log_float',   1e-5,  1e-2),
    'weight_decay': ('log_float',   1e-6,  1e-2),
    'dropout':      ('float',       0.1,   0.5),
    'batch_size':   ('categorical', [32, 64, 128]),
    'num_layers':   ('int',         2,     4),
    'optimizer':    ('categorical', ['adam', 'sgd', 'rmsprop']),
    'scheduler':    ('categorical', ['plateau', 'cosine', 'step']),
}

TRANSFER_SEARCH_SPACE = {
    'lr':           ('log_float',   1e-5,  1e-3),
    'weight_decay': ('log_float',   1e-6,  1e-2),
    'dropout':      ('float',       0.1,   0.5),
    'batch_size':   ('categorical', [32, 64, 128]),
    'optimizer':    ('categorical', ['adam', 'sgd', 'rmsprop']),
    'scheduler':    ('categorical', ['plateau', 'cosine', 'step']),
    # num_layers omitted — ResNet18 backbone is fixed
}
