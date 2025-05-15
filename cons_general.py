import os

# PATHS
ROOT = os.path.dirname(os.path.abspath(__file__))
BASICDATA_DIR = os.path.join(ROOT, 'basicdata')
DATASETS_DIR = os.path.join(ROOT, 'datasets')
FINANDATA_DIR = os.path.join(ROOT, 'finandata')
BACKUP_DIR = os.path.join(ROOT, 'backup')
MODELS_DIR = os.path.join(ROOT, 'models')
PREDICT_DIR = os.path.join(ROOT, 'predict')
TRADE_DIR = os.path.join(ROOT, 'trade')
TEMP_DIR = os.path.join(ROOT, 'temp')

# Create directories if they do not exist
os.makedirs(BASICDATA_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(FINANDATA_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PREDICT_DIR, exist_ok=True)
os.makedirs(TRADE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# FILES
STOCK_LIST_XLS = os.path.join(BASICDATA_DIR, 'stocklist.xlsx')
TRADE_CAL_XLS = os.path.join(BASICDATA_DIR, 'trade_cal.xlsx')
