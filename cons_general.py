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
TEST_DIR = os.path.join(ROOT, 'test')

# Create directories if they do not exist
os.makedirs(BASICDATA_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(FINANDATA_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PREDICT_DIR, exist_ok=True)
os.makedirs(TRADE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# FILES
STOCK_LIST_XLS = os.path.join(BASICDATA_DIR, 'common', 'stocklist.xlsx')
TRADE_CAL_XLS = os.path.join(BASICDATA_DIR, 'common', 'trade_cal.xlsx')
UP_DOWN_LIMIT_XLS = os.path.join(BASICDATA_DIR, 'common', 'up_down_limit.xlsx')
SUSPEND_STOCK_XLS = os.path.join(BASICDATA_DIR, 'common', 'suspend_stock_list.xlsx')

# contains all stock's daily data
DAILY_DATA_TEMP_CSV = os.path.join(BASICDATA_DIR, 'dailytemp', 'daily_data_temp.csv')
DAILY_INDICATOR_TEMP_CSV = os.path.join(BASICDATA_DIR, 'dailytemp', 'daily_indicator_temp.csv')
DAILY_ADJFACTOR_TEMP_CSV = os.path.join(BASICDATA_DIR, 'dailytemp', 'daily_adjfactor_temp.csv')