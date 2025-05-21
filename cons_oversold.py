# OversoldStrategy
MODEL_NAME = 'OversoldStrategy'
TEST_DATASET_PERCENT = 0.15  # 测试集占训练集的比例
MAX_TRADE_DAYS = 130  # 最大交易天数
REST_TRADE_DAYS = 65  # 剩余交易天数=backward_days - waiting_days
MIN_WAITING_DAYS = 7  # 最小等待天数, 一个下跌趋势最后一条记录的日期如果距离今天小于MIN_WAITING_DAYS，则不买入
WAITING_RATE_PCT = 0.60  # 等待天数内，一个下跌趋势最大上涨幅度/预期收益率超过此值，则不买入
MIN_PRED_RATE = 0.45  # 预期收益率下限，用于筛选交易数据集，预期收益率低于此收益率的数据集不列入交易清单
PRED_RATE_PCT = 1.0  # 预期收益率折扣百分比
SHUFFLE = False  # 是否打乱测试集
COST_FEE = 0.0005  # 交易费用和佣金比例
MAX_STOCKS = 85  # 最大持仓股票数
ONE_TIME_FUNDS = 60000  # 单次买入资金
MIN_STOCK_PRICE = 2.88 # 最低股票买入价格，低于此价格不买入
MAX_DOWN_LIMIT = -0.40 # 最大下跌幅度，超过此值要卖出
PAUSE = 0.5  # 暂停时间
initial_funds = 3.6e6  # 初始资金360万
exception_list = ['退市', '退', 'PT', 'ST', 'XR']  # 例外股票列表,不列入买入清单

# files
from cons_general import TRADE_DIR, BACKUP_DIR
BUY_IN_LIST = f'{TRADE_DIR}/oversold/buy_in_list.csv'  # 买入清单
HOLDING_LIST = f'{TRADE_DIR}/oversold/holding_list.csv'  # 持仓清单
DAILY_PROFIT = f'{TRADE_DIR}/oversold/daily_profit.csv'  # 每日盈亏
FUNDS_LIST = f'{BACKUP_DIR}/oversold/funds_change_list.csv'  # 资金清单
TRADE_LOG = f'{BACKUP_DIR}/oversold/trade.log'  # 交易日志

# TO_UPDATE == False, SKIP updating dataset
params_list = [
    {'FORWARD_DAYS': 130, 'BACKWARD_DAYS': 60, 'DOWN_FILTER': -0.50, 'TO_UPDATE': False},
    {'FORWARD_DAYS': 130, 'BACKWARD_DAYS': 130, 'DOWN_FILTER': -0.50, 'TO_UPDATE': False},
    {'FORWARD_DAYS': 150, 'BACKWARD_DAYS': 130, 'DOWN_FILTER': -0.45, 'TO_UPDATE': True},
    {'FORWARD_DAYS': 200, 'BACKWARD_DAYS': 130, 'DOWN_FILTER': -0.50, 'TO_UPDATE': True},
    {'FORWARD_DAYS': 225, 'BACKWARD_DAYS': 130, 'DOWN_FILTER': -0.55, 'TO_UPDATE': True},
    {'FORWARD_DAYS': 250, 'BACKWARD_DAYS': 130, 'DOWN_FILTER': -0.60, 'TO_UPDATE': False},
]

# TIMES == 0, SKIP training dataset
dataset_to_train = [
    {'FORWARD_DAYS': 130, 'BACKWARD_DAYS': 60, 'DOWN_FILTER': -0.50, 'TIMES': 0},
    {'FORWARD_DAYS': 130, 'BACKWARD_DAYS': 130, 'DOWN_FILTER': -0.50, 'TIMES': 0},
    {'FORWARD_DAYS': 150, 'BACKWARD_DAYS': 130, 'DOWN_FILTER': -0.45, 'TIMES': 25},
    {'FORWARD_DAYS': 200, 'BACKWARD_DAYS': 130, 'DOWN_FILTER': -0.50, 'TIMES': 25},
    {'FORWARD_DAYS': 225, 'BACKWARD_DAYS': 130, 'DOWN_FILTER': -0.55, 'TIMES': 25},
    {'FORWARD_DAYS': 250, 'BACKWARD_DAYS': 130, 'DOWN_FILTER': -0.60, 'TIMES': 0},
]

# PRED_MODELS == 0, SKIP predicting dataset
dataset_to_predict_trade = [
    {'FORWARD_DAYS': 130, 'BACKWARD_DAYS': 60, 'DOWN_FILTER': -0.50, 'PRED_MODELS': 0},
    {'FORWARD_DAYS': 130, 'BACKWARD_DAYS': 130, 'DOWN_FILTER': -0.50, 'PRED_MODELS': 0},
    {'FORWARD_DAYS': 150, 'BACKWARD_DAYS': 130, 'DOWN_FILTER': -0.45, 'PRED_MODELS': 12},
    {'FORWARD_DAYS': 200, 'BACKWARD_DAYS': 130, 'DOWN_FILTER': -0.50, 'PRED_MODELS': 12},
    {'FORWARD_DAYS': 225, 'BACKWARD_DAYS': 130, 'DOWN_FILTER': -0.55, 'PRED_MODELS': 8},
    {'FORWARD_DAYS': 250, 'BACKWARD_DAYS': 130, 'DOWN_FILTER': -0.60, 'PRED_MODELS': 0},
]