# cons
MODEL_NAME = 'Downgap Strategy'  # 模型名称
MAX_TRADE_DAYS = 50  # 最大缺口回补天数，即最大持有天数(不含今日),影响训练集的长度
TEST_DATASET_PERCENT = 0.20  # 测试集占训练集的比例
TRADE_COVERAGE_DAYS = 30  # 即买入清单中记录的保留天数，即买入观察天数(不含今日)，影响交易集的长度
MIN_PRED_RATE = 0.15  # 预期收益率下限，用于筛选交易数据集，低于此收益率的数据集不列入交易清单
PRED_RATE_PCT = 0.9 # 预期收益率折扣百分比
SHUFFLE = False  # 是否打乱测试集
COST_FEE = 0.0005  # 交易费用和佣金比例
MAX_STOCKS = 80  # 最大持仓股票数
ONE_TIME_FUNDS = 60000  # 单次买入资金
MIN_STOCK_PRICE = 2.88 # 最低股票买入价格，低于此价格不买入
PRED_MODELS = 12  # 预测交易集的模型数
PAUSE = 0.5  # 每次获取实时价格数据之后的暂停时间
initial_funds = 3.6e6  # 初始资金360万
additionl_rate = 0.06  # 在第一个跌停缺口后，预期第二个交易日会继续出现跌停，提高预期收益阻止交易
exception_list = ['退市', '退', 'PT', 'ST', 'XR', '东方集团']  # 例外股票列表,不列入买入清单

# files
from cons_general import TRADE_DIR, BACKUP_DIR
BUY_IN_LIST = f'{TRADE_DIR}/downgap/max_trade_days_{MAX_TRADE_DAYS}/buy_in_list.csv'  # 买入清单
HOLDING_LIST = f'{TRADE_DIR}/downgap/max_trade_days_{MAX_TRADE_DAYS}/holding_list.csv'  # 持仓清单
DAILY_PROFIT = f'{TRADE_DIR}/downgap/max_trade_days_{MAX_TRADE_DAYS}/daily_profit.csv'  # 每日盈亏
FUNDS_LIST = f'{BACKUP_DIR}/downgap/max_trade_days_{MAX_TRADE_DAYS}/funds_change_list.csv'  # 资金清单
TRADE_LOG = f'{BACKUP_DIR}/downgap/max_trade_days_{MAX_TRADE_DAYS}/trade.log'  # 交易日志