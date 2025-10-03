import os
from cons_general import TRADE_DIR

MODEL_NAME = 'DowngapStrategy'  # 模型名称
trade_root = os.path.join(TRADE_DIR, 'downgap')  # 交易数据根目录
os.makedirs(trade_root, exist_ok=True)  # 确保目录存在

dataset_group_cons = {
    'group_50': {
        'MODEL_NAME': 'DownGap >>> 50',  # 模型名称
        'MAX_TRADE_DAYS': 50,  # group_id, 最大缺口回补天数，即最大持有天数(不含今日),影响训练集的长度
        'TEST_DATASET_PERCENT': 0.20,  # 测试集占训练集的比例
        'TRADE_COVERAGE_DAYS': 30,  # 即买入清单中记录的保留天数，即买入观察天数(不含今日)，影响交易集的长度
        'MIN_PRED_RATE': 0.15,  # 预期收益率下限，用于筛选交易数据集，低于此收益率的数据集不列入交易清单
        'PRED_RATE_PCT': 0.95,  # 预期收益率折扣百分比
        'SHUFFLE': False,  # 是否打乱测试集
        'MAX_STOCKS': 42,  # 最大持仓股票数
        'ONE_TIME_FUNDS': 60000,  # 单次买入资金
        'PRED_MODELS': 6,  # 预测交易集的模型数
        'train_times': 24,  # 训练次数
        'initial_funds': 6e6,  # 初始资金600万
        'BUY_IN_LIST': os.path.join(trade_root, f'max_trade_days_50', 'buy_in_list.csv'),
        'HOLDING_LIST': os.path.join(trade_root, f'max_trade_days_50', 'holding_list.csv'),
        'DAILY_PROFIT': os.path.join(trade_root, f'max_trade_days_50', 'daily_profit.csv'),
        'FUNDS_LIST': os.path.join(trade_root, f'max_trade_days_50', 'funds_change_list.csv'),
        'TRADE_LOG': os.path.join(trade_root, f'max_trade_days_50', 'trade.log'),
        'INDICATOR_CSV': os.path.join(trade_root, f'max_trade_days_50', 'statistic_indicator.csv'),
        'XD_RECORD_HOLDING_CSV': os.path.join(trade_root, f'max_trade_days_50', 'xd_record_holding.csv'),
        'XD_RECORD_BUY_IN_CSV': os.path.join(trade_root, f'max_trade_days_50', 'xd_record_buy_in.csv'),
        'HOLDING_LIST_ORIGIN': os.path.join(trade_root, f'max_trade_days_50', 'holding_list_origin.csv'),
        'BUY_IN_LIST_ORIGIN': os.path.join(trade_root, f'max_trade_days_50', 'buy_in_list_origin.csv')
    },
    'group_45': {
        'MODEL_NAME': 'DownGap >>> 45',  # 模型名称
        'MAX_TRADE_DAYS': 45,
        'TEST_DATASET_PERCENT': 0.20,
        'TRADE_COVERAGE_DAYS': 25,
        'MIN_PRED_RATE': 0.135,
        'PRED_RATE_PCT': 1,
        'SHUFFLE': False,
        'MAX_STOCKS': 45,
        'ONE_TIME_FUNDS': 60000,
        'PRED_MODELS': 6,
        'train_times': 24,  # 训练次数
        'initial_funds': 6e6,
        'BUY_IN_LIST': os.path.join(trade_root, f'max_trade_days_45', 'buy_in_list.csv'),
        'HOLDING_LIST': os.path.join(trade_root, f'max_trade_days_45', 'holding_list.csv'),
        'DAILY_PROFIT': os.path.join(trade_root, f'max_trade_days_45', 'daily_profit.csv'),
        'FUNDS_LIST': os.path.join(trade_root, f'max_trade_days_45', 'funds_change_list.csv'),
        'TRADE_LOG': os.path.join(trade_root, f'max_trade_days_45', 'trade.log'),
        'INDICATOR_CSV': os.path.join(trade_root, f'max_trade_days_45', 'statistic_indicator.csv'),
        'XD_RECORD_HOLDING_CSV': os.path.join(trade_root, f'max_trade_days_45', 'xd_record_holding.csv'),
        'XD_RECORD_BUY_IN_CSV': os.path.join(trade_root, f'max_trade_days_45', 'xd_record_buy_in.csv'),
        'HOLDING_LIST_ORIGIN': os.path.join(trade_root, f'max_trade_days_45', 'holding_list_origin.csv'),
        'BUY_IN_LIST_ORIGIN': os.path.join(trade_root, f'max_trade_days_45', 'buy_in_list_origin.csv')
    },
    'group_60': {
        'MODEL_NAME': 'DownGap >>> 60',  # 模型名称
        'MAX_TRADE_DAYS': 60,
        'TEST_DATASET_PERCENT': 0.20,
        'TRADE_COVERAGE_DAYS': 30,
        'MIN_PRED_RATE': 0.15,
        'PRED_RATE_PCT': 1,
        'SHUFFLE': False,
        'MAX_STOCKS': 45,
        'ONE_TIME_FUNDS': 60000,
        'PRED_MODELS': 6,
        'train_times': 24,  # 训练次数
        'initial_funds': 6e6,
        'BUY_IN_LIST': os.path.join(trade_root, f'max_trade_days_60', 'buy_in_list.csv'),
        'HOLDING_LIST': os.path.join(trade_root, f'max_trade_days_60', 'holding_list.csv'),
        'DAILY_PROFIT': os.path.join(trade_root, f'max_trade_days_60', 'daily_profit.csv'),
        'FUNDS_LIST': os.path.join(trade_root, f'max_trade_days_60', 'funds_change_list.csv'),
        'TRADE_LOG': os.path.join(trade_root, f'max_trade_days_60', 'trade.log'),
        'INDICATOR_CSV': os.path.join(trade_root, f'max_trade_days_60', 'statistic_indicator.csv'),
        'XD_RECORD_HOLDING_CSV': os.path.join(trade_root, f'max_trade_days_60', 'xd_record_holding.csv'),
        'XD_RECORD_BUY_IN_CSV': os.path.join(trade_root, f'max_trade_days_60', 'xd_record_buy_in.csv'),
        'HOLDING_LIST_ORIGIN': os.path.join(trade_root, f'max_trade_days_60', 'holding_list_origin.csv'),
        'BUY_IN_LIST_ORIGIN': os.path.join(trade_root, f'max_trade_days_60', 'buy_in_list_origin.csv')
    },
    'common':{
        'COST_FEE': 0.0006,
        'MIN_STOCK_PRICE': 2.88,
        'PAUSE': 0.5,
        'additionl_rate': 0.02,
        'exception_list': ['退市', '退', 'PT', 'ST', '聚灿光电'],
        'MAX_TRADE_DAYS_LIST': [50, 45, 60],  # 最大交易天数列表
    }
}

# check the max_trade_days setting
for key in dataset_group_cons:
    if 'group' in key:
        max_trade_days = dataset_group_cons[key].get('MAX_TRADE_DAYS')
        max_trade_days_list = dataset_group_cons['common'].get('MAX_TRADE_DAYS_LIST', [])
        if max_trade_days not in max_trade_days_list:
            raise ValueError(f'最大交易日期列表{max_trade_days_list} 未包含 {max_trade_days}')