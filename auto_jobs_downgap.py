import os
import time
import math
import random
import shutil
import datetime
from tensorflow.keras import layers  # type: ignore
import datetime
import pandas as pd
import tqdm
from tensorflow import keras
from concurrent.futures import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from trade_downgap import trade_process
from datasets_downgap import get_gaps_statistic_data, refresh_the_gap_csv, merge_all_gap_data
from cons_general import TRADE_CAL_XLS, TEMP_DIR, MODELS_DIR, PREDICT_DIR, TRADE_DIR, BASICDATA_DIR, DATASETS_DIR, BACKUP_DIR
from cons_downgap import MODEL_NAME, dataset_group_cons
from stocklist import get_all_stocks_info
from utils import calculate_today_series_statistic_indicator

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

temp_root = f'{TEMP_DIR}/downgap'
os.makedirs(temp_root, exist_ok=True)
model_root = f'{MODELS_DIR}/downgap'
os.makedirs(model_root, exist_ok=True)
predict_root = f'{PREDICT_DIR}/downgap'
os.makedirs(predict_root, exist_ok=True)
trade_root = f'{TRADE_DIR}/downgap'
os.makedirs(trade_root, exist_ok=True)
gap_root = f'{DATASETS_DIR}/downgap'
os.makedirs(gap_root, exist_ok=True)
daily_root = f'{BASICDATA_DIR}/dailydata'
os.makedirs(daily_root, exist_ok=True)
backup_root = f'{BACKUP_DIR}/downgap'
os.makedirs(backup_root, exist_ok=True)

def is_trade_day(task: str = None):
    """
    装饰器,判断是否为交易日,如果是则执行数据更新函数.
    :param task: 任务名称
    :return: 装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            df = pd.read_excel(TRADE_CAL_XLS, dtype={'cal_date': str})
            df = df.sort_values(by='cal_date', ascending=False)
            res = df["is_open"][0]
            if res:
                return func(*args, **kwargs)
            else:
                today = datetime.datetime.now().strftime('%Y%m%d')
                print(f'({MODEL_NAME}) {today} 不是交易日, 不执行 <{task}> 任务')
        return wrapper    
    return decorator

def update_dataset():
    """更新缺口数据"""
    all_stocks = get_all_stocks_info()
    codes = [stock[0] for stock in all_stocks]  # 沪深300全部股票代码
    steps = 5
    # 计算缺口数据
    bar = tqdm.tqdm(total=len(codes), desc="更新缺口数据", ncols=100)
    for code in range(0, len(codes), steps):
        with ThreadPoolExecutor(max_workers=steps) as executor:
            executor.map(get_gaps_statistic_data, codes[code:code+steps])
        bar.update(steps)
    bar.close()
    # 刷新缺口数据
    bar = tqdm.tqdm(total=len(codes), desc="刷新缺口数据", ncols=100)
    for code in range(0, len(codes), steps):
        with ThreadPoolExecutor(max_workers=steps) as executor:
            executor.map(refresh_the_gap_csv, codes[code:code+steps])
        bar.update(steps)
    bar.close()
    # 合并缺口数据
    for group in dataset_group_cons:
        max_trade_days = dataset_group_cons[group].get('MAX_TRADE_DAYS')
        model_name = dataset_group_cons[group].get('MODEL_NAME')
        if max_trade_days is None:
            continue
        print(f'({model_name}) 正在合并最大交易天数为{max_trade_days}天组的缺口数据集...')
        merge_all_gap_data(max_trade_days)
    print('缺口数据更新完成！')


def predict_dataset():
    """
    使用模型预测数据集
    生成的文件包括：
    1. down_gap_train_data.csv: 训练集数据
    2. down_gap_trade_data.csv: 交易集数据
    3. test_pred_K.csv: 测试集数据
    4. test_pred_K_filter_gt_{MIN_PRED_RATE}.csv
    5. trade_pred_K.csv: 交易集数据
    """
    for group in dataset_group_cons:
        MAX_TRADE_DAYS = dataset_group_cons[group].get('MAX_TRADE_DAYS')
        TRADE_COVERAGE_DAYS = dataset_group_cons[group].get('TRADE_COVERAGE_DAYS')
        TEST_DATASET_PERCENT = dataset_group_cons[group].get('TEST_DATASET_PERCENT')
        MIN_PRED_RATE = dataset_group_cons[group].get('MIN_PRED_RATE')
        PRED_MODELS = dataset_group_cons[group].get('PRED_MODELS')
        SHUFFLE = dataset_group_cons[group].get('SHUFFLE')
        model_name_1 = dataset_group_cons[group].get('MODEL_NAME')
        if not MAX_TRADE_DAYS or PRED_MODELS == 0:
            continue
        # STEP1: process the gaps dataset
        gap_dir = f'{temp_root}/max_trade_days_{MAX_TRADE_DAYS}'
        gap_csv = f'{gap_dir}/all_gap_data.csv'
        df = pd.read_csv(gap_csv, dtype={'trade_date': str, 'fill_date': str})
        df = df.sort_values(by=['trade_date', 'ts_code'], ascending=[True, True])
        df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last', inplace=True)
        df = df[df['gap'] == 'down']
        # seperate trade dataset from df
        trade_date = df['trade_date'].unique()[-TRADE_COVERAGE_DAYS:]
        trade_df = df[df['trade_date'].isin(trade_date)]
        trade_df.to_csv(gap_dir + f'/down_gap_trade_data.csv', index=False)
        # the rest data as train dataset
        train_df = df[~df['trade_date'].isin(trade_date)]
        train_df = train_df.dropna()
        train_df = train_df[train_df['days'] <= MAX_TRADE_DAYS]
        train_df.to_csv(gap_dir + f'/down_gap_train_data.csv', index=False)
        df_origin = train_df.copy()

        # STEP2: 预处理训练集+验证集+测试集
        train_csv = f'{gap_dir}/down_gap_train_data.csv'
        train_df = pd.read_csv(train_csv, dtype={'trade_date': str, 'fill_date': str})
        # 插一列sum_date，值等于trade_date各位数字平方之和后求正弦
        train_df['sum_date'] = train_df['trade_date'].apply(lambda x: sum([int(i) ** 2 for i in x]))
        train_df['sum_date'] = train_df['sum_date'].apply(lambda x: (math.sin(x / 100) + 1) / 2)
        # 对pe_ttm求正弦值，压缩取值范围
        train_df['pe_ttm'] = train_df['pe_ttm'].apply(lambda x: (math.sin(x / 100) + 1) / 2)
        columns = ['gap_percent', 'vol_ratio', 'pct_chg', 'RSI14', 'RSI7', 'RSI3', 'K', 'MAP14', 'MAP7', 'turnover_rate', 
                'mv_ratio', 'pe_ttm', 'pb', 'dv_ratio',  'sum_date', 'days', 'rise_percent']
        train_df = train_df[columns]

        # STEP3: 把前14列作为特征列，最后1列作为标签列
        x_train = train_df[['gap_percent', 'vol_ratio', 'pct_chg', 'RSI14', 'RSI7', 'RSI3', 'K', 'MAP14', 'MAP7', 
                            'turnover_rate', 'mv_ratio', 'pe_ttm', 'pb', 'dv_ratio',  'sum_date']]
        y_train = train_df['rise_percent']
        # 分割训练集和测试集
        length = len(x_train)
        test_length = int(length * TEST_DATASET_PERCENT)
        x_test = x_train.iloc[-test_length:]
        y_test = y_train.iloc[-test_length:]
        x_train = x_train.iloc[:-test_length]
        y_train = y_train.iloc[:-test_length]

        # shuffle the train dataset and test dataset
        if SHUFFLE:
            from sklearn.utils import shuffle  # type: ignore
            x_train, y_train = shuffle(x_train, y_train)
            x_test, y_test = shuffle(x_test, y_test)

        # STEP4: 准备交易集数据
        trade_csv = f'{gap_dir}/down_gap_trade_data.csv'
        trade_df = pd.read_csv(trade_csv, dtype={'trade_date': str})
        # 插一列sum_date，值等于trade_date各位数字平方之和后求正弦
        trade_df['sum_date'] = trade_df['trade_date'].apply(lambda x: sum([int(i) ** 2 for i in x]))
        trade_df['sum_date'] = trade_df['sum_date'].apply(lambda x: (math.sin(x / 100) + 1) / 2)
        # 对pe_ttm求正弦值，压缩取值范围
        trade_df['pe_ttm'] = trade_df['pe_ttm'].apply(lambda x: (math.sin(x / 100) + 1) / 2)
        x_trade = trade_df[['gap_percent', 'vol_ratio', 'pct_chg', 'RSI14', 'RSI7', 'RSI3', 'K', 'MAP14', 'MAP7', 
                            'turnover_rate', 'mv_ratio', 'pe_ttm', 'pb', 'dv_ratio', 'sum_date']]

        # STEP5: 评价模型在测试集上的表现(使用预测值和真实值的比率衡量)
        models_dir = f'{model_root}/max_trade_days_{MAX_TRADE_DAYS}'
        models = os.listdir(models_dir)
        model_list = []
        for model in models:
            # 按照model中mae后面的值升序排列
            if model.endswith('.keras') and 'mae' in model:
                tmp = model.split('mae')
                if len(tmp) == 2:  # 正常名字中只有一个'mae'
                    model_list.append(tmp)
        model_list = sorted(model_list, key=lambda x: x[-1])
        model_list = [f'{models_dir}/{i[0]}mae{i[1]}' for i in model_list]
        model_list = model_list[:12]
        pred_models = model_list[:PRED_MODELS]
        # delete the models that are not in model_list
        for model in os.listdir(models_dir):
            tmp_model = f'{models_dir}/{model}'
            if tmp_model.endswith('.keras') and tmp_model not in model_list:
                os.remove(tmp_model)
        # use pred_models to predict the test dataset
        y_test_pred_list = []
        for model in pred_models:
            print(f'load model: {model} to predict test dataset')
            model = keras.models.load_model(model)
            pred = model.predict(x_test)
            y_test_pred_list.append(pred)
            del model
        y_test_pred = sum(y_test_pred_list) / len(y_test_pred_list)
        diff = pd.DataFrame({'real': y_test, 'pred': y_test_pred.flatten()})
        diff['ts_code'] = df_origin.iloc[-test_length:]['ts_code'].values
        diff['stock_name'] = df_origin.iloc[-test_length:]['stock_name'].values
        diff['industry'] = df_origin.iloc[-test_length:]['industry'].values
        diff['trade_date'] = df_origin.iloc[-test_length:]['trade_date'].values
        diff['fill_date'] = df_origin.iloc[-test_length:]['fill_date'].values
        diff['days'] = df_origin.iloc[-test_length:]['days'].values
        diff['gap_percent'] = df_origin.iloc[-test_length:]['gap_percent'].values
        diff['pct_chg'] = df_origin.iloc[-test_length:]['pct_chg'].values/100
        diff = diff[
            ['ts_code', 'stock_name', 'industry', 'trade_date', 'fill_date', 'days', 
            'gap_percent', 'pct_chg', 'pred', 'real']
        ]
        diff['diff'] = diff['pred'] - diff['real']
        diff['diff_pct'] = diff['diff'] / diff['real']
        diff['diff_pct'] = diff['diff_pct'].apply(lambda x: f'{x:.2%}')
        pred_path = f'{predict_root}/max_trade_days_{MAX_TRADE_DAYS}'
        os.makedirs(pred_path, exist_ok=True)
        csv_name = f'{pred_path}/test_pred_K.csv'
        diff.to_csv(csv_name, index=False)
        # filter the test dataset with diff['diff'] > MIN_PRED_RATE and plot bar chart
        df_filter = diff[diff['pred'] > MIN_PRED_RATE]
        csv_name = f'{pred_path}/test_pred_K_filter_gt_{MIN_PRED_RATE}.csv'
        df_filter.to_csv(csv_name, index=False)

        # STEP6: use pred_models to predict x_trade and then average the result
        y_trade_pred = []
        for model_name in pred_models:
            print(f'loading model: {model_name}')
            model = keras.models.load_model(model_name)
            pred = model.predict(x_trade)
            y_trade_pred.append(pred)
            del model
        # average the y_trade_pred
        y_trade_pred = sum(y_trade_pred) / len(y_trade_pred)
        #  predict the trade dataset
        diff = pd.DataFrame({'pred': y_trade_pred.flatten(), 'real': trade_df['rise_percent'].values})
        diff['ts_code'] = trade_df['ts_code'].values
        diff['stock_name'] = trade_df['stock_name'].values
        diff['industry'] = trade_df['industry'].values
        diff['trade_date'] = trade_df['trade_date'].values
        diff['fill_date'] = trade_df['fill_date'].values
        diff['fill_date'] = diff['fill_date'].apply(lambda x: str(x)[:8])
        diff['fill_date'] = diff['fill_date'].apply(lambda x: '' if x == 'nan' else x)
        diff['days'] = trade_df['days'].values
        diff['gap_percent'] = trade_df['gap_percent'].values
        diff['pct_chg'] = trade_df['pct_chg'].values/100
        # calculate pred - real
        diff['diff'] = round(diff['pred'] - diff['real'], 4)
        diff['diff_pct'] = diff['diff'] / diff['real']
        diff['diff_pct'] = diff['diff_pct'].apply(lambda x: f'{x:.2%}')
        diff['diff_pct'] = diff['diff_pct'].apply(lambda x: '' if 'nan' in x else x)
        diff = diff[
            ['ts_code', 'stock_name', 'industry', 'trade_date', 'fill_date', 'days', 
            'gap_percent', 'pct_chg', 'pred', 'real', 'diff', 'diff_pct']
        ]
        csv_name = f'{pred_path}/trade_pred_K.csv'
        diff.to_csv(csv_name, index=False)
        print(f'({model_name_1}) 模型预测完成.')

def train_dataset():
    """
    训练模型
    """
    for group in dataset_group_cons:
        MAX_TRADE_DAYS = dataset_group_cons[group].get('MAX_TRADE_DAYS')
        TEST_DATASET_PERCENT = dataset_group_cons[group].get('TEST_DATASET_PERCENT')
        TRADE_COVERAGE_DAYS = dataset_group_cons[group].get('TRADE_COVERAGE_DAYS')
        SHUFFLE = dataset_group_cons[group].get('SHUFFLE')
        train_times = dataset_group_cons[group].get('train_times')
        model_name_1 = dataset_group_cons[group].get('MODEL_NAME')
        if MAX_TRADE_DAYS is None or train_times == 0:
            continue
        # STEP1 process the gaps dataset
        gap_dir = f'{temp_root}/max_trade_days_{MAX_TRADE_DAYS}'
        gap_csv = f'{gap_dir}/all_gap_data.csv'
        df = pd.read_csv(gap_csv, dtype={'trade_date': str, 'fill_date': str})
        df = df.sort_values(by=['trade_date', 'ts_code'], ascending=[True, True])
        df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last', inplace=True)
        df = df[df['gap'] == 'down']
        # seperate trade dataset from df
        trade_date = df['trade_date'].unique()[-TRADE_COVERAGE_DAYS:]
        trade_df = df[df['trade_date'].isin(trade_date)]
        trade_df.to_csv(gap_dir + f'/down_gap_trade_data.csv', index=False)
        # the rest data as train dataset
        train_df = df[~df['trade_date'].isin(trade_date)]
        train_df = train_df.dropna()
        train_df = train_df[train_df['days'] <= MAX_TRADE_DAYS]
        train_df.to_csv(gap_dir + f'/down_gap_train_data.csv', index=False)
        # df_origin = train_df.copy()

        # STEP2 预处理训练集+验证集+测试集
        train_csv = f'{gap_dir}/down_gap_train_data.csv'
        train_df = pd.read_csv(train_csv, dtype={'trade_date': str, 'fill_date': str})
        # 插入一列sum_date，值等于trade_date各位数字平方之和后求正弦
        train_df['sum_date'] = train_df['trade_date'].apply(lambda x: sum([int(i) ** 2 for i in x]))
        train_df['sum_date'] = train_df['sum_date'].apply(lambda x: (math.sin(x) + 1) / 2)
        # 对pe_ttm求正弦值，压缩一下取值范围
        train_df['pe_ttm'] = train_df['pe_ttm'].apply(lambda x: (math.sin(x) + 1) / 2)
        columns = ['gap_percent', 'vol_ratio', 'pct_chg', 'RSI14', 'RSI7', 'RSI3', 'K', 'MAP14', 'MAP7', 'turnover_rate', 
                'mv_ratio', 'pe_ttm', 'pb', 'dv_ratio',  'sum_date', 'days', 'rise_percent']
        train_df = train_df[columns]

        # STEP3 把前14列作为特征列，最后1列作为标签列
        x_train = train_df[['gap_percent', 'vol_ratio', 'pct_chg', 'RSI14', 'RSI7', 'RSI3', 'K', 'MAP14', 'MAP7', 
                            'turnover_rate', 'mv_ratio', 'pe_ttm', 'pb', 'dv_ratio',  'sum_date']]
        y_train = train_df['rise_percent']
        # 分割训练集和测试集
        length = len(x_train)
        test_length = int(length * TEST_DATASET_PERCENT)
        x_test = x_train.iloc[-test_length:]
        y_test = y_train.iloc[-test_length:]
        x_train = x_train.iloc[:-test_length]
        y_train = y_train.iloc[:-test_length]

        # shuffle the train dataset and test dataset
        if SHUFFLE:
            from sklearn.utils import shuffle  # type: ignore
            x_train, y_train = shuffle(x_train, y_train)
            x_test, y_test = shuffle(x_test, y_test)
        
        # build the model
        def get_model(depth: int = 6, dropout_rate: float = 0.5):
            inputs = keras.Input(shape=(x_train.shape[1],))
            feature = layers.BatchNormalization()(inputs)
            for dep in range(depth+3, 4, -1):
                feature = layers.Dense(2**dep, activation='relu')(feature)
                if dep % 3 == 0:
                    feature = layers.BatchNormalization()(feature)
                feature = layers.Dropout(dropout_rate)(feature)
            outputs = layers.Dense(1)(feature)
            model = keras.Model(inputs, outputs)
            optimizer = random.choice(['adam', 'rmsprop', 'sgd'])
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])
            return model

        # STEP5 训练模型
        model_dir = f'{model_root}/max_trade_days_{MAX_TRADE_DAYS}'
        os.makedirs(model_dir, exist_ok=True)
        name_prefix = 'model_with_K'
        print(f'开始训练 ({model_name_1}) 模型...')
        for _ in range(train_times):
            try:
                now = time.strftime('%Y%m%d%H%M%S', time.localtime())
                depth = random.choice([6, 7, 8])
                drop_rate = random.choice([0.2, 0.3, 0.4, 0.5])
                model = get_model(depth=depth, dropout_rate=drop_rate)
                params = model.count_params()
                model_name = f'{model_dir}/{name_prefix}_{now}_{params}.keras'
                callbacks = [
                    keras.callbacks.ModelCheckpoint(filepath=model_name, save_best_only=True, monitor='val_mae', mode='min'),
                    keras.callbacks.EarlyStopping(monitor='val_mae', patience=8, mode='min')
                ]
                epoches = random.choice([50, 40, 30])
                batch_size = random.choice([32, 64, 128, 256])
                validation_split = random.choice([0.1, 0.15, 0.2, 0.25])
                model.fit(
                    x_train, y_train, epochs=epoches, batch_size=batch_size, validation_split=validation_split, callbacks=callbacks
                )
                # 寻找model.history中最小的val_mae和对应的val_loss和val_mape
                min_mae = min(model.history.history['val_mae'])
                min_loss = model.history.history['val_loss'][model.history.history['val_mae'].index(min_mae)]
                min_mape = model.history.history['val_mape'][model.history.history['val_mae'].index(min_mae)]
                # 找到model_dir下最新的文件，将名称后面加上val_loss和val_mae的最小值
                files = os.listdir(model_dir)
                files = [f for f in files if f.endswith('.keras')]
                files.sort(reverse=True)
                model_name = files[0]
                if 'mae' in model_name:  # 如果文件名中已经包含了mae，说明已经重命名过
                    continue
                new_suffix = f'_loss{min_loss:.4f}_mae{min_mae:.4f}_mape{min_mape:.4f}.keras'
                new_model_name = model_name.replace('.keras', new_suffix)
                os.rename(f'{model_dir}/{model_name}', f'{model_dir}/{new_model_name}')
            except Exception as e:
                pass
        print(f'({model_name_1}) 模型训练完成.')

def build_buy_in_list():
    """
    生成买入清单buy_in_list.csv
    """
    for group in dataset_group_cons:
        MAX_TRADE_DAYS = dataset_group_cons[group].get('MAX_TRADE_DAYS')
        MIN_PRED_RATE = dataset_group_cons[group].get('MIN_PRED_RATE')
        PRED_RATE_PCT = dataset_group_cons[group].get('PRED_RATE_PCT')
        exception_list = dataset_group_cons['common'].get('exception_list')
        model_name_1 = dataset_group_cons[group].get('MODEL_NAME')
        if MAX_TRADE_DAYS is None:
            continue
        # prepare data from trade_pred.csv within TRADE_COVERAGE_DAYS days
        pred_path = f'{predict_root}/max_trade_days_{MAX_TRADE_DAYS}'
        csv_name = f'{pred_path}/trade_pred_K.csv'
        trade_df = pd.read_csv(csv_name, dtype={'trade_date': str, 'fill_data': str})
        # add target_price column to trade_df, which is the row's pre_low
        trade_df['target_price'] = None
        for i, row in trade_df.iterrows():
            ts_code = row['ts_code']
            gap_csv = f'{gap_root}/{ts_code}.csv'
            trade_date = row['trade_date']
            gap_df = pd.read_csv(gap_csv, dtype={'trade_date': str})
            gap_df = gap_df[gap_df['trade_date'] == trade_date]
            if not gap_df.empty:
                pre_low = gap_df['pre_low'].values[0]  # target_price
                trade_df.loc[i, 'target_price'] = pre_low
            # calculate days between trade_date and today
            daily_csv = f'{daily_root}/{ts_code}.csv' 
            daily_df = pd.read_csv(daily_csv, dtype={'trade_date': str})
            daily_df = daily_df.sort_values(by='trade_date', ascending=True)
            today = datetime.datetime.now().strftime('%Y%m%d')
            trade_dates = daily_df['trade_date'].tolist()
            trade_date_index = trade_dates.index(trade_date)
            if today not in trade_dates:
                today_index = len(trade_dates) - 1
                days = today_index - trade_date_index + 1  # 今日未收盘，未在daily_df中，所以+1
            else:
                today_index = trade_dates.index(today)
                days = today_index - trade_date_index
            trade_df.loc[i, 'days'] = days
        # add diffrent rate columns to trade_df
        trade_df['pred_100%'] = trade_df['pred'].map(lambda x: f'{x:.2%}')
        col_pct = f'pred_{int(PRED_RATE_PCT * 100)}%'
        trade_df[col_pct] = (trade_df['pred'] * PRED_RATE_PCT).map(lambda x: f'{x:.2%}')
        # filter trade_df
        trade_df = trade_df[trade_df['real'].isnull()]
        trade_df = trade_df[trade_df['pred'] > MIN_PRED_RATE]
        trade_df = trade_df[~trade_df['stock_name'].str.contains('|'.join(exception_list))]
        trade_df = trade_df.reset_index(drop=True)
        trade_dir = f'{trade_root}/max_trade_days_{MAX_TRADE_DAYS}'
        os.makedirs(trade_dir, exist_ok=True)
        csv_name = f'{trade_dir}/buy_in_list.csv'
        trade_df.to_csv(csv_name, index=False)
        print(f'({model_name_1}) 买入清单生成完成')

scheduler = BackgroundScheduler()
scheduler.configure(timezone='Asia/Shanghai')

# 定时任务函数定义
@is_trade_day(task='更新预测数据集')
def update_and_predict_dataset():
    update_dataset()
    predict_dataset()
    today = datetime.datetime.now().strftime('%Y%m%d')
    print(f'({MODEL_NAME}) {today} 数据更新完成')

@is_trade_day(task='更新买入清单')
def update_buy_in_list():
    build_buy_in_list()
    today = datetime.datetime.now().strftime('%Y%m%d')
    print(f'({MODEL_NAME}) {today} 买入清单更新完成')

def train_and_predict_model():
    train_dataset()
    predict_dataset()
    today = datetime.datetime.now().strftime('%Y%m%d')
    print(f'({MODEL_NAME}) {today} 模型训练完成')

@is_trade_day(task='统计各项指标')
def calculate_today_statistics_indicators():
    for group in dataset_group_cons:
        max_trade_days = dataset_group_cons[group].get('MAX_TRADE_DAYS')
        model_name = dataset_group_cons[group].get('MODEL_NAME')
        calculate_today_series_statistic_indicator(name='downgap', max_trade_days=max_trade_days)
        print(f'({model_name}) 统计指标计算完成')
    today = datetime.datetime.now().strftime('%Y%m%d')
    print(f'({MODEL_NAME}) {today} 今日统计数据计算完成！')

@is_trade_day(task='备份交易数据')
def backup_trade_data():
    """
    把TRADE_DIR/downgap/max_trade_days_{MAX_TRADE_DAYS}下的所有文件和目录备份到
    BACKUP_DIR/downgap/max_trade_days_{MAX_TRADE_DAYS}_<datetime>目录下
    NOTE:
    备份清单包括: 买入清单、持有清单、交易日志、资金流水、每日利润、每日指标文件,
    保留最近的6个备份
    """
    for group in dataset_group_cons:
        max_trade_days = dataset_group_cons[group].get('MAX_TRADE_DAYS')
        model_name = dataset_group_cons[group].get('MODEL_NAME')
        if max_trade_days is None:
            continue
        trade_dir = f'{TRADE_DIR}/downgap/max_trade_days_{max_trade_days}'
        now = datetime.datetime.now().strftime('%Y%m%d %H%M%S')
        group_backup_root = f'{backup_root}/max_trade_days_{max_trade_days}'
        os.makedirs(group_backup_root, exist_ok=True)
        backup_dir = f'{group_backup_root}/backup_{now}'
        shutil.copytree(trade_dir, backup_dir, dirs_exist_ok=True)
        files = os.listdir(group_backup_root)
        dirs = [d for d in files if os.path.isdir(os.path.join(group_backup_root, d))]
        dirs.sort(reverse=True)
        [shutil.rmtree(os.path.join(group_backup_root, d)) for d in dirs[6:]]  # 保留最近6个备份
        print(f'({model_name}) 交易数据备份完成！')
    today = datetime.datetime.now().strftime('%Y%m%d')
    print(f'({MODEL_NAME}) {today} 交易数据备份完成！')

# 动态任务am
@is_trade_day(task='股票交易')
def trading_task_am(scheduler, max_trade_days: int):
    """
    上午交易任务
    :param scheduler: apscheduler调度器
    :param max_trade_days: dataset group_id
    """
    now = datetime.datetime.now().time()
    start_time = datetime.time(9, 20)  # 9:20 AM, start of trading
    end_time = datetime.time(11, 30)  # 11:30 AM, end of trading
    if start_time <= now <= end_time:
        try:
            trade_process(max_trade_days=max_trade_days)
        except Exception as e:
            print(f"Error during trading process: {e}")
        # 启动下一个交易loop
        if datetime.datetime.now().time() <= end_time:
            scheduler.add_job(
                trading_task_am,
                args=[scheduler, max_trade_days],
                run_date=datetime.datetime.now()+datetime.timedelta(seconds=1),
                id=f'{MODEL_NAME}_trading_job_am_{max_trade_days}',
            )
        else:
            now = datetime.datetime.now()
            print(f'({MODEL_NAME}) {now.time()} 不在交易时间段内.')
    else:
        now = datetime.datetime.now()
        print(f'({MODEL_NAME}) {now.time()} 不在交易时间段内.')

# 动态任务pm
@is_trade_day(task='股票交易')
def trading_task_pm(scheduler, max_trade_days: int):
    """
    下午交易任务
    :param scheduler: apscheduler调度器
    :param max_trade_days: dataset group_id
    """
    now = datetime.datetime.now().time()
    start_time = datetime.time(12, 50)
    end_time = datetime.time(15, 0)
    if start_time <= now <= end_time:
        try:
            trade_process(max_trade_days=max_trade_days)
        except Exception as e:
            print(f"Error during trading process: {e}")
        # 启动下一个交易loop
        if datetime.datetime.now().time() <= end_time:
            scheduler.add_job(
                trading_task_pm,
                args=[scheduler, max_trade_days],
                run_date=datetime.datetime.now()+datetime.timedelta(seconds=1),
                id=f'{MODEL_NAME}_trading_job_pm_{max_trade_days}',
            )
        else:
            now = datetime.datetime.now()
            print(f'({MODEL_NAME}) {now.time()} 不在交易时间段内.')
    else:
        now = datetime.datetime.now()
        print(f'({MODEL_NAME}) {now.time()} 不在交易时间段内.')

def auto_run():
    """
    自动运行函数
    """
    # 定时任务通过add_job方式添加
    scheduler.add_job(
        update_buy_in_list,
        trigger='cron',
        hour=0, minute=30, misfire_grace_time=300,
        id='build_buy_in_list',
        name='每天00:30创建买入清单'
    )
    scheduler.add_job(
        trading_task_am,
        args=[scheduler, 50],
        trigger='cron',
        hour=9, minute=25, misfire_grace_time=300,
        id=f'{MODEL_NAME}_start_trading_job_am_50',
        name='Start_trading_program_at_9:25_AM_50',
    )
    scheduler.add_job(
        trading_task_am,
        args=[scheduler, 45],
        trigger='cron',
        hour=9, minute=26, misfire_grace_time=300,
        id=f'{MODEL_NAME}_start_trading_job_am_45',
        name='Start_trading_program_at_9:26_AM_45',
    )
    scheduler.add_job(
        trading_task_pm,
        args=[scheduler, 50],
        trigger='cron',
        hour=12, minute=55, misfire_grace_time=300,
        id=f'{MODEL_NAME}_start_trading_job_pm_50',
        name='Start_trading_program_at_12:55_PM_50',
    )
    scheduler.add_job(
        trading_task_pm,
        args=[scheduler, 45],
        trigger='cron',
        hour=12, minute=56, misfire_grace_time=300,
        id=f'{MODEL_NAME}_start_trading_job_pm_45',
        name='Start_trading_program_at_12:56_PM_45',
    )
    scheduler.add_job(
        calculate_today_statistics_indicators,
        trigger='cron',
        hour=15, minute=5, misfire_grace_time=300,
        id='calculate_today_statistics_indicators',
        name='每天15:05计算今日统计数据'
    )
    scheduler.add_job(
        backup_trade_data,
        trigger='cron',
        hour=15, minute=15, misfire_grace_time=300,
        id='backup_trade_data',
        name='每天15:15备份交易数据'
    )
    scheduler.add_job(
        update_and_predict_dataset,
        trigger='cron',
        hour=17, minute=45, misfire_grace_time=300,
        id='update_predict_dataset',
        name='每日17:45更新预测数据集'
    )
    scheduler.add_job(
        train_and_predict_model,
        trigger='cron',
        day_of_week='sun', hour=1, minute=0, misfire_grace_time=300,
        id='train_predict_dataset',
        name='每周日上午1:00训练模型'
    )
    scheduler.start()
    print(f'({MODEL_NAME}) 开始启动自动运行,按CTRL+C退出')
    try:
        while True:
            time.sleep(5)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        print(f'{MODEL_NAME} 自动运行已关闭')

if __name__ == '__main__':
    auto_run()