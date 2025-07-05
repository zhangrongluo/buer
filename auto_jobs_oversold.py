import os
import tqdm 
import time # type: ignore
import datetime
import random
import shutil
from tensorflow.keras import layers  # type: ignore
import math
import pandas as pd  # type: ignore
from tensorflow import keras  
from concurrent.futures import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from utils import calculate_today_series_statistic_indicator
from stocklist import get_all_stocks_info, get_stock_list, get_trade_cal, get_up_down_limit_list, get_suspend_stock_list, load_list_df
from basic_data import update_all_daily_data, update_all_daily_indicator, download_all_dividend_data, update_all_adj_factor_data
from trade_oversold import trade_process, XD_holding_list, XD_buy_in_list, clear_buy_in_list
from cons_general import TEMP_DIR, BASICDATA_DIR, TRADE_CAL_XLS, PREDICT_DIR, MODELS_DIR, TRADE_DIR, BACKUP_DIR
from cons_oversold import (dataset_to_update, dataset_to_predict_trade, dataset_to_train, exception_list, MIN_PRED_RATE, 
                           TEST_DATASET_PERCENT, MODEL_NAME, DROP_ROWS_CSV)
from datasets_oversold import create_stock_max_down_dataset, refresh_oversold_data_csv, merge_all_oversold_dataset

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
    """
    更新数据集
    """
    all_stocks = get_all_stocks_info()
    codes = [stock[0] for stock in all_stocks]
    steps = 5

    trade_cal = pd.read_excel(TRADE_CAL_XLS, dtype={'cal_date': str})
    trade_cal = trade_cal[trade_cal['is_open'] == 1]
    trade_cal = trade_cal.sort_values(by='cal_date', ascending=False)
    last_cal_date = trade_cal.iloc[0]['cal_date']
    print('最新交易日期:', last_cal_date)

    for param in dataset_to_update:
        FORWARD_DAYS = param['FORWARD_DAYS']
        BACKWARD_DAYS = param['BACKWARD_DAYS']
        DOWN_FILTER = param['DOWN_FILTER']
        TO_UPDATE = param['TO_UPDATE']
        if not TO_UPDATE:
            continue
        dataset_save_dir = f'{TEMP_DIR}/oversold/data_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}'
        print(f'向前取{FORWARD_DAYS}天(含基准日)，向后取{BACKWARD_DAYS}天(不含基准日), 下跌幅度过滤值{DOWN_FILTER:.2%}')
        print(f'数据集目录为:{dataset_save_dir}')

        all_dataset_csv = f'{dataset_save_dir}/all_oversold_data_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}.csv'
        if os.path.exists(all_dataset_csv):
            all_dataset_df = pd.read_csv(all_dataset_csv, dtype={'trade_date': str})
            last_trade_date = str(all_dataset_df['trade_date'].max())[:8]
            if last_trade_date == last_cal_date:
                print(f'数据集{all_dataset_csv}已更新到最新交易日期{last_trade_date}')
                print()
                continue

        # build the params list for the thread pool
        params = [(code, FORWARD_DAYS, BACKWARD_DAYS, DOWN_FILTER) for code in codes]

        # build all stock max down dataset
        print('开始构建所有股票的最大下跌数据集，请稍等...')
        bar = tqdm.tqdm(total=len(params), ncols=100)
        for index in range(0, len(params), steps):
            with ThreadPoolExecutor(max_workers=steps) as executor:
                executor.map(create_stock_max_down_dataset, params[index:index+steps])
            bar.update(steps)
        bar.close()

        # refresh all stock max down dataset
        print('开始刷新所有股票的最大下跌数据集，请稍等...')
        bar = tqdm.tqdm(total=len(params), ncols=100)
        for index in range(0, len(params), steps):
            with ThreadPoolExecutor(max_workers=steps) as executor:
                executor.map(refresh_oversold_data_csv, params[index:index+steps])
            bar.update(steps)
        bar.close()

        # merge all stock max down dataset
        print('开始合并所有股票的最大下跌数据集，请稍等...')
        merge_all_oversold_dataset(
            forward_days= FORWARD_DAYS, 
            backward_days= BACKWARD_DAYS, 
            down_filter= DOWN_FILTER
        )
        print('数据集构建完成！')
        print()

def train_dataset():
    """
    训练数据集
    """
    for dataset in dataset_to_train:
        FORWARD_DAYS = dataset['FORWARD_DAYS']
        BACKWARD_DAYS = dataset['BACKWARD_DAYS']
        DOWN_FILTER = dataset['DOWN_FILTER']
        times = dataset['TIMES']
        if times == 0:
            continue

        oversold_data_dir = f'{TEMP_DIR}/oversold/data_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}'
        os.makedirs(oversold_data_dir, exist_ok=True)
        oversold_data_csv = f'{oversold_data_dir}/all_oversold_data_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}.csv'
        df_oversold_data = pd.read_csv(oversold_data_csv, dtype={'trade_date': str, 'max_date_forward': str})
        df_oversold_data = df_oversold_data.sort_values(by=['trade_date', 'code'], ascending=[True, True])
        df_oversold_data = df_oversold_data.reset_index(drop=True)

        # 取出最后BACKWARD_DAYS天的数据,最为交易数据集
        trade_dates_list = df_oversold_data['trade_date'].unique().tolist()
        df_trade = df_oversold_data[df_oversold_data['trade_date'].isin(trade_dates_list[-BACKWARD_DAYS:])]
        df_trade_csv = f'{oversold_data_dir}/oversold_trade_dataset_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}.csv'
        df_trade.to_csv(df_trade_csv, index=False)

        # 其余最为训练数据集
        df_train = df_oversold_data[~df_oversold_data['trade_date'].isin(trade_dates_list[-BACKWARD_DAYS:])]
        df_train = df_train.dropna(subset=['max_date_forward', 'max_up_rate'])
        # df_train = df_train.dropna(subset=['K15', 'MAP15'])
        # NOTE: 对于code和max_date_forward相同的记录，为同一下跌趋势，只保留最后一条记录
        df_train = df_train.drop_duplicates(subset=['code', 'max_date_forward'], keep='last')
        df_train = df_train.sort_values(by=['trade_date', 'code'], ascending=[True, True])
        df_train_csv = f'{oversold_data_dir}/oversold_train_dataset_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}.csv'
        df_train.to_csv(df_train_csv, index=False)
        df_origin = df_train.copy()

        # 制作训练数据集(测试集和验证集)
        oversold_data_dir = f'{TEMP_DIR}/oversold/data_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}'
        df_train_csv = f'{oversold_data_dir}/oversold_train_dataset_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}.csv'
        df_train = pd.read_csv(df_train_csv, dtype={'trade_date': str, 'max_date_forward': str})
        # 对df_train的pe_ttm列求正弦值
        df_train['sina_pe_ttm'] = df_train['pe_ttm'].apply(lambda x: math.sin(x))
        # 对forward_days列求正弦值
        df_train['sina_forward_days'] = df_train['forward_days'].apply(lambda x: math.sin(x))
        columns = ['pct_chg', 'vol_ratio', 'max_down_rate', 'sina_forward_days', 'RSI14', 'RSI7', 'RSI3', 'K9', 'K15', 
                'turnover_rate', 'MAP15', 'MAP7', 'mv_ratio', 'sina_pe_ttm', 'pb', 'dv_ratio']
        df_train_valid_test = df_train[columns]  # 测试验证集的特征值
        df_label = df_train['max_up_rate']
        test_length = int(len(df_train_valid_test) * TEST_DATASET_PERCENT)
        x_train_valid = df_train_valid_test[:-test_length]
        y_train_valid = df_label[:-test_length]
        x_test = df_train_valid_test[-test_length:]
        y_test = df_label[-test_length:]
        
        # build the model
        def get_model(depth: int = 6, drop_rate: float = 0.5):
            inputs = keras.Input(shape=(x_train_valid.shape[1],))
            feature = layers.BatchNormalization()(inputs)
            residual = feature
            for dep in range(depth+3, 4, -1):
                feature = layers.Dense(2**dep, activation='relu')(feature)
                feature = layers.Dropout(drop_rate)(feature)
                if dep % 3 == 0:
                    feature = layers.BatchNormalization()(feature)
                if dep == 7:  # 残差连接
                    if feature.shape[1] != residual.shape[1]:
                        residual = layers.Dense(2**dep)(residual)
                        feature = layers.add([feature, residual])
                    else:
                        feature = layers.add([feature, residual])
            outputs = layers.Dense(1)(feature)
            model = keras.Model(inputs, outputs)
            optimizer = random.choice(['adam', 'rmsprop'])
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])
            return model
        
        # 训练模型
        model_root = f'{MODELS_DIR}/oversold/model_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}'
        os.makedirs(model_root, exist_ok=True)

        print(f'开始训练{dataset}数据集，请稍等...')
        for i in range(times):
            try:
                now = time.strftime('%Y%m%d%H%M%S', time.localtime())
                depth = random.choice([6, 7, 8])
                drop_rate = random.choice([0.2, 0.3, 0.4, 0.5])
                model = get_model(depth=depth, drop_rate=drop_rate)
                params = model.count_params()
                model_name = f'{model_root}/model_{now}_{params}.keras'
                callbacks = [
                    keras.callbacks.ModelCheckpoint(filepath=model_name, save_best_only=True, monitor='val_mae', mode='min'),
                    keras.callbacks.EarlyStopping(monitor='val_mae', patience=8, mode='min')
                ]
                epoches = random.choice([50, 40, 30])
                batch_size = random.choice([32, 64, 128, 256])
                validation_split = random.choice([0.1, 0.15, 0.2, 0.25])
                model.fit(
                    x_train_valid, y_train_valid, epochs=epoches, batch_size=batch_size, validation_split=validation_split, callbacks=callbacks
                )
                # 寻找model.history中最小的val_mae和对应的val_loss和val_mape
                min_mae = min(model.history.history['val_mae'])
                min_loss = model.history.history['val_loss'][model.history.history['val_mae'].index(min_mae)]
                min_mape = model.history.history['val_mape'][model.history.history['val_mae'].index(min_mae)]
                # 找到model_root下最新的文件，将名称后面加上val_loss和val_mae的最小值
                files = os.listdir(model_root)
                files = [f for f in files if f.endswith('.keras')]
                files.sort(reverse=True)
                model_name = files[0]
                if 'mae' not in model_name:  # 如果文件名中没有包含mae，说明还没有重命名过
                    new_suffix = f'_loss{min_loss:.4f}_mae{min_mae:.4f}_mape{min_mape:.4f}.keras'
                    new_model_name = model_name.replace('.keras', new_suffix)
                    os.rename(f'{model_root}/{model_name}', f'{model_root}/{new_model_name}')
            except Exception as e:
                pass
        
def predict_dataset():
    for dataset in dataset_to_predict_trade:
        FORWARD_DAYS = dataset['FORWARD_DAYS']
        BACKWARD_DAYS = dataset['BACKWARD_DAYS']
        DOWN_FILTER = dataset['DOWN_FILTER']
        PRED_MODELS = dataset['PRED_MODELS']
        if PRED_MODELS == 0:
            continue
        print(f'开始预测{dataset}数据集，请稍等...')

        # STEP 1:split trade dataset and train dataset
        oversold_data_dir = f'{TEMP_DIR}/oversold/data_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}'
        os.makedirs(oversold_data_dir, exist_ok=True)
        oversold_data_csv = f'{oversold_data_dir}/all_oversold_data_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}.csv'
        df_oversold_data = pd.read_csv(oversold_data_csv, dtype={'trade_date': str, 'max_date_forward': str})
        df_oversold_data = df_oversold_data.sort_values(by='trade_date', ascending=True)
        df_oversold_data = df_oversold_data.reset_index(drop=True)
        # 取出最后10天的数据,最为交易数据集
        trade_dates_list = df_oversold_data['trade_date'].unique().tolist()
        df_trade = df_oversold_data[df_oversold_data['trade_date'].isin(trade_dates_list[-BACKWARD_DAYS:])]
        df_trade_csv = f'{oversold_data_dir}/oversold_trade_dataset_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}.csv'
        df_trade.to_csv(df_trade_csv, index=False)
        # 其余最为训练数据集
        df_train = df_oversold_data[~df_oversold_data['trade_date'].isin(trade_dates_list[-BACKWARD_DAYS:])]
        df_train = df_train.dropna(subset=['max_date_forward', 'max_up_rate'])
        df_train = df_train.dropna(subset=['K15', 'MAP15'])
        # 删除code和max_date_forward相同的行，保留最后一行
        df_train = df_train.drop_duplicates(subset=['code', 'max_date_forward'], keep='last')
        df_train = df_train.sort_values(by=['trade_date', 'code'], ascending=[True, True])
        df_train_csv = f'{oversold_data_dir}/oversold_train_dataset_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}.csv'
        df_train.to_csv(df_train_csv, index=False)
        df_origin = df_train.copy()

        # STEP 2:制作训练数据集、测试数据集
        df_train_csv = f'{oversold_data_dir}/oversold_train_dataset_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}.csv'
        df_train = pd.read_csv(df_train_csv, dtype={'trade_date': str, 'max_date_forward': str})
        # 对df_train的pe_ttm列求正弦值
        df_train['sina_pe_ttm'] = df_train['pe_ttm'].apply(lambda x: math.sin(x))
        # 对df_train的forward_days列求正弦值
        df_train['sina_forward_days'] = df_train['forward_days'].apply(lambda x: math.sin(x))
        columns = ['pct_chg', 'vol_ratio', 'max_down_rate', 'sina_forward_days', 'RSI14', 'RSI7', 'RSI3', 'K9', 'K15', 
                'turnover_rate', 'MAP15', 'MAP7', 'mv_ratio', 'sina_pe_ttm', 'pb', 'dv_ratio']
        df_train_valid_test = df_train[columns]  # 训练集和验证集的特征值
        df_label = df_train['max_up_rate']
        test_length = int(len(df_train_valid_test) * TEST_DATASET_PERCENT)
        x_train_valid = df_train_valid_test[:-test_length]
        y_train_valid = df_label[:-test_length]
        x_test = df_train_valid_test[-test_length:]
        y_test = df_label[-test_length:]

        # STEP 3:准备交易数据集
        oversold_data_dir = f'{TEMP_DIR}/oversold/data_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}'
        os.makedirs(oversold_data_dir, exist_ok=True)
        df_trade_csv = f'{oversold_data_dir}/oversold_trade_dataset_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}.csv'
        df_trade = pd.read_csv(df_trade_csv, dtype={'trade_date': str, 'max_date_forward': str})
        # 对df_trade的pe_ttm列求正弦值
        df_trade['sina_pe_ttm'] = df_trade['pe_ttm'].apply(lambda x: math.sin(x))
        # 对df_trade的forward_days列求正弦值
        df_trade['sina_forward_days'] = df_trade['forward_days'].apply(lambda x: math.sin(x))
        x_trade = df_trade[columns]

        # STEP 4:评价全部模型在测试集上的表现(使用预测值和真实值的比率衡量)
        model_root = f'{MODELS_DIR}/oversold/model_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}'
        os.makedirs(model_root, exist_ok=True)
        if not os.path.exists(model_root):
            os.makedirs(model_root)
        models = os.listdir(model_root)
        model_list = []
        for model in models:
            # 按照model中mae后面的值升序排列
            if model.endswith('.keras') and 'mae' in model:
                tmp = model.split('mae')
                if len(tmp) == 2:  # 正常名字中只有一个'mae'
                    model_list.append(tmp)
        model_list = sorted(model_list, key=lambda x: x[-1])
        model_list = [f'{model_root}/{i[0]}mae{i[1]}' for i in model_list]
        model_list = model_list[:12]
        pred_models = model_list[:PRED_MODELS]

        for model in os.listdir(model_root):
            tmp_model = f'{model_root}/{model}'
            if tmp_model.endswith('.keras') and tmp_model not in model_list:
                os.remove(tmp_model) 

        # STEP 5:use pred_models to predict the test dataset
        y_test_pred_list = []
        for model in pred_models:
            print(f'load model: {model} to predict test dataset')
            model = keras.models.load_model(model)
            pred = model.predict(x_test)
            y_test_pred_list.append(pred)
            del model
        y_test_pred = sum(y_test_pred_list) / len(y_test_pred_list)
        diff = pd.DataFrame({'real': y_test, 'pred': y_test_pred.flatten()})
        diff['code'] = df_origin.iloc[-test_length:]['code'].values
        diff['name'] = df_origin.iloc[-test_length:]['name'].values
        diff['industry'] = df_origin.iloc[-test_length:]['industry'].values
        diff['trade_date'] = df_origin.iloc[-test_length:]['trade_date'].values
        diff['max_date_forward'] = df_origin.iloc[-test_length:]['max_date_forward'].values
        diff['forward_days'] = df_origin.iloc[-test_length:]['forward_days'].values
        diff['max_down_rate'] = df_origin.iloc[-test_length:]['max_down_rate'].values
        diff['max_date_backward'] = df_origin.iloc[-test_length:]['max_date_backward'].values
        diff['backward_days'] = df_origin.iloc[-test_length:]['backward_days'].values
        diff = diff[
            ['code', 'name', 'industry', 'trade_date', 'max_date_forward', 'forward_days', 
            'max_down_rate', 'max_date_backward', 'backward_days', 'pred', 'real']
        ]
        diff['diff'] = diff['pred'] - diff['real']
        diff['diff_pct'] = diff['diff'] / diff['real']
        diff['diff_pct'] = diff['diff_pct'].apply(lambda x: f'{x:.2%}')
        pred_path = f'{PREDICT_DIR}/oversold/pred_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}'
        os.makedirs(pred_path, exist_ok=True)
        csv_name = f'{pred_path}/test_pred_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}.csv'
        diff.to_csv(csv_name, index=False)
        # filter the test dataset with diff['diff'] > MIN_PRED_RATE and plot bar chart
        df_filter = diff[diff['pred'] > MIN_PRED_RATE]
        df_filter.to_csv(f'{pred_path}/test_pred_filter_gt_{MIN_PRED_RATE:.2f}_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}.csv', index=False)

        # STEP 6:use pred_models to predict x_trade and then average the result
        y_trade_pred = []
        for model_name in pred_models:
            print(f'loading model: {model_name} to predict trade dataset')
            model = keras.models.load_model(model_name)
            pred = model.predict(x_trade)
            y_trade_pred.append(pred)
            del model
        # average the y_trade_pred
        y_trade_pred = sum(y_trade_pred) / len(y_trade_pred)

        #  STEP 7:predict the trade dataset
        diff = pd.DataFrame({'pred': y_trade_pred.flatten(), 'real': df_trade['max_up_rate'].values})
        diff['code'] = df_trade['code'].values
        diff['name'] = df_trade['name'].values
        diff['industry'] = df_trade['industry'].values
        diff['trade_date'] = df_trade['trade_date'].values
        diff['chg_pct'] = df_trade['pct_chg'].values
        diff['max_date_forward'] = df_trade['max_date_forward'].values
        diff['max_down_rate'] = df_trade['max_down_rate'].values
        diff['forward_days'] = df_trade['forward_days'].values
        diff['diff'] = diff['pred'] - diff['real']
        diff['diff_pct'] = diff['diff'] / diff['real']
        diff['diff_pct'] = diff['diff_pct'].apply(lambda x: f'{x:.2%}')
        diff = diff[['code', 'name', 'industry', 'trade_date', 'chg_pct', 'max_date_forward', 'max_down_rate', 'forward_days', 'pred', 'real']]
        diff.to_csv(f'{pred_path}/trade_pred_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}.csv', index=False)

def build_buy_in_list():
    print('creating and saving sub buy_in_list csv files...')
    for dataset in dataset_to_predict_trade:
        FORWARD_DAYS = dataset['FORWARD_DAYS']
        BACKWARD_DAYS = dataset['BACKWARD_DAYS']
        DOWN_FILTER = dataset['DOWN_FILTER']
        PRED_MODELS = dataset['PRED_MODELS']
        if PRED_MODELS == 0:
            continue

        # prepare data from trade_pred.csv within TRADE_COVERAGE_DAYS days
        pred_path = f'{PREDICT_DIR}/oversold/pred_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}'
        os.makedirs(pred_path, exist_ok=True)
        csv_name = f'{pred_path}/trade_pred_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}.csv'
        trade_df = pd.read_csv(csv_name, dtype={'trade_date': str})

        # add trade_date's close price (buy_point_base) and waiting_days to trade_df
        trade_df.insert(4, 'buy_point_base', None)
        trade_df.insert(9, 'waiting_days', None)
        for i , row in trade_df.iterrows():
            code = row['code']
            industry = row['industry']
            trade_date = row['trade_date']
            daily_csv = f'{BASICDATA_DIR}/dailydata/{code}.csv'
            daily_df = pd.read_csv(daily_csv, dtype={'trade_date': str})
            close = daily_df[daily_df['trade_date'] == trade_date]['close'].values[0]
            trade_df.loc[i, 'buy_point_base'] = close

            # calculate days between trade_date and today
            daily_csv = f'{BASICDATA_DIR}/dailydata/{code}.csv' 
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
            trade_df.loc[i, 'waiting_days'] = days
        trade_df['pred_100%'] = trade_df['pred'].map(lambda x: f'{x:.2%}')

        # filter trade_df
        trade_df = trade_df[trade_df['pred'] > MIN_PRED_RATE]
        trade_df = trade_df[~trade_df['name'].str.contains('|'.join(exception_list))]
        trade_df = trade_df.reset_index(drop=True)
        # save trade_df to csv
        trade_dir = f'{TRADE_DIR}/oversold'
        os.makedirs(trade_dir, exist_ok=True)
        trade_df.to_csv(f'{trade_dir}/buy_in_list_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}.csv', index=False)
        print(f'buy_in_list_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}.csv saved')

        all_df = pd.DataFrame()
        for dataset in dataset_to_predict_trade:
            FORWARD_DAYS = dataset['FORWARD_DAYS']
            BACKWARD_DAYS = dataset['BACKWARD_DAYS']
            DOWN_FILTER = dataset['DOWN_FILTER']
            PRED_MODELS = dataset['PRED_MODELS']
            if PRED_MODELS == 0:
                continue
            list_name = f'buy_in_list_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}.csv'
            buy_in_csv = f'{trade_dir}/{list_name}'
            buy_in_df = pd.read_csv(buy_in_csv)
            buy_in_df['src'] = list_name
            all_df = pd.concat([all_df, buy_in_df], ignore_index=True)
        # drop duplicate rows with same code and trade_date, save the highest pred row
        all_df = all_df.sort_values(by=['code', 'trade_date', 'pred'], ascending=[True, True, True])
        all_df = all_df.drop_duplicates(subset=['code', 'trade_date'], keep='last')
        # rename columns and save to buy_in_list.csv
        new_columns = ['ts_code', 'stock_name', 'industry', 'trade_date', 'buy_point_base', 'chg_pct', 'max_date_forward', 
                        'max_down_rate', 'forward_days', 'waiting_days', 'pred', 'real', 'pred_100%', 'src']
        all_df.columns = new_columns
        all_df.to_csv(f'{trade_dir}/buy_in_list.csv', index=False)

scheduler = BackgroundScheduler()
scheduler.configure(timezone='Asia/Shanghai')

def update_trade_cal_and_stock_list():
    get_trade_cal()
    get_stock_list()
    total_stocks = load_list_df()
    today = datetime.datetime.now().date().strftime('%Y%m%d')
    print(f'({MODEL_NAME}) {today} 现追踪的股票列表总数: {total_stocks}')
    print(f'({MODEL_NAME}) {today} 交易日历和股票列表更新完成！')

@is_trade_day(task='更新行情和指标数据')
def update_daily_data_and_indicator():
    update_all_daily_data(step=5)
    update_all_daily_indicator(step=5)
    today = datetime.datetime.now().date().strftime('%Y%m%d')
    print(f'({MODEL_NAME}) {today} 行情和指标数据更新完成！')

@is_trade_day(task='更新和预测数据集')
def update_and_predict_dataset():
    update_dataset()
    predict_dataset()
    today = datetime.datetime.now().date().strftime('%Y%m%d')
    print(f'({MODEL_NAME}) {today} oversold 数据集更新完成！')

@is_trade_day(task='构建买入清单')
def build_buy_in_list_task():
    build_buy_in_list()
    drop_rows = clear_buy_in_list()
    if drop_rows is not None:
        drop_rows.to_csv(DROP_ROWS_CSV, index=False)
    today = datetime.datetime.now().date().strftime('%Y%m%d')
    print(f'({MODEL_NAME}) {today} 买入清单更新完成！')

@is_trade_day(task='获取涨跌停表、停牌清单和分红送股数据')
def get_limit_and_suspend_list_and_dividend_task():
    get_up_down_limit_list()
    today = datetime.datetime.now().date().strftime('%Y%m%d')
    print(f'({MODEL_NAME}) {today} 涨跌停表更新完成！')
    get_suspend_stock_list()
    print(f'({MODEL_NAME}) {today} 停牌清单更新完成！')
    download_all_dividend_data()
    print(f'({MODEL_NAME}) {today} 分红送股数据更新完成！')

@is_trade_day(task='更新复权因子数据')
def update_adj_data_and_XD_stock_and_trading_am_task():
    today = datetime.datetime.now().date().strftime('%Y%m%d')
    update_all_adj_factor_data()
    print(f'({MODEL_NAME}) {today} 复权因子数据更新完成！')
    # 更新复权因子和分红数据后，执行盘中前复权和股数调整
    XD_stock_list_task()
    # 执行上午的股票交易任务
    trading_task_am(scheduler=scheduler)

@is_trade_day(task='盘中前复权和股数调整')
def XD_stock_list_task():
    XD_buy_in_list()
    today = datetime.datetime.now().date().strftime('%Y%m%d')
    print(f'({MODEL_NAME}) {today} 买入清单前复权调整完成！')
    XD_holding_list()
    print(f'({MODEL_NAME}) {today} 持有清单前复权和股数调整完成！')

@is_trade_day(task='计算今日统计指标')
def calculate_today_statistics_indicators():
    calculate_today_series_statistic_indicator(name='oversold')
    today = datetime.datetime.now().date().strftime('%Y%m%d')
    print(f'({MODEL_NAME}) {today} 今日统计数据计算完成！')

@is_trade_day(task='清理屏幕')
def clear_screen_task():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f'({MODEL_NAME}) oversold 模型自动运行中')

def train_and_predict_dataset():
    train_dataset()
    predict_dataset()
    today = datetime.datetime.now().date().strftime('%Y%m%d')
    print(f'({MODEL_NAME}) {today} oversold 模型训练完成！')

@is_trade_day(task='备份交易数据')
def backup_trade_data():
    """
    把TRADE_DIR/oversold目录下的所有文件备份到BACKUP_DIR/oversold/oversold_<备份时间>目录下
    NOTE:
    备份清单包括: 买入清单、持有清单、交易日志、资金流水、每日利润、每日指标文件
    保留最近的6个备份
    """
    trade_dir = f'{TRADE_DIR}/oversold'
    backup_root = f'{BACKUP_DIR}/oversold'
    backup_dir = f'{backup_root}/oversold_{datetime.datetime.now().strftime("%Y%m%d %H%M%S")}'
    shutil.copytree(trade_dir, backup_dir, dirs_exist_ok=True)
    files = os.listdir(backup_root)
    dirs = [d for d in files if os.path.isdir(os.path.join(backup_root, d))]
    dirs.sort(reverse=True)
    [shutil.rmtree(os.path.join(backup_root, d)) for d in dirs[12:]]  # 保留最近12个备份
    print(f'({MODEL_NAME}) oversold 模型交易数据备份完成！')

# 动态任务am
@is_trade_day(task='股票交易')
def trading_task_am(scheduler):
    now = datetime.datetime.now().time()
    start_time = datetime.time(9, 20)  # 9:20 AM, start of trading
    end_time = datetime.time(11, 30)  # 11:30 AM, end of trading
    if start_time <= now <= end_time:
        try:
            trade_process()
        except Exception as e:
            print(f"Error during trading process: {e}")
        # 启动下一个交易loop
        if datetime.datetime.now().time() <= end_time:
            scheduler.add_job(
                trading_task_am,
                args=[scheduler],
                run_date=datetime.datetime.now()+datetime.timedelta(seconds=1),
                id=f'{MODEL_NAME}_trading_job_am_{int(time.time())}',
                name='Run_am_trading_task'
            )
        else:
            now = datetime.datetime.now()
            print(f'({MODEL_NAME}) {now.time()} 不在交易时间段内.')
    else:
        now = datetime.datetime.now()
        print(f'({MODEL_NAME}) {now.time()} 不在交易时间段内.')

# 动态任务pm
@is_trade_day(task='股票交易')
def trading_task_pm(scheduler):
    now = datetime.datetime.now().time()
    start_time = datetime.time(12, 50)
    end_time = datetime.time(15, 0)
    if start_time <= now <= end_time:
        try:
            trade_process()
        except Exception as e:
            print(f"Error during trading process: {e}")
        # 启动下一个交易loop
        if datetime.datetime.now().time() <= end_time:
            scheduler.add_job(
                trading_task_pm,
                args=[scheduler],
                run_date=datetime.datetime.now()+datetime.timedelta(seconds=1),
                id=f'{MODEL_NAME}_trading_job_pm_{int(time.time())}',
                name='Run_pm_trading_task'
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
    # 定时任务注册
    scheduler.add_job(
        update_trade_cal_and_stock_list,
        trigger='cron',
        hour=0, minute=1, misfire_grace_time=300,
        id='update_trade_cal_and_stock_list'
    )
    scheduler.add_job(
        build_buy_in_list_task,
        trigger='cron',
        hour=1, minute=0, misfire_grace_time=300,
        id='update_buy_in_list'
    )
    scheduler.add_job(
        get_limit_and_suspend_list_and_dividend_task,
        trigger='cron',
        hour=9, minute=20, misfire_grace_time=300,
        id='get_limit_and_suspend_list_and_dividend_task'
    )
    scheduler.add_job(
        backup_trade_data,
        trigger='cron',
        hour=11, minute=45, misfire_grace_time=300,
        id='backup_trade_data_am'
    )
    scheduler.add_job(
        calculate_today_statistics_indicators,
        trigger='cron',
        hour=15, minute=5, misfire_grace_time=300,
        id='calculate_today_statistics_indicators'
    )
    scheduler.add_job(
        backup_trade_data,
        trigger='cron',
        hour=15, minute=15, misfire_grace_time=300,
        id='backup_trade_data_pm'
    )
    scheduler.add_job(
        update_daily_data_and_indicator,
        trigger='cron',
        hour=16, minute=30, misfire_grace_time=300,
        id='update_daily_data_and_indicator'
    )
    scheduler.add_job(
        update_and_predict_dataset,
        trigger='cron',
        hour=18, minute=0, misfire_grace_time=300,
        id='update_and_predict_dataset'
    )
    scheduler.add_job(
        train_and_predict_dataset,
        trigger='cron',
        day_of_week='sat', hour=1, minute=0, misfire_grace_time=300,
        id='train_and_predict_dataset'
    )

    # 动态任务
    scheduler.add_job(
        # include XD_stock_list and trading_task_am
        update_adj_data_and_XD_stock_and_trading_am_task,
        trigger='cron',
        hour=9, minute=31, misfire_grace_time=300,
        id='update_adj_factor_data'
    )
    scheduler.add_job(
        trading_task_pm,
        args=[scheduler],
        trigger='cron',
        hour=12, minute=55, misfire_grace_time=300,
        id=f'{MODEL_NAME}_start_trading_job_pm',
        name='Start_trading_program_at_12:55_PM',
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