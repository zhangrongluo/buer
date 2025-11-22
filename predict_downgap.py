import os
from tensorflow.keras import layers  # type: ignore
import pandas as pd
from tensorflow import keras
from cons_general import TEMP_DIR, MODELS_DIR, PREDICT_DIR
from cons_downgap import dataset_group_cons

temp_root = f'{TEMP_DIR}/downgap'
os.makedirs(temp_root, exist_ok=True)
model_root = f'{MODELS_DIR}/downgap'
os.makedirs(model_root, exist_ok=True)
predict_root = f'{PREDICT_DIR}/downgap'
os.makedirs(predict_root, exist_ok=True)

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
        columns = ['gap_percent', 'vol_ratio', 'pct_chg', 'RSI14', 'K', 'MAP14', 'turnover_rate', 
                   'mv_ratio', 'dv_ratio',  'days', 'rise_percent']
        train_df = train_df[columns]

        # STEP3: 把前9列作为特征列，最后1列作为标签列
        x_train = train_df[['gap_percent', 'vol_ratio', 'pct_chg', 'RSI14', 'K', 'MAP14', 'turnover_rate', 
                            'mv_ratio', 'dv_ratio']]
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
        x_trade = trade_df[['gap_percent', 'vol_ratio', 'pct_chg', 'RSI14', 'K', 'MAP14', 'turnover_rate', 
                            'mv_ratio', 'dv_ratio']]

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
