import os
from tensorflow.keras import layers  # type: ignore
import pandas as pd  # type: ignore
from tensorflow import keras  
from cons_general import TEMP_DIR, PREDICT_DIR, MODELS_DIR
from cons_oversold import dataset_to_predict_trade,  MIN_PRED_RATE, TEST_DATASET_PERCENT

def predict_dataset():
    """
    ### 预测交易数据集
    """
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
        columns = ['max_down_rate', 'forward_days', 'RSI7', 'K9', 'turnover_rate', 'MAP15', 'mv_ratio', 'pe_ttm', 'pb', 'dv_ratio']
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
