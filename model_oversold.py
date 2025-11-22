import os
import time # type: ignore
import random
from tensorflow.keras import layers  # type: ignore
import pandas as pd  # type: ignore
from tensorflow import keras  
from cons_general import TEMP_DIR, MODELS_DIR
from cons_oversold import dataset_to_train, TEST_DATASET_PERCENT

def train_dataset():
    """
    ### 训练数据集
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
        columns = ['max_down_rate', 'forward_days', 'RSI7', 'K9', 'turnover_rate', 'MAP15', 'mv_ratio', 'pe_ttm', 'pb', 'dv_ratio']
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
