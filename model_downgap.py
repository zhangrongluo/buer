import os
import time
import random
from tensorflow.keras import layers  # type: ignore
import pandas as pd
from tensorflow import keras
from cons_general import TEMP_DIR, MODELS_DIR
from cons_downgap import dataset_group_cons

temp_root = f'{TEMP_DIR}/downgap'
os.makedirs(temp_root, exist_ok=True)
model_root = f'{MODELS_DIR}/downgap'
os.makedirs(model_root, exist_ok=True)

def train_dataset():
    """
    ### 训练模型
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
        columns = ['gap_percent', 'vol_ratio', 'pct_chg', 'RSI14', 'K', 'MAP14', 'turnover_rate', 
                   'mv_ratio', 'dv_ratio',  'days', 'rise_percent']
        train_df = train_df[columns]

        # STEP3 把前14列作为特征列，最后1列作为标签列
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
        
        # build the model
        def get_model(depth: int = 6, dropout_rate: float = 0.5):
            inputs = keras.Input(shape=(x_train.shape[1],))
            feature = layers.BatchNormalization()(inputs)
            residual = feature
            for dep in range(depth+3, 4, -1):
                feature = layers.Dense(2**dep, activation='relu')(feature)
                if dep % 3 == 0:
                    feature = layers.BatchNormalization()(feature)
                feature = layers.Dropout(dropout_rate)(feature)
                if dep == 7:  # 残差连接
                    if feature.shape[1] != residual.shape[1]:
                        residual = layers.Dense(feature.shape[1])(residual)
                        feature = layers.add([feature, residual])
                    else:
                        feature = layers.add([feature, residual])
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
