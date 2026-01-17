"""
### 定义训练可用的模型
"""
import random
from tensorflow import keras # type: ignore
from tensorflow.keras import layers # type: ignore

def get_simple_dense_model(input_dim: int, dropout_rate: float = 0.3):
    """
    ### 基于 Keras 的全连接神经网络回归模型
    #### :param input_dim: 输入特征维度
    #### :param normalizer: 归一化层
    #### :param dropout_rate: Dropout 比例
    """
    inputs = keras.Input(shape=(input_dim,))
    x = layers.BatchNormalization()(inputs)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
    loss = keras.losses.Huber()  # Huber Loss 对异常值更鲁棒
    model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
    return model

def get_resnet_model_src(input_dim:int, depth: int = 6, dropout_rate: float = 0.5):
    """
    ### 基于 Keras 的残差神经网络回归模型, 原始版本
    #### :param input_dim: 输入特征维度
    #### :param depth: 网络深度
    #### :param dropout_rate: Dropout 比例
    """
    inputs = keras.Input(shape=(input_dim,))
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
    model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
    return model

def get_resnet_model_optimized(input_dim: int, depth: int = 6, dropout_rate: float = 0.3):
    """
    ### 优化版残差神经网络回归模型, Gemini 优化版本
    #### :param input_dim: 输入特征维度
    #### :param depth: 网络深度
    #### :param dropout_rate: Dropout 比例
    """
    inputs = keras.Input(shape=(input_dim,))
    x = layers.BatchNormalization()(inputs)
    x = layers.Dense(128, activation='swish')(x) # Swish 激活函数在金融回归中通常优于 ReLU
    for _ in range(depth):
        res = x
        x = layers.Dense(128, activation='swish')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(128)(x)
        x = layers.Add()([x, res])
        x = layers.Activation('swish')(x)
        x = layers.BatchNormalization()(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
    loss = keras.losses.Huber()  # Huber Loss 对异常值更鲁棒
    model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
    return model
