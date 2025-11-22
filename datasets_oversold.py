"""
calculate、refresh and merge Oversold datasets
"""
import os
import tqdm
import pandas as pd
import datetime
from concurrent.futures import ThreadPoolExecutor
from stocklist import get_name_and_industry_by_code, get_all_stocks_info
from cons_general import BASICDATA_DIR, DATASETS_DIR, TEMP_DIR, TRADE_CAL_CSV
from cons_oversold import dataset_to_update
from utils import get_qfq_price_DF_by_adj_factor

def calculate_RSI_indicator(df: pd.DataFrame, period: int = 14):
    """
    ### 计算RSI指标
    #### :param df: DataFrame, 包含pct_chg列
    #### :param period: 计算周期, 默认14天
    """
    if len(df) < period:
        return None
    df_rsi = df.iloc[-period:]
    # 使用pct_change计算涨跌幅
    up = df_rsi.loc[:, 'pct_chg'].apply(lambda x: x if x > 0 else 0).sum()
    down = abs(df.loc[:, 'pct_chg'].apply(lambda x: x if x < 0 else 0).sum())
    rs = up /(down + 1e-10) if down == 0 else up / down
    rsi = 100 - 100 / (1 + rs)
    return round(rsi, 2)

def calculate_K_indicator(df: pd.DataFrame, period: int = 14):
    """
    ### 计算K指标
    #### :param df: DataFrame, 包含close, high, low列
    #### :param period: 计算周期, 默认14天
    #### NOTE: K = 100 * (close - min) / (max - min)
    """
    if len(df) < period:
        return None
    df_k = df.iloc[-period:]
    max_price = df_k['high'].max()
    min_price = df_k['low'].min()
    close = df_k.iloc[-1]['close']
    if max_price == min_price:
        return None
    K = 100 * (close - min_price) / (max_price - min_price)
    return round(K, 2)

def calculate_MAP_indicator(df: pd.DataFrame, period: int = 15) -> float:
    """
    ### 计算移动平均价格指标
    #### :param df: DataFrame, 包含close列
    #### :param period: 计算周期, 默认15天
    #### NOTE: 
    #### MAP = sum(close) / period
    """
    if len(df) < period:
        return None
    df_map = df.iloc[-period:]
    MAP = df_map['close'].sum() / period
    return round(MAP, 2)

def create_stock_max_down_dataset(params: tuple):
    """ 
    ### 计算创建股票的指定期间最大下跌幅度数据集
    #### :param params: 参数列表,包括code, FORWARD_DAYS, BACKWARD_DAYS, DOWN_FILTER
    #### code: 股票代码, 例如 '600848' 或 '600848.SH'
    #### FORWARD_DAYS: 向前的天数(包含基准日)
    #### BACKWARD_DAYS: 向后的天数(不包含基准日)
    #### DOWN_FILTER: 最大下跌幅度的过滤条件
    """
    code, FORWARD_DAYS, BACKWARD_DAYS, DOWN_FILTER = params
    oversold_root = f'{DATASETS_DIR}/oversold'
    oversold_dataset_dir = f'{oversold_root}/oversolddata_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}'
    os.makedirs(oversold_dataset_dir, exist_ok=True)
    if len(code) == 6:
        code = f'{code}.SH' if code.startswith('6') else f'{code}.SZ'
    # 数据预处理
    msg = get_name_and_industry_by_code(ts_code=code)
    name = msg[0]
    industry = msg[1]
    oversold_csv = f'{oversold_dataset_dir}/{code}.csv'
    daily_csv = f'{BASICDATA_DIR}/dailydata/{code}.csv'
    df_daily = pd.read_csv(daily_csv, dtype={'trade_date': str})
    df_daily['trade_date'] = df_daily['trade_date'].apply(lambda x: str(x)[:8])
    df_daily['trade_date'] = df_daily['trade_date'].apply(lambda x: '' if x == 'nan' else x)  # 去掉nan
    df_daily = df_daily.sort_values(by='trade_date', ascending=True)
    df_daily = df_daily.reset_index(drop=True)
    df_daily['vol_ratio'] =round(df_daily['vol'] / df_daily['vol'].shift(1), 4)
    if os.path.exists(oversold_csv):
        # 打开获取最后一行的last_daily_date
        df_oversold = pd.read_csv(oversold_csv, dtype={'trade_date': str, 'last_daily_date': str})
        df_oversold['trade_date'] = df_oversold['trade_date'].apply(lambda x: str(x)[:8])
        df_oversold['trade_date'] = df_oversold['trade_date'].apply(lambda x: '' if x == 'nan' else x)
        df_oversold['last_daily_date'] = df_oversold['last_daily_date'].apply(lambda x: str(x)[:8])
        df_oversold['last_daily_date'] = df_oversold['last_daily_date'].apply(lambda x: '' if x == 'nan' else x)
        df_oversold = df_oversold.sort_values(by='trade_date', ascending=True)
        df_oversold = df_oversold.reset_index(drop=True)
        last_date = df_oversold.iloc[-1]['last_daily_date']
        if last_date == df_daily.iloc[-1]['trade_date']:
            return
        trade_date_index = df_daily[df_daily['trade_date'] == last_date].index[0]  # 从此开始计算
        df_daily = df_daily.loc[trade_date_index-FORWARD_DAYS:]
        df_daily = df_daily.reset_index(drop=True)

    df_daily = get_qfq_price_DF_by_adj_factor(df_daily)  # 前复权价格序列
    trade_csv = f'{BASICDATA_DIR}/dailyindicator/{code}.csv'
    df_trade = pd.read_csv(trade_csv, dtype={'trade_date': str})
    df_trade['trade_date'] = df_trade['trade_date'].apply(lambda x: str(x)[:8])
    df_trade['trade_date'] = df_trade['trade_date'].apply(lambda x: '' if x == 'nan' else x)  # 去掉nan
    df_trade['mv_ratio'] = round(df_trade['circ_mv']/df_trade['total_mv'], 4)

    #  遍历df_daily, 计算每个交易日的最大下跌幅度
    max_down_list = []
    for i, row in df_daily.iterrows():
        if i < FORWARD_DAYS-1:
            continue
        trade_date = row['trade_date']
        pct_chg = row['pct_chg']
        vol_ratio = row['vol_ratio']
        tmp_row = [code, name, industry, trade_date, pct_chg, vol_ratio]
        date_close = row['close']  # 当天的收盘价
        df_forward = df_daily.loc[i-FORWARD_DAYS+1:i]  # 向前forward天的数据(包括基准日)
        max_price_forward = df_forward['high'].max()
        # 获取最大价格的日期
        max_date_forward = df_forward[df_forward['high'] == max_price_forward]['trade_date'].values[0]
        tmp_row.append(max_date_forward)
        max_down_rate = round((date_close - max_price_forward) / max_price_forward, 4)
        tmp_row.append(max_down_rate)
        # 计算两者的天数
        max_price_index = df_forward[df_forward['high'] == max_price_forward].index[0]
        forward_days = i - max_price_index + 1  # 向前的天数要+1，因为包括基准日本身
        tmp_row.append(forward_days)
        # 计算14 7 3日RSI
        # df_rsi = df_daily.loc[i-14:i]
        RSI14 = calculate_RSI_indicator(period=14, df=df_daily.loc[i-14:i])
        RSI7 = calculate_RSI_indicator(period=7, df=df_daily.loc[i-7:i])
        RSI3 = calculate_RSI_indicator(period=3, df=df_daily.loc[i-3:i])
        tmp_row.append(RSI14)
        tmp_row.append(RSI7)
        tmp_row.append(RSI3)
        # 计算K指标, 9日
        K9 = calculate_K_indicator(period=9, df=df_daily.loc[i-9:i])
        tmp_row.append(K9)
        K15 = calculate_K_indicator(period=15, df=df_daily.loc[i-15:i])
        tmp_row.append(K15)
        # 计算MAP指标, 15日
        MAP15 = calculate_MAP_indicator(period=15, df=df_daily.loc[i-15:i])
        tmp_row.append(MAP15)
        MAP7 = calculate_MAP_indicator(period=7, df=df_daily.loc[i-7:i])
        tmp_row.append(MAP7)
        if max_down_rate > DOWN_FILTER:
            continue
        # 计算向后的最大涨幅
        rest_rows = len(df_daily) - (i + 1)
        if rest_rows < BACKWARD_DAYS:
            max_date_backward = ''
            max_up_rate = None
            backward_days = None
            tmp_row.extend([max_date_backward, max_up_rate, backward_days])
        else:
            df_backward = df_daily.loc[i+1:i+BACKWARD_DAYS]  # 向后backward天的数据, 不包括基准日，因为无法在当天买入
            max_price_backward = df_backward['high'].max()
            # 获取最大价格的日期
            max_date_backward = df_backward[df_backward['high'] == max_price_backward]['trade_date'].values[0]
            tmp_row.append(max_date_backward)
            max_up_rate = round((max_price_backward - date_close) / date_close, 4)
            tmp_row.append(max_up_rate)
            max_price_index = df_backward[df_backward['high'] == max_price_backward].index[0]
            backward_days = max_price_index - i  # 向后的天数不用+1，因为不包括基准日
            tmp_row.append(backward_days)
        # 在df_trade中查找trade_date 日的'turnover_rate', 'mv_ratio', 'pe_ttm', 'pb', 'dv_ratio'指标
        trade_row = df_trade[df_trade['trade_date'] == trade_date]
        if trade_row.empty:
            tmp_row.extend([0, 0, 0, 0, 0])
        else:
            tmp_row.extend([
                trade_row['turnover_rate'].values[0],
                trade_row['mv_ratio'].values[0],
                trade_row['pe_ttm'].values[0],
                trade_row['pb'].values[0],
                trade_row['dv_ratio'].values[0]
            ])
        # 获取df_daily最后一天的日期
        last_daily_date = df_daily.iloc[-1]['trade_date']
        tmp_row.append(last_daily_date)
        max_down_list.append(tmp_row)
    if len(max_down_list) == 0:
        return
    columns = [
        'code', 'name', 'industry', 'trade_date', 'pct_chg', 'vol_ratio', 'max_date_forward', 'max_down_rate', 
        'forward_days', 'RSI14', 'RSI7', 'RSI3', 'K9', 'K15', 'MAP15', 'MAP7', 'max_date_backward', 'max_up_rate', 
        'backward_days', 'turnover_rate', 'mv_ratio', 'pe_ttm', 'pb', 'dv_ratio', 'last_daily_date'
    ]  
    df_max_down = pd.DataFrame(max_down_list, columns=columns)
    columns_new = [
        'code', 'name', 'industry', 'trade_date', 'pct_chg', 'vol_ratio', 'max_date_forward', 'max_down_rate', 
        'forward_days', 'RSI14', 'RSI7', 'RSI3', 'K9', 'K15', 'MAP15', 'MAP7', 'max_date_backward', 'turnover_rate', 
        'mv_ratio', 'pe_ttm', 'pb', 'dv_ratio', 'backward_days', 'max_up_rate', 'last_daily_date'
    ]
    df_max_down = df_max_down[columns_new]
    df_max_down = df_max_down.sort_values(by='trade_date', ascending=True)
    df_max_down = df_max_down.reset_index(drop=True)
    if os.path.exists(oversold_csv):
        df_old = pd.read_csv(oversold_csv, dtype={'trade_date': str})
        df_max_down = pd.concat([df_old, df_max_down])
        df_max_down = df_max_down.drop_duplicates(subset=['trade_date'], keep='last')
    df_max_down.to_csv(oversold_csv, index=False)

def refresh_oversold_data_csv(params: tuple):
    """
    ### 刷新oversold数据集csv文件
    #### :param params: 参数列表,包括code, FORWARD_DAYS, BACKWARD_DAYS, DOWN_FILTER
    #### code: 股票代码, 例如 '600848' 或 '600848.SH'
    #### FORWARD_DAYS: 向前的天数(包含基准日)
    #### BACKWARD_DAYS: 向后的天数(不包含基准日)
    #### DOWN_FILTER: 最大下跌幅度的过滤条件
    #### NOTE: 
    #### 删除重复行、检查标签空白行是否已经经过了BACKWARD_DAYS个交易日, 计算向后的最大涨幅
    """
    code, FORWARD_DAYS, BACKWARD_DAYS, DOWN_FILTER = params
    oversold_root = f'{DATASETS_DIR}/oversold'
    oversold_dataset_dir = f'{oversold_root}/oversolddata_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}'
    os.makedirs(oversold_dataset_dir, exist_ok=True)
    if len(code) == 6:
        code = f'{code}.SH' if code.startswith('6') else f'{code}.SZ'
    oversold_csv = f'{oversold_dataset_dir}/{code}.csv'
    if not os.path.exists(oversold_csv):
        return
    df_oversold = pd.read_csv(oversold_csv, dtype={'trade_date': str})
    df_oversold['trade_date'] = df_oversold['trade_date'].apply(lambda x: str(x)[:8])
    df_oversold['trade_date'] = df_oversold['trade_date'].apply(lambda x: '' if x == 'nan' else x)  # 去掉nan
    df_oversold['last_daily_date'] = df_oversold['last_daily_date'].apply(lambda x: str(x)[:8])
    df_oversold['last_daily_date'] = df_oversold['last_daily_date'].apply(lambda x: '' if x == 'nan' else x)  # 去掉nan
    df_oversold = df_oversold.drop_duplicates(subset=['trade_date'], keep='last')
    daily_csv = f'{BASICDATA_DIR}/dailydata/{code}.csv'
    if not os.path.exists(daily_csv):
        return
    df_daily = pd.read_csv(daily_csv, dtype={'trade_date': str})
    df_daily['trade_date'] = df_daily['trade_date'].apply(lambda x: str(x)[:8])
    df_daily['trade_date'] = df_daily['trade_date'].apply(lambda x: '' if x == 'nan' else x)  # 去掉nan
    df_daily = df_daily.sort_values(by='trade_date', ascending=True)
    df_daily = df_daily.reset_index(drop=True)
    df_daily = get_qfq_price_DF_by_adj_factor(df_daily)  # 前复权价格序列

    # 检查标签空白行是否已经经过了backword个交易日
    df_oversold = df_oversold.sort_values(by='trade_date', ascending=True)
    df_oversold = df_oversold.reset_index(drop=True)
    for i, row in df_oversold.iterrows():
        if row['max_date_backward'] != '':
            continue
        ts_code = row['code']
        industry = row['industry']
        trade_date = row['trade_date']
        trade_date_list = df_daily['trade_date'].values.tolist()
        trade_date_index = trade_date_list.index(trade_date)
        today = datetime.datetime.now().date().strftime('%Y%m%d')
        if today in trade_date_list:
            today_index = trade_date_list.index(today)
            days = abs(today_index - trade_date_index)
        else:
            today_index = len(trade_date_list) - 1
            days = abs(today_index - trade_date_index) + 1  # 今天尚未收盘,但也要算一天
        if days < BACKWARD_DAYS:  # 还未经过backword个交易日
            continue
        # 计算向后的最大涨幅
        df_backward = df_daily.loc[trade_date_index+1:trade_date_index+BACKWARD_DAYS]  # 向后不包括基准日，因为无法在基准日买入
        max_price_backward = df_backward['high'].max()
        # 获取最大价格的日期
        max_date_backward = df_backward[df_backward['high'] == max_price_backward]['trade_date'].values[0]
        trade_date_close = df_daily.loc[trade_date_index, 'close']
        max_up_rate = round((max_price_backward - trade_date_close) / trade_date_close, 4)
        backward_days = days
        df_oversold.loc[i, 'max_date_backward'] = max_date_backward
        df_oversold.loc[i, 'max_up_rate'] = max_up_rate
        df_oversold.loc[i, 'backward_days'] = backward_days
    # 用df_daily最后一天的日期更新df_oversold最后一行的last_daily_date
    last_daily_date = df_daily.iloc[-1]['trade_date']
    df_oversold.loc[i, 'last_daily_date'] = last_daily_date
    df_oversold = df_oversold.sort_values(by=['trade_date','code'], ascending=True)
    df_oversold = df_oversold.reset_index(drop=True)
    df_oversold.to_csv(oversold_csv, index=False)

# 把oversold_dataset_dir下的所有csv文件合并成一个csv文件
def merge_all_oversold_dataset(forward_days, backward_days, down_filter):
    """
    ### 合并所有的oversold数据集csv文件
    #### :param forward_days: 向前的天数(包含基准日)
    #### :param backward_days: 向后的天数(不包含基准日)
    #### :param down_filter: 最大下跌幅度的过滤条件
    """
    oversold_root = f'{DATASETS_DIR}/oversold'
    oversold_dataset_dir = f'{oversold_root}/oversolddata_{forward_days}_{backward_days}_{-down_filter:.2f}'
    if not os.path.exists(oversold_dataset_dir):
        return
    oversold_temp = f'{TEMP_DIR}/oversold'
    all_oversold_dir = f'{oversold_temp}/data_{forward_days}_{backward_days}_{-down_filter:.2f}'
    os.makedirs(all_oversold_dir, exist_ok=True)
    all_oversold_csv = os.path.join(
        all_oversold_dir, f'all_oversold_data_{forward_days}_{backward_days}_{-down_filter:.2f}.csv'
    )
    df_list = []
    for csv in os.listdir(oversold_dataset_dir):
        try:
            csv_path = os.path.join(oversold_dataset_dir, csv)
            df = pd.read_csv(csv_path, dtype={'trade_date': str})
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue
    df_all = pd.concat(df_list, ignore_index=True)
    # 删除 ts_code name industry 为空的行
    df_all = df_all.dropna(subset=['code', 'name', 'industry'])
    # 对'turnover_rate', 'mv_ratio', 'pe_ttm', 'pb', 'dv_ratio'为空的行以 0 填充
    columns_to_fill = ['turnover_rate', 'mv_ratio', 'pe_ttm', 'pb', 'dv_ratio']
    df_all[columns_to_fill] = df_all[columns_to_fill].fillna(0)
    df_all = df_all.sort_values(by=['trade_date', 'code'], ascending=True)
    df_all = df_all.reset_index(drop=True)
    if os.path.exists(all_oversold_csv):
        os.remove(all_oversold_csv)
    df_all.to_csv(all_oversold_csv, index=False)
    # NOTE: 此处保留了同一下跌过程的全部数据，但是在训练过程中同一个下跌过程保留最后一条数据。
    # 判断是否重复的依据是记录是否具有相同的'code', 'max_date_forward'。

def update_dataset():
    """
    ### 更新数据集
    """
    all_stocks = get_all_stocks_info()
    codes = [stock[0] for stock in all_stocks]
    steps = 5

    trade_cal = pd.read_csv(TRADE_CAL_CSV, dtype={'cal_date': str})
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
