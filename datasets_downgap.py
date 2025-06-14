"""
calculate、refresh and merge Downgap datasets
"""
import os
import pandas as pd
import datetime
import tushare as ts
from cons_general import BASICDATA_DIR, DATASETS_DIR, TEMP_DIR
from stocklist import get_name_and_industry_by_code
from utils import get_qfq_price_DF_by_adj_factor


daily_root = os.path.join(BASICDATA_DIR, 'dailydata')
os.makedirs(daily_root, exist_ok=True)
gap_root = os.path.join(DATASETS_DIR, 'downgap')
os.makedirs(gap_root, exist_ok=True)
temp_root = os.path.join(TEMP_DIR, 'downgap')
os.makedirs(temp_root, exist_ok=True)

def get_gaps_statistic_data(code: str):
    """ 
    获取股票和指数缺口及回补情况
    :param code: 股票代码, 例如: '600000' or '000001.SH'
    """
    if len(code) == 6:
        code = f'{code}.SH' if code.startswith('6') else f'{code}.SZ'
    # industry = tssw.get_name_and_class_by_code(code=code[:6])[1]
    daily_data_csv = os.path.join(f'{daily_root}', f'{code}.csv')
    if not os.path.exists(daily_data_csv):
        return None
    # prepare raw data
    df = pd.read_csv(daily_data_csv, dtype={'trade_date': str})
    trade_dates = df['trade_date'].unique().tolist()  # to calculate days between trade_date and fill_date
    gap_csv = os.path.join(f'{gap_root}', f'{code}.csv')
    if os.path.exists(gap_csv):
        df_gap = pd.read_csv(gap_csv, dtype={'trade_date': str})
        df_gap = df_gap.sort_values(by='trade_date', ascending=True)
        df_gap.reset_index(drop=True, inplace=True)  # 重置索引
        last_row_trade_date = df_gap.iloc[-1]['trade_date']
        last_row_trade_date = ''.join(last_row_trade_date.split('-'))
        last_trade_date = datetime.datetime.strptime(last_row_trade_date, '%Y%m%d') + \
            datetime.timedelta(days=-30)  # 向后推移30天,保留RSI计算所需数据
        last_trade_date = last_trade_date.strftime('%Y%m%d')
        df = df[df['trade_date'] > last_trade_date]
    # preprocess data
    df = get_qfq_price_DF_by_adj_factor(df)  # 获取前复权数据
    df = df.sort_values(by='trade_date', ascending=True)
    df.reset_index(drop=True, inplace=True)  # 重置索引
    df['pre_high'] = df['high'].shift(1)
    df['pre_low'] = df['low'].shift(1)
    # if high<pre_low, down, if low>pre_high, up
    df['gap'] = None
    df.loc[df['high'] < df['pre_low'], 'gap'] = 'down'
    df.loc[df['low'] > df['pre_high'], 'gap'] = 'up'
    df['vol_ratio'] = round(df['vol'] / df['vol'].shift(1), 4)  # 当日和前一日交易量比例
    df['gap_percent'] = None      # 计算添加gap_percent列
    df.loc[df['gap'] == 'down', 'gap_percent'] = round((df['high'] - df['pre_low']) / df['pre_low'], 4)
    df.loc[df['gap'] == 'up', 'gap_percent'] = round((df['low'] - df['pre_high']) / df['pre_high'], 4)
    # 使用前14天pct_chg 数据计算RSI = 100-100/(1+RS)
    period = 14
    df['RSI14'] = None
    for i in range(len(df)):
        if i < period:
            continue
        up = df.loc[i-period:i, 'pct_chg'].apply(lambda x: x if x > 0 else 0).sum()
        down = abs(df.loc[i-period:i, 'pct_chg'].apply(lambda x: x if x < 0 else 0).sum())
        rs = up /(down + 1e-10) if down == 0 else up / down
        rsi = 100 - 100 / (1 + rs)
        df.loc[i, 'RSI14'] = round(rsi, 2)
    # 使用前7天pct_chg 数据计算RSI = 100-100/(1+RS)
    period = 7
    df['RSI7'] = None
    for i in range(len(df)):
        if i < period:
            continue
        up = df.loc[i-period:i, 'pct_chg'].apply(lambda x: x if x > 0 else 0).sum()
        down = abs(df.loc[i-period:i, 'pct_chg'].apply(lambda x: x if x < 0 else 0).sum())
        rs = up /(down + 1e-10) if down == 0 else up / down
        rsi = 100 - 100 / (1 + rs)
        df.loc[i, 'RSI7'] = round(rsi, 2)
    # 使用前3天pct_chg 数据计算RSI = 100-100/(1+RS)
    period = 3
    df['RSI3'] = None
    for i in range(len(df)):
        if i < period:
            continue
        up = df.loc[i-period:i, 'pct_chg'].apply(lambda x: x if x > 0 else 0).sum()
        down = abs(df.loc[i-period:i, 'pct_chg'].apply(lambda x: x if x < 0 else 0).sum())
        rs = up /(down + 1e-10) if down == 0 else up / down
        rsi = 100 - 100 / (1 + rs)
        df.loc[i, 'RSI3'] = round(rsi, 2)
    # 使用前14天价格计算K指标=100*(close-min)/(max-min)
    period = 14
    df['K'] = None
    for i in range(len(df)):
        if i < period:
            continue
        close = df.loc[i, 'close']
        high = df.loc[i-period:i, 'high'].max()
        low = df.loc[i-period:i, 'low'].min()
        k = 100 * (close - low) / (high - low)
        df.loc[i, 'K'] = round(k, 2)
    # 使用前14天价格计算MAP指标=sum(close)/14
    period = 14
    df['MAP14'] = None
    for i in range(len(df)):
        if i < period:
            continue
        map14 = df.loc[i-period:i, 'close'].sum() / period
        df.loc[i, 'MAP14'] = round(map14, 2)
    # 使用前7天价格计算MAP指标=sum(close)/7
    period = 7
    df['MAP7'] = None
    for i in range(len(df)):
        if i < period:
            continue
        map7 = df.loc[i-period:i, 'close'].sum() / period
        df.loc[i, 'MAP7'] = round(map7, 2)
    # 遍历df每一行,获取回补的日期,计算回补期间的涨幅
    # 如果某行的gap为down,记录该行的pre_low值为low0, 然后从该行下一行开始查找
    # high>low0的行， 记录该行trade_date日期为fill_date
    # 如果某行的gap为up,记录该行的pre_high值为high0, 然后从该行下一行开始查找
    # low<high0的行， 记录该行trade_date日期为fill_date
    df['fill_date'] = ''  # 避免日期型数据转换为float, 先设置为空字符串
    df['rise_percent'] = None
    for i, row in df.iterrows():
        if row['gap'] == 'down':
            low0 = row['pre_low']
            date0 = row['trade_date']
            tmp_df = df[df['trade_date'] > date0]
            tmp_df = tmp_df[tmp_df['high'] > low0]
            if not tmp_df.empty:
                tmp_df.sort_values(by='trade_date', ascending=True, inplace=True)
                df.loc[i, 'fill_date'] = tmp_df.iloc[0]['trade_date']
                fill_date = tmp_df.iloc[0]['trade_date']
                # 计算回补的涨幅: 获取date0到fill_date之间的交易数据df1
                # 计算df1中low的最小值low1到date0的date0_pre_low值的涨幅
                # NOTE 按照计算过程来看，rise_percent的结果是理想的状态，实际要打个折扣。
                df1 = df[(df['trade_date'] >= date0) & (df['trade_date'] <= fill_date)]
                low1 = df1['low'].min()
                rise_percent = round((low0 - low1) / low1, 4)
                df.loc[i, 'rise_percent'] = rise_percent
        elif row['gap'] == 'up':
            high0 = row['pre_high']
            date0 = row['trade_date']
            tmp_df = df[df['trade_date'] > date0]
            tmp_df = tmp_df[tmp_df['low'] < high0]
            if not tmp_df.empty:
                tmp_df.sort_values(by='trade_date', ascending=True, inplace=True)
                df.loc[i, 'fill_date'] = tmp_df.iloc[0]['trade_date']
                fill_date = tmp_df.iloc[0]['trade_date']
                # 计算回补的涨幅: 获取date0到fill_date之间的交易数据df1
                # 计算date0close到df1的high的最大值high1的涨幅
                # TODO: up gap rise_percent need to optimize
                df1 = df[(df['trade_date'] >= date0) & (df['trade_date'] <= fill_date)]
                high1 = df1['high'].max()
                close1 = df1[df1['trade_date'] == date0]['close'].values[0]
                rise_percent = round((high1 - close1) / close1, 4)
                df.loc[i, 'rise_percent'] = rise_percent
    # calculate days between trade_date and fill_date
    df.dropna(subset=['gap'], inplace=True)
    df['days'] = None
    for i, row in df.iterrows():
        try:
            trade_date = row['trade_date']
            trade_date_index = trade_dates.index(trade_date)
            fill_date = row['fill_date']
            if fill_date is not None:
                fill_date_index = trade_dates.index(fill_date)
                days = abs(trade_date_index - fill_date_index)
                df.loc[i, 'days'] = days
        except Exception as e:
            continue
    # open trade record and insert turnover_rate mv_ratio ... to df
    trade_file = os.path.join(BASICDATA_DIR, 'dailyindicator', f"{code}.csv")
    if not os.path.exists(trade_file):
        return None
    df_trade = pd.read_csv(trade_file, dtype={'trade_date': str})
    df_trade['mv_ratio'] = round(df_trade['circ_mv']/df_trade['total_mv'], 4)
    df_trade = df_trade[['trade_date', 'turnover_rate', 'mv_ratio', 'pe_ttm', 'pb', 'dv_ratio']]
    df = pd.merge(df, df_trade, on='trade_date', how='left')
    # res = tssw.get_name_and_class_by_code(code[:6])
    msg = get_name_and_industry_by_code(code)
    df['stock_name'] = msg[0]
    df['industry'] = msg[1]
    df = df[['ts_code', 'stock_name', 'industry', 'open', 'high', 'low', 'close', 'pre_low', 'pre_high',
            'gap', 'gap_percent', 'vol_ratio', 'pct_chg', 'RSI14', 'RSI7', 'RSI3', 'K', 'MAP14','MAP7',
            'turnover_rate', 'mv_ratio', 'pe_ttm', 'pb', 'dv_ratio', 'trade_date', 'fill_date', 'days', 
            'rise_percent']]
    # concat df to gapt_csv
    gap_csv = os.path.join(gap_root, f'{code}.csv')
    if os.path.exists(gap_csv):
        gap_df = pd.read_csv(gap_csv, dtype={'trade_date': str, 'fill_date': str})
        df = pd.concat([gap_df, df])
        df = df.drop_duplicates(subset='trade_date', keep='first')
        df = df.sort_values(by='trade_date', ascending=True)
        df.to_csv(gap_csv, index=False)
    else:
        df.to_csv(gap_csv, index=False)

def refresh_the_gap_csv(code: str):
    """
    replace gap csv data  
    :param code: stock code like 600000 or 600000.SH
    NOTE: delete duplicate data and check gap filled or not
    """
    if len(code) == 6:
        code = f'{code}.SH' if code.startswith('6') else f'{code}.SZ'
    gap_csv = f'{gap_root}/{code}.csv'
    daily_csv = f'{daily_root}/{code}.csv'
    if not os.path.exists(gap_csv):
        return None
    gap_df = pd.read_csv(gap_csv, dtype={'trade_date': str, 'fill_date': str})
    gap_df['fill_date'] = gap_df['fill_date'].apply(lambda x: str(x)[:8])
    gap_df['fill_date'] = gap_df['fill_date'].apply(lambda x: '' if x == 'nan' else x)
    gap_df = gap_df.drop_duplicates(subset='trade_date', keep='last')
    gap_df = gap_df.sort_values(by='trade_date', ascending=True)
    daily_df = pd.read_csv(daily_csv, dtype={'trade_date': str})
    daily_df = get_qfq_price_DF_by_adj_factor(daily_df)  # 获取前复权数据
    daily_df = daily_df.sort_values(by='trade_date', ascending=True)
    daily_df['pre_low'] = daily_df['low'].shift(1)
    daily_df['pre_high'] = daily_df['high'].shift(1)
    # check the gap filled or not, update fill_date days rise_percent
    for i, row in gap_df.iterrows():
        if row['fill_date'] != '':
            continue
        if row['gap'] == 'down':
            low0 = row['pre_low']
            date0 = row['trade_date']
            tmp_daily_df = daily_df[daily_df['trade_date'] > date0]
            tmp_daily_df = tmp_daily_df[tmp_daily_df['high'] > low0]
            if not tmp_daily_df.empty:
                tmp_daily_df.sort_values(by='trade_date', ascending=True, inplace=True)
                fill_date = tmp_daily_df.iloc[0]['trade_date']
                gap_df.loc[i, 'fill_date'] = fill_date
                # 计算回补的涨幅: 获取date0到fill_date之间的交易数据df1
                # 计算df1中low的最小值low1到date0的date0_pre_low值的涨幅
                # NOTE 按照计算过程来看，rise_percent的结果是理想的状态，实际要打个折扣。
                df1 = daily_df[(daily_df['trade_date'] >= date0) & (daily_df['trade_date'] <= fill_date)]
                low1 = df1['low'].min()
                rise_percent = round((low0 - low1) / low1, 4)
                gap_df.loc[i, 'rise_percent'] = rise_percent
        elif row['gap'] == 'up':
            high0 = row['pre_high']
            date0 = row['trade_date']
            tmp_daily_df = daily_df[daily_df['trade_date'] > date0]
            tmp_daily_df = tmp_daily_df[tmp_daily_df['low'] < high0]
            if not tmp_daily_df.empty:
                tmp_daily_df.sort_values(by='trade_date', ascending=True, inplace=True)
                gap_df.loc[i, 'fill_date'] = tmp_daily_df.iloc[0]['trade_date']
                fill_date = tmp_daily_df.iloc[0]['trade_date']
                # 计算回补的涨幅: 获取date0到fill_date之间的交易数据df1
                # 计算date0close到df1的high的最大值high1的涨幅
                # TODO: up gap rise_percent need to optimize
                df1 = daily_df[(daily_df['trade_date'] >= date0) & (daily_df['trade_date'] <= fill_date)]
                high1 = df1['high'].max()
                close1 = df1[df1['trade_date'] == date0]['close'].values[0]
                rise_percent = round((high1 - close1) / close1, 4)
                gap_df.loc[i, 'rise_percent'] = rise_percent
    # calculate days between trade_date and fill_date
    for i, row in gap_df.iterrows():
        try:
            trade_date = row['trade_date']
            trade_date_index = daily_df[daily_df['trade_date'] == trade_date].index[0]
            fill_date = row['fill_date']
            if fill_date is not None:
                fill_date_index = daily_df[daily_df['trade_date'] == fill_date].index[0]
                days = abs(trade_date_index - fill_date_index)
                gap_df.loc[i, 'days'] = days
        except Exception as e:
            continue
    gap_df.to_csv(gap_csv, index=False)

# 把gap_data目录下的所有csv文件合并到一个csv文件
def merge_all_gap_data(max_trade_days: int):
    dest_dir = f'{temp_root}/max_trade_days_{max_trade_days}'
    os.makedirs(dest_dir, exist_ok=True)
    all_gaps_csv = os.path.join(f'{dest_dir}', 'all_gap_data.csv')
    if os.path.exists(all_gaps_csv):
        os.remove(all_gaps_csv)
    df_list = []
    for root, dirs, files in os.walk(gap_root):
        for file in files:
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(root, file), dtype={'trade_date': str, 'fill_date': str})
                df_list.append(df)
    if df_list:
        df = pd.concat(df_list)
        df = df.sort_values(by=['trade_date', 'ts_code'], ascending=True)
        df = df[df['ts_code'].str.contains('|'.join(['.SH', '.SZ']))]
        five_columns = ['turnover_rate','mv_ratio', 'pe_ttm', 'pb', 'dv_ratio']
        df[five_columns] = df[five_columns].fillna(0) 
        df.to_csv(all_gaps_csv, index=False)
