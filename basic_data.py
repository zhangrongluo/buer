"""
download and update daily data and daily indicator data for all stocks
"""
import os
import datetime
import pandas as pd
import tqdm
from concurrent.futures import ThreadPoolExecutor
from stocklist import get_name_and_industry_by_code, get_all_stocks_info, pro
from cons_general import BASICDATA_DIR, FINANDATA_DIR

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)  # pandas concat warning

all_stocks_info = get_all_stocks_info()

def download_daily_data(ts_code: str):
    """
    download daily data from tushare
    :param ts_code: stock code
    :return: None
    """
    if len(ts_code) == 6:
        ts_code = ts_code + '.SH' if ts_code[0] == '6' else ts_code + '.SZ'
    daily_df = pro.daily(ts_code=ts_code)
    msg = get_name_and_industry_by_code(ts_code)
    daily_df.insert(1, 'name', msg[0])
    daily_df.insert(2, 'industry', msg[1])
    daily_df = daily_df.sort_values(by='trade_date', ascending=False)
    dest_dir = f'{BASICDATA_DIR}/dailydata'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    daily_df.to_csv(f'{dest_dir}/{ts_code}.csv', index=False)

def update_daily_data(ts_code: str):
    """
    update daily data for all stocks
    :param ts_code: stock code, 600036 or 600036.SH
    :return: None
    """
    if len(ts_code) == 6:
        ts_code = ts_code + '.SH' if ts_code[0] == '6' else ts_code + '.SZ'
    dest_dir = f'{BASICDATA_DIR}/dailydata'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    dest_csv = f'{dest_dir}/{ts_code}.csv'
    if not os.path.exists(dest_csv):
        download_daily_data(ts_code)
        return
    # update stock daily data
    df = pd.read_csv(dest_csv, dtype={'trade_date': str})
    df = df.sort_values(by='trade_date', ascending=False)
    df.reset_index(drop=True, inplace=True)  # 重置索引
    last_trade_date = df.iloc[0]['trade_date']
    today = datetime.datetime.now().strftime('%Y%m%d')
    if today == last_trade_date:
        return
    df_new = pro.daily(ts_code=ts_code, start_date=last_trade_date, end_date=today)
    if df_new.empty:
        return
    msg = get_name_and_industry_by_code(ts_code)
    df_new.insert(1, 'name', msg[0])
    df_new.insert(2, 'industry', msg[1])
    df = pd.concat([df, df_new], ignore_index=True)
    df = df.sort_values(by='trade_date', ascending=False)
    df = df.drop_duplicates(subset='trade_date', keep='first')
    df.to_csv(dest_csv, index=False)

def download_daily_indicator(ts_code: str):
    """
    download daily indicator data from tushare
    :param ts_code: stock code, 600036 or 600036.SH
    :return: None
    """
    if len(ts_code) == 6:
        ts_code = ts_code + '.SH' if ts_code[0] == '6' else ts_code + '.SZ'
    daily_indicator_df = pro.daily_basic(ts_code=ts_code)
    msg = get_name_and_industry_by_code(ts_code)
    daily_indicator_df.insert(1, 'name', msg[0])
    daily_indicator_df.insert(2, 'industry', msg[1])
    daily_indicator_df = daily_indicator_df.sort_values(by='trade_date', ascending=False)
    dest_dir = f'{BASICDATA_DIR}/dailyindicator'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    daily_indicator_df.to_csv(f'{dest_dir}/{ts_code}.csv', index=False)

def update_daily_indicator(ts_code: str):
    """
    update daily indicator data for all stocks
    :param ts_code: stock code, 600036 or 600036.SH
    :return: None
    """
    if len(ts_code) == 6:
        ts_code = ts_code + '.SH' if ts_code[0] == '6' else ts_code + '.SZ'
    dest_dir = f'{BASICDATA_DIR}/dailyindicator'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    dest_csv = f'{dest_dir}/{ts_code}.csv'
    if not os.path.exists(dest_csv):
        download_daily_indicator(ts_code)
        return
    # update stock daily indicator data
    df = pd.read_csv(dest_csv, dtype={'trade_date': str})
    df = df.sort_values(by='trade_date', ascending=False)
    df.reset_index(drop=True, inplace=True)  # 重置索引
    last_trade_date = df.iloc[0]['trade_date']
    today = datetime.datetime.now().strftime('%Y%m%d')
    if today == last_trade_date:
        return
    df_new = pro.daily_basic(ts_code=ts_code, start_date=last_trade_date, end_date=today)
    if df_new.empty or df_new.isna().all().all():
        return
    msg = get_name_and_industry_by_code(ts_code)
    df_new.insert(1, 'name', msg[0])
    df_new.insert(2, 'industry', msg[1])
    df = pd.concat([df, df_new], ignore_index=True)
    df = df.sort_values(by='trade_date', ascending=False)
    df = df.drop_duplicates(subset='trade_date', keep='first')
    df.to_csv(dest_csv, index=False)

def update_all_daily_data(step: int = 5):
    """
    update daily data for all stocks
    :param step: number of stocks to update at a time
    :return: None
    """
    dest_dir = f'{BASICDATA_DIR}/dailydata'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    all_codes = [item[0] for item in all_stocks_info]
    bar = tqdm.tqdm(total=len(all_codes), desc='更新每日行情数据', unit='stock', ncols=100)
    for i in range(0, len(all_codes), step):
        with ThreadPoolExecutor() as executor:
            executor.map(update_daily_data, all_codes[i:i + step])
        bar.update(step)
    bar.close()

def update_all_daily_indicator(step: int = 5):
    """
    update daily indicator data for all stocks
    :param step: number of stocks to update at a time
    :return: None
    """
    dest_dir = f'{BASICDATA_DIR}/dailyindicator'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    all_codes = [item[0] for item in all_stocks_info]
    bar = tqdm.tqdm(total=len(all_codes), desc='更新每日指标数据', unit='stock', ncols=100)
    for i in range(0, len(all_codes), step):
        with ThreadPoolExecutor() as executor:
            executor.map(update_daily_indicator, all_codes[i:i + step])
        bar.update(step)
    bar.close()

def download_dividend_data(ts_code: str):
    """
    download dividend data from tushare
    :param ts_code: stock code, 600036 or 600036.SH
    :return: None
    """
    if len(ts_code) == 6:
        ts_code = ts_code + '.SH' if ts_code[0] == '6' else ts_code + '.SZ'
    dividend_df = pro.dividend(ts_code=ts_code)
    msg = get_name_and_industry_by_code(ts_code)
    dividend_df.insert(1, 'name', msg[0])
    dividend_df.insert(2, 'industry', msg[1])
    dest_dir = f'{FINANDATA_DIR}/dividend'
    os.makedirs(dest_dir, exist_ok=True)
    dividend_df.to_csv(f'{dest_dir}/{ts_code}.csv', index=False)

def download_all_dividend_data(step: int = 5):
    """
    download dividend data for all stocks
    :param step: number of stocks to update one time
    :return: None
    """
    dest_dir = f'{FINANDATA_DIR}/dividend'
    os.makedirs(dest_dir, exist_ok=True)
    all_codes = [item[0] for item in all_stocks_info]
    bar = tqdm.tqdm(total=len(all_codes), desc='下载分红派息数据', unit='stock', ncols=100)
    for i in range(0, len(all_codes), step):
        with ThreadPoolExecutor() as executor:
            executor.map(download_dividend_data, all_codes[i:i + step])
        bar.update(step)
    bar.close()

def download_adj_factor_data(ts_code: str):
    """
    download adj factor data from tushare
    :param ts_code: stock code, 600036 or 600036.SH
    :return: None
    """
    if len(ts_code) == 6:
        ts_code = ts_code + '.SH' if ts_code[0] == '6' else ts_code + '.SZ'
    adj_factor_df = pro.adj_factor(ts_code=ts_code)
    msg = get_name_and_industry_by_code(ts_code)
    adj_factor_df.insert(1, 'name', msg[0])
    adj_factor_df.insert(2, 'industry', msg[1])
    dest_dir = f'{BASICDATA_DIR}/adjfactor'
    os.makedirs(dest_dir, exist_ok=True)
    adj_factor_df.to_csv(f'{dest_dir}/{ts_code}.csv', index=False)

def update_adj_factor_data(ts_code: str):
    """
    update or download(if not) adj factor data from last trade date to today
    :param ts_code: stock code, 600036 or 600036.SH
    :return: None
    """
    if len(ts_code) == 6:
        ts_code = ts_code + '.SH' if ts_code[0] == '6' else ts_code + '.SZ'
    dest_dir = f'{BASICDATA_DIR}/adjfactor'
    os.makedirs(dest_dir, exist_ok=True)
    dest_csv = f'{dest_dir}/{ts_code}.csv'
    if not os.path.exists(dest_csv):
        download_adj_factor_data(ts_code)
        return
    # update stock adj factor data
    df = pd.read_csv(dest_csv, dtype={'trade_date': str})
    df = df.sort_values(by='trade_date', ascending=False)
    df.reset_index(drop=True, inplace=True)  # 重置索引
    last_trade_date = df.iloc[0]['trade_date']
    today = datetime.datetime.now().strftime('%Y%m%d')
    if today == last_trade_date:
        return
    df_new = pro.adj_factor(ts_code=ts_code, start_date=last_trade_date, end_date=today)
    if df_new.empty or df_new.isna().all().all():
        return
    msg = get_name_and_industry_by_code(ts_code)
    df_new.insert(1, 'name', msg[0])
    df_new.insert(2, 'industry', msg[1])
    df = pd.concat([df, df_new], ignore_index=True)
    df = df.sort_values(by='trade_date', ascending=False)
    df = df.drop_duplicates(subset='trade_date', keep='first')
    df.to_csv(dest_csv, index=False)

def update_all_adj_factor_data(step: int = 5):
    """
    update or download(if not) adj factor data for all stocks
    :param step: number of stocks to update at a time
    :return: None
    """
    dest_dir = f'{BASICDATA_DIR}/adjfactor'
    os.makedirs(dest_dir, exist_ok=True)
    all_codes = [item[0] for item in all_stocks_info]
    bar = tqdm.tqdm(total=len(all_codes), desc='更新复权因子数据', unit='stock', ncols=100)
    for i in range(0, len(all_codes), step):
        with ThreadPoolExecutor() as executor:
            executor.map(update_adj_factor_data, all_codes[i:i + step])
        bar.update(step)
    bar.close()

# 遍历trade-record目录下的所有csv文件,获取最新的trade_date==today的数量
def get_indicator_date_equal_today_nums():
    total_nums = 0
    equal_nums = 0
    date_list = []
    indicator_dir = f'{BASICDATA_DIR}/dailyindicator'
    files = os.listdir(indicator_dir)
    for file in files:
        try:
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(indicator_dir, file), dtype={'trade_date': str})
                trade_date = df.iloc[0]['trade_date']
                date_list.append(trade_date)
                today = datetime.datetime.now().strftime('%Y%m%d')
                total_nums += 1
                if trade_date == today:
                    equal_nums += 1
        except Exception as e:
            pass
    last_trade_date = max(date_list)
    last_trade_date_count = date_list.count(last_trade_date)
    return equal_nums, total_nums, last_trade_date, last_trade_date_count

# 遍历daily_data目录下的所有csv文件,获取最新的trade_date==today的数量
def get_daily_data_equal_today_nums():
    total_nums = 0
    equal_nums = 0
    date_list = []
    daily_dir = f'{BASICDATA_DIR}/dailydata'
    files = os.listdir(daily_dir)
    for file in files:
        try:
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(daily_dir, file), dtype={'trade_date': str})
                trade_date = df.iloc[0]['trade_date']
                date_list.append(trade_date)
                today = datetime.datetime.now().strftime('%Y%m%d')
                total_nums += 1
                if trade_date == today:
                    equal_nums += 1
        except Exception as e:
            pass
    last_trade_date = max(date_list)
    last_trade_date_count = date_list.count(last_trade_date)
    return equal_nums, total_nums, last_trade_date, last_trade_date_count


if __name__ == '__main__':
    update_all_daily_data()
    update_all_daily_indicator()
    download_all_dividend_data()
