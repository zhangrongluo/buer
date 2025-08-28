"""
download and update daily data and daily indicator data for all stocks
basic_data_alt_edition已替代本模块。但在一次性需要更新多日数据的情况下,本模块仍然是可用的。
"""
import os
import time
import datetime
import pandas as pd
import re
import requests
import tqdm
from io import StringIO
from DrissionPage import ChromiumOptions, Chromium
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

def download_dividend_data_from_sina(ts_code: str):
    """
    get dividend data from sina finance
    :param ts_code: stock code, 600036 or 600036.SH
    :return: None
    NOTE: 
    ts_code should be in the format of 600036.SH or 000036.SZ,
    then 600036.SH -> 600036, 000036.SZ -> 000036
    """
    ts_code = ts_code[:6] if len(ts_code) == 9 else ts_code
    url = f'https://vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/{ts_code}.phtml'
    response = requests.get(url)
    if response.status_code != 200:
        return
    try:
        div_df = pd.read_html(StringIO(response.text))[12]
        if div_df.empty:
            return
        div_df.columns = ['ann_date', 'stk_bo_rate', 'stk_co_rate', 'cash_div_tax', 'div_proc', \
                        'ex_date', 'record_date', 'div_listdate', 'detail']
        div_df = div_df[div_df.columns[:-1]]
        ts_code = ts_code + '.SH' if ts_code[0] == '6' else ts_code + '.SZ'
        div_df.insert(0, 'ts_code', ts_code)
        msg = get_name_and_industry_by_code(ts_code)
        div_df.insert(1, 'name', msg[0])
        div_df.insert(2, 'industry', msg[1])
        div_df['stk_bo_rate'] = div_df['stk_bo_rate'].astype(float) / 10
        div_df['stk_co_rate'] = div_df['stk_co_rate'].astype(float) / 10
        div_df['cash_div_tax'] = div_df['cash_div_tax'].astype(float) / 10
        div_df['stk_div'] = div_df['stk_bo_rate'] + div_df['stk_co_rate']
        div_df['ann_date'] = div_df['ann_date'].apply(lambda x: x.replace('-', ''))
        div_df['ex_date'] = div_df['ex_date'].apply(lambda x: x.replace('-', ''))
        div_df['record_date'] = div_df['record_date'].apply(lambda x: x.replace('-', ''))
        div_df['div_listdate'] = div_df['div_listdate'].apply(lambda x: x.replace('-', ''))
        div_df = div_df[div_df['div_proc'] == '实施']
        columns = ['ts_code', 'name', 'industry', 'ann_date', 'stk_div', 'stk_bo_rate', 'stk_co_rate', \
                'cash_div_tax', 'div_proc', 'ex_date', 'record_date', 'div_listdate']
        div_df = div_df[columns]
        dest_dir = f'{FINANDATA_DIR}/sina_dividend'
        os.makedirs(dest_dir, exist_ok=True)
        div_df.to_csv(f'{dest_dir}/{ts_code}.csv', index=False)
    except Exception as e:
        print(f'download {ts_code} {msg[0]} dividend data from sina Error: {e}')

def download_dividend_data_from_xueqiu(ts_code: str):
    """
    get dividend data from xueqiu
    :param ts_code: stock code, 600036 or 600036.SH
    :return: None
    NOTE:
    ts_code should be in the format of 600036.SH or 000036.SZ,
    then 600036.SH -> SH600036, 000036.SZ -> SZ000036
    """
    ts_code = f'SH{ts_code[:6]}' if ts_code[0] == '6' else f'SZ{ts_code[:6]}'
    url = f'https://xueqiu.com/snowman/S/{ts_code}/detail#/FHPS'
    try:
        co = ChromiumOptions()
        co.headless().no_imgs().no_js()
        browser = Chromium(co)
        tab = browser.latest_tab
        tab.get(url)
        tab.wait.eles_loaded('tag:table@class=table table-bordered table-hover')
        html = tab.ele('tag:table@class=table table-bordered table-hover')
        div_df = pd.read_html(StringIO(html.html))[0]
        if div_df.empty:
            return
        ts_code = ts_code[2:] + '.SH' if ts_code[0:2] == 'SH' else ts_code[2:] + '.SZ'
        div_df.insert(0, 'ts_code', ts_code)
        msg = get_name_and_industry_by_code(ts_code)
        div_df.insert(1, 'name', msg[0])
        div_df.insert(2, 'industry', msg[1])
        div_df['stk_div'] = None
        div_df['stk_bo_rate'] = None
        div_df['stk_co_rate'] = None
        div_df['cash_div_tax'] = None
        div_df['div_proc'] = None
        zg_pattern = re.compile(r'转(\d+\.?\d*)')
        sg_pattern = re.compile(r'送(\d+\.?\d*)')
        px_pattern = re.compile(r'派(\d+\.?\d*)')
        for idx, row in div_df.iterrows():
            try:
                div_plan = row['方案']
                zg = float(zg_pattern.search(div_plan).group(1)) if zg_pattern.search(div_plan) else None
                sg = float(sg_pattern.search(div_plan).group(1)) if sg_pattern.search(div_plan) else None
                px = float(px_pattern.search(div_plan).group(1)) if px_pattern.search(div_plan) else None
                div_df.at[idx, 'stk_co_rate'] = zg/10 if zg else 0
                div_df.at[idx, 'stk_bo_rate'] = sg/10 if sg else 0
                div_df.at[idx, 'cash_div_tax'] = px/10 if px else 0
                div_df.at[idx, 'stk_div'] = div_df.at[idx, 'stk_co_rate'] + div_df.at[idx, 'stk_bo_rate']
                div_df.at[idx, 'div_proc'] = '实施' if '实施' in div_plan else None
            except Exception as e:
                print(f'Error processing row {idx} in dividend data: {e}')
        div_df.columns = ['ts_code', 'name', 'industry', 'report', 'div_plan', 'reg_date', 'ex_date', 'pay_date', \
                        'stk_div', 'stk_bo_rate', 'stk_co_rate', 'cash_div_tax', 'div_proc']
        columns = ['ts_code', 'name', 'industry', 'report', 'div_plan', 'stk_div', 'stk_bo_rate', 'stk_co_rate', \
                'cash_div_tax', 'div_proc', 'ex_date', 'pay_date']
        div_df = div_df[columns]
        div_df = div_df[div_df['div_proc'] == '实施']
        div_df['ex_date'] = div_df['ex_date'].apply(lambda x: x.replace('-', ''))
        div_df['pay_date'] = div_df['pay_date'].apply(lambda x: x.replace('-', ''))
        dest_dir = f'{FINANDATA_DIR}/xueqiu_dividend'
        os.makedirs(dest_dir, exist_ok=True)
        div_df.to_csv(f'{dest_dir}/{ts_code}.csv', index=False)
        browser.quit()
    except Exception as e:
        print(f'download {ts_code} {msg[0]} dividend data from xueqiu Error: {e}')

def download_dividend_data_in_multi_ways(ts_code: str) -> str | None:
    """
    下载分红数据
    :param ts_code: 股票代码, 如 000001 或 000001.SZ
    :return: 下载的文件路径(成功)或者 None(失败)
    NOTE: 
    如果未能从 tushare 下载到数据, 则继续从 sina 和 xueqiu 下载数据
    """
    if len(ts_code) == 6:
        ts_code = ts_code + '.SH' if ts_code[0] == '6' else ts_code + '.SZ'
    try:
        download_dividend_data(ts_code=ts_code)
        dividend_csv = f'{FINANDATA_DIR}/dividend/{ts_code}.csv'
        if not os.path.exists(dividend_csv):
            today = datetime.datetime.now().strftime('%Y%m%d')
            dividend_csv = f'{FINANDATA_DIR}/sina_dividend/{ts_code}.csv'
            if os.path.exists(dividend_csv):
                mtime = os.path.getmtime(dividend_csv)
                mtime = time.strftime('%Y%m%d', time.localtime(mtime))
                if mtime != today:
                    download_dividend_data_from_sina(ts_code=ts_code)
            else:
                download_dividend_data_from_sina(ts_code=ts_code)
            try_xueqiu_again = (os.path.exists(dividend_csv) and \
                time.strftime('%Y%m%d', time.localtime(os.path.getmtime(dividend_csv))) != today)
            if not os.path.exists(dividend_csv) or try_xueqiu_again:
                dividend_csv = f'{FINANDATA_DIR}/xueqiu_dividend/{ts_code}.csv'
                if os.path.exists(dividend_csv):
                    mtime = os.path.getmtime(dividend_csv)
                    mtime = time.strftime('%Y%m%d', time.localtime(mtime))
                    if mtime != today:
                        download_dividend_data_from_xueqiu(ts_code=ts_code)
                else:
                    download_dividend_data_from_xueqiu(ts_code=ts_code)
        return dividend_csv
    except Exception as e:
        print(f'download {ts_code} dividend data Error: {e}')
        return None

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
