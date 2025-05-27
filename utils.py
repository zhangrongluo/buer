import os
import re
import time
import pandas as pd
import datetime
import requests
import tushare as ts
from DrissionPage import ChromiumOptions, Chromium
from cons_general import TRADE_CAL_XLS, FINANDATA_DIR
from cons_oversold import PAUSE


# send wechat message
def send_wechat_message_via_bark(device_key, title, message):
    """
    send wechat message via bark
    :param device_key: bark device key
    :param title: message title
    :param message: message content
    :return: response
    """
    url = f"https://api.day.app/{device_key}/{title}/{message}"
    response = requests.get(url).json()
    return response

def get_stock_price_from_sina(code: str) -> float | None:
    """ 
    get stock realtime price from sina finance
    :param code: stock code, like 000001 or 000001.SH
    :return: stock price
    NOTE:
    get stock price by chromium, very slow, last way.
    """
    try:
        code = code[:6]
        code = 'sh' + code if code.startswith('6') else 'sz' + code
        co = ChromiumOptions().headless().auto_port().no_imgs().no_js()
        page = Chromium(co)
        tab = page.latest_tab
        url = f'https://finance.sina.com.cn/realstock/company/{code}/nc.shtml'  # 实时页面
        tab.get(url)
        p = tab.ele('tag:div@id=price').text.strip()
        p = float(p)
        page.quit()
        return p
    except Exception as e:
        return None

def get_stock_price_from_tencent(code: str) -> float | None:
    """
    get stock realtime price from tencent finance
    :param code: stock code, like 000001 or 000001.SH
    :return: realtime stock price
    NOTE: written by Grok
    """
    stock_code = code[:6]
    stock_code = 'sh'+stock_code if code[0] == '6' else 'sz'+stock_code
    try:
        url = f"https://qt.gtimg.cn/q={stock_code}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()  # 检查请求是否成功
        # 解析数据腾讯财经返回的数据,格式:v_sh600036="1~招商银行~600036~40.74...";
        data = response.text
        if not data or "v_" not in data:
            return None
        data_content = data.split("=")[1]
        data_list = data_content.split("~")
        price = data_list[3]  # realtime price
        price = float(price)
        return price
    except Exception as e:
        return None
    
def get_stock_realtime_price(code: str) -> float | None:
    """
    采用多重模式获取股票实时交易价格
    """
    if len(code) == 6:
        code = code + '.SH' if code.startswith('6') else code + '.SZ'
    df_Price = ts.realtime_quote(ts_code=code, src='sina')
    price_now = df_Price['PRICE'][0] if not df_Price.empty else None
    if price_now is None:
        df_price = ts.realtime_quote(ts_code=code, src='dc')
        price_now = df_price['price'][0] if not df_price.empty else None
    if price_now is None:
        price_now = get_stock_price_from_tencent(code=code)
    if price_now is None:
        price_now = get_stock_price_from_sina(code=code)
    else:
        time.sleep(PAUSE)
    return price_now

def get_pre_XD_XR_DR_price_df(src_data: pd.DataFrame) -> pd.DataFrame:
    """
    将价格恢复到除权除息前的水平
    :param src_data: basicdata/dailydata下日行情数据(按trade_date升序排列)
    :return: src_data(open、high、low、close)除权除息前的价格序列(按trade_date升序排列)
    NOTE: 
    pre_price = price * (1 + div_per_stock) + cash_divdend_per_stock
    div_per_stock: 每股转送股数
    cash_dividend_per_stock: 每股派息税前金额
    如有多次除权除息，按实施时间顺序从后到前依次计算 (向前复权)
    """
    scr_data_cp = src_data.copy()
    ts_code = scr_data_cp['ts_code'].iloc[0]
    dividend_csv = f'{FINANDATA_DIR}/dividend/{ts_code}.csv'
    if not os.path.exists(dividend_csv):
        return scr_data_cp
    dividend_df = pd.read_csv(dividend_csv, dtype={'ex_date': str})
    if dividend_df.empty:
        return scr_data_cp
    columns = ['ts_code', 'name', 'industry', 'stk_div', 'cash_div_tax', 'ex_date']
    dividend_df = dividend_df[columns]
    dividend_df = dividend_df.dropna(subset=['ex_date'])
    dividend_df = dividend_df.sort_values(by='ex_date', ascending=False)  # 降序排列
    dividend_df.reset_index(drop=True, inplace=True)  # 重置索引
    # 遍历dividend_df，如果ex_date在scr_data_cp的trade_date中，
    # 计算除权除息前价格(from ex_date to scr_data_cp's last trade_date)
    for _, row in dividend_df.iterrows():
        ex_date = row['ex_date']
        if ex_date not in scr_data_cp['trade_date'].values:
            continue
        idx = scr_data_cp[scr_data_cp['trade_date'] == ex_date].index[0]
        div_per_stock = row['stk_div'] if pd.notna(row['stk_div']) else 0
        cash_dividend_per_stock = row['cash_div_tax'] if pd.notna(row['cash_div_tax']) else 0
        scr_data_cp.loc[idx:, ['open', 'high', 'low', 'close']] *= (1 + div_per_stock)
        scr_data_cp.loc[idx:, ['open', 'high', 'low', 'close']] += cash_dividend_per_stock
    return scr_data_cp

def get_pre_XD_XR_DR_price(code: str, price_now: float, start: float, end: str=None) -> float:
    """
    将实时价格price_now恢复到start日除权除息前的价格
    :param code: 股票代码, 如 000001 或 000001.SZ
    :param price_now: 当前价格
    :param start: 起始日期(YYYYMMDD)
    :param end: 结束日期(YYYYMMDD), None表示到今天
    :return: 除权除息前的价格
    NOTE: 
    pre_price = price * (1 + div_per_stock) + cash_divdend_per_stock
    div_per_stock: 每股转送股数
    cash_dividend_per_stock: 每股派息税前金额
    如有多次除权除息，按实施时间顺序从后到前依次计算(向前复权)
    """
    if len(code) == 6:
        code = code + '.SH' if code.startswith('6') else code + '.SZ'
    if end is None:
        end = datetime.datetime.now().strftime('%Y%m%d')
    date_regex = r'^(19[89]\d|20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])$'
    pattern = re.compile(date_regex)
    if not pattern.match(start) or not pattern.match(end):
        raise ValueError("start 和 end 必须是 YYYYMMDD 格式")
    if start > end:
        raise ValueError("start 不得晚于 end")
    dividend_csv = f'{FINANDATA_DIR}/dividend/{code}.csv'
    if not os.path.exists(dividend_csv):
        return price_now
    dividend_df = pd.read_csv(dividend_csv, dtype={'ex_date': str})
    if dividend_df.empty:
        return price_now
    columns = ['ts_code', 'name', 'industry', 'stk_div', 'cash_div_tax', 'ex_date']
    dividend_df = dividend_df[columns]
    dividend_df = dividend_df.dropna(subset=['ex_date'])
    dividend_df = dividend_df.sort_values(by='ex_date', ascending=False)  # 降序排列
    dividend_df.reset_index(drop=True, inplace=True)  # 重置索引
    # 遍历dividend_df，如果ex_date在start和end之间，
    # 计算除权除息前价格(from start to end)
    pre_price = price_now
    for _, row in dividend_df.iterrows():
        ex_date = row['ex_date']
        if not (start <= ex_date <= end):
            continue
        div_per_stock = row['stk_div'] if pd.notna(row['stk_div']) else 0
        cash_dividend_per_stock = row['cash_div_tax'] if pd.notna(row['cash_div_tax']) else 0
        pre_price = pre_price * (1 + div_per_stock) + cash_dividend_per_stock
    return pre_price

def is_trade_date_or_not():
    """ 
    check if today is trade date or not
    :return: True if today is trade date, False otherwise
    """
    today = datetime.datetime.now().strftime('%Y%m%d')
    trade_cal = pd.read_excel(TRADE_CAL_XLS, dtype={'cal_date': str})
    trade_cal = trade_cal[trade_cal['is_open'] == 1]
    dates = trade_cal['cal_date'].tolist()
    return today in dates

def is_within_trading_hours():
    """
    check if current time is within trading hours
    """
    now = datetime.datetime.now().time()
    time1 = datetime.time(9, 30)  # 上午 9:30
    time2 = datetime.time(11, 30)  # 上午 11:00
    time3 = datetime.time(13, 0)  # 下午 13:00
    time4 = datetime.time(15, 0)    # 下午 15:00
    return (time1 <= now <= time2) or (time3 <= now <= time4)
