import os
import re
import time
import pandas as pd
import datetime
import requests
import tushare as ts
from DrissionPage import ChromiumOptions, Chromium
from cons_general import TRADE_CAL_XLS, FINANDATA_DIR, UP_DOWN_LIMIT_XLS
from cons_oversold import PAUSE
from cons_hidden import xq_a_token


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
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
                (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
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

def get_history_realtime_price_DF_from_sina(code, scale=1, datalen=10) -> pd.DataFrame:
    """
    获取新浪财经今日历史实时价格数据(分钟数据)
    :param code: 股票代码, 如 000001 或 000001.SZ, 要转换为 sh600000 或 sz000001 格式
    :param scale: 分钟周期, 1(1分钟)、5(5分钟)、15(15分钟)、30(30分钟)、60(60分钟)
    :param datalen: 返回的数据节点数量, 最大值为1023
    :return: 实时价格数据序列或则空 DataFrame
    NOTE:
    url = 'https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData?\
    symbol=[股票代码]&scale=[分钟周期]&ma=no&datalen=[数据长度]'
    ma=no 表示不返回均线数据
    """
    code = f'sh{code[:6]}' if code.startswith('6') else f'sz{code[:6]}'
    url = f'https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData?\
        symbol={code}&scale={scale}&ma=no&datalen={datalen}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
            (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(url, headers=headers)
    data = response.json() if response.status_code == 200 else None
    res_df = pd.DataFrame(data) if data else pd.DataFrame()
    if not res_df.empty:
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        res_df = res_df[res_df['day'].str.startswith(today)]
        res_df[['open', 'high']] = res_df[['open', 'high']].apply(pd.to_numeric, errors='coerce')
        res_df[['low', 'close']] = res_df[['low', 'close']].apply(pd.to_numeric, errors='coerce')
        res_df = res_df[['day', 'open', 'high', 'low', 'close']]
        res_df = res_df.reset_index(drop=True)
    return res_df

def get_history_realtime_price_DF_from_xueqiu(code, datalen=10) -> pd.DataFrame:
    """
    获取雪球今日历史实时价格数据(分钟数据)
    :param code: 股票代码, 如 000001 或 600000.SH, 需要转换为 SZ000001 或 SH600000 格式
    :param datalen: 返回的数据节点数量
    :return: 实时价格数据序列或者空 DataFrame
    NOTE:
    url = 'https://stock.xueqiu.com/v5/stock/chart/minute.json?symbol=[股票代码]&period=1d'
    period: '1d' 表示获取当天的分钟数据
    """
    code = f'SZ{code[:6]}' if code.startswith('0') else f'SH{code[:6]}'
    url = f'https://stock.xueqiu.com/v5/stock/chart/minute.json?symbol={code}&period=1d'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
            (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Cookie': xq_a_token
    }
    response = requests.get(url, headers=headers)
    data = response.json() if response.status_code == 200 else None
    res = data['data']['items'] if data else None
    res_df = pd.DataFrame(res) if res else pd.DataFrame(columns=['day', 'close', 'high', 'low'])
    if not res_df.empty:
        res_df = res_df[['timestamp', 'current', 'high', 'low']]
        res_df['timestamp'] = res_df['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000))
        res_df['timestamp'] = res_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        res_df.columns = ['day', 'close', 'high', 'low']
        res_df = res_df.tail(datalen)
        res_df = res_df.reset_index(drop=True)
    else:
        print(f'获取雪球数据失败:{response.status_code},请检查 token 设置、网络连接或是否为交易日')
    return res_df

def get_XD_XR_DR_qfq_price_DF(src_data: pd.DataFrame) -> pd.DataFrame:
    """
    计算除权除息后的价格序列(前复权),并对change、pct_chg字段重新计算.
    :param src_data: basicdata/dailydata下日行情数据(按trade_date升序排列)
    :return: src_data(open、high、low、close、pre_close)除权除息前的价格序列(按trade_date升序排列)
    NOTE: 
    xr_price = (pre_price - cash_divdend_per_stock) / (1 + div_per_stock)
    div_per_stock: 每股转送股数
    cash_dividend_per_stock: 每股派息税前金额
    如有多次除权除息,按实施时间顺序从前到后依次计算(前复权)
    XD:除息 XR:转送股除权 DR:除息及除权, 未考虑配股除权
    """
    src_data_cp = src_data.copy()
    src_data_cp = src_data_cp.sort_values(by='trade_date', ascending=True)  # 升序排列
    src_data_cp.reset_index(drop=True, inplace=True)  # 重置索引
    ts_code = src_data_cp['ts_code'].iloc[0]
    dividend_csv = f'{FINANDATA_DIR}/dividend/{ts_code}.csv'
    if not os.path.exists(dividend_csv):
        return src_data_cp
    dividend_df = pd.read_csv(dividend_csv, dtype={'ex_date': str})
    if dividend_df.empty:
        return src_data_cp
    columns = ['ts_code', 'name', 'industry', 'stk_div', 'cash_div_tax', 'ex_date']
    dividend_df = dividend_df[columns]
    dividend_df = dividend_df.dropna(subset=['ex_date'])
    dividend_df = dividend_df.sort_values(by='ex_date', ascending=True)  # 升序排列
    dividend_df.reset_index(drop=True, inplace=True)  # 重置索引
    # 遍历dividend_df，如果ex_date在scr_data_cp的trade_date中，
    # 计算除权除息后价格(from ex_date to src_data_cp's last trade_date)
    for _, row in dividend_df.iterrows():
        ex_date = row['ex_date']
        if ex_date not in src_data_cp['trade_date'].values:
            continue
        div_per_stock = row['stk_div'] if pd.notna(row['stk_div']) else 0
        cash_dividend_per_stock = row['cash_div_tax'] if pd.notna(row['cash_div_tax']) else 0
        idx = src_data_cp[src_data_cp['trade_date'] == ex_date].index[0]
        src_data_cp.loc[:idx-1, ['open', 'high', 'low']] = \
            (src_data_cp.loc[:idx-1, ['open', 'high', 'low']] - cash_dividend_per_stock) / (1 + div_per_stock)
        src_data_cp.loc[:idx-1, ['close', 'pre_close']] = \
            (src_data_cp.loc[:idx-1, ['close', 'pre_close']] - cash_dividend_per_stock) / (1 + div_per_stock)
        src_data_cp.loc[:idx-1, 'change'] = \
            src_data_cp.loc[:idx-1, 'close'] - src_data_cp.loc[:idx-1, 'pre_close']
        src_data_cp.loc[:idx-1, 'pct_chg'] = \
            src_data_cp.loc[:idx-1, 'change'] / src_data_cp.loc[:idx-1, 'pre_close']
    return src_data_cp

def get_XD_XR_DR_qfq_price_and_amount(code, pre_price, amount, start:str, end:str = None) -> tuple[float, float]:
    """
    将start日pre_price和股数转换到end日除权除息后的价格和股数(前复权)
    :param code: 股票代码, 如 000001 或 000001.SZ
    :param pre_price: 未除权的价格
    :param amount: 未除权的股数
    :param start: 起始日期(YYYYMMDD)
    :param end: 结束日期(YYYYMMDD), None表示到今天
    :return: (end日除权除息后的价格, end日除权除息后的股数)
    NOTE: 
    xr_price = (pre_price - cash_dividend_per_stock) / (1 + div_per_stock)
    xr_amount = amount * (1 + div_per_stock)
    div_per_stock: 每股转送股数
    cash_dividend_per_stock: 每股派息税前金额
    如有多次除权除息，按实施时间顺序从前到后依次计算(前复权)
    XD:除息 XR:转送股除权 DR:除息及除权, 未考虑配股除权
    """
    if len(code) == 6:
        code = code + '.SH' if code.startswith('6') else code + '.SZ'
    if end is None:
        end = datetime.datetime.now().strftime('%Y%m%d')
    date_regex = r'^(19[89]\d|20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])$'  # from 19800101
    pattern = re.compile(date_regex)
    if not pattern.match(start) or not pattern.match(end):
        raise ValueError("start 和 end 必须是 YYYYMMDD 格式")
    if start > end:
        raise ValueError("start 不得晚于 end")
    dividend_csv = f'{FINANDATA_DIR}/dividend/{code}.csv'
    if not os.path.exists(dividend_csv):
        return pre_price, amount
    dividend_df = pd.read_csv(dividend_csv, dtype={'ex_date': str})
    if dividend_df.empty:
        return pre_price, amount
    columns = ['ts_code', 'name', 'industry', 'stk_div', 'cash_div_tax', 'ex_date']
    dividend_df = dividend_df[columns]
    dividend_df = dividend_df.dropna(subset=['ex_date'])
    dividend_df = dividend_df.sort_values(by='ex_date', ascending=True)  # 升序排列
    dividend_df.reset_index(drop=True, inplace=True)  # 重置索引
    # 遍历dividend_df，如果ex_date在start和end之间，
    # 计算除权除息后的价格(from start to end)
    xr_price = pre_price
    xr_amount = amount
    for _, row in dividend_df.iterrows():
        ex_date = row['ex_date']
        if not (start < ex_date <= end):
            continue
        div_per_stock = row['stk_div'] if pd.notna(row['stk_div']) else 0
        cash_dividend_per_stock = row['cash_div_tax'] if pd.notna(row['cash_div_tax']) else 0
        xr_price = (xr_price - cash_dividend_per_stock) / (1 + div_per_stock)
        xr_amount = xr_amount * (1 + div_per_stock)
    xr_price = round(xr_price, 2)  # 保留两位小数
    if xr_price < 0:
        xr_price = 0.0  # 如果计算结果小于0，则返回0
    return xr_price, xr_amount

def get_up_down_limit(code: str) -> tuple[float, float]:
    """
    获取股票的涨跌停价格
    :param code: 股票代码, 如 000001 或 000001.SZ
    :return: (涨停价, 跌停价)
    """
    if len(code) == 6:
        code = code + '.SH' if code.startswith('6') else code + '.SZ'
    today = datetime.datetime.now().strftime('%Y%m%d')
    up_down_df = pd.read_excel(UP_DOWN_LIMIT_XLS, dtype={'trade_date': str})
    if up_down_df['trade_date'].iloc[0] != today:
        return None, None
    res_df = up_down_df[up_down_df['ts_code'] == code]
    if res_df.empty:
        return None, None
    up_limit = res_df['up_limit'].values[0]
    down_limit = res_df['down_limit'].values[0]
    return up_limit, down_limit

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
