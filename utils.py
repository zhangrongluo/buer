import os
import re
import time
import json
import pandas as pd
import aiohttp
import asyncio
import datetime
import requests
import tushare as ts
from typing import Literal
from DrissionPage import ChromiumOptions, Chromium
from basic_data_alt_edition import download_dividend_data_in_multi_ways
from cons_general import TRADE_CAL_CSV, UP_DOWN_LIMIT_CSV, BASICDATA_DIR, TRADE_DIR, SUSPEND_STOCK_CSV, TEMP_DIR, DAILY_ADJFACTOR_TEMP_CSV
from cons_oversold import PAUSE
from cons_downgap import dataset_group_cons


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

def send_wechat_message_via_pushover(user_key, app_token, title, message):
    """
    send wechat message via pushover
    :param user_key: pushover user key
    :param app_token: pushover app token
    :param title: message title
    :param message: message content
    :return: response
    NOTE:
    backup method when bark not works, 10000 messages/month free
    need to feed for more messages
    """
    url = "https://api.pushover.net/1/messages.json"
    data = {
        "token": app_token,
        "user": user_key,
        "title": title,
        "message": message
    }
    response = requests.post(url, data=data).json()
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
    
async def async_get_stock_price_from_tencent(code: str) -> float | None:
    """
    get stock realtime price from tencent finance using asyncio
    :param code: stock code, like 000001 or 000001.SH
    :return: realtime stock price
    NOTE: 
    Async version of get_stock_price_from_tencent
    """
    stock_code = code[:6]
    stock_code = 'sh' + stock_code if code[0] == '6' else 'sz' + stock_code
    try:
        url = f"https://qt.gtimg.cn/q={stock_code}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
                (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=5) as response:
                response.raise_for_status()  # 检查请求是否成功
                # 解析数据腾讯财经返回的数据,格式:v_sh600036="1~招商银行~600036~40.74...";
                data = await response.text()
                if not data or "v_" not in data:
                    return None
                data_content = data.split("=")[1]
                data_list = data_content.split("~")
                price = data_list[3]  # realtime price
                price = float(price)
                return price
    except Exception as e:
        return None

async def get_all_prices_async(ts_codes: list[str]) -> dict[str, float | None]:
    """
    异步并发获取多个股票股价
    :param ts_codes: 股票代码列表, 如 ['000001.SZ', '600000.SH']
    :return: 字典, 股票代码为键, 实时价格为值
    NOTE:
    采用腾讯财经接口单渠道获取股票价格
    """
    tasks = [async_get_stock_price_from_tencent(code) for code in ts_codes]
    prices = await asyncio.gather(*tasks, return_exceptions=True)
    results = {}
    for code, price in zip(ts_codes, prices):
        if isinstance(price, Exception):
            results[code] = None
        else:
            results[code] = price
    return results
    
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

def get_history_realtime_price_DF_from_sina(code, scale=1, datalen=15) -> pd.DataFrame:
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
    res_df = pd.DataFrame(data) if data else pd.DataFrame(columns=['day', 'open', 'high', 'low', 'close'])
    if not res_df.empty:
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        res_df = res_df[res_df['day'].str.startswith(today)]
        res_df[['open', 'high']] = res_df[['open', 'high']].apply(pd.to_numeric, errors='coerce')
        res_df[['low', 'close']] = res_df[['low', 'close']].apply(pd.to_numeric, errors='coerce')
        res_df = res_df[['day', 'open', 'high', 'low', 'close']]
        res_df = res_df.reset_index(drop=True)
    else:
        print(f'获取新浪数据失败:{response.status_code},请检查 token 设置、网络连接或是否为交易日')
    return res_df

def get_history_realtime_price_DF_from_dc(code, klt=1, datalen=15) -> pd.DataFrame:
    """
    从东方财富获取股票分钟级别价格数据
    :param code: 股票代码, 000001或者000001.SZ, 需转化成 sz000001 或则sh600000 格式
    :param klt: K线周期, 1(1分钟)、5(5分钟)、15(15分钟)、30(30分钟)、60(60分钟)
    :param datalen: 返回的数据节点数量
    :return: DataFrame   
    NOTE:
    written by Grok, 参数详细说明见 https://www.sanrenjz.com/2023/03/31/
    """
    code = 'sh' + code[:6] if code.startswith('6') else 'sz' + code[:6]
    market = '1' if code.startswith('sh') else '0'
    secid = f"{market}.{code[2:]}"
    url = "http://push2.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": str(klt),  # K线周期
        "fqt": "1",      # 复权类型，1=前复权
        "beg": "0",      # 开始日期，0表示最新数据
        "end": "20500101",  # 结束日期，设置为未来日期以获取最新数据
        "lmt": "240",    # 获取最近 240 条记录
        "_": str(int(time.time() * 1000))  # 时间戳，防止缓存
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
            (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()  # 检查请求是否成功
        data = json.loads(response.text)
        if data["data"] is None:
            print(f"未获取到股票 {code} 的数据，可能代码错误或无分钟数据")
            return pd.DataFrame(columns=["day", "open", "close", "high", "low"])
        kline_data = data["data"]["klines"]
        columns = ["时间", "开盘价", "收盘价", "最高价", "最低价", "成交量", "成交额", 
                   "振幅%", "涨跌幅%", "涨跌额", "换手率%"]
        df = pd.DataFrame([x.split(",") for x in kline_data], columns=columns)
        # 数据类型转换
        for col in ["开盘价", "收盘价", "最高价", "最低价", "成交量", "成交额", "振幅%", "涨跌幅%", "涨跌额", "换手率%"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        col_out = ["时间", "开盘价", "收盘价", "最高价", "最低价"]
        df = df[col_out]
        df.columns = ["day", "open", "close", "high", "low"]
        df = df.tail(datalen).sort_values(by="day").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"获取东方财富实时价格数据序列失败: {e}")
        return pd.DataFrame(columns=["day", "open", "close", "high", "low"])

def get_qfq_price_DF_by_adj_factor(src_data: pd.DataFrame) -> pd.DataFrame:
    """
    通过复权因子计算前复权价格序列
    :param src_data: basicdata/dailydata下日行情数据(按trade_date升序排列)
    :return: src_data(open、high、low、close、pre_close、change、pct_chg)
    前复权的价格序列(按trade_date升序排列)
    NOTE:
    复权后价格为空的行被删除
    """
    src_data_cp = src_data.copy()
    src_data_cp = src_data_cp.sort_values(by='trade_date', ascending=True)  # 升序排列
    src_data_cp.reset_index(drop=True, inplace=True)  # 重置索引
    ts_code = src_data_cp['ts_code'].iloc[0]
    factor_csv = f'{BASICDATA_DIR}/adjfactor/{ts_code}.csv'
    if not os.path.exists(factor_csv):
        return src_data_cp
    factor_df = pd.read_csv(factor_csv, dtype={'trade_date': str})
    if factor_df.empty:
        return src_data_cp
    factor_df = factor_df.sort_values(by='trade_date', ascending=True)  # 升序排列
    factor_df.reset_index(drop=True, inplace=True)  # 重置索引
    # merge src_data_cp with factor_df on trade_date
    src_data_cp = pd.merge(src_data_cp, factor_df[['trade_date', 'adj_factor']], on='trade_date', how='left')
    last_adj_factor = factor_df['adj_factor'].iloc[-1]
    src_data_cp['last_adj_factor'] = last_adj_factor
    src_data_cp['cal_factor'] = src_data_cp['adj_factor'] / last_adj_factor
    src_data_cp['open'] = src_data_cp['open'] * src_data_cp['cal_factor']
    src_data_cp['high'] = src_data_cp['high'] * src_data_cp['cal_factor']
    src_data_cp['low'] = src_data_cp['low'] * src_data_cp['cal_factor']
    src_data_cp['close'] = src_data_cp['close'] * src_data_cp['cal_factor']
    src_data_cp['pre_close'] = src_data_cp['pre_close'] * src_data_cp['cal_factor']
    src_data_cp.dropna(subset=['open', 'high', 'low', 'close', 'pre_close'], inplace=True)
    src_data_cp.drop(columns=['adj_factor', 'last_adj_factor', 'cal_factor'], inplace=True)
    src_data_cp['change'] = src_data_cp['close'] - src_data_cp['pre_close']
    src_data_cp['pct_chg'] = src_data_cp['change'] / src_data_cp['pre_close'] * 100
    src_data_cp[['open', 'high', 'low', 'close', 'pre_close', 'change']] = \
        src_data_cp[['open', 'high', 'low', 'close', 'pre_close', 'change']].round(2)
    src_data_cp['pct_chg'] = src_data_cp['pct_chg'].round(4)
    return src_data_cp

def get_XR_adjust_amount_by_dividend_data(code, amount, start:str, end:str = None) -> float:
    """
    根据XR送转股比例将start日股数转换到end日的股数
    :param code: 股票代码, 如 000001 或 000001.SZ
    :param amount: 未除权的股数
    :param start: 起始日期(YYYYMMDD)
    :param end: 结束日期(YYYYMMDD), None表示到今天
    :return: end日调整后的股数(一般为放大股数)
    NOTE: 
    xr_amount = amount * (1 + div_per_stock)
    div_per_stock: 每股转送股数
    如有多次除权，按实施时间顺序从前到后依次计算(前复权)
    NOTE:
    参数错误直接返回原股数,不再抛出异常
    如未能从 tushare 下载到数据, 则继续从 sina 和 xueqiu 下载数据
    """
    if len(code) == 6:
        code = code + '.SH' if code.startswith('6') else code + '.SZ'
    if end is None:
        end = datetime.datetime.now().strftime('%Y%m%d')
    date_regex = r'^(19[89]\d|20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])$'  # from 19800101
    pattern = re.compile(date_regex)
    if not pattern.match(start) or not pattern.match(end):
        return amount
    if start > end:
        return amount
    dividend_csv = download_dividend_data_in_multi_ways(ts_code=code)  # download when needed
    if dividend_csv is None:
        return amount
    dividend_df = pd.read_csv(dividend_csv, dtype={'ex_date': str})
    if dividend_df.empty:
        return amount
    columns = ['ts_code', 'name', 'industry', 'stk_div', 'ex_date']
    dividend_df = dividend_df[columns]
    dividend_df = dividend_df.dropna(subset=['ex_date'])
    dividend_df = dividend_df.sort_values(by='ex_date', ascending=True)  # 升序排列
    dividend_df.reset_index(drop=True, inplace=True)  # 重置索引
    xr_amount = amount
    for _, row in dividend_df.iterrows():
        ex_date = row['ex_date']  # 除权除息日期
        if not (start < ex_date <= end):
            continue
        div_per_stock = row['stk_div'] if pd.notna(row['stk_div']) else 0
        xr_amount = xr_amount * (1 + div_per_stock)
    xr_amount = round(xr_amount, 0)  # 保留整数股数
    return xr_amount

def get_qfq_price_by_adj_factor(code, pre_price, start: str, end: str = None) -> float:
    """
    获取前复权价格
    :param code: 股票代码, 如 000001 或 000001.SZ
    :param pre_price: 未复权的价格
    :param start: 起始日期(YYYYMMDD)
    :param end: 结束日期(YYYYMMDD), None表示到今天
    :return: 复权后的价格
    NOTE:
    参数错误直接返回原价格和股数,不再抛出异常
    """
    if len(code) == 6:
        code = code + '.SH' if code.startswith('6') else code + '.SZ'
    if end is None:
        end = datetime.datetime.now().strftime('%Y%m%d')
    date_regex = r'^(19[89]\d|20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])$'  # from 19800101 to 20991231
    pattern = re.compile(date_regex)
    if not pattern.match(start) or not pattern.match(end):
        return pre_price
    if start > end:
        return pre_price
    factor_csv = f'{BASICDATA_DIR}/adjfactor/{code}.csv'
    if not os.path.exists(factor_csv):
        return pre_price
    factor_df = pd.read_csv(factor_csv, dtype={'trade_date': str})
    if factor_df.empty:
        return pre_price
    factor_df = factor_df.sort_values(by='trade_date', ascending=True)  # 升序排列
    factor_df.reset_index(drop=True, inplace=True)  # 重置索引
    start_factor = factor_df[factor_df['trade_date'] == start]['adj_factor'].values
    end_factor = factor_df[factor_df['trade_date'] == end]['adj_factor'].values
    if start_factor.size == 0 :
        # 获取最接近 start 的前复权因子
        start_factor = factor_df[factor_df['trade_date'] < start]['adj_factor'].max()
    else:
        start_factor = start_factor[0]
    if end_factor.size == 0:
        # 获取最接近 end 的前复权因子
        end_factor = factor_df[factor_df['trade_date'] <= end]['adj_factor'].max()
    else:
        end_factor = end_factor[0]
    if start_factor == 0 or end_factor == 0:
        return pre_price
    # 计算复权后的价格和股数
    qfq_price = pre_price * (start_factor / end_factor)
    qfq_price = round(qfq_price, 2)  # 保留两位小数
    if qfq_price < 0:
        qfq_price = 0.0  # 如果计算结果小于0，则返回0
    return qfq_price

def get_up_down_limit(code: str) -> tuple[float, float, float]:
    """
    获取股票的涨跌停价格
    :param code: 股票代码, 如 000001 或 000001.SZ
    :return: (涨停价, 跌停价, 涨停幅度)
    """
    if len(code) == 6:
        code = code + '.SH' if code.startswith('6') else code + '.SZ'
    today = datetime.datetime.now().strftime('%Y%m%d')
    if not os.path.exists(UP_DOWN_LIMIT_CSV):
        from stocklist import get_up_down_limit_list
        get_up_down_limit_list()
    up_down_df = pd.read_csv(UP_DOWN_LIMIT_CSV, dtype={'trade_date': str})
    if up_down_df.empty:
        from stocklist import get_up_down_limit_list
        get_up_down_limit_list()
        up_down_df = pd.read_csv(UP_DOWN_LIMIT_CSV, dtype={'trade_date': str})
    if up_down_df.empty:
        return None, None, None
    if up_down_df['trade_date'].iloc[0] != today:
        return None, None, None
    res_df = up_down_df[up_down_df['ts_code'] == code]
    if res_df.empty:
        return None, None, None
    up_limit = res_df['up_limit'].values[0]
    down_limit = res_df['down_limit'].values[0]
    pre_close = (up_limit + down_limit) / 2
    up_limit_rate = round((up_limit - pre_close) / pre_close, 2)
    return up_limit, down_limit, up_limit_rate

def check_pre_trade_data_update_status() -> dict[str, bool]:
    """
    ### 检查交易前数据是否更新完成
    #### 检查四个文件: TRADE_CAL_CSV, UP_DOWN_LIMIT_CSV, SUSPEND_STOCK_CSV, DAILY_ADJFACTOR_TEMP_CSV
    #### 检查内容: 文件是否存在且是否已更新至最新交易日
    :return: 字典, 包含各数据更新状态
    """
    today = datetime.datetime.now().strftime('%Y%m%d')
    status = {
        'trade_cal_updated': False,
        'up_down_limit_updated': False,
        'suspend_stock_updated': False,
        'daily_adj_factor_updated': False
    }
    # check trade calendar
    if os.path.exists(TRADE_CAL_CSV):
        trade_cal_df = pd.read_csv(TRADE_CAL_CSV, dtype={'cal_date': str})
        if not trade_cal_df.empty:
            last_trade_date = trade_cal_df['cal_date'].max()
            if last_trade_date >= today:
                status['trade_cal_updated'] = True
    # check up down limit
    if os.path.exists(UP_DOWN_LIMIT_CSV):
        up_down_df = pd.read_csv(UP_DOWN_LIMIT_CSV, dtype={'trade_date': str})
        if not up_down_df.empty:
            last_trade_date = up_down_df['trade_date'].max()
            if last_trade_date >= today:
                status['up_down_limit_updated'] = True
    # check suspend stock
    if os.path.exists(SUSPEND_STOCK_CSV):
        suspend_stock_df = pd.read_csv(SUSPEND_STOCK_CSV, dtype={'trade_date': str})
        if not suspend_stock_df.empty:
            last_trade_date = suspend_stock_df['trade_date'].max()
            if last_trade_date >= today:
                status['suspend_stock_updated'] = True
    # check daily adj factor tmp  csv
    if os.path.exists(DAILY_ADJFACTOR_TEMP_CSV):
        daily_adj_factor_df = pd.read_csv(DAILY_ADJFACTOR_TEMP_CSV, dtype={'trade_date': str})
        if not daily_adj_factor_df.empty:
            last_trade_date = daily_adj_factor_df['trade_date'].max()
            if last_trade_date >= today:
                status['daily_adj_factor_updated'] = True
    return status

def early_sell_standard_oversold(holding_days: int, rate_current: float, rate_yearly: float) -> bool:
    """
    oversold 提前卖出标准
    :param holding_days: 持有天数
    :param rate_current: 当前收益率
    :param rate_yearly: 年化收益率
    :return: True if should sell, False otherwise
    NOTE:
    holding_days < 15 and rate_current >= 0.20
    30 > holding_days >= 15 and rate_yearly >= 3.65
    60 > holding_days >= 30 and rate_yearly >= 2.23
    90 > holding_days >= 60 and rate_yearly >= 1.58
    rate_yearly 按照年 365 天计算
    3.65 = 365/((15+30)/2)*((0.20+0.25)/2)
    2.23 = 365/((30+60)/2)*((0.25+0.30)/2)
    1.58 = 365/((60+90)/2)*((0.30+0.35)/2)
    """
    if holding_days < 15 and rate_current >= 0.20:
        return True
    elif 15 <= holding_days < 30 and rate_yearly >= 3.65:
        return True
    elif 30 <= holding_days < 60 and rate_yearly >= 2.23:
        return True
    elif 60 <= holding_days < 90 and rate_yearly >= 1.58:
        return True
    else:
        return False

def early_sell_standard_downgap(holding_days: int, rate_current: float, rate_yearly: float) -> bool:
    """
    downgap 提前卖出标准
    :param holding_days: 持有天数
    :param rate_current: 当前收益率
    :param rate_yearly: 年化收益率
    :return: True if should sell, False otherwise
    NOTE:
    rate_yearly 按照年 365 天计算
    3.65 = 365/((10+20)/2)*((0.12+0.18)/2)
    """
    if holding_days < 10 and rate_current >= 0.10:
        return True
    elif 10 <= holding_days < 20 and rate_yearly >= 3.65:
        return True
    else:
        return False

def is_rising_or_not(code, price_now: float, method: Literal['max', 'mean'] = 'mean') -> bool:
    """
    判断股票是否上涨
    :param code: 股票代码, 如 000001 或 000001.SZ
    :param price_now: 当前价格
    :param method: 判断方法, 'max' 或 'mean', 默认 'mean'
    :return: True if stock is rising, False otherwise
    NOTE:
    如果 price_now 超过前 10 分钟平均价(最高价), 则认为上涨
    """
    if len(code) == 6:
        code = code + '.SH' if code.startswith('6') else code + '.SZ'
    rt_price_df = get_history_realtime_price_DF_from_sina(code)
    if rt_price_df.empty:
        rt_price_df = get_history_realtime_price_DF_from_dc(code)
    if rt_price_df.empty:
        return False
    if method == 'mean':
        return price_now > rt_price_df['close'].mean()
    elif method == 'max':
        return price_now > rt_price_df['close'].max()
    return False

def is_decreasing_or_not(code, price_now: float, method: Literal['min', 'mean'] = 'mean') -> bool:
    """
    判断股票是否下跌
    :param code: 股票代码, 如 000001 或 000001.SZ
    :param price_now: 当前价格
    :param method: 判断方法, 'min' 或 'mean', 默认 'mean'
    :return: True if stock is decreasing, False otherwise
    NOTE:
    如果 price_now 低于前 10 分钟平均价(最低价), 则认为下跌
    """
    if len(code) == 6:
        code = code + '.SH' if code.startswith('6') else code + '.SZ'
    rt_price_df = get_history_realtime_price_DF_from_sina(code)
    if rt_price_df.empty:
        rt_price_df = get_history_realtime_price_DF_from_dc(code)
    if rt_price_df.empty:
        return False
    if method == 'mean':
        return price_now < rt_price_df['close'].mean()
    elif method == 'min':
        return price_now < rt_price_df['close'].min()
    return False

def is_trade_date_or_not():
    """ 
    check if today is trade date or not
    :return: True if today is trade date, False otherwise
    """
    today = datetime.datetime.now().strftime('%Y%m%d')
    trade_cal = pd.read_csv(TRADE_CAL_CSV, dtype={'cal_date': str})
    trade_cal = trade_cal[trade_cal['is_open'] == 1]
    dates = trade_cal['cal_date'].tolist()
    return today in dates

def is_suspended_or_not(code: str) -> bool:
    """
    check if stock is suspended
    :param code: stock code
    :return: True if suspended, False otherwise
    """
    if len(code) == 6:
        code = code + '.SH' if code.startswith('6') else code + '.SZ'
    if not is_trade_date_or_not():
        return False
    if not os.path.exists(SUSPEND_STOCK_CSV):
        from stocklist import get_suspend_stock_list
        get_suspend_stock_list()
    suspend_df = pd.read_csv(SUSPEND_STOCK_CSV, dtype={'ts_code': str})
    if suspend_df.empty:
        from stocklist import get_suspend_stock_list
        get_suspend_stock_list()
        suspend_df = pd.read_csv(SUSPEND_STOCK_CSV, dtype={'ts_code': str})
    if suspend_df.empty:
        return False
    return code in suspend_df['ts_code'].values

### statistics functions
def calculate_win_rate_of_days(
        name : Literal['oversold', 'downgap'], start=None, end=None, days=1, **kwargs
):
    """
    Calculate the win rate based on number of days
    :param name: Name of the strategy or model, oversold or downgap
    :param start: 'YYYYMMDD' format, default is None (all data)
    :param end: 'YYYYMMDD' format, default is None (all data)
    :param days: Number of days to calculate profit, default is 1
    :param kwargs: e.g., max_trade_days for downgap strategy
    :return: Win rate (percentage)
    """
    if name.upper() not in ['OVERSOLD', 'DOWNGAP']:
        raise ValueError(f"Name {name} not in ['oversold', 'downgap']")
    if name.upper() == 'OVERSOLD':
        trade_root = f'{TRADE_DIR}/oversold'
    if name.upper() == 'DOWNGAP':
        max_trade_days = kwargs.get('max_trade_days')
        if max_trade_days is None:
            raise ValueError("max_trade_days is required for downgap strategy")
        if not isinstance(max_trade_days, (int, float)):
            raise ValueError("max_trade_days must be an integer or float for downgap strategy")
        if int(max_trade_days) not in dataset_group_cons['common']['MAX_TRADE_DAYS_LIST']:
            raise ValueError(
                f"max_trade_days must be in {dataset_group_cons['common']['MAX_TRADE_DAYS_LIST']}"
            )
        max_trade_days = int(max_trade_days)
        trade_root = f'{TRADE_DIR}/downgap/max_trade_days_{max_trade_days}'
    profit_csv = f'{trade_root}/daily_profit.csv'
    if not os.path.exists(profit_csv):
        return 0.0
    profit_df = pd.read_csv(profit_csv, dtype={'trade_date': str})
    if start is not None:
        profit_df = profit_df[profit_df['trade_date'] >= start]
    if end is not None:
        profit_df = profit_df[profit_df['trade_date'] <= end]
    if profit_df.empty:
        return 0.0
    profit_df = profit_df.sort_values(by='trade_date', ascending=True)
    profit_df = profit_df.reset_index(drop=True)
    win_count = 0
    for i in range(0, len(profit_df), days):
        group_df = profit_df.iloc[i:i + days]
        if group_df.empty:
            continue
        total_profit = group_df['delta'].sum()
        if total_profit > 0:
            win_count += 1
    total_groups = len(profit_df) // days + (1 if len(profit_df) % days > 0 else 0)
    if total_groups == 0:
        return 0.0
    win_rate = (win_count / total_groups) * 100
    return win_rate

def calculate_win_rate_of_stocks(
        name: Literal['oversold', 'downgap'], start=None, end=None, **kwargs
):
    """
    Calculate the win rate based on number of sold_out stocks
    :param name: Name of the strategy or model, oversold or downgap
    :param start: 'YYYYMMDD' format, default is None (all data)
    :param end: 'YYYYMMDD' format, default is None (all data)
    :param kwargs: e.g., max_trade_days for downgap strategy
    :return: Win rate (percentage)
    """
    if name.upper() not in ['OVERSOLD', 'DOWNGAP']:
        raise ValueError(f"Name {name} not in ['oversold', 'downgap']")
    if name.upper() == 'OVERSOLD':
        trade_root = f'{TRADE_DIR}/oversold'
    if name.upper() == 'DOWNGAP':
        max_trade_days = kwargs.get('max_trade_days')
        if max_trade_days is None:
            raise ValueError("max_trade_days is required for downgap strategy")
        if not isinstance(max_trade_days, (int, float)):
            raise ValueError("max_trade_days must be an integer or float for downgap strategy")
        if int(max_trade_days) not in dataset_group_cons['common']['MAX_TRADE_DAYS_LIST']:
            raise ValueError(
                f"max_trade_days must be in {dataset_group_cons['common']['MAX_TRADE_DAYS_LIST']}"
            )
        max_trade_days = int(max_trade_days)
        trade_root = f'{TRADE_DIR}/downgap/max_trade_days_{max_trade_days}'
    holding_csv = f'{trade_root}/holding_list.csv'
    if not os.path.exists(holding_csv):
        return 0.0
    holding_df = pd.read_csv(holding_csv)
    sold_stocks_df = holding_df[holding_df['status'] == 'sold_out']
    if start is not None:
        sold_stocks_df = sold_stocks_df[sold_stocks_df['date_out'] >= start]
    if end is not None:
        sold_stocks_df = sold_stocks_df[sold_stocks_df['date_out'] <= end]
    if sold_stocks_df.empty:
        return 0.0
    total_sold_stocks = len(sold_stocks_df)
    win_count = sold_stocks_df[sold_stocks_df['profit'] > 0].shape[0]
    if total_sold_stocks == 0:
        return 0.0
    win_rate = (win_count / total_sold_stocks) * 100
    return win_rate

def calculate_omega_ratio(
        name: Literal['oversold', 'downgap'], start=None, end=None, **kwargs
):
    """
    Calculate omega ratio (total profit to total loss)
    :param name: Name of the strategy or model, oversold or downgap
    :param start: 'YYYYMMDD' format, default is None (all data)
    :param end: 'YYYYMMDD' format, default is None (all data)
    :param kwargs: e.g., max_trade_days for downgap strategy
    :return: Ratio of total profit to total loss
    """
    if name.upper() not in ['OVERSOLD', 'DOWNGAP']:
        raise ValueError(f"Name {name} not in ['oversold', 'downgap']")
    if name.upper() == 'OVERSOLD':
        trade_root = f'{TRADE_DIR}/oversold'
    if name.upper() == 'DOWNGAP':
        max_trade_days = kwargs.get('max_trade_days')
        if max_trade_days is None:
            raise ValueError("max_trade_days is required for downgap strategy")
        if not isinstance(max_trade_days, (int, float)):
            raise ValueError("max_trade_days must be an integer or float for downgap strategy")
        if int(max_trade_days) not in dataset_group_cons['common']['MAX_TRADE_DAYS_LIST']:
            raise ValueError(
                f"max_trade_days must be in {dataset_group_cons['common']['MAX_TRADE_DAYS_LIST']}"
            )
        max_trade_days = int(max_trade_days)
        trade_root = f'{TRADE_DIR}/downgap/max_trade_days_{max_trade_days}'
    holding_csv = f'{trade_root}/holding_list.csv'
    if not os.path.exists(holding_csv):
        return 0.0
    holding_df = pd.read_csv(holding_csv, dtype={'date_in': str, 'date_out': str})
    if start is not None:
        holding_df = holding_df[holding_df['date_out'] >= start]
    if end is not None:
        holding_df = holding_df[holding_df['date_out'] <= end]
    loss_df = holding_df[holding_df['profit'] < 0]
    profit_df = holding_df[holding_df['profit'] >= 0]
    total_loss = -loss_df['profit'].sum()
    total_profit = profit_df['profit'].sum()
    if total_loss == 0:
        return 0.0
    return total_profit / total_loss if total_loss != 0 else 0.0

def get_stock_list_of_specific_date(
        name: Literal['oversold', 'downgap'], date: str, **kwargs
) -> pd.DataFrame:
    """
    Get stock list of specific date
    :param name: Name of the strategy or model, oversold or downgap
    :param date: 'YYYYMMDD' format
    :param kwargs: e.g., max_trade_days for downgap strategy
    :return: List of stock codes for the specific date
    """
    if name.upper() not in ['OVERSOLD', 'DOWNGAP']:
        raise ValueError(f"Name {name} not in ['oversold', 'downgap']")
    if name.upper() == 'OVERSOLD':
        trade_root = f'{TRADE_DIR}/oversold'
    if name.upper() == 'DOWNGAP':
        max_trade_days = kwargs.get('max_trade_days')
        if max_trade_days is None:
            raise ValueError("max_trade_days is required for downgap strategy")
        if not isinstance(max_trade_days, (int, float)):
            raise ValueError("max_trade_days must be an integer or float for downgap strategy")
        if int(max_trade_days) not in dataset_group_cons['common']['MAX_TRADE_DAYS_LIST']:
            raise ValueError(
                f"max_trade_days must be in {dataset_group_cons['common']['MAX_TRADE_DAYS_LIST']}"
            )
        max_trade_days = int(max_trade_days)
        trade_root = f'{TRADE_DIR}/downgap/max_trade_days_{max_trade_days}'
    hd_csv = f'{trade_root}/holding_list.csv'
    hd_df = pd.read_csv(hd_csv, dtype={'date_in': str, 'date_out': str})
    hd_df = hd_df[hd_df['date_in'] <= date]
    hd_df = hd_df[(hd_df['date_out'] > date) | (hd_df['date_out'].isnull())]
    columns = ['ts_code', 'stock_name', 'industry', 'date_in', 'date_out', 'amount']
    hd_df = hd_df[columns]
    hd_df['price'] = None
    codes = hd_df['ts_code'].unique().tolist()
    for code in codes:
        daily_csv = f'basicdata/dailydata/{code}.csv'
        daily_df = pd.read_csv(daily_csv, dtype={'trade_date': str})
        daily_df = daily_df[daily_df['trade_date'] <= date]
        if not daily_df.empty:
            daily_df = daily_df.sort_values(by='trade_date', ascending=True)
            hd_df.loc[hd_df['ts_code'] == code, 'price'] = daily_df.iloc[-1]['close']
    hd_df['value'] = hd_df['amount'] * hd_df['price']
    return hd_df

def calculate_profit_of_specific_date(
        name: Literal['oversold', 'downgap'], date: str, **kwargs
) -> float:
    """
    Calculate the profit of specific date
    :param name: Name of the strategy or model, oversold or downgap
    :param date: 'YYYYMMDD' format
    :param kwargs: e.g., max_trade_days for downgap strategy
    :return: Profit of the specific date
    """
    if name.upper() not in ['OVERSOLD', 'DOWNGAP']:
        raise ValueError(f"Name {name} not in ['oversold', 'downgap']")
    if name.upper() == 'OVERSOLD':
        trade_root = f'{TRADE_DIR}/oversold'
    if name.upper() == 'DOWNGAP':
        max_trade_days = kwargs.get('max_trade_days')
        if max_trade_days is None:
            raise ValueError("max_trade_days is required for downgap strategy")
        if not isinstance(max_trade_days, (int, float)):
            raise ValueError("max_trade_days must be an integer or float for downgap strategy")
        if int(max_trade_days) not in dataset_group_cons['common']['MAX_TRADE_DAYS_LIST']:
            raise ValueError(
                f"max_trade_days must be in {dataset_group_cons['common']['MAX_TRADE_DAYS_LIST']}"
            )
        max_trade_days = int(max_trade_days)
        trade_root = f'{TRADE_DIR}/downgap/max_trade_days_{max_trade_days}'
    profit_csv = f'{trade_root}/daily_profit.csv'
    if not os.path.exists(profit_csv):
        return 0.0
    profit_df = pd.read_csv(profit_csv, dtype={'trade_date': str})
    profit_df = profit_df[profit_df['trade_date'] == date]
    if profit_df.empty:
        return 0.0
    return profit_df['delta'].sum()

def calculate_return_ratio_of_specific_date(
        name: Literal['oversold', 'downgap'], date: str, **kwargs
) -> float:
    """
    Calculate the return ratio of specific date
    :param name: Name of the strategy or model, oversold or downgap
    :param date: 'YYYYMMDD' format
    :param kwargs: e.g., max_trade_days for downgap strategy
    :return: Return ratio of the specific date
    NOTE:
    Return ratio = (total profit) / (total value of the stocks)
    """
    if name.upper() not in ['OVERSOLD', 'DOWNGAP']:
        raise ValueError(f"Name {name} not in ['oversold', 'downgap']")
    if name.upper() == 'DOWNGAP':
        max_trade_days = kwargs.get('max_trade_days')
        if max_trade_days is None:
            raise ValueError("max_trade_days is required for downgap strategy")
        if not isinstance(max_trade_days, (int, float)):
            raise ValueError("max_trade_days must be an integer or float for downgap strategy")
        if int(max_trade_days) not in dataset_group_cons['common']['MAX_TRADE_DAYS_LIST']:
            raise ValueError(
                f"max_trade_days must be in {dataset_group_cons['common']['MAX_TRADE_DAYS_LIST']}"
            )
    profit = calculate_profit_of_specific_date(name, date, **kwargs)
    hd_df = get_stock_list_of_specific_date(name, date, **kwargs)
    if hd_df.empty:
        return 0.0
    total_value = hd_df['value'].sum()
    if total_value == 0:
        return 0.0
    return round(profit / total_value, 4)

def calculate_today_series_statistic_indicator(
        name: Literal['oversold', 'downgap'], **kwargs
):
    """
    Calculate today's series statistic indicators
    :param name: Name of the strategy or model, oversold or downgap
    :param kwargs: e.g., max_trade_days for downgap strategy
    NOTE:
    - 'win_rate': Win rate of the strategy(1、5、10、20、30 days)
    - 'omega_ratio': Omega ratio of the strategy
    - 'return_ratio': Return ratio of the strategy
    """
    if not is_trade_date_or_not():
        return
    if name.upper() not in ['OVERSOLD', 'DOWNGAP']:
        raise ValueError(f"Name {name} not in ['oversold', 'downgap']")
    if name.upper() == 'OVERSOLD':
        trade_root = f'{TRADE_DIR}/oversold'
    if name.upper() == 'DOWNGAP':
        max_trade_days = kwargs.get('max_trade_days')
        if max_trade_days is None:
            raise ValueError("max_trade_days is required for downgap strategy")
        if not isinstance(max_trade_days, (int, float)):
            raise ValueError("max_trade_days must be an integer or float for downgap strategy")
        if int(max_trade_days) not in dataset_group_cons['common']['MAX_TRADE_DAYS_LIST']:
            raise ValueError(
                f"max_trade_days must be in {dataset_group_cons['common']['MAX_TRADE_DAYS_LIST']}"
            )
        max_trade_days = int(max_trade_days)
        trade_root = f'{TRADE_DIR}/downgap/max_trade_days_{max_trade_days}'
    indicator_csv = f'{trade_root}/statistic_indicator.csv'
    win_rate_1 = calculate_win_rate_of_days(name, days=1, **kwargs)
    win_rate_5 = calculate_win_rate_of_days(name, days=5, **kwargs)
    win_rate_10 = calculate_win_rate_of_days(name, days=10, **kwargs)
    win_rate_20 = calculate_win_rate_of_days(name, days=20, **kwargs)
    win_rate_30 = calculate_win_rate_of_days(name, days=30, **kwargs)
    omega_ratio = calculate_omega_ratio(name, **kwargs)
    today = datetime.datetime.now().strftime('%Y%m%d')
    return_ratio = calculate_return_ratio_of_specific_date(name, today, **kwargs)
    win_rate_stocks = calculate_win_rate_of_stocks(name, **kwargs)
    data = {
        'trade_date': today,
        'win_rate_1': round(win_rate_1, 4),
        'win_rate_5': round(win_rate_5, 4),
        'win_rate_10': round(win_rate_10, 4),
        'win_rate_20': round(win_rate_20, 4),
        'win_rate_30': round(win_rate_30, 4),
        'omega_ratio': round(omega_ratio, 4),
        'return_ratio': round(return_ratio, 4),
        'win_rate_stocks': round(win_rate_stocks, 4)
    }
    df = pd.DataFrame([data])
    if not os.path.exists(indicator_csv):
        df.to_csv(indicator_csv, index=False)
    else:
        df.to_csv(indicator_csv, mode='a', header=False, index=False)

def calculate_information_ratio(
        name: Literal['oversold', 'downgap'], start=None, end=None, **kwargs
):
    """
    Calculate information ratio
    :param name: Name of the strategy or model, oversold or downgap
    :param start: 'YYYYMMDD' format, default is None (all data)
    :param end: 'YYYYMMDD' format, default is None (all data)
    :param kwargs: e.g., max_trade_days for downgap strategy
    :return: Information ratio
    NOTE:
    Information ratio = (mean return) / (standard deviation of return) * sqrt(252)
    """
    if name.upper() not in ['OVERSOLD', 'DOWNGAP']:
        raise ValueError(f"Name {name} not in ['oversold', 'downgap']")
    if name.upper() == 'OVERSOLD':
        trade_root = f'{TRADE_DIR}/oversold'
    if name.upper() == 'DOWNGAP':
        max_trade_days = kwargs.get('max_trade_days')
        if max_trade_days is None:
            raise ValueError("max_trade_days is required for downgap strategy")
        if not isinstance(max_trade_days, (int, float)):
            raise ValueError("max_trade_days must be an integer or float for downgap strategy")
        if int(max_trade_days) not in dataset_group_cons['common']['MAX_TRADE_DAYS_LIST']:
            raise ValueError(
                f"max_trade_days must be in {dataset_group_cons['common']['MAX_TRADE_DAYS_LIST']}"
            )
        max_trade_days = int(max_trade_days)
        trade_root = f'{TRADE_DIR}/downgap/max_trade_days_{max_trade_days}'
    indicator_csv = f'{trade_root}/statistic_indicator.csv'
    if not os.path.exists(indicator_csv):
        return 0.0
    indicator_df = pd.read_csv(indicator_csv, dtype={'trade_date': str})
    if start is not None:
        indicator_df = indicator_df[indicator_df['trade_date'] >= start]
    if end is not None:
        indicator_df = indicator_df[indicator_df['trade_date'] <= end]
    if indicator_df.empty:
        return 0.0
    indicator_df = indicator_df.sort_values(by='trade_date', ascending=True)
    indicator_df = indicator_df.reset_index(drop=True)
    return_mean = indicator_df['return_ratio'].mean()
    return_std = indicator_df['return_ratio'].std()
    if return_std == 0:
        return 0.0
    information_ratio = return_mean / return_std * (252 ** 0.5)  # 年化波动率
    return round(information_ratio, 4)

def calculate_sharpe_ratio(
        name: Literal['oversold', 'downgap'], rf: float, start=None, end=None, **kwargs
):
    """
    Calculate Sharpe ratio
    :param name: Name of the strategy or model, oversold or downgap
    :param rf: Risk-free rate, e.g., 0.03 for 3%
    :param start: 'YYYYMMDD' format, default is None (all data)
    :param end: 'YYYYMMDD' format, default is None (all data)
    :param kwargs: e.g., max_trade_days for downgap strategy
    :return: Sharpe ratio
    NOTE:
    Sharpe ratio = (mean return - risk-free rate) / (standard deviation of return) * sqrt(252)
    risk-free rate normally is equal to 10-year government bond yield
    """
    if name.upper() not in ['OVERSOLD', 'DOWNGAP']:
        raise ValueError(f"Name {name} not in ['oversold', 'downgap']")
    if name.upper() == 'OVERSOLD':
        trade_root = f'{TRADE_DIR}/oversold'
    if name.upper() == 'DOWNGAP':
        max_trade_days = kwargs.get('max_trade_days')
        if max_trade_days is None:
            raise ValueError("max_trade_days is required for downgap strategy")
        if not isinstance(max_trade_days, (int, float)):
            raise ValueError("max_trade_days must be an integer or float for downgap strategy")
        if int(max_trade_days) not in dataset_group_cons['common']['MAX_TRADE_DAYS_LIST']:
            raise ValueError(
                f"max_trade_days must be in {dataset_group_cons['common']['MAX_TRADE_DAYS_LIST']}"
            )
        max_trade_days = int(max_trade_days)
        trade_root = f'{TRADE_DIR}/downgap/max_trade_days_{max_trade_days}'
    indicator_csv = f'{trade_root}/statistic_indicator.csv'
    if not os.path.exists(indicator_csv):
        return 0.0
    indicator_df = pd.read_csv(indicator_csv, dtype={'trade_date': str})
    if start is not None:
        indicator_df = indicator_df[indicator_df['trade_date'] >= start]
    if end is not None:
        indicator_df = indicator_df[indicator_df['trade_date'] <= end]
    if indicator_df.empty:
        return 0.0
    indicator_df = indicator_df.sort_values(by='trade_date', ascending=True)
    indicator_df = indicator_df.reset_index(drop=True)
    return_mean = indicator_df['return_ratio'].mean()
    return_std = indicator_df['return_ratio'].std()
    if return_std == 0:
        return 0.0
    rf_daily = rf / 252  # 将年化无风险利率转换为日化
    sharpe_ratio = (return_mean - rf_daily) / return_std * (252 ** 0.5)  # 年化夏普比率
    return round(sharpe_ratio, 4)

### 缺口统计相关函数
def get_all_gaps_statistic_general_infomation() -> pd.DataFrame | None:
    """
    获取所有缺口的统计信息
    :return: DataFrame with statistics or None if no data
    NOTE:
    统计信息包括:
    - total_gaps: 总缺口数量
    - filled_gaps: 已回补的缺口数量
    - gaps_filled_rate: 全部缺口回补率
    - down_gaps: 向下缺口数量
    - filled_down_gaps: 已回补的向下缺口数量
    - down_gaps_filled_rate: 向下缺口回补率
    - up_gaps: 向上缺口数量
    - filled_up_gaps: 已回补的向上缺口数量
    - up_gaps_filled_rate: 向上缺口回补率
    返回 DataFrame 格式如下(20250930数据):
    | total_gaps | filled_gaps | gaps_filled_rate  | down_gaps | filled_down_gaps  | down_gaps_filled_rate | up_gaps | filled_up_gaps | up_gaps_filled_rate |
    |------------|-------------|-------------------|-----------|-------------------|-----------------------|---------|----------------|---------------------|
    | 530874     | 507351      | 0.9557            | 258177    | 250720            | 0.9711                | 272697  | 256631         | 0.9411              |
    """
    max_trade_days = dataset_group_cons['common'].get('MAX_TRADE_DAYS_LIST')
    if max_trade_days is None:
        return
    max_trade_days = max(max_trade_days)
    all_gaps_csv = f'{TEMP_DIR}/downgap/max_trade_days_{int(max_trade_days)}/all_gap_data.csv'
    if not os.path.exists(all_gaps_csv):
        return
    all_gaps_df = pd.read_csv(all_gaps_csv, dtype={'trade_date': str})
    if all_gaps_df.empty:
        return
    result = {}
    result['total_gaps'] = len(all_gaps_df)
    filled_gaps = all_gaps_df[all_gaps_df['fill_date'].notna()]
    result['filled_gaps'] = len(filled_gaps)
    result['gaps_filled_rate'] = round(result['filled_gaps'] / result['total_gaps'], 4)
    down_gaps = all_gaps_df[all_gaps_df['gap'] == 'down']
    result['down_gaps'] = len(down_gaps)
    filled_down_gaps = down_gaps[down_gaps['fill_date'].notna()]
    result['filled_down_gaps'] = len(filled_down_gaps)
    result['down_gaps_filled_rate'] = round(result['filled_down_gaps'] / result['down_gaps'], 4)
    up_gaps = all_gaps_df[all_gaps_df['gap'] == 'up']
    result['up_gaps'] = len(up_gaps)
    filled_up_gaps = up_gaps[up_gaps['fill_date'].notna()]
    result['filled_up_gaps'] = len(filled_up_gaps)
    result['up_gaps_filled_rate'] = round(result['filled_up_gaps'] / result['up_gaps'], 4)
    result_df = pd.DataFrame([result])
    return result_df

def get_gaps_earning_to_days_probability(
        trade_days: int, gap: Literal['up', 'down'] = 'down',
        rate0: float = 0.06, step: float = 0.02, times: int = 3
) -> pd.DataFrame | None:
    """
    计算指定类型的缺口在小于 trade_days 时，涨幅大于 rate0序列的概率
    : param trade_days: 交易天数
    : param gap: 缺口类型
    : param rate0: 起始涨幅
    : param step: 涨幅步长
    : param times: 涨幅步长次数
    : return: 概率
    NOTE:
    例如: trade_days=60, gap='down', rate0=0.10, step=0.02, times=5
    计算在所有向下缺口中, 60天内涨幅大于 10%、12%、14%、16%、18%的概率
    返回 DataFrame 格式如下(20250930数据):
    | trade_days | gap  | rate | count  | total  | probability |
    |------------|------|------|--------|--------|-------------|
    | 60         | down | 0.10 | 101261 | 211534 | 0.4789      |
    | 60         | down | 0.12 |  81220 | 211534 | 0.3842      |
    | 60         | down | 0.14 |  65145 | 211534 | 0.3079      |
    | 60         | down | 0.16 |  52436 | 211534 | 0.2479      |
    | 60         | down | 0.18 |  42149 | 211534 | 0.1992      |
    """
    max_trade_days = dataset_group_cons['common'].get('MAX_TRADE_DAYS_LIST')
    if max_trade_days is None:
        return
    max_trade_days = max(max_trade_days)
    all_gaps_csv = f'{TEMP_DIR}/downgap/max_trade_days_{int(max_trade_days)}/all_gap_data.csv'
    if not os.path.exists(all_gaps_csv):
        return
    all_gaps_df = pd.read_csv(all_gaps_csv, dtype={'trade_date': str})
    if all_gaps_df.empty:
        return
    if gap not in ['up', 'down']:
        return
    gap_df = all_gaps_df[all_gaps_df['gap'] == gap]
    if gap_df.empty:
        return
    gap_df_0 = gap_df[gap_df['days'] <= trade_days]
    if gap_df_0.empty:
        return
    # 返回 df 为多行格式，每一个涨幅对应一行
    results = []
    for i in range(times):
        rate = rate0 + i * step
        gap_df_1 = gap_df_0[gap_df_0['rise_percent'] >= rate]
        count_1 = len(gap_df_1)
        count_0 = len(gap_df_0)
        probability = round(count_1 / count_0, 4) if count_0 != 0 else 0.0
        results.append({
            'trade_days': trade_days,
            'gap': gap,
            'rate': round(rate, 4),
            'count': count_1,
            'total': count_0,
            'probability': probability
        })
    results_df = pd.DataFrame(results)
    return results_df

def get_gaps_agg_days_information_groupby_rate(
        gap: Literal['up', 'down'] = 'down', 
        rate0: float = 0.08, step: float = 0.01, times: int = 12
) -> pd.DataFrame | None:
    """
    按照对缺口的 rise_percent 分组后统计 days 的信息
    : param gap: 缺口类型
    : param rate0: 起始涨幅
    : param step: 涨幅步长
    : param times: 涨幅步长次数
    : return: DataFrame with grouped statistics or None if no data
    NOTE:
    - days 统计方法: count, mean, median, min, max, quantile(0.75), quantile(0.90)
    - 以下是rate0=0.05, step=0.05, times=5的输出, 返回 DataFrame 格式如下(20250930数据):
    | rise_percent_group | count_days | mean_days | median_days | quantile_75_days | quantile_90_days | min_days | max_days |
    |--------------------|------------|-----------|-------------|------------------|------------------|----------|----------|
    | (0.00, 0.05]       |      47497 |  2.078468 |         1.0 |              2.0 |              4.0 |      1.0 |     68.0 |
    | (0.05, 0.10]       |      63075 |  4.962901 |         3.0 |              6.0 |             11.0 |      1.0 |    146.0 |
    | (0.10, 0.15]       |      38469 | 10.174634 |         6.0 |             13.0 |             23.0 |      1.0 |    171.0 |
    | (0.15, 0.20]       |      21525 | 19.368595 |        13.0 |             26.0 |             44.0 |      1.0 |    395.0 |
    | (0.20, 0.25]       |      17415 | 25.453000 |        17.0 |             34.0 |             58.0 |      1.0 |    572.0 |
    | (0.25, ....]       |      62739 | 233.89735 |        81.0 |            234.0 |            612.0 |      1.0 |   4148.0 |    
    """
    max_trade_days = dataset_group_cons['common'].get('MAX_TRADE_DAYS_LIST')
    if max_trade_days is None:
        return
    max_trade_days = max(max_trade_days)
    all_gaps_csv = f'{TEMP_DIR}/downgap/max_trade_days_{int(max_trade_days)}/all_gap_data.csv'
    if not os.path.exists(all_gaps_csv):
        return
    all_gaps_df = pd.read_csv(all_gaps_csv, dtype={'trade_date': str})
    if all_gaps_df.empty:
        return
    if gap not in ['up', 'down']:
        return
    gap_df = all_gaps_df[all_gaps_df['gap'] == gap]
    if gap_df.empty:
        return
    gap_df = gap_df.copy()
    bins = [-float('inf')] + [round(rate0 + i * step, 4) for i in range(times)] + [float('inf')]
    labels = []
    for i in range(times + 1):
        if i == 0:
            labels.append(f'({0.00:.2f}, {bins[i+1]:.2f}]')
        elif i == times:
            labels.append(f'({bins[i]:.2f}, ....]')
        else:
            labels.append(f'({bins[i]:.2f}, {bins[i+1]:.2f}]')
    gap_df['rise_percent_group'] = pd.cut(gap_df['rise_percent'], bins=bins, labels=labels)
    grouped = gap_df.groupby('rise_percent_group')['days'].agg(
        ['count', 'mean', 'median', lambda x: x.quantile(0.75), lambda x: x.quantile(0.90), 'min', 'max']
    )
    new_columns = ['count_days', 'mean_days', 'median_days', 'quantile_75_days', 'quantile_90_days', 'min_days', 'max_days']
    grouped.columns = new_columns
    grouped = grouped.reset_index()
    return grouped

### 和指数相关统计函数
def calculate_correlation_between_portfolio_and_index(
        name: Literal['oversold', 'downgap'], index_code: str = '000001.SH',
        start: str = None, end: str = None, **kwargs
) -> float:
    """
    ### 计算投资组合和指定指数之间日收益率的相关系数
    #### :param name: 投资策略的名称: oversold 或者 downgap
    #### :param index_code: 指数代码, 例如 '000001.SH' 表示上证综合指数
    #### :param start: 'YYYYMMDD' 格式, 默认为 None (所有数据)
    #### :param end: 'YYYYMMDD' 格式, 默认为 None (所有数据)
    #### :param kwargs: 例如, downgap 策略的 max_trade_days(必须提供)
    #### :return: 相关系数
    #### NOTE:
    #### 计算投资组合的每日收益率与指定指数的每日收益率之间的相关系数
    """
    if name.upper() not in ['OVERSOLD', 'DOWNGAP']:
        raise ValueError(f"Name {name} not in ['oversold', 'downgap']")
    if name.upper() == 'OVERSOLD':
        trade_root = f'{TRADE_DIR}/oversold'
    if name.upper() == 'DOWNGAP':
        max_trade_days = kwargs.get('max_trade_days')
        if max_trade_days is None:
            raise ValueError("max_trade_days is required for downgap strategy")
        if not isinstance(max_trade_days, (int, float)):
            raise ValueError("max_trade_days must be an integer or float for downgap strategy")
        if int(max_trade_days) not in dataset_group_cons['common']['MAX_TRADE_DAYS_LIST']:
            raise ValueError(
                f"max_trade_days must be in {dataset_group_cons['common']['MAX_TRADE_DAYS_LIST']}"
            )
        max_trade_days = int(max_trade_days)
        trade_root = f'{TRADE_DIR}/downgap/max_trade_days_{max_trade_days}'
    indicator_csv = f'{trade_root}/statistic_indicator.csv'
    if not os.path.exists(indicator_csv):
        return 0.0
    indicator_df = pd.read_csv(indicator_csv, dtype={'trade_date': str})
    if start is None:
        start = indicator_df['trade_date'].min()
    if end is None:
        end = indicator_df['trade_date'].max()
    indicator_df = indicator_df[(indicator_df['trade_date'] >= start) & (indicator_df['trade_date'] <= end)]
    if indicator_df.empty:
        return 0.0
    indicator_df = indicator_df[['trade_date', 'return_ratio']]
    indicator_df = indicator_df.sort_values(by='trade_date', ascending=True)
    indicator_df = indicator_df.reset_index(drop=True)
    # 下载指数数据
    pro = ts.pro_api()
    index_df = pro.index_daily(ts_code=index_code, start_date=start, end_date=end)
    if index_df.empty:
        return 0.0
    index_df = index_df[['trade_date', 'pct_chg']]
    index_df = index_df.sort_values(by='trade_date', ascending=True)
    index_df = index_df.reset_index(drop=True)
    # 合并数据
    merged_df = pd.merge(indicator_df, index_df, on='trade_date', how='inner')
    if merged_df.empty:
        return 0.0
    correlation = merged_df['return_ratio'].corr(merged_df['pct_chg'])
    return round(correlation, 4)