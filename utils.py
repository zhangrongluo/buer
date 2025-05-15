import time
import pandas as pd
import datetime
import requests
import tushare as ts
from DrissionPage import ChromiumOptions, Chromium
from cons_general import TRADE_CAL_XLS
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
