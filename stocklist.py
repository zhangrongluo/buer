"""
股票列表、交易日历等基本信息
"""
import datetime
import tushare as ts 
import pandas as pd
from cons_general import STOCK_LIST_XLS, TRADE_CAL_XLS, UP_DOWN_LIMIT_XLS, SUSPEND_STOCK_XLS

__all__ = ['get_name_and_industry_by_code', 'get_all_stocks_info', 'STOCK_LIST_NUMS', 'LIST_DF', 'pro']

pro = ts.pro_api()
LIST_DF = None

def get_stock_list():
    """
    get stock list and save to the basicdata dir
    """
    stocklist_df = pro.stock_basic(exchange='', list_status='L')
    stocklist_df.to_excel(STOCK_LIST_XLS, index=False)

get_stock_list()

def get_trade_cal():
    """
    get trade calendar and save to the basicdata dir
    NOTE: coverage 90 days from today
    """
    today = datetime.date.today()
    start_date = (today - datetime.timedelta(days=90)).strftime('%Y%m%d')
    end_date = today.strftime('%Y%m%d')
    trade_cal_df = pro.trade_cal(exchange='', start_date=start_date, end_date=end_date)
    # trade_cal_df = trade_cal_df[trade_cal_df['is_open'] == 1]
    trade_cal_df = trade_cal_df[['cal_date', 'is_open']]
    trade_cal_df.to_excel(TRADE_CAL_XLS, index=False)

get_trade_cal()  # build trade calendar

def load_list_df():
    """ 
    load the latest stock list DataFrame from STOCK_LIST_XLS
    and filter out the suffix for Shanghai and Shenzhen stocks.
    """
    global LIST_DF
    suffix = ['.sz', '.sh', '.SZ', '.SH']
    LIST_DF = pd.read_excel(STOCK_LIST_XLS, dtype=str)
    LIST_DF = LIST_DF[LIST_DF['ts_code'].str.contains('|'.join(suffix))]
    total_stocks = LIST_DF.shape[0]
    return total_stocks

load_list_df()

def get_name_and_industry_by_code(ts_code: str) -> str|None:
    """
    get stock name by ts_code in the LIST_DF
    :params ts_code: 600036.SH or 600036
    :return: stock name or None(if not found)
    """
    if len(ts_code) == 6:
        ts_code = ts_code + '.SH' if ts_code[0] == '6' else ts_code + '.SZ'
    if ts_code not in LIST_DF['ts_code'].values:
        return None
    return LIST_DF[LIST_DF['ts_code'] == ts_code][['name', 'industry']].values[0].tolist()

def get_all_stocks_info() -> list:
    """
    get all stock codes and names in the LIST_DF
    :return: list of stock code、name、industry and cnspell
    """
    return LIST_DF[['ts_code', 'name', 'industry', 'cnspell']].values.tolist()

def get_all_stock_industry() -> list:
    """
    get all stock industry in the LIST_DF
    :return: list of stock industry
    """
    res =  LIST_DF['industry'].unique().tolist()
    res = [item for item in res if isinstance(item, str)]
    return res

def get_up_down_limit_list():
    """
    get today up down limit list and save to the basicdata dir
    """
    today = datetime.date.today().strftime('%Y%m%d')
    up_down_limit_df = pro.stk_limit(trade_date=today)
    up_down_limit_df = up_down_limit_df.dropna(how='any')
    up_down_limit_df.to_excel(UP_DOWN_LIMIT_XLS, index=False)

def get_suspend_stock_list():
    """
    get today suspend stock list and save to the basicdata dir
    """
    today = datetime.date.today().strftime('%Y%m%d')
    suspend_df = pro.suspend_d(suspend_type='S', trade_date=today)
    suspend_df.to_excel(SUSPEND_STOCK_XLS, index=False)

if __name__ == '__main__':
    # test
    print(get_name_and_industry_by_code('600036'))
    print(get_name_and_industry_by_code('000001.SZ'))
    print(get_all_stocks_info()[5:10])
    print(get_all_stock_industry())
    print(len(get_all_stock_industry()))
    print(LIST_DF.shape)