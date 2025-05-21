import os
import re
import pandas as pd
import subprocess
import datetime
import logging
from threading import Lock
from stocklist import get_name_and_industry_by_code
from concurrent.futures import ThreadPoolExecutor
from cons_oversold import (initial_funds, COST_FEE, MIN_STOCK_PRICE, ONE_TIME_FUNDS, MAX_STOCKS, 
                           PRED_RATE_PCT, MIN_PRED_RATE, MIN_WAITING_DAYS, MAX_TRADE_DAYS, MAX_DOWN_LIMIT,
                           REST_TRADE_DAYS, WAITING_RATE_PCT, MODEL_NAME, BUY_IN_LIST, HOLDING_LIST,
                            DAILY_PROFIT, FUNDS_LIST, TRADE_LOG)
from cons_general import BACKUP_DIR, TRADE_DIR, BASICDATA_DIR
from cons_hidden import bark_device_key
from utils import send_wechat_message_via_bark, get_stock_realtime_price, is_trade_date_or_not


backup_dir = f'{BACKUP_DIR}/oversold'
os.makedirs(backup_dir, exist_ok=True)
trade_dir = f'{TRADE_DIR}/oversold'
os.makedirs(trade_dir, exist_ok=True)
lock = Lock()
# config the trade log
trade_log = logging.getLogger(name='trade_log')
trade_log.setLevel(logging.INFO)
trade_handle = logging.FileHandler(filename=TRADE_LOG, mode='a')
trade_handle.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
trade_log.addHandler(trade_handle)

def _get_holding_stocks_number() -> int:
    """
    get holding stocks number and set signal to
    control the number of holding stocks
    NOTE:
    signal is HOLDING_STOCKS, signal += 1 when buy in, signal -= 1 when sell out
    """
    if not os.path.exists(HOLDING_LIST):
        return 0
    holding_df = pd.read_csv(HOLDING_LIST)
    holding_stocks = holding_df[holding_df['status'] == 'holding'].shape[0]
    return holding_stocks
HOLDING_STOCKS = _get_holding_stocks_number()
print(f'({MODEL_NAME}) HOLDING_STOCKS is {HOLDING_STOCKS} now')

def copy_holding_list_to_backup_root():
    """
    copy holding list to backup to avoid data loss
    """
    dest_holding_list = f'{backup_dir}/holding_list.csv'
    if not os.path.exists(HOLDING_LIST):
        return
    if not os.path.exists(dest_holding_list):
        subprocess.run(['cp', HOLDING_LIST, backup_dir])
    else:  # 如果src行数大于等于dest行数，则拷贝
        src_df = pd.read_csv(HOLDING_LIST)  # lock or unlock?
        dest_df = pd.read_csv(dest_holding_list)
        if src_df.shape[0] >= dest_df.shape[0]:
            subprocess.run(['cp', HOLDING_LIST, backup_dir])

# buy in and sell out
def buy_in(code: str, price: float, amount: int, trade_date: str, buy_point_base: float, target_rate: float) -> None:
    """
    buy in stock
    :param code: stock code, like 000001 or 000001.SH
    :param price: stock price
    :param amount: stock amount
    :param trade_date: trade date link to row, like '20210804'
    :param buy_point_base: target price link to row
    :param target_rate: target rate link to row
    NOTE: 
    the last second parameters are used to carry more information for holding_list.csv,
    but it is not necessary to use them in the function.
    """
    if len(code) != 9:
        code = code +'.SH' if code.startswith('6') else code + '.SZ'
    msg = get_name_and_industry_by_code(code)
    stock_name = msg[0]
    pattern = re.compile(r'[*]*[sS][tT]|退市|退|[pP][Tt]|[xX][rR]')  # 例外股票不投
    if pattern.findall(stock_name):
        return
    industry = msg[1]
    trade_date = trade_date
    date_in = datetime.datetime.now().strftime('%Y%m%d')
    date_out = ''  # 避免日期型数据被转换为数值型
    days = None
    holding_days = 1
    buy_point_base = buy_point_base
    target_rate = target_rate
    price_in = price
    price_out = None
    price_now = price
    amount = amount
    cost_fee = COST_FEE
    profit = None
    rate_pred = MIN_PRED_RATE
    rate_pct = PRED_RATE_PCT
    rate_current = None
    rate_yearly = None
    status = 'holding'
    row_data = [code, stock_name, industry, trade_date, date_in, date_out, days, holding_days, buy_point_base, target_rate, price_in, 
                price_out, price_now, amount, cost_fee, profit, rate_pred, rate_pct, rate_current, rate_yearly, status]
    column = ['ts_code', 'stock_name', 'industry', 'trade_date', 'date_in', 'date_out', 'days', 'holding_days', 'buy_point_base', 
              'target_rate', 'price_in', 'price_out', 'price_now', 'amount', 'cost_fee', 'profit', 'rate_pred', 'rate_pct', 
              'rate_current', 'rate_yearly', 'status']
    new_row = pd.DataFrame([row_data, ], columns=column)
    # adjust stock numbers signal and cash amount
    global HOLDING_STOCKS
    if HOLDING_STOCKS >= MAX_STOCKS:
        return
    HOLDING_STOCKS += 1
    cash_amount_buy = round(price * (1 + cost_fee) * amount, 2)
    note = f'买入 {code} {stock_name} at {price} total {amount}'
    res = create_or_update_funds_change_list(-cash_amount_buy, note)
    if not res:  # cash balance is not enough
        HOLDING_STOCKS -= 1
        return
    new_row.to_csv(HOLDING_LIST, mode='a', header=False, index=False)
    now = datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')
    trade_log.info(f'买入 {code} {stock_name} {industry} at {price} total {amount} at {now}')
    os.system(f'afplay /System/Library/Sounds/Ping.aiff')  # play sound to remind buy in
    os.system(f'afplay /System/Library/Sounds/Ping.aiff')  # play sound second times
    os.system(f'afplay /System/Library/Sounds/Ping.aiff')  # play sound third times
    title = '买入股票::OverSold'
    message = f'{stock_name}-{code}-买入价:{price}元-买入数量:{amount}股-{now}'
    send_wechat_message_via_bark(bark_device_key, title, message)

def sell_out(code: str, price: float, trade_date: str) -> None:
    """
    sell out stock
    :param code: stock code, like 000001 or 000001.SH
    :param price: stock price
    :param amount: stock amount
    :param trade_date: trade date link to row, like '20210804'
    """
    if len(code) != 9:
        code = code +'.SH' if code.startswith('6') else code + '.SZ'
    if not os.path.exists(HOLDING_LIST):
        return
    holding_df = pd.read_csv(HOLDING_LIST, dtype={'trade_date': str, 'date_out': str})
    # no stock to sell out, return  
    to_sell_row = holding_df[(holding_df['ts_code'] == code) & (holding_df['trade_date'] == trade_date)]
    if to_sell_row.empty:
        return
    # update date_out, price_out, price_now, profit, rate_current, rate_yearly, status
    stock_name = to_sell_row['stock_name'].values[0]
    industry = to_sell_row['industry'].values[0]
    today = datetime.datetime.now().strftime('%Y%m%d')
    holding_df.loc[
        (holding_df['ts_code'] == code) & (holding_df['trade_date'] == trade_date), 'date_out'
    ] = today
    holding_df.loc[
        (holding_df['ts_code'] == code) & (holding_df['trade_date'] == trade_date), 'price_out'
    ] = price
    holding_df.loc[
        (holding_df['ts_code'] == code) & (holding_df['trade_date'] == trade_date), 'price_now'
    ] = None
    price_in = to_sell_row['price_in'].values[0]
    amount = to_sell_row['amount'].values[0]
    cost_fee = to_sell_row['cost_fee'].values[0]
    profit = (price*(1-cost_fee) - price_in*(1+cost_fee)) * amount
    profit = round(profit, 4)
    holding_df.loc[
        (holding_df['ts_code'] == code) & (holding_df['trade_date'] == trade_date), 'profit'
    ] = profit
    rate_current = profit / (price_in * amount)
    rate_current = round(rate_current, 4)
    holding_df.loc[
        (holding_df['ts_code'] == code) & (holding_df['trade_date'] == trade_date), 'rate_current'
    ] = rate_current
    rate_yearly = rate_current * 365 / to_sell_row['holding_days'].values[0]  # 自然日历年化收益率
    rate_yearly = round(rate_yearly, 4)
    holding_df.loc[
        (holding_df['ts_code'] == code) & (holding_df['trade_date'] == trade_date), 'rate_yearly'
    ] = rate_yearly
    holding_df.loc[
        (holding_df['ts_code'] == code) & (holding_df['trade_date'] == trade_date), 'status'
    ] = 'sold_out'
    # adjust the stock numbers signal
    cash_amount_sell = round(price * (1 - cost_fee) * amount, 2)
    note = f'卖出 {code} {stock_name} at {price} total {amount}'
    create_or_update_funds_change_list(cash_amount_sell, note)
    global HOLDING_STOCKS
    if HOLDING_STOCKS <= 0:
        return
    HOLDING_STOCKS -= 1
    holding_df.to_csv(HOLDING_LIST, index=False)
    now = datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')
    trade_log.info(f'卖出 {code} {stock_name} {industry} at {price} at {now}')
    trade_log.info(f'profit: {profit:.2f}, rate_current: {rate_current:.2%}, rate_yearly: {rate_yearly:.2%}')
    os.system(f'afplay /System/Library/Sounds/Hero.aiff')  # play sound to remind sell out
    os.system(f'afplay /System/Library/Sounds/Hero.aiff')  # play sound second times
    os.system(f'afplay /System/Library/Sounds/Hero.aiff')  # play sound third times
    title = '卖出股票::OverSold'
    message = f'{stock_name}-{code}-卖出价:{price}元-卖出数量:{amount}股-{now}'
    send_wechat_message_via_bark(bark_device_key, title, message)

def calculate_buy_in_amount(funds, price) -> int | None:
    """
    calculate buy_in amount
    :param funds: funds to buy in
    :param price: stock price
    :return: buy_in amount
    """
    amount = int(funds*(1-COST_FEE) / price)
    amount = amount // 100 * 100
    return amount

def calculate_cash_and_stock_total_value() -> tuple:
    """
    calculate cash and stock total value
    :return: cash amount, stock total value, total value
    """
    if not os.path.exists(HOLDING_LIST):
        return
    with lock:
        holding_df = pd.read_csv(HOLDING_LIST)
        cash_row = holding_df[holding_df['ts_code'] == 'cash_ini']
        cash_amount = round(cash_row['amount'].values[0], 2)
        holding_df = holding_df[holding_df['ts_code'] != 'cash_ini']
        holding_df = holding_df[holding_df['status'] == 'holding']
        holding_df = holding_df.copy()
        holding_df['total_value'] = holding_df['price_now'] * holding_df['amount']
        stock_total_value = round(holding_df['total_value'].sum(), 2)
    return cash_amount, stock_total_value, cash_amount + stock_total_value

def create_daily_profit_list():
    """
    create and update daily profit list
    NOTE: 
    contains columns: trade_date, profit, delta
    """
    if not is_trade_date_or_not():
        return
    if not os.path.exists(HOLDING_LIST):
        return
    today = datetime.datetime.now().date().strftime('%Y%m%d')
    with lock:
        holding_df = pd.read_csv(HOLDING_LIST, dtype={'trade_date': str})
        total_profit = round(holding_df['profit'].sum(), 2)
    if not os.path.exists(DAILY_PROFIT):
        fisrt_row = pd.DataFrame([[today, total_profit, total_profit]], columns=['trade_date', 'profit', 'delta'])
        fisrt_row.to_csv(DAILY_PROFIT, index=False)
        subprocess.run(['cp', DAILY_PROFIT, backup_dir])
        return
    profit_df = pd.read_csv(DAILY_PROFIT, dtype={'trade_date': str})
    profit_df = profit_df.sort_values(by='trade_date', ascending=True)
    last_row = profit_df.iloc[-1]
    if last_row['trade_date'] != today:
        new_row = pd.DataFrame([[today, total_profit, None], ], columns=['trade_date', 'profit', 'delta'])
        profit_df = pd.concat([profit_df, new_row], ignore_index=True)
    if profit_df.shape[0] == 1:
        profit_df.loc[profit_df['trade_date'] == today, 'profit'] = total_profit
        profit_df.loc[profit_df['trade_date'] == today, 'delta'] = total_profit
        profit_df.to_csv(DAILY_PROFIT, index=False)
        subprocess.run(['cp', DAILY_PROFIT, backup_dir])
        return
    reverse_second_profit = profit_df.iloc[-2]['profit']
    delta = round(total_profit-reverse_second_profit, 2)
    profit_df.loc[profit_df['trade_date'] == today, 'profit'] = total_profit
    profit_df.loc[profit_df['trade_date'] == today, 'delta'] = delta
    profit_df = profit_df.sort_values(by='trade_date', ascending=True)
    profit_df.to_csv(DAILY_PROFIT, index=False)
    # copy daily_profit.csv to backup
    subprocess.run(['cp', DAILY_PROFIT, backup_dir])

def create_or_update_funds_change_list(funds: float, note: str) -> bool:
    """
    create or refresh funds change list
    :param funds: funds to add or reduce
    :param note: note for funds change
    :return: True if success, False if failed
    NOTE: contains columns: datetime, amount, balance, note
    """
    if not os.path.exists(FUNDS_LIST):
        now = datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')
        balance = funds
        first_row = pd.DataFrame([[now, funds, balance, note]], columns=['datetime', 'amount', 'balance', 'note'])
        first_row.to_csv(FUNDS_LIST, index=False)
        return True
    funds_change_df = pd.read_csv(FUNDS_LIST, dtype={'datetime': str})
    now = datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')
    old_balence = funds_change_df.iloc[-1]['balance']
    new_balence = old_balence + funds
    if new_balence < 0:
        print('可用资金不足, 无法完成交易')
        return False
    new_row = pd.DataFrame([[now, funds, new_balence, note]], columns=['datetime', 'amount', 'balance', 'note'])
    funds_change_df = pd.concat([funds_change_df, new_row], ignore_index=True)
    funds_change_df.to_csv(FUNDS_LIST, index=False)
    return True

def create_holding_list(initial_cash: float = initial_funds):
    """
    build holding list
    :param initial_cash: initial cash
    NOTE: contains columns:
    ts_code, stock_name, industry, trade_date, date_in, date_out, days(trade days), holding_days(calender days), 
    buy_point_base, target_rate, price_in, price_out, amount, rate_current, rate_yearly, status
    """
    columns = [
        'ts_code', 'stock_name', 'industry', 'trade_date', 'date_in', 'date_out', 'days', 'holding_days','buy_point_base', 
        'target_rate', 'price_in', 'price_out', 'price_now', 'amount', 'cost_fee', 'profit', 'rate_pred', 'rate_pct', 
        'rate_current', 'rate_yearly', 'status'
    ]
    holding_df = pd.DataFrame(columns=columns)
    holding_df.to_csv(HOLDING_LIST, index=False)
    # create funds change list
    create_or_update_funds_change_list(initial_cash, '初始资金')

# 持续扫描buy_in_list.csv, 买入股票
def scan_buy_in_list():
    if not os.path.exists(BUY_IN_LIST):
        return
    if not os.path.exists(HOLDING_LIST):
        create_holding_list()
    with lock:
        buy_in_df = pd.read_csv(BUY_IN_LIST, dtype={'trade_date': str})
        holding_df = pd.read_csv(HOLDING_LIST, dtype={'trade_date': str})
    # select the last day's data of a down trend
    buy_in_df = buy_in_df.sort_values(by=['ts_code', 'trade_date'], ascending=[True, True])
    buy_in_df = buy_in_df.drop_duplicates(subset='ts_code', keep='last')
    # delete rows in holding_df with same ts_code and status is holding from buy_in_df
    rows_not_holding = []
    for i, row in buy_in_df.iterrows():
        code = row['ts_code']
        if holding_df[(holding_df['ts_code'] == code) & (holding_df['status'] == 'holding')].empty:
            rows_not_holding.append(row)
    buy_in_df = pd.DataFrame(rows_not_holding)
    buy_in_df = buy_in_df.sort_values(by='trade_date', ascending=True)
    buy_in_df = buy_in_df.reset_index(drop=True)

    def get_max_price_between_today_and_trade_date(idx_row):
        i, row = idx_row
        code = row['ts_code']
        trade_date = row['trade_date']
        daily_csv = f'{BASICDATA_DIR}/dailydata/{code}.csv'
        with lock:
            daily_df = pd.read_csv(daily_csv, dtype={'trade_date': str})
        daily_df = daily_df.sort_values(by='trade_date', ascending=True)
        trade_date_list = daily_df['trade_date'].tolist()
        today = datetime.datetime.now().date().strftime('%Y%m%d')
        if today in trade_date_list:
            today_index = trade_date_list.index(today)
        else:
            today_index = len(trade_date_list) - 1
        trade_date_index = trade_date_list.index(trade_date)
        price_max = daily_df.loc[trade_date_index:today_index, 'high'].max()
        return price_max

    def scan_buy_in_list_row(idx_row):
        # if not in trading time 9:35-11:30, 13:00-14:55, skip
        now = datetime.datetime.now().time()
        am_begin = datetime.time(9, 35)
        am_end = datetime.time(11, 30)
        pm_begin = datetime.time(13, 0)
        pm_end = datetime.time(14, 55)
        if not (am_begin <= now <= am_end or pm_begin <= now <= pm_end):
            return
        i, row = idx_row
        code = row['ts_code']
        name = row['stock_name']
        trade_date = row['trade_date']
        buy_point_base = row['buy_point_base']
        target_rate = row['pred']  # 预期收益率
        waiting_days = row['waiting_days']
        price_now = get_stock_realtime_price(code)
        print(f'({MODEL_NAME}) {row['ts_code']} {row['stock_name']} price_now: {price_now}')
        if price_now is None:
            return
        if price_now >= buy_point_base:
            return
        if price_now <= MIN_STOCK_PRICE:
            return
        # wait for the down trend to end
        if waiting_days <= MIN_WAITING_DAYS:  
            return
        # avoid too short holding days after buying in
        src = row['src']
        backward_days = int(src.split('_')[-2])
        if backward_days - waiting_days <= REST_TRADE_DAYS:  
            return
        # if max price between today and trade_date >= buy_point_base * (1+target_rate*WAITING_RATE_PCT), skip
        # in other words, the stock has reached the WAITING_RATE_PCT target rate in the waiting days
        price_max = get_max_price_between_today_and_trade_date(idx_row)
        if price_max >= buy_point_base * (1 + target_rate * WAITING_RATE_PCT):
            return
        # if holding_stocks >= MAX_STOCKS, skip
        # holding_stocks = holding_df[holding_df['status'] == 'holding'].shape[0]
        global HOLDING_STOCKS
        if HOLDING_STOCKS >= MAX_STOCKS:
            return
        amount = calculate_buy_in_amount(funds=ONE_TIME_FUNDS, price=price_now)
        if amount == 0:
            return
        with lock:
            buy_in(code, price_now, amount, trade_date, buy_point_base, target_rate)

    # 多线程扫描buy_in_list.csv
    idx_rows = list(buy_in_df.iterrows())
    with ThreadPoolExecutor() as executor:
        executor.map(scan_buy_in_list_row, idx_rows)

# 持续刷新holding_list.csv
def refresh_holding_list():
    """ 
    refresh columns: days, holding_days, price_now, profit, rate_current, rate_yearly
    """
    if not os.path.exists(HOLDING_LIST):
        create_holding_list()
        return
    with lock:
        holding_df = pd.read_csv(
            HOLDING_LIST, dtype={'trade_date': str, 'date_in': str}
        )
        holding_df['date_in'] = holding_df['date_in'].apply(lambda x: str(x)[:8])
        holding_df['date_in'] = holding_df['date_in'].apply(lambda x: x if x != 'nan' else '')

    def refresh_holding_list_row(idx_row):
        """
        refresh holding list single row infomation
        :param idx_row: index, row
        """
        i, row = idx_row
        trade_date = row['trade_date']
        # if status is sold_out, skip
        if row['status'] == 'sold_out':
            return
        # if status is holding, update holding_days, days, price_now, profit, rate_current, rate_yearly
        # holding_days(自然日历间隔天数)
        date_in = datetime.datetime.strptime(row['date_in'], '%Y%m%d')  # date format problem
        today = datetime.datetime.now()
        holding_days = (today - date_in).days + 1
        holding_df.loc[i, 'holding_days'] = holding_days
        # days(交易日历间隔天数today - trade_date)
        daily_csv = f'{BASICDATA_DIR}/dailydata/{row["ts_code"]}.csv'
        daily_df = pd.read_csv(daily_csv, dtype={'trade_date': str})
        daily_df = daily_df.sort_values(by='trade_date', ascending=True)
        trade_date_list = daily_df['trade_date'].tolist()
        today_str = today.strftime('%Y%m%d')
        trade_date_index = trade_date_list.index(trade_date)
        if today_str in trade_date_list:
            today_index = trade_date_list.index(today_str)
            days = abs(today_index - trade_date_index) 
        else:
            today_index = len(trade_date_list) - 1
            days = abs(today_index - trade_date_index) + 1  # today还未收盘，未在daily_df中，所以+1
        holding_df.loc[i, 'days'] = days
        # price_now, profit, rate_current, rate_yearly
        price_now = get_stock_realtime_price(row['ts_code'])
        print(f'({MODEL_NAME}) {row['ts_code']} {row['stock_name']} price_now: {price_now}')
        if price_now is None:
            with lock:
                holding_df.to_csv(HOLDING_LIST, index=False)
            return
        price_in = row['price_in']
        amount = row['amount']  # 股数量
        cost_fee = row['cost_fee']
        holding_df.loc[i, 'price_now'] = price_now
        profit = (price_now*(1-cost_fee) - price_in*(1+cost_fee)) * amount
        profit = round(profit, 4)
        holding_df.loc[i, 'profit'] = profit
        rate_current = profit / (price_in * amount)
        rate_current = round(rate_current, 4)
        holding_df.loc[i, 'rate_current'] = rate_current
        rate_yearly = rate_current * 365 / holding_days  # 自然日历年化收益率
        rate_yearly = round(rate_yearly, 4)
        holding_df.loc[i, 'rate_yearly'] = rate_yearly
        with lock:
            holding_df.to_csv(HOLDING_LIST, index=False)
    
    # 多线程刷新holding_list.csv
    idx_rows = list(holding_df.iterrows())
    with ThreadPoolExecutor() as executor:
        executor.map(refresh_holding_list_row, idx_rows)

# 持续扫描holding_list.csv, 卖出股票
def scan_holding_list():
    """
    scan holding list, sell out stocks
    """
    if not os.path.exists(HOLDING_LIST):
        create_holding_list()
        return
    with lock:
        holding_df = pd.read_csv(HOLDING_LIST, dtype={'trade_date': str, 'date_out': str})
        holding_df['date_out'] = holding_df['date_out'].apply(lambda x: str(x)[:8])
        holding_df['date_out'] = holding_df['date_out'].apply(lambda x: x if x != 'nan' else '')
    
    def scan_holding_list_row(idx_row):
        # if not in trading time 9:35-11:30, 13:00-14:55, skip
        now = datetime.datetime.now().time()
        am_begin = datetime.time(9, 35)
        am_end = datetime.time(11, 30)
        pm_begin = datetime.time(13, 0)
        pm_end = datetime.time(14, 55)
        if not (am_begin <= now <= am_end or pm_begin <= now <= pm_end):
            return
        i, row = idx_row
        if row['date_out'] != '':
            return
        if row['status'] == 'sold_out':
            return
        holding_days = row['holding_days']
        if holding_days == 1:
            return
        price_now = get_stock_realtime_price(row['ts_code'])
        print(f'({MODEL_NAME}) {row['ts_code']} {row['stock_name']} price_now: {price_now}')
        if price_now is None:
            return
        # if days > MAX_TRADE_DAYS, sell out
        days = row['days']
        if days >= MAX_TRADE_DAYS:
            with lock:
                sell_out(row['ts_code'], price_now, row['trade_date'])
        # if reach the target rate, sell out
        base_point = row['buy_point_base']
        target_rate = row['target_rate']
        if price_now >= base_point * (1 + target_rate):
            with lock:
                sell_out(row['ts_code'], price_now, row['trade_date'])
        # if rate_current <= MAX_DOWN_LIMIT, sell out
        rate_current = row['rate_current']
        if rate_current <= MAX_DOWN_LIMIT:
            with lock:
                sell_out(row['ts_code'], price_now, row['trade_date'])
        # if rate_yearly > target_rate_yearly, sell out in advance
        # 30 >= holding_days > 15, rate_yearly >= 4.5, sell out
        # 60 >= holding_days > 30, rate_yearly >= 3.0, sell out
        # 90 >= holding_days > 60, rate_yearly >= 2.4, sell out
        rate_yearly = row['rate_yearly']
        if 30 >= holding_days > 15 and rate_yearly >= 4.5:
            with lock:
                sell_out(row['ts_code'], price_now, row['trade_date'])
        if 60 >= holding_days > 30 and rate_yearly >= 3.0:
            with lock:
                sell_out(row['ts_code'], price_now, row['trade_date'])
        if 90 >= holding_days > 60 and rate_yearly >= 2.4:
            with lock:
                sell_out(row['ts_code'], price_now, row['trade_date'])
    
    # 多线程扫描holding_list.csv
    idx_rows = list(holding_df.iterrows())
    with ThreadPoolExecutor() as executor:
        executor.map(scan_holding_list_row, idx_rows)

def trade_process():
    """
    trade period: buy_in sell_out refresh and backup
    NOTE: 
    buy_in_list.csv -> holding_list.csv -> daily_profit.csv
    """
    scan_buy_in_list()
    refresh_holding_list()
    scan_holding_list()
    create_daily_profit_list()
    copy_holding_list_to_backup_root()
