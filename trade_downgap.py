import os
import re
import pandas as pd
import datetime
import logging
from typing import Literal
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from cons_general import DATASETS_DIR, BASICDATA_DIR, TRADE_DIR
from cons_downgap import dataset_group_cons
from cons_hidden import bark_device_key
from utils import (send_wechat_message_via_bark, get_stock_realtime_price, is_trade_date_or_not,
                   get_up_down_limit, is_decreasing_or_not, is_rising_or_not, is_suspended_or_not,
                   get_qfq_price_by_adj_factor, get_XR_adjust_amount_by_dividend_data, early_sell_standard_downgap)
from stocklist import get_name_and_industry_by_code

daily_root = f'{BASICDATA_DIR}/dailydata'
os.makedirs(daily_root, exist_ok=True)
gap_root = f'{DATASETS_DIR}/downgap'
os.makedirs(gap_root, exist_ok=True)
lock = Lock()

# 日志对象缓存，避免重复添加handler
trade_loggers = {}

def get_trade_logger(trade_log_path):
    if trade_log_path not in trade_loggers:
        logger = logging.getLogger(trade_log_path)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            trade_handle = logging.FileHandler(filename=trade_log_path, mode='a')
            trade_handle.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(trade_handle)
        trade_loggers[trade_log_path] = logger
    return trade_loggers[trade_log_path]

def get_holding_stocks_number(max_trade_days: int) -> int:
    """
    get holding stocks number and set signal to
    control the number of holding stocks
    :param max_trade_days: dataset group_id, like 50, 45, 40, etc.
    :return: number of holding stocks
    """
    for group in dataset_group_cons:
        if str(int(max_trade_days)) in group:
            HOLDING_LIST = dataset_group_cons[group].get('HOLDING_LIST')
            break
    if not os.path.exists(HOLDING_LIST):
        return 0
    holding_df = pd.read_csv(HOLDING_LIST)
    holding_stocks = holding_df[holding_df['status'] == 'holding'].shape[0]
    return holding_stocks

for days in dataset_group_cons['common'].get('MAX_TRADE_DAYS_LIST'):
    stock_numbers = get_holding_stocks_number(days)
    model_name = dataset_group_cons[f'group_{days}'].get('MODEL_NAME')
    print(f'({model_name}) 当前持仓股票数: {stock_numbers}')

# buy in and sell out
def buy_in(code: str, price: float, amount: int, trade_date: str, target_price: float, max_trade_days) -> None:
    """
    buy in stock
    :param code: stock code, like 000001 or 000001.SH
    :param price: stock price
    :param amount: stock amount
    :param trade_date: trade date link to gap, like '20210804'
    :param target_price: target price link to gap
    :param max_trade_days: dataset group_id, like 50, 45, 40, etc.
    """
    COST_FEE = dataset_group_cons['common'].get('COST_FEE')
    for group in dataset_group_cons:
        if str(int(max_trade_days)) in group:
            HOLDING_LIST = dataset_group_cons[group].get('HOLDING_LIST')
            MIN_PRED_RATE = dataset_group_cons[group].get('MIN_PRED_RATE')
            PRED_RATE_PCT = dataset_group_cons[group].get('PRED_RATE_PCT')
            TRADE_LOG = dataset_group_cons[group].get('TRADE_LOG')
            MODEL_NAME = dataset_group_cons[group].get('MODEL_NAME')
            HOLDING_LIST_ORIGIN = dataset_group_cons[group].get('HOLDING_LIST_ORIGIN')
            break
    trade_log = get_trade_logger(TRADE_LOG)
    if len(code) != 9:
        code = code +'.SH' if code.startswith('6') else code + '.SZ'
    msg = get_name_and_industry_by_code(code)
    stock_name = msg[0]
    industry = msg[1]
    pattern = re.compile(r'[*]?[sS][tT]|退市|退|[pP][Tt]')  # 例外股票，不投
    if pattern.findall(stock_name):
        return
    trade_date = trade_date
    fill_date = ''  # 避免日期型数据被转换为数值型
    date_in = datetime.datetime.now().strftime('%Y%m%d')
    date_out = ''  # 避免日期型数据被转换为数值型
    days = None
    holding_days = 1
    target_price = target_price
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
    row_data = [code, stock_name, industry, trade_date, fill_date, date_in, date_out, days, holding_days, target_price, 
                price_in, price_out, price_now, amount, cost_fee, profit, rate_pred, rate_pct, rate_current, rate_yearly, status]
    column = ['ts_code', 'stock_name', 'industry', 'trade_date', 'fill_date', 'date_in', 'date_out', 'days', 'holding_days',
                'target_price', 'price_in', 'price_out', 'price_now', 'amount', 'cost_fee', 'profit', 'rate_pred', 'rate_pct',
                'rate_current', 'rate_yearly', 'status']
    new_row = pd.DataFrame([row_data, ], columns=column)
    # adjust stock number signal and cash amount
    cash_amount_buy = round(price * (1 + cost_fee) * amount, 2)
    note = f'买入 {code} {stock_name} at {price} total {amount}'
    res = create_or_update_funds_change_list(-cash_amount_buy, note, max_trade_days=max_trade_days)
    if not res:  # cash balance is not enough
        return
    new_row.to_csv(HOLDING_LIST, mode='a', header=False, index=False)
    # update origin holding list for xd holding_list
    origin_columns = ['ts_code', 'stock_name', 'industry', 'trade_date', 'date_in', 'price_in', 'amount', 'target_price']
    origin_row = [code, stock_name, industry, trade_date, date_in, price_in, amount, target_price]
    origin_df = pd.DataFrame([origin_row, ], columns=origin_columns)
    if not os.path.exists(HOLDING_LIST_ORIGIN):
        origin_df.to_csv(HOLDING_LIST_ORIGIN, mode='w', header=True, index=False)
    else:
        origin_df.to_csv(HOLDING_LIST_ORIGIN, mode='a', header=False, index=False)
    now = datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')
    trade_log.info(f'买入 {code} {stock_name} {industry} at {price} total {amount} at {now}')
    os.system(f'afplay /System/Library/Sounds/Ping.aiff')  # play sound to remind buy in
    os.system(f'afplay /System/Library/Sounds/Ping.aiff')  # play sound second times
    os.system(f'afplay /System/Library/Sounds/Ping.aiff')  # play sound third times
    title = f'买入股票::{MODEL_NAME}'
    message = f'{stock_name}-{code}-买入价:{price}元-买入数量:{amount}股-{now}'
    send_wechat_message_via_bark(bark_device_key, title, message)

def sell_out(code: str, price: float, trade_date: str, max_trade_days) -> None:
    """
    sell out stock
    :param code: stock code, like 000001 or 000001.SH
    :param price: stock price
    :param amount: stock amount
    :param trade_date: trade date link to gap, like '20210804'
    :param max_trade_days: dataset group_id, like 50, 45, 40, etc.
    """
    for group in dataset_group_cons:
        if str(max_trade_days) in group:
            HOLDING_LIST = dataset_group_cons[group].get('HOLDING_LIST')
            MODEL_NAME = dataset_group_cons[group].get('MODEL_NAME')
            TRADE_LOG = dataset_group_cons[group].get('TRADE_LOG')
            break
    trade_log = get_trade_logger(TRADE_LOG)
    if len(code) != 9:
        code = code +'.SH' if code.startswith('6') else code + '.SZ'
    holding_df = pd.read_csv(HOLDING_LIST, dtype={'trade_date': str, 'fill_date': str, 'date_in': str, 'date_out': str})
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
    # adjust cash amount and stock number signal
    cash_amount_sell = round(price * (1 - cost_fee) * amount, 2)
    note = f'卖出 {code} {stock_name} at {price} total {amount}'
    create_or_update_funds_change_list(cash_amount_sell, note, max_trade_days=max_trade_days)
    holding_df.to_csv(HOLDING_LIST, index=False)
    now = datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')
    msg = f'卖出 {code} {stock_name} {industry} at {price} total {amount}, profit: {profit:.2f}, rate_current: {rate_current:.2%}, rate_yearly: {rate_yearly:.2%}'
    trade_log.info(msg)
    os.system(f'afplay /System/Library/Sounds/Hero.aiff')  # play sound to remind sell out
    os.system(f'afplay /System/Library/Sounds/Hero.aiff')  # play sound second times
    os.system(f'afplay /System/Library/Sounds/Hero.aiff')  # play sound third times
    title = f'卖出股票::{MODEL_NAME}'
    message = f'{stock_name}-{code}-卖出价:{price}元-卖出数量:{amount}股-{now}'
    send_wechat_message_via_bark(bark_device_key, title, message)

def calculate_buy_in_amount(funds, price) -> int | None:
    """
    calculate buy_in amount
    :param funds: funds to buy in
    :param price: stock price
    :return: buy_in amount
    """
    COST_FEE = dataset_group_cons['common'].get('COST_FEE')
    amount = int(funds*(1-COST_FEE) / price)
    amount = amount // 100 * 100
    return amount

def create_daily_profit_list(max_trade_days):
    """
    create and update daily profit list
    :param max_trade_days: dataset group_id, like 50, 45, 40, etc.
    NOTE: contains columns: trade_date, profit, delta
    """
    for group in dataset_group_cons:
        if str(int(max_trade_days)) in group:
            HOLDING_LIST = dataset_group_cons[group].get('HOLDING_LIST')
            DAILY_PROFIT = dataset_group_cons[group].get('DAILY_PROFIT')
            break
    if not is_trade_date_or_not():
        return
    if not os.path.exists(HOLDING_LIST):
        return
    today = datetime.datetime.now().date().strftime('%Y%m%d')
    with lock:
        holding_df = pd.read_csv(HOLDING_LIST, dtype={'trade_date': str, 'fill_date': str})
        total_profit = round(holding_df['profit'].sum(), 2)
    if not os.path.exists(DAILY_PROFIT):
        fisrt_row = pd.DataFrame([[today, total_profit, total_profit]], columns=['trade_date', 'profit', 'delta'])
        fisrt_row.to_csv(DAILY_PROFIT, index=False)
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
        return
    reverse_second_profit = profit_df.iloc[-2]['profit']
    delta = round(total_profit-reverse_second_profit, 2)
    profit_df.loc[profit_df['trade_date'] == today, 'profit'] = total_profit
    profit_df.loc[profit_df['trade_date'] == today, 'delta'] = delta
    profit_df = profit_df.sort_values(by='trade_date', ascending=True)
    profit_df.to_csv(DAILY_PROFIT, index=False)

def create_or_update_funds_change_list(funds: float, note: str, max_trade_days: int) -> bool:
    """
    create or refresh funds change list
    :param funds: funds to add or reduce
    :param note: note for funds change
    :param max_trade_days: dataset group_id, like 50, 45, 40, etc.
    :return: True if success, False if failed
    NOTE: contains columns: datetime, amount, balance, note
    """
    for group in dataset_group_cons:
        if str(int(max_trade_days)) in group:
            FUNDS_LIST = dataset_group_cons[group].get('FUNDS_LIST')
            break
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
        print('现金余额不足，无法完成交易')
        return False
    new_row = pd.DataFrame([[now, funds, new_balence, note]], columns=['datetime', 'amount', 'balance', 'note'])
    funds_change_df = pd.concat([funds_change_df, new_row], ignore_index=True)
    funds_change_df.to_csv(FUNDS_LIST, index=False)
    return True

def create_holding_list(max_trade_days: int):
    """
    build holding list
    :param max_trade_days: dataset group_id, like 50, 45, 40, etc.
    NOTE: contains columns:
    ts_code, stock_name, industry, trade_date, fill_date, date_in, date_out, days(trade days), 
    holding_days(calender days), target_price, price_in, price_out, amount, rate_current, rate_yearly, status
    """
    for group in dataset_group_cons:
        if str(int(max_trade_days)) in group:
            HOLDING_LIST = dataset_group_cons[group].get('HOLDING_LIST')
            initial_funds = dataset_group_cons[group].get('initial_funds')
            break
    columns = [
        'ts_code', 'stock_name', 'industry', 'trade_date', 'fill_date', 'date_in', 'date_out', 'days', 
        'holding_days','target_price', 'price_in', 'price_out', 'price_now', 'amount', 'cost_fee', 
        'profit', 'rate_pred', 'rate_pct', 'rate_current', 'rate_yearly', 'status'
    ]
    holding_df = pd.DataFrame(columns=columns)
    holding_df.to_csv(HOLDING_LIST, index=False)
    create_or_update_funds_change_list(initial_funds, '初始资金', max_trade_days=max_trade_days)

# 持续扫描buy_in_list.csv, 买入股票
def scan_buy_in_list(max_trade_days:int):
    MIN_STOCK_PRICE = dataset_group_cons['common'].get('MIN_STOCK_PRICE')
    additionl_rate = dataset_group_cons['common'].get('additionl_rate')
    for group in dataset_group_cons:
        if str(int(max_trade_days)) in group:
            BUY_IN_LIST = dataset_group_cons[group].get('BUY_IN_LIST')
            HOLDING_LIST = dataset_group_cons[group].get('HOLDING_LIST')
            MIN_PRED_RATE = dataset_group_cons[group].get('MIN_PRED_RATE')
            ONE_TIME_FUNDS = dataset_group_cons[group].get('ONE_TIME_FUNDS')
            PRED_RATE_PCT = dataset_group_cons[group].get('PRED_RATE_PCT')
            MAX_STOCKS = dataset_group_cons[group].get('MAX_STOCKS')
            MODEL_NAME = dataset_group_cons[group].get('MODEL_NAME')
            break
    if not os.path.exists(BUY_IN_LIST):
        return
    if not os.path.exists(HOLDING_LIST):
        create_holding_list(max_trade_days=max_trade_days)
    with lock:
        buy_in_df = pd.read_csv(BUY_IN_LIST, dtype={'trade_date': str, 'fill_data': str})
        holding_df = pd.read_csv(HOLDING_LIST, dtype={'trade_date': str, 'fill_date': str})
    # delete rows in holding_df with same ts_code and trade_date from buy_in_df
    rows_not_holding = []
    for i, row in buy_in_df.iterrows():
        code = row['ts_code']
        trade_date = row['trade_date']
        if holding_df[(holding_df['ts_code'] == code) & (holding_df['trade_date'] == trade_date)].empty:
            rows_not_holding.append(row)
    buy_in_df = pd.DataFrame(rows_not_holding)
    buy_in_df = buy_in_df.sort_values(by='trade_date', ascending=True)
    buy_in_df = buy_in_df.reset_index(drop=True)

    def calculate_gaps_buy_in_points(code: str) -> float | None:
        """
        buy_in_list中同一只股票存在多个缺口可买入时,计算每个缺口的买点
        :param code: stock code
        :return: 返回每个缺口的买点(全部买点的最低值)。如果全部缺口已买入,或者存在三个
        以上缺口未买入,则返回None。
        NOTE:
        如果只存在一个缺口,返回该缺口的买点。
        如果同时存在两个缺口,且均未买入,返回二个缺口的买点及第二个缺口中的最低值。
        如果同时存在三个以上缺口,且均未买入,返回三个缺口的买点中的最低值。
        如果缺口连续跌停,则返回买点和最后一个缺口跌停价的最低值。
        如果同时存在三个以上未买入的缺口,作为异常处理,返回None。
        当存在多个缺口未买入时,只要买点,则全部缺口一起买入。这比较符合常识。
        """
        gaps_df = buy_in_df[buy_in_df['ts_code'] == code]
        up_limit_rate = get_up_down_limit(code=code)[2]
        pct = 0.95
        # 逐行检查是否已买入
        gaps_list = []
        for i, row in gaps_df.iterrows():
            trade_date = row['trade_date']
            holding_row = holding_df[(holding_df['ts_code'] == code) & (holding_df['trade_date'] == trade_date)]
            if holding_row.empty:
                gaps_list.append(row)
        if not gaps_list:
            return None
        # 计算每个缺口的买点,采用直观通俗的写法
        elif len(gaps_list) == 1:
            traget_price = gaps_list[0]['target_price']
            pred = gaps_list[0]['pred']
            p = traget_price / (1 + pred)
            return p
        elif len(gaps_list) == 2:
            target_price1 = gaps_list[0]['target_price']
            pred1 = gaps_list[0]['pred']
            p1 = target_price1 / (1 + pred1)  # 买点1
            pct_chg1 = gaps_list[0]['pct_chg']
            target_price2 = gaps_list[1]['target_price']
            pred2 = gaps_list[1]['pred']
            p2 = target_price2 / (1 + pred2)  # 买点2
            pct_chg2 = gaps_list[1]['pct_chg']
            p_down_limit = target_price2 * 0.9 * 0.9  # 缺口2的跌停价
            if pct_chg1 <= -up_limit_rate * pct and pct_chg2 <= -up_limit_rate * pct:
                return min(p1, p2, p_down_limit)  # 买点1,2和缺口2的跌停价最低值
            return min(p1, p2)
        elif len(gaps_list) == 3:
            target_price1 = gaps_list[0]['target_price']
            pred1 = gaps_list[0]['pred']
            p1 = target_price1 / (1 + pred1)  # 买点1
            pct_chg1 = gaps_list[0]['pct_chg']
            target_price2 = gaps_list[1]['target_price']
            pred2 = gaps_list[1]['pred']
            p2 = target_price2 / (1 + pred2)  # 买点2
            pct_chg2 = gaps_list[1]['pct_chg']
            target_price3 = gaps_list[2]['target_price']
            pred3 = gaps_list[2]['pred']
            p3 = target_price3 / (1 + pred3)  # 买点3
            pct_chg3 = gaps_list[2]['pct_chg']
            p_down_limit = target_price3 * 0.9 * 0.9  # 缺口3的跌停价
            if pct_chg1 <= -up_limit_rate * pct and pct_chg2 <= -up_limit_rate * pct and pct_chg3 <= -up_limit_rate * pct:
                return min(p1, p2, p3, p_down_limit)  # 买点1,2,3和缺口3的跌停价最低值
            return min(p1, p2, p3)
        else:  # 3个以上缺口
            return None

    def scan_buy_in_list_row(idx_row):
        i, row = idx_row
        code = row['ts_code']
        name = row['stock_name']
        trade_date = row['trade_date']
        target_price = row['target_price']
        pred = row['pred']
        pct_chg = row['pct_chg']
        price_now = get_stock_realtime_price(code)
        print(f'({MODEL_NAME}) {row['ts_code']} {row['stock_name']} price_now: {price_now}')
        if price_now is None:
            return
        if price_now <= MIN_STOCK_PRICE:
            return
        # 停牌不买
        if is_suspended_or_not(code):
            return
        # 下跌不买
        if is_decreasing_or_not(code, price_now):
            return
        # 跌停不买
        down_limit = get_up_down_limit(code=code)[1]
        if down_limit is not None and price_now / down_limit <= 1.02: 
            return
        # 如果同一只股票存在多个缺口未买入时,重新计算确定每个缺口的买点p
        stock_df = buy_in_df[buy_in_df['ts_code'] == code]
        stock_nums = stock_df.shape[0]
        if pred > MIN_PRED_RATE * 1.3:  # 预期收益率大于最低收益率的30%,加大买入资金15%
            buy_in_amount = ONE_TIME_FUNDS * 1.15
        else:
            buy_in_amount = ONE_TIME_FUNDS
        if stock_nums > 1:
            p = calculate_gaps_buy_in_points(code)
            if p is None:
                return
            if price_now > p:  # 现价高于买点,不买入
                return
            amount = calculate_buy_in_amount(funds=buy_in_amount/stock_nums, price=price_now)
        else:  # 只有一个缺口
            pred_now = (target_price - price_now) / price_now  # 以现价买入后回补缺口的预期收益率
            if pred_now < pred * PRED_RATE_PCT:
                return
            # 当日接近跌停且推断还会继续大跌或者跌停,合理预期会出现第二个较大的缺口,该缺口的买点
            # 会比当前缺口的买点更低,故不能以当前缺口的买点成交。此时采用强制提高收益率的方式
            # （即人为设定更低的买点）阻止交易,等待第二个较大的缺口出现。
            up_limit_rate = get_up_down_limit(code=code)[2]
            pct = 0.95
            pred_2_down_limit = 1/(1-up_limit_rate*pct)**2 -1  # 连续 2 个跌停后回补的最低收益率
            if pct_chg <= -up_limit_rate * pct and pred >= pred_2_down_limit:
                if pred_now < pred + additionl_rate:
                    return
            amount = calculate_buy_in_amount(funds=buy_in_amount, price=price_now)
        if amount == 0:
            return
        with lock:
            current_holding_stocks = get_holding_stocks_number(max_trade_days)
            if current_holding_stocks >= MAX_STOCKS:
                return
            buy_in(code, price_now, amount, trade_date, target_price, max_trade_days)

    # 多线程扫描buy_in_list.csv
    idx_rows = list(buy_in_df.iterrows())
    with ThreadPoolExecutor() as executor:
        executor.map(scan_buy_in_list_row, idx_rows)

# 持续刷新holding_list.csv
def refresh_holding_list(max_trade_days: int):
    """ 
    refresh columns: fill_date, days, holding_days, price_now, profit, rate_current, rate_yearly
    :param max_trade_days: dataset group_id, like 50, 45, 40, etc.
    """
    for group in dataset_group_cons:
        if str(int(max_trade_days)) in group:
            HOLDING_LIST = dataset_group_cons[group].get('HOLDING_LIST')
            MODEL_NAME = dataset_group_cons[group].get('MODEL_NAME')
            break
    if not os.path.exists(HOLDING_LIST):
        create_holding_list(max_trade_days=max_trade_days)
        return
    with lock:
        holding_df = pd.read_csv(
            HOLDING_LIST, dtype={'trade_date': str, 'fill_date': str, 'date_in': str}
        )
        holding_df['fill_date'] = holding_df['fill_date'].apply(lambda x: str(x)[:8])
        holding_df['fill_date'] = holding_df['fill_date'].apply(lambda x: x if x != 'nan' else '')
        holding_df['date_in'] = holding_df['date_in'].apply(lambda x: str(x)[:8])
        holding_df['date_in'] = holding_df['date_in'].apply(lambda x: x if x != 'nan' else '')

    def refresh_holding_list_row(idx_row):
        """
        refresh holding list single row infomation
        :param idx_row: index, row
        """
        i, row = idx_row
        try:
            gap_csv = f'{gap_root}/{row["ts_code"]}.csv'
            if not os.path.exists(gap_csv):
                print(f"Warning: {gap_csv} not found, skip row {row['ts_code']}")
                return i, row
            gap_df = pd.read_csv(gap_csv, dtype={'trade_date': str, 'fill_date': str})
            gap_df['fill_date'] = gap_df['fill_date'].apply(lambda x: str(x)[:8])
            gap_df['fill_date'] = gap_df['fill_date'].apply(lambda x: x if x != 'nan' else '')
            gap_df = gap_df.sort_values(by='trade_date', ascending=True)
            trade_date = row['trade_date']
            if trade_date not in gap_df['trade_date'].values:
                print(f"Warning: trade_date {trade_date} not in {gap_csv}, skip row {row['ts_code']}")
                return i, row
            fill_date = gap_df[gap_df['trade_date'] == trade_date]['fill_date'].values[0]
            if fill_date != '':
                days = gap_df[gap_df['trade_date'] == trade_date]['days'].values[0]
                row['fill_date'] = fill_date
                row['days'] = days
            # if status is sold_out, update fill_date and days
            if row['status'] == 'sold_out':
                return i, row
            # if status is holding, update holding_days, days, price_now, profit, rate_current, rate_yearly
            if row['status'] == 'holding':
                # holding_days(自然日历间隔天数)
                try:
                    date_in = datetime.datetime.strptime(str(row['date_in']), '%Y%m%d')
                except Exception as e:
                    print(f"Warning: date_in parse error for {row['ts_code']}: {row['date_in']}, {e}")
                    return i, row
                today = datetime.datetime.now()
                holding_days = (today - date_in).days + 1
                row['holding_days'] = holding_days
                # days(交易日历间隔天数today - trade_date)
                daily_csv = f'{daily_root}/{row["ts_code"]}.csv'
                if not os.path.exists(daily_csv):
                    print(f"Warning: {daily_csv} not found, skip row {row['ts_code']}")
                    return i, row
                daily_df = pd.read_csv(daily_csv, dtype={'trade_date': str})
                daily_df = daily_df.sort_values(by='trade_date', ascending=True)
                trade_date_list = daily_df['trade_date'].tolist()
                today_str = today.strftime('%Y%m%d')
                if trade_date not in trade_date_list:
                    print(f"Warning: trade_date {trade_date} not in {daily_csv}, skip row {row['ts_code']}")
                    return i, row
                trade_date_index = trade_date_list.index(trade_date)
                if today_str in trade_date_list:
                    today_index = trade_date_list.index(today_str)
                    days = abs(today_index - trade_date_index) 
                else:
                    today_index = len(trade_date_list) - 1
                    days = abs(today_index - trade_date_index) + 1  # today还未收盘，未在daily_df中，所以+1
                row['days'] = days
                # price_now, profit, rate_current, rate_yearly
                price_now = get_stock_realtime_price(row['ts_code'])
                print(f'({MODEL_NAME}) {row["ts_code"]} {row["stock_name"]} price_now: {price_now}')
                if price_now is None:
                    # 保持原有 price_now，不更新
                    return i, row
                price_in = row['price_in']
                amount = row['amount']  # 股数量
                cost_fee = row['cost_fee']
                row['price_now'] = price_now
                profit = (price_now*(1-cost_fee) - price_in*(1+cost_fee)) * amount
                profit = round(profit, 4)
                row['profit'] = profit
                rate_current = profit / (price_in * amount)
                rate_current = round(rate_current, 4)
                row['rate_current'] = rate_current
                rate_yearly = rate_current * 365 / holding_days  # 自然日历年化收益率
                rate_yearly = round(rate_yearly, 4)
                row['rate_yearly'] = rate_yearly
            return i, row
        except Exception as e:
            print(f"Exception in refresh_holding_list_row for {row['ts_code']}: {e}")
            return i, row

    # 多线程刷新holding_list.csv，收集结果，主线程写回
    idx_rows = list(holding_df.iterrows())
    results = []
    with ThreadPoolExecutor() as executor:
        for res in executor.map(refresh_holding_list_row, idx_rows):
            results.append(res)
    # 合并结果
    for i, updated_row in results:
        for col in holding_df.columns:
            if col in updated_row:
                holding_df.at[i, col] = updated_row[col]
    with lock:
        holding_df.to_csv(HOLDING_LIST, index=False)

# 持续扫描holding_list.csv, 卖出股票
def scan_holding_list(max_trade_days: int):
    """
    scan holding list, sell out stocks
    :param max_trade_days: dataset group_id, like 50, 45, 40, etc.
    """
    for group in dataset_group_cons:
        if str(int(max_trade_days)) in group:
            HOLDING_LIST = dataset_group_cons[group].get('HOLDING_LIST')
            MAX_TRADE_DAYS = dataset_group_cons[group].get('MAX_TRADE_DAYS')
            MODEL_NAME = dataset_group_cons[group].get('MODEL_NAME')
            TRADE_LOG = dataset_group_cons[group].get('TRADE_LOG')
            break
    trade_log = get_trade_logger(TRADE_LOG)
    if not os.path.exists(HOLDING_LIST):
        create_holding_list(max_trade_days=max_trade_days)
        return
    with lock:
        holding_df = pd.read_csv(
            HOLDING_LIST, dtype={'trade_date': str, 'fill_date': str, 'date_out': str}
        )
        holding_df['fill_date'] = holding_df['fill_date'].apply(lambda x: str(x)[:8])
        holding_df['fill_date'] = holding_df['fill_date'].apply(lambda x: x if x != 'nan' else '')
        holding_df['date_out'] = holding_df['date_out'].apply(lambda x: str(x)[:8])
        holding_df['date_out'] = holding_df['date_out'].apply(lambda x: x if x != 'nan' else '')
    
    def scan_holding_list_row(idx_row):
        i, row = idx_row
        ts_code = row['ts_code']
        stock_name = row['stock_name']
        trade_date = row['trade_date']
        if row['date_out'] != '':
            return
        if row['status'] == 'sold_out':
            return
        holding_days = row['holding_days']
        if holding_days == 1:
            return
        price_now = get_stock_realtime_price(ts_code)
        print(f'({MODEL_NAME}) {ts_code} {stock_name} price_now: {price_now}')
        if price_now is None:
            return
        # 停牌不卖
        if is_suspended_or_not(ts_code):
            return
        # 上涨不卖
        if is_rising_or_not(ts_code, price_now):
            return
        # 涨停不卖
        up_limit = get_up_down_limit(code=ts_code)[0]
        if up_limit is not None and price_now / up_limit >= 0.98:
            return
        # if the down gap is filled, sell out
        fill_date = row['fill_date']
        if price_now >= row['target_price'] or fill_date != '':
            with lock:
                sell_out(ts_code, price_now, trade_date=trade_date, max_trade_days=max_trade_days)
            msg = f'卖出 {ts_code} {stock_name}: the down gap is filled'
            trade_log.info(msg)
            return
        # if days > MAX_TRADE_DAYS, sell out
        days = row['days']
        if days >= MAX_TRADE_DAYS:
            with lock:
                sell_out(ts_code, price_now, trade_date=trade_date, max_trade_days=max_trade_days)
            msg = f'卖出 {ts_code} {stock_name}: days > {MAX_TRADE_DAYS}'
            trade_log.info(msg)
            return
        # if rate_yearly >= 3.0 and holding_days >= 10, sell out in advance
        rate_current = row['rate_current']
        rate_yearly = row['rate_yearly']
        early_or_not =  early_sell_standard_downgap(holding_days, rate_current, rate_yearly)
        if early_or_not:
            with lock:
                sell_out(ts_code, price_now, trade_date=trade_date, max_trade_days=max_trade_days)
            msg = f'卖出 {ts_code} {stock_name}: trigger early sell standard, rate_yearly {rate_yearly:.2%} within {holding_days} days'
            trade_log.info(msg)
            return
    # 多线程扫描holding_list.csv
    idx_rows = list(holding_df.iterrows())
    with ThreadPoolExecutor() as executor:
        executor.map(scan_holding_list_row, idx_rows)

def XD_buy_in_list(max_trade_days: int):
    """
    盘中前复权 target_price, 即 trade_date 前一日的最低价
    NOTE:
    save the result to XD_RECORD_BUY_IN_CSV(only one row),
    XD_RECORD_BUY_IN_CSV contains columns: today, xd_or_not
    """
    for group in dataset_group_cons:
        if str(int(max_trade_days)) in group:
            BUY_IN_LIST = dataset_group_cons[group].get('BUY_IN_LIST')
            XD_RECORD_BUY_IN_CSV = dataset_group_cons[group].get('XD_RECORD_BUY_IN_CSV')
            break
    if not os.path.exists(BUY_IN_LIST):
        return
    with lock:
        buy_in_df = pd.read_csv(BUY_IN_LIST, dtype={'trade_date': str})
    today = datetime.datetime.now().strftime('%Y%m%d')
    if not os.path.exists(XD_RECORD_BUY_IN_CSV):
        xd_or_not = False
        xd_record_df = pd.DataFrame([[today, xd_or_not]], columns=['today', 'xd_or_not'])
        xd_record_df.to_csv(XD_RECORD_BUY_IN_CSV, index=False)
    else:
        xd_record_df = pd.read_csv(XD_RECORD_BUY_IN_CSV, dtype={'today': str})
        xd_or_not = xd_record_df[xd_record_df['today'] == today]['xd_or_not'].values
        if xd_or_not.size > 0:
            xd_or_not = xd_or_not[0]
        else:
            xd_or_not = False
        if xd_or_not:
            return
    for idx_row in buy_in_df.iterrows():
        i, row = idx_row
        ts_code = row['ts_code']
        trade_date = row['trade_date']
        target_price = row['target_price']
        # 获取 trade_date 的前一日(该日最低价即 target_price)
        daily_csv = f'{daily_root}/{ts_code}.csv'
        if not os.path.exists(daily_csv):
            continue
        daily_df = pd.read_csv(daily_csv, dtype={'trade_date': str})
        daily_df = daily_df.sort_values(by='trade_date', ascending=True)
        daily_df = daily_df.reset_index(drop=True)
        if trade_date not in daily_df['trade_date'].values:
            continue
        trade_date_index = daily_df[daily_df['trade_date'] == trade_date].index
        if trade_date_index.empty:
            continue
        trade_date_index = trade_date_index[0]
        if trade_date_index == 0:
            continue
        prev_trade_date = daily_df.iloc[trade_date_index - 1]['trade_date']
        xd_target_price = get_qfq_price_by_adj_factor(
            code=ts_code, pre_price=target_price, start=prev_trade_date, end=today,
        )
        if xd_target_price == target_price:
            continue
        buy_in_df.loc[i, 'target_price'] = xd_target_price
    with lock:
        buy_in_df.to_csv(BUY_IN_LIST, index=False)
    xd_or_not = True
    xd_record_df = pd.DataFrame([[today, xd_or_not]], columns=['today', 'xd_or_not'])
    xd_record_df.to_csv(XD_RECORD_BUY_IN_CSV, index=False)

def XD_holding_list_bak(max_trade_days: int):
    """
    盘中前复权 target_price 和 price_in, 对 amount 进行股数调整
    前复权和股数调整记录在 XD_RECORD_HOLDING_CSV 中
    :param max_trade_days: dataset group_id, like 50, 45, 40, etc.
    NOTE:
    XD_RECORD_CSV contains columns: ts_code, trade_date, pre_trade_date, xd_date, target_price, 
    price_in, amount, xd_target_price, xd_price_in, xd_amount
    NOTE: 已弃用   
    """
    for group in dataset_group_cons:
        if str(int(max_trade_days)) in group:
            HOLDING_LIST = dataset_group_cons[group].get('HOLDING_LIST')
            XD_RECORD_HOLDING_CSV = dataset_group_cons[group].get('XD_RECORD_HOLDING_CSV')
            break
    if not os.path.exists(HOLDING_LIST):
        return
    with lock:
        holding_df = pd.read_csv(HOLDING_LIST, dtype={'trade_date': str, 'date_in': str})
        holding_df['date_in'] = holding_df['date_in'].apply(lambda x: str(x)[:8])
        holding_df['date_in'] = holding_df['date_in'].apply(lambda x: x if x != 'nan' else '')
    today = datetime.datetime.now().strftime('%Y%m%d')
    xd_record_df = []
    columns = ['ts_code', 'trade_date', 'pre_trade_date', 'xd_date', 'target_price', 'price_in', 
               'amount', 'xd_target_price', 'xd_price_in', 'xd_amount']
    for idx_row in holding_df.iterrows():
        i, row = idx_row
        if row['status'] == 'sold_out':
            continue
        ts_code = row['ts_code']
        trade_date = row['trade_date']
        date_in = row['date_in']
        target_price = row['target_price']
        price_in = row['price_in']
        amount = row['amount']
        # 获取 trade_date 的前一日(该日最低价即 target_price)
        daily_csv = f'{daily_root}/{ts_code}.csv'
        if not os.path.exists(daily_csv):
            continue
        daily_df = pd.read_csv(daily_csv, dtype={'trade_date': str})
        daily_df = daily_df.sort_values(by='trade_date', ascending=True)
        daily_df = daily_df.reset_index(drop=True)
        if trade_date not in daily_df['trade_date'].values:
            continue
        trade_date_index = daily_df[daily_df['trade_date'] == trade_date].index
        if trade_date_index.empty:
            continue
        trade_date_index = trade_date_index[0]
        if trade_date_index == 0:
            continue
        prev_trade_date = daily_df.iloc[trade_date_index - 1]['trade_date']
        tmp_list = [ts_code, trade_date, prev_trade_date, today, target_price, price_in, amount, ]
        if not os.path.exists(XD_RECORD_HOLDING_CSV):
            start_target_price = prev_trade_date
            start_price_in = date_in
        else:
            xd_df = pd.read_csv(XD_RECORD_HOLDING_CSV, dtype={'trade_date': str, 'xd_date': str})
            xd_df = xd_df.sort_values(by=['ts_code', 'xd_date'], ascending=[True, True])
            xd_df = xd_df[xd_df['ts_code'] == ts_code]
            if xd_df.empty:
                start_target_price = prev_trade_date
                start_price_in = date_in
            else:
                last_xd_date = xd_df.iloc[-1]['xd_date']
                start_target_price = last_xd_date if last_xd_date > prev_trade_date else prev_trade_date
                start_price_in = last_xd_date if last_xd_date > date_in else date_in
        xd_target_price = get_qfq_price_by_adj_factor(
            code=ts_code, pre_price=target_price, start=start_target_price, end=today,
        )
        if xd_target_price == target_price:
            continue
        xd_price_in = get_qfq_price_by_adj_factor(
            code=ts_code, pre_price=price_in, start=start_price_in, end=today,
        )
        if xd_price_in == price_in:
            xd_amount = amount
            tmp_list.extend([xd_target_price, xd_price_in, xd_amount])
            xd_record_df.append(tmp_list)
            holding_df.loc[i, 'target_price'] = xd_target_price
            continue
        xd_amount = get_XR_adjust_amount_by_dividend_data(
            code=ts_code, amount=amount, start=start_price_in, end=today,
        )
        tmp_list.extend([xd_target_price, xd_price_in, xd_amount])
        xd_record_df.append(tmp_list)
        holding_df.loc[i, 'target_price'] = xd_target_price
        holding_df.loc[i, 'price_in'] = xd_price_in
        holding_df.loc[i, 'amount'] = xd_amount
    xd_df = pd.DataFrame(xd_record_df, columns=columns)
    if xd_df.empty:
        return
    columns = ['ts_code', 'trade_date', 'pre_trade_date', 'xd_date', 'target_price', 'xd_target_price',
               'price_in', 'xd_price_in', 'amount', 'xd_amount']
    xd_df = xd_df[columns]
    if not os.path.exists(XD_RECORD_HOLDING_CSV):
        xd_df.to_csv(XD_RECORD_HOLDING_CSV, index=False)
    else:
        xd_df.to_csv(XD_RECORD_HOLDING_CSV, mode='a', header=False, index=False)
    with lock:
        holding_df.to_csv(HOLDING_LIST, index=False)
    # refresh_holding_list(max_trade_days=max_trade_days)

def XD_holding_list(max_trade_days: int):
    """
    盘中前复权HOLDING_LIST中 target_price 和 price_in, 对 amount 进行股数调整
    从买入的原始记录 HOLDING_LIST_ORIGIN 中获取数据结合复权因子进行复权
    :param max_trade_days: dataset group_id, like 50, 45, 40, etc.
    """
    for group in dataset_group_cons:
        if str(int(max_trade_days)) in group:
            HOLDING_LIST = dataset_group_cons[group].get('HOLDING_LIST')
            HOLDING_LIST_ORIGIN = dataset_group_cons[group].get('HOLDING_LIST_ORIGIN')
            break
    if not os.path.exists(HOLDING_LIST):
        return
    if not os.path.exists(HOLDING_LIST_ORIGIN):
        return
    with lock:
        holding_df = pd.read_csv(HOLDING_LIST, dtype={'trade_date': str, 'date_in': str})
        holding_df['date_in'] = holding_df['date_in'].apply(lambda x: str(x)[:8])
        holding_df['date_in'] = holding_df['date_in'].apply(lambda x: x if x != 'nan' else '')
    origin_holding_df = pd.read_csv(HOLDING_LIST_ORIGIN, dtype={'trade_date': str, 'date_in': str})
    today = datetime.datetime.now().strftime('%Y%m%d')
    for idx_row in holding_df.iterrows():
        i, row = idx_row
        if row['status'] == 'sold_out':
            continue
        ts_code = row['ts_code']
        trade_date = row['trade_date']
        # 从origin_holding_df中获取date_in target_price price_in 和 amount
        origin_res_row = origin_holding_df[(origin_holding_df['trade_date'] == trade_date) & (origin_holding_df['ts_code'] == ts_code)]
        if origin_res_row.empty:
            continue
        date_in = origin_res_row.iloc[0]['date_in']
        target_price = origin_res_row.iloc[0]['target_price']
        price_in = origin_res_row.iloc[0]['price_in']
        amount = origin_res_row.iloc[0]['amount']
        # 获取 trade_date 的前一日(该日最低价即 target_price)
        adj_csv = f'{BASICDATA_DIR}/adjfactor/{ts_code}.csv'
        if not os.path.exists(adj_csv):
            continue
        adj_df = pd.read_csv(adj_csv, dtype={'trade_date': str})
        adj_df = adj_df.sort_values(by='trade_date', ascending=True)
        adj_df = adj_df.reset_index(drop=True)
        if trade_date not in adj_df['trade_date'].values:
            continue
        trade_date_index = adj_df[adj_df['trade_date'] == trade_date].index
        if trade_date_index.empty:
            continue
        trade_date_index = trade_date_index[0]
        if trade_date_index == 0:
            continue
        prev_trade_date = adj_df.iloc[trade_date_index - 1]['trade_date']
        # 除权
        xd_target_price = get_qfq_price_by_adj_factor(
            code=ts_code, pre_price=target_price, start=prev_trade_date, end=today,
        )
        xd_price_in = get_qfq_price_by_adj_factor(
            code=ts_code, pre_price=price_in, start=date_in, end=today,
        )
        xd_amount = get_XR_adjust_amount_by_dividend_data(
            code=ts_code, amount=amount, start=date_in, end=today
        )
        holding_df.loc[i, 'target_price'] = xd_target_price
        holding_df.loc[i, 'price_in'] = xd_price_in
        holding_df.loc[i, 'amount'] = xd_amount
    with lock:
        holding_df.to_csv(HOLDING_LIST, index=False)
    # refresh_holding_list(max_trade_days=max_trade_days)

def trade_process(max_trade_days: int, mode: Literal['trade', 'test'] = 'trade'):
    """
    trade logic process
    :param max_trade_days: dataset group_id, like 50, 45, 40, etc.
    :param mode: 'trade' or 'test'
    NOTE:
    - 'trade' 模式下, 在实际交易时间内执行交易逻辑。
    - 'test' 模式下, 在非交易时间执行交易逻辑, 主要为了检测交易逻辑是否正确。
    """
    def is_within_trading_hours():
        now = datetime.datetime.now().time()
        am_begin = datetime.time(9, 30)
        am_end = datetime.time(11, 30)
        pm_begin = datetime.time(13, 0)
        pm_end = datetime.time(15, 0)
        return (am_begin <= now <= am_end or pm_begin <= now <= pm_end)
    
    def one_trade_loop(max_trade_days: int):
        refresh_holding_list(max_trade_days=max_trade_days)
        scan_buy_in_list(max_trade_days=max_trade_days)
        scan_holding_list(max_trade_days=max_trade_days)
        create_daily_profit_list(max_trade_days=max_trade_days)

    if mode == 'trade' and is_within_trading_hours():
        one_trade_loop(max_trade_days=max_trade_days)
    if mode == 'test' and not is_within_trading_hours():
        import shutil
        trade_dir = f'{TRADE_DIR}/downgap/max_trade_days_{max_trade_days}'
        shutil.copytree(trade_dir, f'{trade_dir}_test_copy', dirs_exist_ok=True)  # 备份交易数据
        one_trade_loop(max_trade_days=max_trade_days)
    if mode not in ['trade', 'test']:
        print(f'Invalid mode: {mode}. Use "trade" or "test".')
        return

if __name__ == '__main__':
    trade_process(max_trade_days=45, mode='test')