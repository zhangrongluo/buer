import os
import re
import shutil
import pandas as pd
import datetime
import logging
from typing import Literal
from threading import Lock
from stocklist import get_name_and_industry_by_code
from concurrent.futures import ThreadPoolExecutor
from cons_oversold import (initial_funds, COST_FEE, MIN_STOCK_PRICE, ONE_TIME_FUNDS, MAX_STOCKS, STOP_BUYING,
                           PRED_RATE_PCT, MIN_PRED_RATE, MIN_WAITING_DAYS, MAX_TRADE_DAYS, MAX_DOWN_LIMIT,
                           REST_TRADE_DAYS, WAITING_RATE_PCT, MODEL_NAME, BUY_IN_LIST, HOLDING_LIST, MAX_BUY_UP_RATE,
                            DAILY_PROFIT, FUNDS_LIST, TRADE_LOG, XD_RECORD_HOLDGING_CSV, XD_RECORD_BUY_IN_CSV,
                            HOLDING_LIST_ORIGIN, BUY_IN_LIST_ORIGIN, exception_list, dataset_to_predict_trade)
from cons_general import BACKUP_DIR, TRADE_DIR, BASICDATA_DIR, TEST_DIR, PREDICT_DIR
from cons_hidden import bark_device_key
from utils import (send_message_via_bark, get_stock_realtime_price, is_trade_date_or_not, 
                   get_up_down_limit, early_sell_standard_oversold_v2, is_rising_or_not, is_decreasing_or_not, 
                   is_suspended_or_not, get_qfq_price_by_adj_factor, get_XR_adjust_amount_by_dividend_data)

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

def get_holding_stocks_number() -> int:
    """
    ### 获取当前持仓股票数
    """
    if not os.path.exists(HOLDING_LIST):
        return 0
    holding_df = pd.read_csv(HOLDING_LIST)
    holding_stocks = holding_df[holding_df['status'] == 'holding'].shape[0]
    return holding_stocks
HOLDING_STOCKS = get_holding_stocks_number()
print(f'({MODEL_NAME}) HOLDING_STOCKS is {HOLDING_STOCKS} now')

# buy in and sell out
def buy_in(code: str, price: float, amount: int, trade_date: str, buy_point_base: float, target_rate: float) -> None:
    """
    ### 买入股票
    #### :param code: 股票代码, 格式为 000001 or 000001.SZ
    #### :param price: 买入价格
    #### :param amount: 买入数量
    #### :param trade_date: 交易日期, 标记下跌趋势的买入点, 用于识别买入清单中的特定的行
    #### :param buy_point_base: 买入基准点, 即交易日期当天的收盘价
    #### :param target_rate: 目标收益率, 买入清单中 oversold 模型推理得到的
    """
    if len(code) != 9:
        code = code +'.SH' if code.startswith('6') else code + '.SZ'
    msg = get_name_and_industry_by_code(code)
    stock_name = msg[0]
    pattern = re.compile(r'[*]*[sS][tT]|退市|退|[pP][Tt]')  # 例外股票不投
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
    cash_amount_buy = round(price * (1 + cost_fee) * amount, 2)
    note = f'买入 {code} {stock_name} at {price} total {amount}'
    res = create_or_update_funds_change_list(-cash_amount_buy, note)
    if not res:  # cash balance is not enough
        return
    new_row.to_csv(HOLDING_LIST, mode='a', header=False, index=False)
    # update origin holding list for xd holding_list
    origin_columns = ['ts_code', 'stock_name', 'industry', 'trade_date', 'date_in', 'price_in', 'amount', 'buy_point_base']
    origin_row = [code, stock_name, industry, trade_date, date_in, price_in, amount, buy_point_base]
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
    title = '买入股票::OverSold'
    message = f'{stock_name}-{code}-买入价:{price}元-买入数量:{amount}股-{now}'
    send_message_via_bark(bark_device_key, title, message)

def sell_out(code: str, price: float, trade_date: str) -> None:
    """
    ### 卖出股票
    #### :param code: 股票代码, 格式为 000001 或者 000001.SZ
    #### :param price: 卖出价格
    #### :param amount: 卖出数量
    #### :param trade_date: 交易日期, 标记下跌趋势的买入点, 用于识别持有清单中的具体持仓记录
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
    holding_df.to_csv(HOLDING_LIST, index=False)
    now = datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')
    msg = f'卖出 {code} {stock_name} {industry} at {price} total {amount}, profit: {profit:.2f}, rate_current: {rate_current:.2%}, rate_yearly: {rate_yearly:.2%}'
    trade_log.info(msg)
    os.system(f'afplay /System/Library/Sounds/Hero.aiff')  # play sound to remind sell out
    os.system(f'afplay /System/Library/Sounds/Hero.aiff')  # play sound second times
    os.system(f'afplay /System/Library/Sounds/Hero.aiff')  # play sound third times
    title = '卖出股票::OverSold'
    message = f'{stock_name}-{code}-卖出价:{price}元-卖出数量:{amount}股-{now}'
    send_message_via_bark(bark_device_key, title, message)

def calculate_buy_in_amount(funds, price) -> int | None:
    """
    ### 计算买入数量
    #### :param funds: 可用资金
    #### :param price: 股票价格
    #### :return: 买入数量
    """
    amount = int(funds*(1-COST_FEE) / price)
    amount = amount // 100 * 100
    return amount

def create_daily_profit_list():
    """
    ### 创建和更新每日收益列表
    #### :param max_trade_days: 最大交易天数, 用于区别数据集, 50, 45, 60.
    #### NOTE: 包含的列名: trade_date, profit, delta
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

def create_or_update_funds_change_list(funds: float, note: str) -> bool:
    """
    ### 创建或更新资金变动列表
    #### :param funds: 资金变动金额, 正数为增加, 负数为减少
    #### :param note: 备注
    #### :param max_trade_days: 最大交易天数, 用于区别数据集, 50, 45, 60.
    #### :return: 成功返回True, 失败返回False(余额不足)
    #### :包含的列名: datetime, amount, balance, note
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
    new_row.to_csv(FUNDS_LIST, mode='a', header=False, index=False)
    return True

def create_holding_list(initial_cash: float = initial_funds):
    """
    ### 创建持仓列表
    #### :param max_trade_days: 最大交易天数, 用于区别数据集, 50, 45, 60.
    #### NOTE: 包含的列名:
    #### ts_code, stock_name, industry, trade_date, date_in, date_out, days(trade days), holding_days(calender days), 
    #### buy_point_base, target_rate, price_in, price_out, amount, rate_current, rate_yearly, status
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
    """
    ### 扫描买入列表, 买入股票
    #### :param max_trade_days: 最大交易天数, 用于区别数据集, 50, 45, 60.
    """
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
    # save all_rows which are not in holding_df with same ts_code and holding status
    rows_not_holding = []
    for i, row in buy_in_df.iterrows():
        code = row['ts_code']
        if holding_df[(holding_df['ts_code'] == code) & (holding_df['status'] == 'holding')].empty:
            rows_not_holding.append(row)
    buy_in_df = pd.DataFrame(rows_not_holding)
    buy_in_df = buy_in_df.sort_values(by='trade_date', ascending=True)
    buy_in_df = buy_in_df.reset_index(drop=True)

    def get_max_price_between_today_and_trade_date(idx_row):
        """
        ### 获取从trade_date到今天的最高价
        """
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
        """
        ### 扫描买入列表单行信息
        #### :param idx_row: index, row
        #### 买入逻辑
        - 如果设置了停止买入新股票，跳过
        - 未获取到实时价格，跳过
        - 价格高于买入基准点buy_point_base的限定幅度MAX_BUY_UP_RATE, 跳过
        - 价格低于最低股票价格，跳过
        - 停牌股票，跳过
        - 跌停板附近，跳过
        - 等待天数不足(下跌趋势触底天数不足)，跳过
        - 避免买入后持有天数过短, 剩余交易天数不足, 跳过
        - 买入前最高价达到预期收益率目标的WAITING_RATE_PCT倍数的, 跳过
        - 价格持续下降，跳过
        - 当前持仓股票数达到上限，跳过
        """
        i, row = idx_row
        code = row['ts_code']
        name = row['stock_name']
        trade_date = row['trade_date']
        buy_point_base = row['buy_point_base']
        target_rate = row['pred']  # 预期收益率
        waiting_days = row['waiting_days']
        price_now = get_stock_realtime_price(code)
        print(f'({MODEL_NAME}) {row["ts_code"]} {row["stock_name"]} price_now: {price_now}')
        if STOP_BUYING:
            return
        if price_now is None or price_now <= 0:
            return
        if price_now >= buy_point_base * (1 + MAX_BUY_UP_RATE): 
            return
        if price_now <= MIN_STOCK_PRICE:
            return
        # suspended stocks, skip
        if is_suspended_or_not(code=code):
            return
        down_limit = get_up_down_limit(code=code)[1]
        if down_limit is not None and price_now / down_limit <= 1.02:  # if nearly down limit, dont buy in 
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
        amount = calculate_buy_in_amount(funds=ONE_TIME_FUNDS, price=price_now)
        if amount == 0:
            return
        # if price is decreasing, dont buy in
        decreasing_or_not = is_decreasing_or_not(code=code, price_now=price_now)
        if decreasing_or_not:  # if decreasing, dont buy in
            return
        # 优化：实时获取持仓数量，避免多线程下全局变量不同步
        with lock:
            current_holding_stocks = get_holding_stocks_number()
            if current_holding_stocks >= MAX_STOCKS:
                return
            buy_in(code, price_now, amount, trade_date, buy_point_base, target_rate)

    # 循环+多线程扫描buy_in_list.csv
    idx_rows = list(buy_in_df.iterrows())
    steps = 8
    for start in range(0, len(idx_rows), steps):
        end = start + steps if start + steps < len(idx_rows) else len(idx_rows)
        idx_rows_batch = idx_rows[start:end]
        # 使用线程池执行多线程扫描  
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(scan_buy_in_list_row, idx_row) for idx_row in idx_rows_batch]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f'({MODEL_NAME}) scan_buy_in_list_row error: {e}')

# 持续刷新holding_list.csv
def refresh_holding_list():
    """ 
    ### 刷新持仓列表
    #### 刷新的列名: days, holding_days, price_now, profit, rate_current, rate_yearly
    """
    if not os.path.exists(HOLDING_LIST):
        create_holding_list()
        return
    with lock:
        holding_df = pd.read_csv(HOLDING_LIST, dtype={'trade_date': str, 'date_in': str})

    def refresh_holding_list_row(idx_row):
        """
        ### 刷新持仓列表单行信息
        #### :param idx_row: index, row
        """
        i, row = idx_row
        trade_date = row['trade_date']
        # if status is sold_out, skip
        if row['status'] == 'sold_out':
            return
        # if status is holding, update holding_days, days, price_now, price_in, amount, profit
        # rate_current, rate_yearly. holding_days(自然日历间隔天数)
        try:
            date_in = datetime.datetime.strptime(row['date_in'], '%Y%m%d')  # date format problem
            today = datetime.datetime.now()
            holding_days = (today - date_in).days + 1
            holding_df.loc[i, 'holding_days'] = holding_days
            # days(交易日历间隔天数today - trade_date)
            daily_csv = f'{BASICDATA_DIR}/dailydata/{row["ts_code"]}.csv'
            try:
                daily_df = pd.read_csv(daily_csv, dtype={'trade_date': str})
                daily_df = daily_df.sort_values(by='trade_date', ascending=True)
                trade_date_list = daily_df['trade_date'].tolist()
                today_str = today.strftime('%Y%m%d')
                if today_str in trade_date_list:
                    today_index = trade_date_list.index(today_str)
                else:
                    today_index = len(trade_date_list) - 1
                trade_date_index = trade_date_list.index(trade_date)
                days = abs(today_index - trade_date_index) 
                holding_df.loc[i, 'days'] = days
            except Exception as e:
                print(f"读取daily_csv失败: {daily_csv}, 错误: {e}")
                return
            # price_now, price_in, amount, profit, rate_current, rate_yearly
            price_now = get_stock_realtime_price(row['ts_code'])
            if price_now is None:
                return
            print(f'({MODEL_NAME}) {row["ts_code"]} {row["stock_name"]} price_now: {price_now}')
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
        except Exception as e:
            print(f"refresh_holding_list_row error: {e}")
    
    # 多线程刷新holding_list.csv
    idx_rows = list(holding_df.iterrows())
    steps = 8
    for start in range(0, len(idx_rows), steps):
        end = start + steps if start + steps < len(idx_rows) else len(idx_rows)
        idx_rows_batch = idx_rows[start:end]
        # 使用线程池执行多线程刷新
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(refresh_holding_list_row, idx_row) for idx_row in idx_rows_batch]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f'({MODEL_NAME}) refresh_holding_list_row error: {e}')
    with lock:
        holding_df.to_csv(HOLDING_LIST, index=False)

# 持续扫描holding_list.csv, 卖出股票
def scan_holding_list():
    """
    ### 扫描持仓列表，卖出股票
    """
    if not os.path.exists(HOLDING_LIST):
        create_holding_list()
        return
    with lock:
        holding_df = pd.read_csv(HOLDING_LIST, dtype={'trade_date': str})
    
    def scan_holding_list_row(idx_row):
        """
        ### 扫描持仓列表单行信息
        #### :param idx_row: index, row
        #### 卖出逻辑
        - 已经卖出的股票，跳过
        - 持有天数为1天(当天买入), 跳过
        - 未获取到实时价格，跳过
        - 停牌股票，跳过
        - 涨停板附近，跳过
        - 上涨过程中，跳过
        - 持有天数超过最大交易天数，卖出
        - 当前跌幅超过最大跌幅限制，卖出
        - 达到目标收益率，卖出
        - 提前卖出标准触发，卖出
        """
        i, row = idx_row
        if row['status'] == 'sold_out':
            return
        holding_days = row['holding_days']
        if holding_days == 1:
            return
        price_now = get_stock_realtime_price(row['ts_code'])
        print(f'({MODEL_NAME}) {row["ts_code"]} {row["stock_name"]} price_now: {price_now}')
        if price_now is None or price_now <= 0:
            return
        # if suspended stocks, skip
        if is_suspended_or_not(code=row['ts_code']):
            return
        # if nearly up limit, dont sell out
        up_limit = get_up_down_limit(code=row['ts_code'])[0]
        if up_limit is not None and price_now / up_limit >= 0.98:
            return
        # if rising now, dont sell out
        rising_or_not = is_rising_or_not(row['ts_code'], price_now)
        if rising_or_not:  # if rising, dont sell out
            return
        # if days > MAX_TRADE_DAYS, sell out
        days = row['days']
        if days >= MAX_TRADE_DAYS:
            with lock:
                sell_out(row['ts_code'], price_now, row['trade_date'])
            msg = f'卖出 {row["ts_code"]} {row["stock_name"]}: days {days} >= {MAX_TRADE_DAYS}'
            trade_log.info(msg)
            return
        # if rate_current <= MAX_DOWN_LIMIT, sell out
        rate_current = row['rate_current']
        if rate_current <= MAX_DOWN_LIMIT:
            with lock:
                sell_out(row['ts_code'], price_now, row['trade_date'])
            msg = f'卖出 {row["ts_code"]} {row["stock_name"]}: rate_current {rate_current:.2%} <= {MAX_DOWN_LIMIT:.2%}'
            trade_log.info(msg)
            return
        # if reach the target rate, sell out
        base_point = row['buy_point_base']
        target_rate = row['target_rate']
        if price_now >= base_point * (1 + target_rate):
            with lock:
                sell_out(row['ts_code'], price_now, row['trade_date'])
            msg = f'卖出 {row["ts_code"]} {row["stock_name"]}: reach the target_rate: {target_rate:.2%}'
            trade_log.info(msg)
            return
        # sell out early or not
        rate_yearly = row['rate_yearly']
        early_or_not = early_sell_standard_oversold_v2(
            holding_days=holding_days, target_rate=target_rate, rate_current=rate_current
        )
        if early_or_not:
            with lock:
                sell_out(row['ts_code'], price_now, row['trade_date'])
            msg = f'卖出 {row["ts_code"]} {row["stock_name"]}: trigger early sell standard: holding_days: {holding_days}, rate_yearly: {rate_yearly:.2%}'
            trade_log.info(msg)
            return
    # 多线程扫描holding_list.csv
    idx_rows = list(holding_df.iterrows())
    steps = 8
    for start in range(0, len(idx_rows), steps):
        end = start + steps if start + steps < len(idx_rows) else len(idx_rows)
        idx_rows_batch = idx_rows[start:end]
        # 使用线程池执行多线程扫描
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(scan_holding_list_row, idx_row) for idx_row in idx_rows_batch]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f'({MODEL_NAME}) scan_holding_list_row error: {e}')
    
def XD_buy_in_list_bak():
    """
    ### 盘中前复权 buy_point_base
    #### NOTE:
    #### 保存结果到 XD_RECORD_BUY_IN_CSV(只有一行),
    #### XD_RECORD_BUY_IN_CSV 包含列名: today, xd_or_not
    #### NOTE: 已弃用
    """
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
            # print(f'({MODEL_NAME}) {today} 买入清单今日已前复权, 不再处理')
            return

    def xd_buy_in_list_row(idx_row):
        i, row = idx_row
        ts_code = row['ts_code']
        trade_date = row['trade_date']
        buy_point_base = row['buy_point_base']
        xd_buy_point_base = get_qfq_price_by_adj_factor(
            code=ts_code, pre_price=buy_point_base, start=trade_date
        )
        if xd_buy_point_base == buy_point_base:
            return
        buy_in_df.loc[i, 'buy_point_base'] = xd_buy_point_base

    idx_rows = list(buy_in_df.iterrows())
    # 单一多线程模式优化为外层循环，内层多线程模式，每次循环使用 8 个线程, 减少线程创建和销毁的开销
    all_rows = len(idx_rows)
    step = 8
    for start in range(0, all_rows, step):
        end = start + step if start + step < all_rows else all_rows
        idx_rows_batch = idx_rows[start:end]
        with ThreadPoolExecutor() as executor:
            executor.map(xd_buy_in_list_row, idx_rows_batch)
    with lock:
        buy_in_df.to_csv(BUY_IN_LIST, index=False)
    xd_or_not = True
    xd_record_df = pd.DataFrame([[today, xd_or_not]], columns=['today', 'xd_or_not'])
    xd_record_df.to_csv(XD_RECORD_BUY_IN_CSV, index=False)

def XD_buy_in_list():
    """
    ### 通过BUY_IN_LIST_ORIGIN盘中前复权 buy_point_base
    """
    if not os.path.exists(BUY_IN_LIST):
        return
    if not os.path.exists(BUY_IN_LIST_ORIGIN):
        return
    with lock:
        buy_in_df = pd.read_csv(BUY_IN_LIST, dtype={'trade_date': str})
    origin_buy_in_df = pd.read_csv(BUY_IN_LIST_ORIGIN, dtype={'trade_date': str})
    today = datetime.datetime.now().strftime('%Y%m%d')

    def xd_buy_in_list_row(idx_row):
        i, row = idx_row
        # 两个文件顺序完全一致，通过索引号从BUY_IN_LIST_ORIGIN获取原始 buy_point_base,
        origin_row = origin_buy_in_df.iloc[i]
        ts_code = origin_row['ts_code']
        trade_date = origin_row['trade_date']
        buy_point_base = origin_row['buy_point_base']
        # 对BUY_IN_LIST除权
        xd_buy_point_base = get_qfq_price_by_adj_factor(
            code=ts_code, pre_price=buy_point_base, start=trade_date, end=today
        )
        if xd_buy_point_base == buy_point_base:
            return
        buy_in_df.loc[i, 'buy_point_base'] = xd_buy_point_base

    idx_rows = list(buy_in_df.iterrows())
    # 单一多线程模式优化为外层循环，内层多线程模式，每次循环使用 8 个线程, 减少线程创建和销毁的开销
    all_rows = len(idx_rows)
    step = 8
    for start in range(0, all_rows, step):
        end = start + step if start + step < all_rows else all_rows
        idx_rows_batch = idx_rows[start:end]
        with ThreadPoolExecutor() as executor:
            executor.map(xd_buy_in_list_row, idx_rows_batch)
    with lock:
        buy_in_df.to_csv(BUY_IN_LIST, index=False)

def XD_holding_list_bak():
    """
    ### 盘中前复权 buy_point_base 和 price_in, 对 amount 进行股数调整
    #### 前复权和股数调整记录在 XD_RECORD_CSV 中
    #### NOTE:
    #### XD_RECORD_CSV contains columns: ts_code, trade_date, xd_date, buy_point_base, 
    #### price_in, amount, xd_buy_point_base, xd_price_in, xd_amount  
    #### NOTE: 已弃用
    """
    if not os.path.exists(HOLDING_LIST):
        return
    with lock:
        holding_df = pd.read_csv(HOLDING_LIST, dtype={'trade_date': str, 'date_in': str})
        holding_df['date_in'] = holding_df['date_in'].apply(lambda x: str(x)[:8])
        holding_df['date_in'] = holding_df['date_in'].apply(lambda x: x if x != 'nan' else '')
    today = datetime.datetime.now().strftime('%Y%m%d')
    xd_record_df = []
    columns = ['ts_code', 'trade_date', 'xd_date', 'buy_point_base', 'price_in', 'amount', 
               'xd_buy_point_base', 'xd_price_in', 'xd_amount']

    def xd_holding_list_row(idx_row):
        i, row = idx_row
        if row['status'] == 'sold_out':
            return None
        ts_code = row['ts_code']
        trade_date = row['trade_date']
        date_in = row['date_in']
        buy_point_base = row['buy_point_base']
        price_in = row['price_in']
        amount = row['amount']
        tmp_list = [ts_code, trade_date, today, buy_point_base, price_in, amount]
        if not os.path.exists(XD_RECORD_HOLDGING_CSV):
            start_buy_point_base = trade_date
            start_price_in = date_in
        else:
            xd_df = pd.read_csv(XD_RECORD_HOLDGING_CSV, dtype={'trade_date': str, 'xd_date': str})
            xd_df = xd_df.sort_values(by=['ts_code', 'xd_date'], ascending=[True, True])
            xd_df = xd_df[xd_df['ts_code'] == ts_code]
            if xd_df.empty:
                start_buy_point_base = trade_date
                start_price_in = date_in
            else:
                last_xd_date = xd_df.iloc[-1]['xd_date']
                start_buy_point_base = last_xd_date if last_xd_date > trade_date else trade_date
                start_price_in = last_xd_date if last_xd_date > date_in else date_in
        xd_buy_point_base = get_qfq_price_by_adj_factor(
            code=ts_code, pre_price=buy_point_base, start=start_buy_point_base, end=today,
        )
        if xd_buy_point_base == buy_point_base:
            return None
        xd_price_in = get_qfq_price_by_adj_factor(
            code=ts_code, pre_price=price_in, start=start_price_in, end=today,
        )
        if xd_price_in == price_in:
            xd_amount = amount
            tmp_list.extend([xd_buy_point_base, xd_price_in, xd_amount])
            # 更新持仓表
            holding_df.loc[i, 'buy_point_base'] = xd_buy_point_base
            return tmp_list
        xd_amount = get_XR_adjust_amount_by_dividend_data(
            code=ts_code, amount=amount, start=start_price_in, end=today,
        )
        tmp_list.extend([xd_buy_point_base, xd_price_in, xd_amount])
        holding_df.loc[i, 'buy_point_base'] = xd_buy_point_base
        holding_df.loc[i, 'price_in'] = xd_price_in
        holding_df.loc[i, 'amount'] = xd_amount
        return tmp_list

    # 多线程处理持仓表
    idx_rows = list(holding_df.iterrows())
    step = 8
    for start in range(0, len(idx_rows), step):
        end = start + step if start + step < len(idx_rows) else len(idx_rows)
        idx_rows_batch = idx_rows[start:end]
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(xd_holding_list_row, idx_rows_batch))
        # 收集非None结果
        for r in results:
            if r is not None:
                xd_record_df.append(r)

    xd_df = pd.DataFrame(xd_record_df, columns=columns)
    if xd_df.empty:
        return
    columns_out = ['ts_code', 'trade_date', 'xd_date', 'buy_point_base', 'xd_buy_point_base',
                   'price_in', 'xd_price_in', 'amount', 'xd_amount']
    xd_df = xd_df[columns_out]
    if not os.path.exists(XD_RECORD_HOLDGING_CSV):
        xd_df.to_csv(XD_RECORD_HOLDGING_CSV, index=False)
    else:
        xd_df.to_csv(XD_RECORD_HOLDGING_CSV, mode='a', header=False, index=False)
    with lock:
        holding_df.to_csv(HOLDING_LIST, index=False)

def XD_holding_list():
    """
    ### 盘中前复权HOLDING_LIST中 buy_point_base 和 price_in, 对 amount 进行股数调整
    ### 从买入的原始记录 HOLDING_LIST_ORIGIN 中获取数据结合复权因子进行复权
    """
    if not os.path.exists(HOLDING_LIST):
        return
    if not os.path.exists(HOLDING_LIST_ORIGIN):
        return
    with lock:
        holding_df = pd.read_csv(HOLDING_LIST, dtype={'trade_date': str, 'date_in': str})
        # holding_df['date_in'] = holding_df['date_in'].apply(lambda x: str(x)[:8])
        # holding_df['date_in'] = holding_df['date_in'].apply(lambda x: x if x != 'nan' else '')
    origin_holding_df = pd.read_csv(HOLDING_LIST_ORIGIN, dtype={'trade_date': str, 'date_in': str})
    today = datetime.datetime.now().strftime('%Y%m%d')

    def xd_holding_list_row(idx_row):
        i, row = idx_row
        if row['status'] == 'sold_out':
            return None
        ts_code = row['ts_code']
        trade_date = row['trade_date']
        # 从origin_holding_df中获取date_in buy_point_base price_in 和 amount
        origin_res_row = origin_holding_df[(origin_holding_df['trade_date'] == trade_date) & (origin_holding_df['ts_code'] == ts_code)]
        if origin_res_row.empty:
            return
        date_in = origin_res_row.iloc[0]['date_in']
        buy_point_base = origin_res_row.iloc[0]['buy_point_base']
        price_in = origin_res_row.iloc[0]['price_in']
        amount = origin_res_row.iloc[0]['amount']
        # 除权
        xd_buy_point_base = get_qfq_price_by_adj_factor(
            code=ts_code, pre_price=buy_point_base, start=trade_date, end=today,
        )
        xd_price_in = get_qfq_price_by_adj_factor(
            code=ts_code, pre_price=price_in, start=date_in, end=today,
        )
        xd_amount = get_XR_adjust_amount_by_dividend_data(
            code=ts_code, amount=amount, start=trade_date, end=today,
        )
        holding_df.loc[i, 'buy_point_base'] = xd_buy_point_base
        holding_df.loc[i, 'price_in'] = xd_price_in
        holding_df.loc[i, 'amount'] = xd_amount

    # 多线程处理持仓表
    idx_rows = list(holding_df.iterrows())
    step = 8
    for start in range(0, len(idx_rows), step):
        end = start + step if start + step < len(idx_rows) else len(idx_rows)
        idx_rows_batch = idx_rows[start:end]
        with ThreadPoolExecutor() as executor:
            list(executor.map(xd_holding_list_row, idx_rows_batch))
    with lock:
        holding_df.to_csv(HOLDING_LIST, index=False)

def trade_process(mode: Literal['trade', 'test'] = 'trade'):
    """
    ### 交易流程
    #### :param mode: trade or test, default is trade
    #### :return: None
    #### NOTE: 
    - 'trade' 模式下, 在实际交易时间内执行交易逻辑。
    - 'test' 模式下, 在非交易时间执行交易逻辑, 主要为了检测交易逻辑是否正确。
    """
    if mode not in ['trade', 'test']:
        print(f'Invalid mode: {mode}. Use "trade" or "test".')
        return
    
    def is_within_trading_hours():
        now = datetime.datetime.now().time()
        am_begin = datetime.time(9, 30)
        am_end = datetime.time(11, 30)
        pm_begin = datetime.time(13, 0)
        pm_end = datetime.time(15, 0)
        return (am_begin <= now <= am_end or pm_begin <= now <= pm_end)

    def one_trade_loop(mode=mode):
        if mode == 'trade':
            refresh_holding_list()
        scan_buy_in_list()
        scan_holding_list()
        create_daily_profit_list()
        if mode == 'test':
            refresh_holding_list()

    if mode == 'trade' and is_within_trading_hours():
        # 在实际交易时间内执行交易逻辑
        one_trade_loop()
    if mode == 'test' and not is_within_trading_hours():
        # 在非交易时间测试交易逻辑
        import shutil
        shutil.copytree(trade_dir, f'{trade_dir}_copy', dirs_exist_ok=True)
        # 删除 trade_dir中不含有'buy_in_list'的文件
        for root, dirs, files in os.walk(trade_dir):
            for file in files:
                if 'buy_in_list' not in file:
                    os.remove(os.path.join(root, file))
        # 测试交易逻辑
        one_trade_loop()
        # 把 trade_dir 复制到 TEST_DIR/oversold
        shutil.copytree(trade_dir, f'{TEST_DIR}/oversold', dirs_exist_ok=True)
        # 恢复原始交易目录
        shutil.copytree(f'{trade_dir}_copy', trade_dir, dirs_exist_ok=True)
        shutil.rmtree(f'{trade_dir}_copy')

def build_buy_in_list():
    """
    ### 构建买入清单 buy_in_list.csv
    """
    print('creating and saving sub buy_in_list csv files...')
    for dataset in dataset_to_predict_trade:
        FORWARD_DAYS = dataset['FORWARD_DAYS']
        BACKWARD_DAYS = dataset['BACKWARD_DAYS']
        DOWN_FILTER = dataset['DOWN_FILTER']
        PRED_MODELS = dataset['PRED_MODELS']
        if PRED_MODELS == 0:
            continue

        # prepare data from trade_pred.csv within TRADE_COVERAGE_DAYS days
        pred_path = f'{PREDICT_DIR}/oversold/pred_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}'
        os.makedirs(pred_path, exist_ok=True)
        csv_name = f'{pred_path}/trade_pred_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}.csv'
        trade_df = pd.read_csv(csv_name, dtype={'trade_date': str})

        # add trade_date's close price (buy_point_base) and waiting_days to trade_df
        trade_df.insert(4, 'buy_point_base', None)
        trade_df.insert(9, 'waiting_days', None)
        for i , row in trade_df.iterrows():
            code = row['code']
            industry = row['industry']
            trade_date = row['trade_date']
            daily_csv = f'{BASICDATA_DIR}/dailydata/{code}.csv'
            daily_df = pd.read_csv(daily_csv, dtype={'trade_date': str})
            close = daily_df[daily_df['trade_date'] == trade_date]['close'].values[0]
            trade_df.loc[i, 'buy_point_base'] = close

            # calculate days between trade_date and today
            daily_csv = f'{BASICDATA_DIR}/dailydata/{code}.csv' 
            daily_df = pd.read_csv(daily_csv, dtype={'trade_date': str})
            daily_df = daily_df.sort_values(by='trade_date', ascending=True)
            today = datetime.datetime.now().strftime('%Y%m%d')
            trade_dates = daily_df['trade_date'].tolist()
            trade_date_index = trade_dates.index(trade_date)
            if today not in trade_dates:
                today_index = len(trade_dates) - 1
                days = today_index - trade_date_index + 1  # 今日未收盘，未在daily_df中，所以+1
            else:
                today_index = trade_dates.index(today)
                days = today_index - trade_date_index
            trade_df.loc[i, 'waiting_days'] = days
        trade_df['pred_100%'] = trade_df['pred'].map(lambda x: f'{x:.2%}')

        # filter trade_df
        trade_df = trade_df[trade_df['pred'] > MIN_PRED_RATE]
        trade_df = trade_df[~trade_df['name'].str.contains('|'.join(exception_list))]
        trade_df = trade_df.reset_index(drop=True)
        # save trade_df to csv
        trade_dir = f'{TRADE_DIR}/oversold'
        os.makedirs(trade_dir, exist_ok=True)
        trade_df.to_csv(f'{trade_dir}/buy_in_list_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}.csv', index=False)
        print(f'buy_in_list_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}.csv saved')

    trade_dir = f'{TRADE_DIR}/oversold'
    os.makedirs(trade_dir, exist_ok=True)
    all_df = pd.DataFrame()
    for dataset in dataset_to_predict_trade:
        FORWARD_DAYS = dataset['FORWARD_DAYS']
        BACKWARD_DAYS = dataset['BACKWARD_DAYS']
        DOWN_FILTER = dataset['DOWN_FILTER']
        PRED_MODELS = dataset['PRED_MODELS']
        if PRED_MODELS == 0:
            continue
        list_name = f'buy_in_list_{FORWARD_DAYS}_{BACKWARD_DAYS}_{-DOWN_FILTER:.2f}.csv'
        buy_in_csv = f'{trade_dir}/{list_name}'
        buy_in_df = pd.read_csv(buy_in_csv)
        buy_in_df['src'] = list_name
        all_df = pd.concat([all_df, buy_in_df], ignore_index=True)
    # drop duplicate rows with same code and trade_date, save the highest pred row
    all_df = all_df.sort_values(by=['code', 'trade_date', 'pred'], ascending=[True, True, True])
    all_df = all_df.drop_duplicates(subset=['code', 'trade_date'], keep='last')
    # rename columns and save to buy_in_list.csv
    new_columns = ['ts_code', 'stock_name', 'industry', 'trade_date', 'buy_point_base', 'chg_pct', 'max_date_forward', 
                    'max_down_rate', 'forward_days', 'waiting_days', 'pred', 'real', 'pred_100%', 'src']
    all_df.columns = new_columns
    # 检查 ts_code 相同的行，选出其中最小的 waiting_days，如果 MAX_TRADE_DAYS - 最小waiting_days < REST_TRADE_DAYS，
    # 则舍弃这些行，以保证买入后的股票有足够的交易天数等待股价的反弹。舍弃这些行的目的是为了减少后续买入程序中检查的计算量。当一个
    # 下跌趋势在横盘了较长时间后，又开始下跌，其新的序列的 waiting_days从 1 重新开始，原先不被删除的某些行会重新加入到买入清单中。 
    ts_codes = all_df['ts_code'].unique().tolist()
    indices_to_drop = []
    for ts_code in ts_codes:
        df_code = all_df[all_df['ts_code'] == ts_code]
        min_waiting_days = df_code['waiting_days'].min()
        if MAX_TRADE_DAYS - min_waiting_days < REST_TRADE_DAYS:
            indices_to_drop.extend(df_code.index.tolist())
    all_df = all_df.drop(indices_to_drop)
    all_df.to_csv(f'{trade_dir}/buy_in_list.csv', index=False)  # BUY_IN_LIST
    shutil.copy(f'{trade_dir}/buy_in_list.csv', BUY_IN_LIST_ORIGIN)  # BUY_IN_LIST_ORIGIN for xd
    print(f'buy_in_list.csv saved with {len(all_df)} records and {len(ts_codes)} stocks')

if __name__ == '__main__':
    import time
    start = time.time()
    trade_process(mode='test')  # or mode='trade' for actual trading
    end = time.time()
    print(f'Execution time: {end - start:.2f} seconds')
