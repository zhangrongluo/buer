import os
import time # type: ignore
import datetime
import shutil
from tensorflow.keras import layers  # type: ignore
import pandas as pd  # type: ignore
from tensorflow import keras  
from apscheduler.schedulers.background import BackgroundScheduler
from utils import calculate_today_series_statistic_indicator, check_pre_trade_data_update_status, send_message_via_bark, check_daily_temp_data_update_status
from stocklist import get_stock_list, get_trade_cal, get_up_down_limit_list, get_suspend_stock_list, load_list_df
from basic_data_alt_edition import (update_all_daily_data, update_all_daily_indicator, update_all_daily_simple_quant_factor, update_all_adj_factor_data,
                                    download_all_stocks_daily_temp_adjfactor_data, download_all_stocks_daily_temp_data, 
                                    download_all_stocks_daily_temp_indicator_data, download_all_stocks_daily_simple_temp_quant_factor)
from trade_oversold import trade_process, XD_holding_list, XD_buy_in_list, build_buy_in_list
from cons_hidden import bark_device_key
from cons_general import TRADE_CAL_CSV, TRADE_DIR, BACKUP_DIR, BASICDATA_DIR
from cons_oversold import MODEL_NAME
from datasets_oversold import update_dataset
from model_oversold import train_dataset
from predict_oversold import predict_dataset

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def is_trade_day(task: str = None):
    """
    装饰器,判断是否为交易日,如果是则执行数据更新函数.
    :param task: 任务名称
    :return: 装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            df = pd.read_csv(TRADE_CAL_CSV, dtype={'cal_date': str})
            df = df.sort_values(by='cal_date', ascending=False)
            res = df["is_open"][0]
            if res:
                return func(*args, **kwargs)
            else:
                today = datetime.datetime.now().strftime('%Y%m%d')
                print(f'({MODEL_NAME}) {today} 不是交易日, 不执行 <{task}> 任务')
        return wrapper    
    return decorator

scheduler = BackgroundScheduler()
scheduler.configure(timezone='Asia/Shanghai')

def update_trade_cal_and_stock_list():
    get_trade_cal()
    get_stock_list()
    total_stocks = load_list_df()
    today = datetime.datetime.now().date().strftime('%Y%m%d')
    print(f'({MODEL_NAME}) {today} 现追踪的股票列表总数: {total_stocks}')
    print(f'({MODEL_NAME}) {today} 交易日历和股票列表更新完成！')

@is_trade_day(task='更新行情和指标数据')
def update_daily_data_and_indicator():
    download_all_stocks_daily_temp_data()
    download_all_stocks_daily_temp_indicator_data()
    download_all_stocks_daily_simple_temp_quant_factor()
    update_all_daily_data(step=5)
    update_all_daily_indicator(step=5)
    update_all_daily_simple_quant_factor(step=5)
    today = datetime.datetime.now().date().strftime('%Y%m%d')
    print(f'({MODEL_NAME}) {today} 行情和指标数据更新完成！')

@is_trade_day(task='更新和预测数据集')
def update_and_predict_dataset():
    update_dataset()
    predict_dataset()
    today = datetime.datetime.now().date().strftime('%Y%m%d')
    print(f'({MODEL_NAME}) {today} oversold 数据集更新完成！')

@is_trade_day(task='清理实时价格序列文件')
def delete_realtime_price_csv_task():
    rt_dir = f'{BASICDATA_DIR}/realtime'
    files = os.listdir(rt_dir)
    for file in files:
        file_path = os.path.join(rt_dir, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f'删除文件 {file_path} 失败: {e}')
    today = datetime.datetime.now().date().strftime('%Y%m%d')
    print(f'({MODEL_NAME}) {today} 清理实时价格文件完成！')

@is_trade_day(task='构建买入清单')
def build_buy_in_list_task():
    build_buy_in_list()
    today = datetime.datetime.now().date().strftime('%Y%m%d')
    print(f'({MODEL_NAME}) {today} 买入清单更新完成！')

@is_trade_day(task='获取涨跌停表、停牌清单数据')
def get_limit_and_suspend_list_task():
    get_up_down_limit_list()
    today = datetime.datetime.now().date().strftime('%Y%m%d')
    print(f'({MODEL_NAME}) {today} 涨跌停表更新完成！')
    get_suspend_stock_list()
    print(f'({MODEL_NAME}) {today} 停牌清单更新完成！')

@is_trade_day(task='更新复权因子数据')
def download_and_update_adj_data_task():
    today = datetime.datetime.now().date().strftime('%Y%m%d')
    download_all_stocks_daily_temp_adjfactor_data()
    update_all_adj_factor_data()
    print(f'({MODEL_NAME}) {today} 复权因子数据更新完成！')

@is_trade_day(task='盘前前复权和股数调整')
def XD_stock_list_task():
    XD_buy_in_list()
    today = datetime.datetime.now().date().strftime('%Y%m%d')
    print(f'({MODEL_NAME}) {today} 买入清单前复权调整完成！')
    XD_holding_list()
    print(f'({MODEL_NAME}) {today} 持有清单前复权和股数调整完成！')

@is_trade_day(task='检查盘前数据更新状态')
def check_pre_trade_data_update_status_task():
    status = check_pre_trade_data_update_status()
    send_message_via_bark(device_key=bark_device_key, message=status, title='检查盘前数据更新状态')

@is_trade_day(task='检查盘后数据更新状态')
def check_daily_temp_data_update_status_task():
    status = check_daily_temp_data_update_status()
    send_message_via_bark(device_key=bark_device_key, message=status, title='检查盘后数据更新状态')

@is_trade_day(task='计算今日统计指标')
def calculate_today_statistics_indicators():
    calculate_today_series_statistic_indicator(name='oversold')
    today = datetime.datetime.now().date().strftime('%Y%m%d')
    print(f'({MODEL_NAME}) {today} 今日统计数据计算完成！')

@is_trade_day(task='清理屏幕')
def clear_screen_task():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f'({MODEL_NAME}) oversold 模型自动运行中')

def train_and_predict_dataset():
    train_dataset()
    predict_dataset()
    today = datetime.datetime.now().date().strftime('%Y%m%d')
    print(f'({MODEL_NAME}) {today} oversold 模型训练完成！')

@is_trade_day(task='备份交易数据')
def backup_trade_data():
    """
    把TRADE_DIR/oversold目录下的所有文件备份到BACKUP_DIR/oversold/oversold_<备份时间>目录下
    NOTE:
    备份清单包括: 买入清单、持有清单、交易日志、资金流水、每日利润、每日指标文件
    保留最近的6个备份
    """
    trade_dir = f'{TRADE_DIR}/oversold'
    backup_root = f'{BACKUP_DIR}/oversold'
    backup_dir = f'{backup_root}/oversold_{datetime.datetime.now().strftime("%Y%m%d %H%M%S")}'
    shutil.copytree(trade_dir, backup_dir, dirs_exist_ok=True)
    files = os.listdir(backup_root)
    dirs = [d for d in files if os.path.isdir(os.path.join(backup_root, d))]
    dirs.sort(reverse=True)
    [shutil.rmtree(os.path.join(backup_root, d)) for d in dirs[12:]]  # 保留最近12个备份
    print(f'({MODEL_NAME}) oversold 模型交易数据备份完成！')

# 动态任务am
@is_trade_day(task='股票交易')
def trading_task_am(scheduler):
    now = datetime.datetime.now().time()
    start_time = datetime.time(9, 20)  # 9:20 AM, start of trading
    end_time = datetime.time(11, 30)  # 11:30 AM, end of trading
    if start_time <= now <= end_time:
        try:
            trade_process()
        except Exception as e:
            print(f"Error during trading process: {e}")
        # 启动下一个交易loop
        if datetime.datetime.now().time() <= end_time:
            scheduler.add_job(
                trading_task_am,
                args=[scheduler],
                run_date=datetime.datetime.now()+datetime.timedelta(seconds=1),
                id=f'{MODEL_NAME}_trading_job_am_{int(time.time())}',
                name='Run_am_trading_task'
            )
        else:
            now = datetime.datetime.now()
            print(f'({MODEL_NAME}) {now.time()} 不在交易时间段内.')
    else:
        now = datetime.datetime.now()
        print(f'({MODEL_NAME}) {now.time()} 不在交易时间段内.')

# 动态任务pm
@is_trade_day(task='股票交易')
def trading_task_pm(scheduler):
    now = datetime.datetime.now().time()
    start_time = datetime.time(12, 50)
    end_time = datetime.time(15, 0)
    if start_time <= now <= end_time:
        try:
            trade_process()
        except Exception as e:
            print(f"Error during trading process: {e}")
        # 启动下一个交易loop
        if datetime.datetime.now().time() <= end_time:
            scheduler.add_job(
                trading_task_pm,
                args=[scheduler],
                run_date=datetime.datetime.now()+datetime.timedelta(seconds=1),
                id=f'{MODEL_NAME}_trading_job_pm_{int(time.time())}',
                name='Run_pm_trading_task'
            )
        else:
            now = datetime.datetime.now()
            print(f'({MODEL_NAME}) {now.time()} 不在交易时间段内.')
    else:
        now = datetime.datetime.now()
        print(f'({MODEL_NAME}) {now.time()} 不在交易时间段内.')

def auto_run():
    """
    自动运行函数
    """
    # 定时任务注册
    scheduler.add_job(
        update_trade_cal_and_stock_list,
        trigger='cron',
        hour=0, minute=1, misfire_grace_time=300,
        id='update_trade_cal_and_stock_list',
        name='更新交易日历和股票列表'
    )
    scheduler.add_job(
        delete_realtime_price_csv_task,
        trigger='cron',
        hour=0, minute=15, misfire_grace_time=300,
        id='delete_realtime_price_csv_task',
        name='删除实时价格序列文件'
    )
    scheduler.add_job(
        build_buy_in_list_task,
        trigger='cron',
        hour=1, minute=0, misfire_grace_time=300,
        id='update_buy_in_list',
        name='创建股票买入清单'
    )
    scheduler.add_job(
        get_limit_and_suspend_list_task,
        trigger='cron',
        hour=9, minute=20, misfire_grace_time=300,
        id='get_limit_and_suspend_list_task',
        name='获取涨跌停和停牌列表'
    )
    scheduler.add_job(
        get_up_down_limit_list,
        trigger='cron',
        hour=9, minute=28, misfire_grace_time=300,
        id='get_up_down_limit_list_again',
        name='再次获取涨跌停列表'
    )
    scheduler.add_job(
        download_and_update_adj_data_task,
        trigger='cron',
        hour=9, minute=30, second=15, misfire_grace_time=300,
        id='download_and_update_adj_data_task',
        name='下载并更新复权数据'
    )
    scheduler.add_job(
        XD_stock_list_task,
        trigger='cron',
        hour=9, minute=32, second=45, misfire_grace_time=300,
        id='XD_stock_list_task',
        name='前复权股票买入清单和股票持有清单'
    )
    scheduler.add_job(
        check_pre_trade_data_update_status_task,
        trigger='cron',
        hour=9, minute=34, second=30, misfire_grace_time=300,
        id='check_pre_trade_data_update_status_task',
        name='检查交易前数据更新状态'
    )
    scheduler.add_job(
        trading_task_am,
        args=[scheduler],
        trigger='cron',
        hour=9, minute=35, misfire_grace_time=300,
        id=f'{MODEL_NAME}_start_trading_job_am',
        name='Start_trading_program_at_09:35_AM',
    )
    scheduler.add_job(
        backup_trade_data,
        trigger='cron',
        hour=11, minute=45, misfire_grace_time=300,
        id='backup_trade_data_am',
        name='备份上午交易数据'
    )
    scheduler.add_job(
        trading_task_pm,
        args=[scheduler],
        trigger='cron',
        hour=12, minute=55, misfire_grace_time=300,
        id=f'{MODEL_NAME}_start_trading_job_pm',
        name='Start_trading_program_at_12:55_PM',
    )
    scheduler.add_job(
        calculate_today_statistics_indicators,
        trigger='cron',
        hour=15, minute=5, misfire_grace_time=300,
        id='calculate_today_statistics_indicators',
        name='计算当天统计指标'
    )
    scheduler.add_job(
        backup_trade_data,
        trigger='cron',
        hour=15, minute=15, misfire_grace_time=300,
        id='backup_trade_data_pm',
        name='备份下午交易数据'
    )
    scheduler.add_job(
        update_daily_data_and_indicator,
        trigger='cron',
        hour=17, minute=5, misfire_grace_time=300,
        id='update_daily_data_and_indicator',
        name='更新每日数据、指标和简版量化因子数据'
    )
    scheduler.add_job(
        update_and_predict_dataset,
        trigger='cron',
        hour=17, minute=45, misfire_grace_time=300,
        id='update_and_predict_dataset',
        name='更新和预测数据集'
    )
    scheduler.add_job(
        check_daily_temp_data_update_status_task,
        trigger='cron',
        hour=19, minute=15, misfire_grace_time=300,
        id='check_daily_temp_data_update_status_task',
        name='检查每日盘后数据更新状态'
    )
    scheduler.add_job(
        train_and_predict_dataset,
        trigger='cron',
        day_of_week='sun', hour=1, minute=0, misfire_grace_time=300,
        id='train_and_predict_dataset',
        name='周日训练和预测数据集'
    )
    scheduler.start()
    print(f'({MODEL_NAME}) 开始启动自动运行,按CTRL+C退出')
    try:
        while True:
            time.sleep(5)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        print(f'{MODEL_NAME} 自动运行已关闭')

if __name__ == '__main__':
    auto_run()