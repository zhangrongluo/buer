import os
import time
import shutil
import datetime
import datetime
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from trade_downgap import trade_process, XD_buy_in_list, XD_holding_list, build_buy_in_list
from cons_general import TRADE_CAL_CSV, TRADE_DIR, BACKUP_DIR
from cons_downgap import MODEL_NAME, dataset_group_cons
from utils import calculate_today_series_statistic_indicator
from datasets_downgap import update_dataset
from model_downgap import train_dataset
from predict_downgap import predict_dataset

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

backup_root = f'{BACKUP_DIR}/downgap'
os.makedirs(backup_root, exist_ok=True)

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

# 定时任务函数定义
@is_trade_day(task='更新预测数据集')
def update_and_predict_dataset():
    update_dataset()
    predict_dataset()
    today = datetime.datetime.now().strftime('%Y%m%d')
    print(f'({MODEL_NAME}) {today} 数据更新完成')

@is_trade_day(task='更新买入清单')
def update_buy_in_list():
    build_buy_in_list()
    today = datetime.datetime.now().strftime('%Y%m%d')
    print(f'({MODEL_NAME}) {today} 买入清单更新完成')

def train_and_predict_model():
    train_dataset()
    predict_dataset()
    today = datetime.datetime.now().strftime('%Y%m%d')
    print(f'({MODEL_NAME}) {today} 模型训练完成')

@is_trade_day(task='盘中前复权和股数调整')
def XD_stock_list_task():
    today = datetime.datetime.now().strftime('%Y%m%d')
    for group in dataset_group_cons:
        max_trade_days = dataset_group_cons[group].get('MAX_TRADE_DAYS')
        model_name = dataset_group_cons[group].get('MODEL_NAME')
        if max_trade_days is None:
            continue
        XD_buy_in_list(max_trade_days=max_trade_days)
        print(f'({model_name}) {today} 买入清单盘中前复权完成')
        XD_holding_list(max_trade_days=max_trade_days)
        print(f'({model_name}) {today} 持有清单盘中前复权和股数调整完成')
    print(f'({MODEL_NAME}) {today} 盘中前复权和股数调整完成')

@is_trade_day(task='统计各项指标')
def calculate_today_statistics_indicators():
    for group in dataset_group_cons:
        max_trade_days = dataset_group_cons[group].get('MAX_TRADE_DAYS')
        if max_trade_days is None:
            continue
        model_name = dataset_group_cons[group].get('MODEL_NAME')
        calculate_today_series_statistic_indicator(name='downgap', max_trade_days=max_trade_days)
        print(f'({model_name}) 统计指标计算完成')
    today = datetime.datetime.now().strftime('%Y%m%d')
    print(f'({MODEL_NAME}) {today} 今日统计数据计算完成！')

@is_trade_day(task='备份交易数据')
def backup_trade_data():
    """
    把TRADE_DIR/downgap/max_trade_days_{MAX_TRADE_DAYS}下的所有文件和目录备份到
    BACKUP_DIR/downgap/max_trade_days_{MAX_TRADE_DAYS}_<datetime>目录下
    NOTE:
    备份清单包括: 买入清单、持有清单、交易日志、资金流水、每日利润、每日指标文件,
    保留最近的6个备份
    """
    for group in dataset_group_cons:
        max_trade_days = dataset_group_cons[group].get('MAX_TRADE_DAYS')
        model_name = dataset_group_cons[group].get('MODEL_NAME')
        if max_trade_days is None:
            continue
        trade_dir = f'{TRADE_DIR}/downgap/max_trade_days_{max_trade_days}'
        now = datetime.datetime.now().strftime('%Y%m%d %H%M%S')
        group_backup_root = f'{backup_root}/max_trade_days_{max_trade_days}'
        os.makedirs(group_backup_root, exist_ok=True)
        backup_dir = f'{group_backup_root}/backup_{now}'
        shutil.copytree(trade_dir, backup_dir, dirs_exist_ok=True)
        files = os.listdir(group_backup_root)
        dirs = [d for d in files if os.path.isdir(os.path.join(group_backup_root, d))]
        dirs.sort(reverse=True)
        [shutil.rmtree(os.path.join(group_backup_root, d)) for d in dirs[12:]]  # 保留最近12个备份
        print(f'({model_name}) 交易数据备份完成！')
    today = datetime.datetime.now().strftime('%Y%m%d')
    print(f'({MODEL_NAME}) {today} 交易数据备份完成！')

# 动态任务am
@is_trade_day(task='股票交易')
def trading_task_am(scheduler, max_trade_days: int):
    """
    上午交易任务
    :param scheduler: apscheduler调度器
    :param max_trade_days: dataset group_id
    """
    now = datetime.datetime.now().time()
    start_time = datetime.time(9, 20)  # 9:20 AM, start of trading
    end_time = datetime.time(11, 30)  # 11:30 AM, end of trading
    if start_time <= now <= end_time:
        try:
            trade_process(max_trade_days=max_trade_days)
        except Exception as e:
            print(f"Error during trading process: {e}")
        # 启动下一个交易loop
        if datetime.datetime.now().time() <= end_time:
            scheduler.add_job(
                trading_task_am,
                args=[scheduler, max_trade_days],
                run_date=datetime.datetime.now()+datetime.timedelta(seconds=1),
                id=f'{MODEL_NAME}_trading_job_am_{max_trade_days}',
            )
        else:
            now = datetime.datetime.now()
            print(f'({MODEL_NAME}) {now.time()} 不在交易时间段内.')
    else:
        now = datetime.datetime.now()
        print(f'({MODEL_NAME}) {now.time()} 不在交易时间段内.')

# 动态任务pm
@is_trade_day(task='股票交易')
def trading_task_pm(scheduler, max_trade_days: int):
    """
    下午交易任务
    :param scheduler: apscheduler调度器
    :param max_trade_days: dataset group_id
    """
    now = datetime.datetime.now().time()
    start_time = datetime.time(12, 50)
    end_time = datetime.time(15, 0)
    if start_time <= now <= end_time:
        try:
            trade_process(max_trade_days=max_trade_days)
        except Exception as e:
            print(f"Error during trading process: {e}")
        # 启动下一个交易loop
        if datetime.datetime.now().time() <= end_time:
            scheduler.add_job(
                trading_task_pm,
                args=[scheduler, max_trade_days],
                run_date=datetime.datetime.now()+datetime.timedelta(seconds=1),
                id=f'{MODEL_NAME}_trading_job_pm_{max_trade_days}',
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
    # 定时任务通过add_job方式添加
    scheduler.add_job(
        update_buy_in_list,
        trigger='cron',
        hour=0, minute=30, misfire_grace_time=300,
        id='build_buy_in_list',
        name='每天00:30创建买入清单'
    )
    scheduler.add_job(
        XD_stock_list_task,
        trigger='cron',
        hour=9, minute=32, second=45, misfire_grace_time=300,
        id='XD_stock_list_task',
        name='每天9:32盘中前复权和股数调整'
    )
    scheduler.add_job(
        trading_task_am,
        args=[scheduler, 50],
        trigger='cron',
        hour=9, minute=35, second=5, misfire_grace_time=300,
        id=f'{MODEL_NAME}_start_trading_job_am_50',
        name='Start_trading_program_at_9:35_AM_50',
    )
    scheduler.add_job(
        trading_task_am,
        args=[scheduler, 45],
        trigger='cron',
        hour=9, minute=35, second=10, misfire_grace_time=300,
        id=f'{MODEL_NAME}_start_trading_job_am_45',
        name='Start_trading_program_at_9:35_AM_45',
    )
    scheduler.add_job(
        trading_task_am,
        args=[scheduler, 60],
        trigger='cron',
        hour=9, minute=35, second=15, misfire_grace_time=300,
        id=f'{MODEL_NAME}_start_trading_job_am_60',
        name='Start_trading_program_at_9:35_AM_60',
    )
    scheduler.add_job(
        backup_trade_data,
        trigger='cron',
        hour=11, minute=45, misfire_grace_time=300,
        id='backup_trade_data_am'
    )
    scheduler.add_job(
        trading_task_pm,
        args=[scheduler, 50],
        trigger='cron',
        hour=12, minute=55, misfire_grace_time=300,
        id=f'{MODEL_NAME}_start_trading_job_pm_50',
        name='Start_trading_program_at_12:55_PM_50',
    )
    scheduler.add_job(
        trading_task_pm,
        args=[scheduler, 45],
        trigger='cron',
        hour=12, minute=56, misfire_grace_time=300,
        id=f'{MODEL_NAME}_start_trading_job_pm_45',
        name='Start_trading_program_at_12:56_PM_45',
    )
    scheduler.add_job(
        trading_task_pm,
        args=[scheduler, 60],
        trigger='cron',
        hour=12, minute=57, misfire_grace_time=300,
        id=f'{MODEL_NAME}_start_trading_job_pm_60',
        name='Start_trading_program_at_12:57_PM_60',
    )
    scheduler.add_job(
        calculate_today_statistics_indicators,
        trigger='cron',
        hour=15, minute=5, misfire_grace_time=300,
        id='calculate_today_statistics_indicators',
        name='每天15:05计算今日统计数据'
    )
    scheduler.add_job(
        backup_trade_data,
        trigger='cron',
        hour=15, minute=15, misfire_grace_time=300,
        id='backup_trade_data',
        name='每天15:15备份交易数据'
    )
    scheduler.add_job(
        update_and_predict_dataset,
        trigger='cron',
        hour=17, minute=15, misfire_grace_time=300,
        id='update_predict_dataset',
        name='每日17:15更新预测数据集'
    )
    scheduler.add_job(
        train_and_predict_model,
        trigger='cron',
        day_of_week='sun', hour=1, minute=0, misfire_grace_time=300,
        id='train_predict_dataset',
        name='每周日上午1:00训练模型'
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