"""
oversold and downgap general auto task entrance
"""
import os
import time
import subprocess
import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from basic_data import update_all_daily_data, update_all_daily_indicator
from auto_jobs_oversold import is_trade_day
from auto_jobs_oversold import update_dataset as update_dataset_oversold
from auto_jobs_oversold import train_dataset as train_dataset_oversold
from auto_jobs_oversold import predict_dataset as predict_dataset_oversold
from auto_jobs_oversold import build_buy_in_list as build_buy_in_list_oversold
from auto_jobs_oversold import trading_task as trading_task_oversold
from auto_jobs_downgap import update_dataset as update_dataset_downgap
from auto_jobs_downgap import train_dataset as train_dataset_downgap
from auto_jobs_downgap import predict_dataset as predict_dataset_downgap
from auto_jobs_downgap import build_buy_in_list as build_buy_in_list_downgap
from auto_jobs_downgap import trading_task as trading_task_downgap

general_scheduler = BackgroundScheduler()

# 每个小时清理一次屏幕
@general_scheduler.scheduled_job(
    trigger='cron',
    hour='*',
    minute=0,
    misfire_grace_time=300,
    id='clear_screen'
)
def clear_screen():
    """
    clear screen every hour
    """
    print("clear screen")
    subprocess.call('cls' if os.name == 'nt' else 'clear', shell=True)

# 每天 4:30 PM 更新基础数据
@general_scheduler.scheduled_job(
    trigger='cron',
    hour=16,
    minute=30,
    misfire_grace_time=300,
    id='update_all_daily_data_indicator',
)
@is_trade_day
def update_all_daily_data_indicator():
    """
    update all daily data and indicator
    """
    update_all_daily_data()
    update_all_daily_indicator()
    today = datetime.date.today().strftime('%Y%m%d')
    print(f"{today} 基础数据更新完成")

# 每天 6:00 PM 更新 oversol 和 downgap 数据集
@general_scheduler.scheduled_job(
    trigger='cron',
    hour=18,
    minute=0,
    misfire_grace_time=300,
    id='update_and_predict_dataset'
)
@is_trade_day
def update_and_predict_dataset():
    """
    update oversold and downgap dataset
    """
    print("update oversold and downgap dataset")
    update_dataset_oversold()
    predict_dataset_oversold()
    update_dataset_downgap()
    predict_dataset_downgap()
    today = datetime.date.today().strftime('%Y%m%d')
    print(f"{today} oversold 和 downgap 数据集更新完成")

# 每天 1：00 AM 构建 oversol 和 downgap 买入列表
@general_scheduler.scheduled_job(
    trigger='cron',
    hour=1,
    minute=0,
    misfire_grace_time=300,
    id='build_buy_in_list'
)
@is_trade_day
def build_buy_in_list():
    """
    build oversold and downgap buy in list
    """
    print("build oversold and downgap buy in list")
    build_buy_in_list_oversold()
    build_buy_in_list_downgap()
    today = datetime.date.today().strftime('%Y%m%d')
    print(f"{today} oversold 和 downgap 买入列表构建完成")

# 周六 1：00 AM 训练 oversold  模型
@general_scheduler.scheduled_job(
    trigger='cron',
    day_of_week='sat',
    hour=1,
    minute=0,
    misfire_grace_time=300,
    id='train_oversold_model'
)
def train_oversold_model():
    """
    train oversold model
    """
    print("train oversold model")
    train_dataset_oversold()
    today = datetime.date.today().strftime('%Y%m%d')
    print(f"{today} oversold 模型训练完成")

# 周日 1：00 AM 训练 downgap 模型
@general_scheduler.scheduled_job(
    trigger='cron',
    day_of_week='sun',
    hour=1,
    minute=0,
    misfire_grace_time=300
)
def train_downgap_model():
    """
    train downgap model
    """
    print("train downgap model")
    train_dataset_downgap()
    today = datetime.date.today().strftime('%Y%m%d')
    print(f"{today} downgap 模型训练完成")

# auto_run, 9:30分钟开始添加trade
def auto_run():
    """
    自动运行函数
    """
    # load the oversold dynamic task at 9:30 AM
    general_scheduler.add_job(
        trading_task_oversold,
        args=[general_scheduler],
        trigger='cron',
        hour=9,
        minute=30,
        id='start_trading_job_oversold',
        misfire_grace_time=300
    )
    # load the downgap dynamic task at 9:30 AM
    general_scheduler.add_job(
        trading_task_downgap,
        args=[general_scheduler],
        trigger='cron',
        hour=9,
        minute=30,
        id='start_trading_job_downgap',
        misfire_grace_time=300
    )
    general_scheduler.start()
    print('开始启动自动运行,按CTRL+C退出')
    try:
        while True:
            time.sleep(5)
    except (KeyboardInterrupt, SystemExit):
        general_scheduler.shutdown()
        print('自动运行已关闭')

if __name__ == '__main__':
    auto_run()