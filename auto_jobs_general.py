"""
oversold and downgap general auto task entrance
"""
import os
import time
import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from stocklist import get_up_down_limit_list, get_stock_list, get_trade_cal
from basic_data import update_all_daily_data, update_all_daily_indicator, download_all_XD_XR_DR_dividend_data
import auto_jobs_oversold as jobs_oversold
import auto_jobs_downgap as jobs_downgap
import cons_oversold as cons_oversold
import cons_downgap as cons_downgap

general_scheduler = BackgroundScheduler()
general_scheduler.configure(timezone='Asia/Shanghai')

# 每日 00:01 AM 更新股票清单、交易日历、除权除息数据    
@general_scheduler.scheduled_job(
    trigger='cron',
    hour=0,
    minute=1,
    misfire_grace_time=300,
    id='update_basic_daily_data'
)
@jobs_oversold.is_trade_day
def update_basic_daily_data():
    """
    update stocklist、trade_cal、XR、XD、DR data
    """
    get_stock_list
    get_trade_cal()
    today = datetime.date.today().strftime('%Y%m%d')
    print(f'({cons_oversold.MODEL_NAME}) {today} 股票清单和交易日历更新完成')
    download_all_XD_XR_DR_dividend_data()
    print(f'({cons_oversold.MODEL_NAME}) {today} 除权除息更新完成')
    # 60 秒以后加载build_and_refresh_buy_in_list
    general_scheduler.add_job(
        build_and_refresh_buy_in_list,
        trigger='date',
        run_date=datetime.datetime.now() + datetime.timedelta(seconds=60),
        id='build_and_refresh_buy_in_list'
    )

@jobs_oversold.is_trade_day
def build_and_refresh_buy_in_list():
    """
    build and refresh buy in list
    """
    jobs_oversold.build_buy_in_list()
    jobs_oversold.refresh_buy_in_list()
    today = datetime.date.today().strftime('%Y%m%d')
    print(f'({cons_oversold.MODEL_NAME}) {today} oversold 买入列表构建刷新完成')
    jobs_downgap.build_buy_in_list()
    print(f'({cons_downgap.MODEL_NAME}) {today} downgap 买入列表构建完成')

# 每日 09:15 AM 更新up_limit 和 down_limit 数据
@general_scheduler.scheduled_job(
    trigger='cron',
    hour=9,
    minute=15,
    misfire_grace_time=300,
    id='update_up_down_limit_data'
)
@jobs_oversold.is_trade_day
def update_up_down_limit_data():
    """
    update up limit and down limit data
    """
    get_up_down_limit_list()
    today = datetime.date.today().strftime('%Y%m%d')
    print(f'({cons_oversold.MODEL_NAME}) {today} 涨跌停数据更新完成')

# 每天 9:25 AM 加载 oversold 交易任务
general_scheduler.add_job(
    jobs_oversold.trading_task_am,
    args=[general_scheduler],
    trigger='cron',
    hour=9, minute=25, misfire_grace_time=300,
    id=f'{cons_oversold.MODEL_NAME}_start_trading_job_am',
    name='Start_oversold_trading_program_at_9:25_AM',
)
# 每天 12:55 PM 加载 oversold 交易任务
general_scheduler.add_job(
    jobs_oversold.trading_task_pm,
    args=[general_scheduler],
    trigger='cron',
    hour=12, minute=55, misfire_grace_time=300,
    id=f'{cons_oversold.MODEL_NAME}_start_trading_job_pm',
    name='Start_oversold_trading_program_at_12:55_PM',
)

# 每天 9:25 AM 加载 downgap 交易任务
general_scheduler.add_job(
    jobs_downgap.trading_task_am,
    args=[general_scheduler],
    trigger='cron',
    hour=9, minute=25, misfire_grace_time=300,
    id=f'{cons_downgap.MODEL_NAME}_start_trading_job_am',
    name='Start_downgap_trading_program_at_9:25_AM',
)
# 每天 12:55 PM 加载 downgap 交易任务
general_scheduler.add_job(
    jobs_downgap.trading_task_pm,
    args=[general_scheduler],
    trigger='cron',
    hour=12, minute=55, misfire_grace_time=300,
    id=f'{cons_downgap.MODEL_NAME}_start_trading_job_pm',
    name='Start_downgap_trading_program_at_12:55_PM',
)

# 每天15:30 PM 清屏
@general_scheduler.scheduled_job(
    trigger='cron',
    hour=15, minute=30, misfire_grace_time=300,
    id='clear_screen_task'
)
@jobs_oversold.is_trade_day
def clear_screen_task():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f'({cons_oversold.MODEL_NAME}) oversold 和 downgap 模型运行中...')

# 每天 16:30 PM 更新 daily and indicator data
@general_scheduler.scheduled_job(
    trigger='cron',
    hour=16,
    minute=30,
    misfire_grace_time=300,
    id='update_daily_and_indicator_data'
)
@jobs_oversold.is_trade_day
def update_daily_and_indicator_data():
    """
    update daily and indicator data
    """
    update_all_daily_data()
    update_all_daily_indicator()
    today = datetime.date.today().strftime('%Y%m%d')
    print(f'({cons_oversold.MODEL_NAME}) {today} daily and indicator data 更新完成')
    # 60 秒以后加载 update_and_predict_dataset
    general_scheduler.add_job(
        update_and_predict_dataset,
        trigger='date',
        run_date=datetime.datetime.now() + datetime.timedelta(seconds=60),
        id='update_and_predict_dataset'
    )

@jobs_oversold.is_trade_day
def update_and_predict_dataset():
    """
    update and predict dataset
    """
    jobs_oversold.update_dataset()
    jobs_oversold.predict_dataset()
    today = datetime.date.today().strftime('%Y%m%d')
    print(f'({cons_oversold.MODEL_NAME}) {today} oversold 数据集更新和预测完成')
    jobs_downgap.update_dataset()
    jobs_downgap.predict_dataset()
    print(f'({cons_downgap.MODEL_NAME}) {today} downgap 数据集更新和预测完成')

# 每周六上午 01:00 AM 训练 oversold模型
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
    jobs_oversold.train_dataset()
    jobs_oversold.predict_dataset()
    today = datetime.date.today().strftime('%Y%m%d')
    print(f'({cons_oversold.MODEL_NAME}) {today} oversold 模型训练完成')

# 每周日上午 01:00 AM 训练 downgap模型
@general_scheduler.scheduled_job(
    trigger='cron',
    day_of_week='sun',
    hour=1,
    minute=0,
    misfire_grace_time=300,
    id='train_downgap_model'
)
def train_downgap_model():
    """
    train downgap model
    """
    jobs_downgap.train_dataset()
    jobs_downgap.predict_dataset()
    today = datetime.date.today().strftime('%Y%m%d')
    print(f'({cons_downgap.MODEL_NAME}) {today} downgap 模型训练完成')

# auto_run, load trade tasks and start scheduler
def auto_run():
    """
    自动运行函数
    """
    general_scheduler.start()
    print('开始启动自动运行,按CTRL+C退出')
    try:
        while True:
            time.sleep(5)
    except (KeyboardInterrupt, SystemExit):
        general_scheduler.shutdown()
        print(f'({cons_oversold.MODEL_NAME}) 自动运行已关闭')

if __name__ == '__main__':
    auto_run()