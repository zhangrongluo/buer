"""create、refresh macd datasets"""

import pathlib
import polars as pl  # polars is faster than pandas 20251002
from cons_general import BASICDATA_DIR, DATASETS_DIR, TEMP_DIR

def get_macd_dataset(ts_code: str, k_lines: int = 3, backward_days: int = 5) -> None:
    """
    ### 创建指定股票的MACD数据集(构建数据集备用, 未进入训练, 推理和交易流程)
    #### :param ts_code: 股票代码, 格式为 '000001.SZ' 或 '000001'
    #### :param k_lines: 一个 MACD 金叉序列中包含的开始几个K线数量, 默认值为3
    #### :param backward_days: 自今天向后的天数, 默认值为5, 不含trade_date当天
    #### 逻辑说明:
    - MACD 数据集的入选对象为 macd_dif > 0 且 macd_dea > 0 且 macd > 0 的金叉K线(即所谓的红柱)
    - 每个 MACD 金叉序列, 最多包含该序列中的前 k_lines 根K线数据(通过计算当天和第一个金叉日之间的天数来控制)
    - 对于入选的K线, 以第二日开盘价为买入价, 计算之后 backward_days 天内的最高涨幅和最大回撤幅度及相关指标
    - 如果 MACD 数据集已存在, 则仅处理自上次数据集最后交易日之后的新数据
    #### 数据集存储路径:DATASETS_DIR/macd/ts_code.csv
    """
    if len(ts_code) == 6:
        ts_code = f'{ts_code}.SH' if ts_code.startswith('6') else f'{ts_code}.SZ'
    factor_dir = pathlib.Path(BASICDATA_DIR) / 'dailyquantfactor'
    if not factor_dir.exists():
        return
    factor_csv = factor_dir / f'{ts_code}.csv'
    if not factor_csv.exists():
        return
    dataset_dir = pathlib.Path(DATASETS_DIR) / 'macd'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_csv = dataset_dir / f'{ts_code}.csv'
    factor_df = pl.read_csv(factor_csv).sort('trade_date', descending=False)
    if dataset_csv.exists():
        # 获取 last_trade_date 的最大值
        dataset_df = pl.read_csv(dataset_csv)
        last_trade_date = dataset_df['last_trade_date'].max()
        factor_df = factor_df.filter(pl.col('trade_date') > last_trade_date - k_lines - 1)
    if factor_df.is_empty():
        return
    rows_result = []
    factor_dict = factor_df.to_dicts()
    last_trade_date = factor_df[-1, 'trade_date']
    for index, row in enumerate(factor_dict):
        if index < k_lines - 1:
            continue
        if not (row['macd_dif'] > 0 and row['macd_dea'] > 0 and row['macd'] > 0):  # 快慢线非金叉 非零轴以上
            continue
        # # 计算 index 之后 5 天内（不含 index）的最高价格和最低价格
        if len(factor_dict) - (index + 1) < backward_days:  # 不足5天
            row['max_rise_pct'] = None
            row['max_down_ptc'] = None
        else:
            future_data = factor_dict[index + 1: index + 1 + backward_days]
            high_prices = [item.get('high') for item in future_data]
            low_prices = [item.get('low') for item in future_data]
            price_buy = factor_dict[index+1].get('open')
            rise_pct = round((max(high_prices) - price_buy) / price_buy, 4) if price_buy and high_prices else None
            down_ptc = round((min(low_prices) - price_buy) / price_buy, 4) if price_buy and low_prices else None
            row['max_rise_pct'] = rise_pct
            row['max_down_ptc'] = down_ptc
        for i in range(index-1, -1, -1):  # 向前搜索
            if factor_dict[i]['macd'] > 0:
                continue
            gold_cross_index = i + 1  # 最近出现的第一个金叉的索引
            days_between = index - gold_cross_index
            if days_between > k_lines - 1:
                break  # 超过 k_lines 天数范围
            # 添加 gold_date 和 days 字段，表示 trade_date 和距离第一个金叉的天数
            row['gold_date'] = factor_dict[gold_cross_index]['trade_date']
            row['days'] = days_between 
            # 计算添加 index 之前12天(含index)的macd 的均值
            macd_values = [factor_dict[j]['macd'] for j in range(max(0, index - 12), index + 1)]
            if not macd_values:
                continue
            row['macd_12_avg'] = round(sum(macd_values) / len(macd_values), 4)
            row['macd_12_avg_rate'] = round((row['macd_12_avg'] / row['macd_dif']), 4) if row['macd_dif'] else None
            row['last_trade_date'] = last_trade_date
            rows_result.append(row)
            break
    if not rows_result:
        return
    result_df = pl.DataFrame(rows_result)
    result_df = result_df.unique(subset=['trade_date', 'ts_code'], keep='last')
    result_df = result_df.sort(by=['trade_date', 'ts_code'], descending=[False, False])
    result_df.write_csv(dataset_csv)

def refresh_macd_dataset(ts_code:str) -> None:
    """
    ### 刷新指定股票的MACD数据集中的 max_rise_pct 和 max_down_ptc 字段
    #### :param ts_code: 股票代码, 格式为 '000001.SZ' 或 '000001'
    #### 逻辑说明:
    - 填充 MACD 数据集中入选记录 5 日内的 max_rise_pct 和 max_down_ptc 字段
    """
    if len(ts_code) == 6:
        ts_code = f'{ts_code}.SH' if ts_code.startswith('6') else f'{ts_code}.SZ'
    factor_dir = pathlib.Path(BASICDATA_DIR) / 'dailyquantfactor'
    if not factor_dir.exists():
        return
    factor_csv = factor_dir / f'{ts_code}.csv'
    if not factor_csv.exists():
        return
    dataset_dir = pathlib.Path(DATASETS_DIR) / 'macd'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_csv = dataset_dir / f'{ts_code}.csv'
    if not dataset_csv.exists():
        return
    dataset_df = pl.read_csv(dataset_csv)
    last_trade_date = dataset_df['last_trade_date'].max()
    factor_df = pl.read_csv(factor_csv).sort('trade_date', descending=False)
    factor_df = factor_df.filter(pl.col('trade_date') >= last_trade_date)
    # 遍历 dataset_df，从 factor_df 中取数计算每一行的 max_rise_pct 和 max_down_ptc
    # 计算方法为 取该行 trade_date 之后的 5 天内的最高价和最低价（不含 trade_date 当天）
    if factor_df.is_empty():
        return
    factor_dict = factor_df.to_dicts()
    for row in dataset_df.to_dicts():
        if row.get('max_rise_pct') is not None and row.get('max_down_ptc') is not None:
            continue
        trade_date = row['trade_date']
        # 在 factor_dict 中找到 trade_date 所在的索引
        trade_index = next((i for i, item in enumerate(factor_dict) if item['trade_date'] == trade_date), None)
        if trade_index is None or len(factor_dict) - (trade_index + 1) < 5:  # 不足5天
            continue
        else:
            future_data = factor_dict[trade_index + 1: trade_index + 1 + 5]
            high_prices = [item.get('high') for item in future_data]
            low_prices = [item.get('low') for item in future_data]
            price_buy = factor_dict[trade_index + 1].get('open')
            rise_pct = round((max(high_prices) - price_buy) / price_buy, 4) if price_buy and high_prices else None
            down_ptc = round((min(low_prices) - price_buy) / price_buy, 4) if price_buy and low_prices else None
            row['max_rise_pct'] = rise_pct
            row['max_down_ptc'] = down_ptc
    dataset_df.write_csv(dataset_csv)

# 把datasets/macd目录下的所有文件合并成一个文件all_macd_data.csv
def merge_all_macd_dataset() -> None:
    """
    ### 合并 DATASETS_DIR/macd 目录下的所有 MACD 数据集文件
    #### 目标文件保存路径: TEMP_DIR/macd/all_macd_data.csv
    """
    macd_dir = pathlib.Path(DATASETS_DIR) / 'macd'
    if not macd_dir.exists():
        return
    dataset_dir = pathlib.Path(TEMP_DIR) / 'macd'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_csv = dataset_dir / 'all_macd_data.csv'
    all_files = list(macd_dir.glob('*.csv'))
    if not all_files:
        return
    all_dfs = []
    for file in all_files:
        tmp_df = pl.read_csv(file)
        if tmp_df.is_empty():
            continue
        for col in tmp_df.columns[4:]:
            if tmp_df[col].dtype not in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16]:
                tmp_df = tmp_df.with_columns(pl.col(col).cast(pl.Float64))
        all_dfs.append(tmp_df)
    if not all_dfs:
        return
    combined_df : pl.DataFrame = pl.concat(all_dfs)
    combined_df = combined_df.unique(subset=['trade_date', 'ts_code'], keep='last')
    combined_df = combined_df.sort(by=['trade_date', 'ts_code'], descending=[False, False])
    combined_df.write_csv(dataset_csv)
