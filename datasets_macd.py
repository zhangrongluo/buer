"""create、refresh macd datasets"""

import pathlib
import polars as pl  # polars is faster than pandas 20251002
from cons_general import BASICDATA_DIR, DATASETS_DIR, TEMP_DIR


def get_macd_dataset(ts_code: str, k_lines: int = 3) -> None:
    """
    create or refresh macd dataset for a given stock code

    Args:
        ts_code (str): stock code, e.g. '000001.SZ'
        k_lines (int, optional): how many k lines to contain in the dataset. Defaults to 2.
    """
    if len(ts_code) == 6:
        ts_code = f'{ts_code}.SH' if ts_code.startswith('6') else f'{ts_code}.SZ'
    factor_dir = pathlib.Path(BASICDATA_DIR) / 'dailyquantfactor'
    if not factor_dir.exists():
        return
    factor_csv = factor_dir / f'{ts_code}.csv'
    if not factor_csv.exists():
        return
    dest_dir = pathlib.Path(DATASETS_DIR) / 'macd'
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_csv = dest_dir / f'{ts_code}.csv'
    factor_df = pl.read_csv(factor_csv).sort('trade_date', descending=False)
    if dest_csv.exists():
        # 获取 last_trade_date 的最大值
        existing_df = pl.read_csv(dest_csv)
        last_trade_date = existing_df['last_trade_date'].max()
        factor_df = factor_df.filter(pl.col('trade_date') > last_trade_date - k_lines)
    if factor_df.is_empty():
        return
    rows_result = []
    factor_dict = factor_df.to_dicts()
    for index, row in enumerate(factor_dict):
        if index < k_lines - 1:
            continue
        if not (row['macd_dif'] > 0 and row['macd_dea'] > 0 and row['macd'] > 0):
            continue
        for i in range(index-1, -1, -1):  # 向前搜索 定位到最近的金叉
            if factor_dict[i]['macd'] > 0:
                continue
            gold_cross_index = i + 1
            days_between = index - gold_cross_index
            if days_between > k_lines - 1:
                continue
            # 添加 gold_date 和 days 字段，表示最近金叉的日期和距离金叉的天数
            row['gold_date'] = factor_dict[gold_cross_index]['trade_date']
            row['days'] = days_between 
            # 计算添加 index 之前12天(含index)的macd 的均值
            macd_values = [factor_dict[j]['macd'] for j in range(max(0, index - 12), index + 1)]
            if not macd_values:
                continue
            row['macd_12_avg'] = round(sum(macd_values) / len(macd_values), 4)
            rows_result.append(row)
            break
    if not rows_result:
        return
    result_df = pl.DataFrame(rows_result)
    last_trade_date = factor_df[-1, 'trade_date']
    result_df = result_df.with_columns(
        (round(pl.col('macd_12_avg')/ pl.col('macd_dif'), 4)).alias('macd_12_avg_rate'),
        pl.lit(last_trade_date).alias('last_trade_date')
    ).sort(
        by=['trade_date', 'ts_code'], descending=[False, False]
    ).unique(
        subset=['trade_date', 'ts_code'], keep='last'
    )
    result_df.write_csv(dest_csv)

# 把datasets/macd目录下的所有文件合并成一个文件all_macd_data.csv
def merge_all_macd_data() -> None:
    macd_dir = pathlib.Path(DATASETS_DIR) / 'macd'
    if not macd_dir.exists():
        return
    dest_dir = pathlib.Path(TEMP_DIR) / 'macd'
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_csv = dest_dir / 'all_macd_data.csv'
    all_files = list(macd_dir.glob('*.csv'))
    if not all_files:
        return
    all_dfs = []
    for file in all_files:
        tmp_df = pl.read_csv(file)
        if tmp_df.is_empty():
            continue
        tmp_df = tmp_df.drop_nulls()
        for col in tmp_df.columns[4:]:
            if tmp_df[col].dtype not in [pl.Float64, pl.Float32]:
                tmp_df = tmp_df.with_columns(pl.col(col).cast(pl.Float64))
        all_dfs.append(tmp_df)
    if not all_dfs:
        return
    combined_df : pl.DataFrame = pl.concat(all_dfs)
    combined_df = combined_df.sort(
        ['trade_date', 'ts_code'], descending=[False, False]
    ).unique(subset=['trade_date', 'ts_code'], keep='last')
    combined_df.write_csv(dest_csv)
