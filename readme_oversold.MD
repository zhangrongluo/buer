#### 注意事项
##### 每日需要定期更新数据，以获取最新的OverSold数据集：
- 更新TRADE_RECORD_PATH目录下的全部股票交易记录，里面保存了沪深300行业中股票的每日的财务指标信息。更新后需要调用get_trade_date_equal_today_nums接口检查数据更新情况。
- 更新dailydata_root目录下全部文件数据，里面保存了沪深300行业中股票每日的交易价格信息，含open、high、low、close、pre_close、chg_pct、vol等，交易信息更新后需要调用get_daily_data_equal_today_nums接口检查数据更新情况。
- 以上两项基础数据更新已经打包在dataset项目中，运行dataset项目下的Roe_data.ipynb文件即可完成数据更新。
- 运行本项目下dataset.ipynb文件，完成OverSold数据集的创建和更新。
- 有时会缺少财务记录文件，从而导致合并数据集报错。需要到quant-stock项目下data.py或者dataset项目下Roe_data.ipynb中运行check_stockcodes_integrity检查记录完整性。
#### 参数说明（cons.py）:
##### OverSold模型由以下参数来控制：
- FORWARD_DAYS和DOWN_FILTER：基准日向前交易日的天数（包含基准日）。模型计算该天数内的最大跌幅（最高值和基准日收盘价比较），如果小于DOWN_FILTER，则该数据列入OverSold数据集。该参数是以交易日历天数（非自然日历天数）来计算的。
- BACKWARD_DAYS：基准日向后交易日的天数（不包含基准日）。模型计算该天数内的最大涨幅（最高值和基准日收盘价比较），该涨幅是数据集的标签值。该参数是以交易日历天数（非自然日历天数）来计算的。该参数决定了交易集的大小。交易集确定后，剩余部分列入训练集、验证集和测试集。
- MAX_TRADE_DAYS：最大交易天数，它和BACKWARD_DAYS是同一个数据。该参数是以交易日历天数（非自然日历天数）来计算的。
- MIN_WAITING_DAYS：最小等待天数，一般以BACKWARD_DAYS的10%取整来确定。当buy_in_list.csv文件中的某股票全部下跌趋势数据最后一个数据（按trade_date排序）的waiting_days小于该参数时，推断下跌趋势没有结束，不可买入。该参数是以交易日历天数（非自然日历天数）来计算的。
- WAITING_RATE_PCT：等待期间内最大上涨幅度/预期收益率的比例，如果等待期内的最大上涨幅度大于该参数，则不买入。
- REST_TRADE_DAYS：剩余交易天数。在股票进入了holding_list.csv后，该参数决定了股票剩余的持有天数（等于MAX_TRADE_DAYS - waiting_days）。该参数是以交易日历天数（非自然日历天数）来计算的。waiting_days指一个下跌趋推断势终止后，基准日至今天已经发生的交易天数，即在holding_list.csv中，days列的值。
- TEST_DATASET_PERCENT：测试数据集占训练数据集+验证集+测试集的比例。
- MIN_PRED_RATE：最低预期收益率。该参数决定了buy_in_list.csv的大小。只用当trade_pred_suffix.csv文件中下跌趋势的预期收益率大于该参数时，才能入选buy_in_list.csv文件。
- PRED_RATE_PCT：预期收益率折扣百分比。该参数和MIN_PRED_RATE一起可以影响卖出的价格(该参数在本系统无实际意义)。
- COST_FEE：交易税费合计，设定为万分之5，申万宏源证券的实际交易水平为万分之3.5。
- MAX_STOCKS：最大持股数，当持股数超过此参数值时，系统不再买入新的股票。
- ONE_TIME_FUNDS：单次买入投入的资金额度。
- MIN_STOCK_PRICE：买入股票的最低价格，低于此价格的股票，系统不会买入。
- MAX_DOWN_LIMIT：最大下跌幅度，持有期间跌幅超过该参数值，系统会卖出该股票。
- SHUFFLE：决定训练时是否打乱数据集。
- PRED_MODELS：用于预测交易集预期收益率的模型数。
- initial_funds：初始投入资金，可以通过trade.change_trade_funds来调整资金额。
- PAUSE：采用TS模式获取实时价格的情况下，获取价格后的时间间隔。
- exception_list：排除列表，股票名称中含有列表中的文字，不会列入买入清单。
- params_list：该字典参数控制了不同数据集的参数设置和下载更新设置，前三个参数（FORWARD_DAYS、BACKWARD_DAYS和DOWN_FILTER）决定数据集的特征，第四个参数（TO_UPDATE）决定是否创建和更新该数据集。
- dataset_to_train：该字典参数控制了训练数据的选择设置，前三个参数（FORWARD_DAYS、BACKWARD_DAYS和DOWN_FILTER）决定选择数据集的特征，第四个参数(TIMES)决定是否训练该数据集或者训练的次数。
- dataset_to_predict_trade：该字典参数控制了交易数据的选择设置，前三个参数（FORWARD_DAYS、BACKWARD_DAYS和DOWN_FILTER）决定数据集的特征，第四个参数(PRED_MODELS)决定是否推测测试集和交易集或则使用多少模型来推测。
#### 风险来源
##### OverSold 模型风险可能来自于方方面面
- 模型失效。模型在训练集和测试集上较好拟合，在交易集上拟合无效。
- 数据集风险。训练集、验证集、测试集和交易集采用了不同的选择标准，比如交易集使用了排除列表筛选，而训练集、测试集未使用排除列表筛选。
- 执行风险。即在模型发出买入卖出信号后，实际操作和模型存在差异。
- 股票实时价格获取速度。如果股票实时价格获取速度太慢，可能会丧失交易机会。
- 研究不够深入，未能发现下跌趋势行为模式的变化。
- 代码错误。这种风险无处不在，防不胜防。如前次把time.sleep()写成了time.slee(),由于在线程池中运行，没有报错，导致卖出模块失效，原因难寻。比如yyyymmdd型日期格式的处理，在保存过程中会出现string向float的自主转换，这些小问题，如果关注不到，也会造成大麻烦。今后处理日期型字段，将采用yyyymmdd H:m:s格式。比如np.nan、None的区别，也会造成不少问题。
- 多线程运行过程中，可能会发生不可预知的错误。
#### 风险控制
##### OverSold 模型采用以下风险控制措施
- 通过最大持股个数MAX_STOCKS来控制单个股票最大投资比例。该参数一般要超过80，从而控制单个股票最大持仓比例。
- 通过单次买入资金ONE_TIME_FUNDS来控制单次买入的金额。
- 通过最大持股个数MAX_STOCKS 和 单次买入资金ONE_TIME_FUNDS来控制最大投资金额占可投资金额的比例。
- 通过exception_list强制排除风险股票:退市股票、st股票、pt股票等。
- 通过最低买入价格MIN_STOCK_PRICE来回避股价过低的股票。
- 买入时，跌停不买，处于下跌过程不买。
- 卖出时，涨停不卖，处于上涨过程不卖。
#### 几个文件中的指标说明
##### OverSold数据集文件
- trade_date：下跌趋势的基准日。
- max_date_forward：基准日向前FORWARD_DAYS天的最大价格出现的日期。
- max_down_rate：max_date_forward的最高价至基准日收盘价的最大跌幅。
- forward_days：max_date_forward和基准日之间的交易天数（包含基准日）。
- max_date_backward：基准日向后BACKWARD_DAYS天的最大价格出现的日期。
- max_up_rate：max_date_backward的最高价至基准日收盘价的最大涨幅（即数据集的标签值）。
- backward_days：max_date_backward和基准日之间的交易天数（不包含基准日）。
- last_daily_data：下跌趋势数据集更新的最后日期，下次更新时从该日期开始。
- pct_chg：基准日当日涨跌幅。
- vol_ratio：量比，即今日成交量/昨日成交量。 
- RSI和K指标：即是股票技术分析中的RSI指标和K指标。
- turnover_rate：换手率。
##### buy_in_list.csv
- trade_date：下跌趋势的基准日。
- buy_point_base：基准日的收盘价。股价小于等于该价格时，才可以买入。
- pct_chg：基准日当日涨跌幅。
- max_date_forward：基准日向前FORWARD_DAYS天的最大价格出现的日期。
- max_down_rate：基准日向前FORWARD_DAYS天的最大跌幅。
- forward_days：基准日向前FORWARD_DAYS天的交易天数（包含基准日）。
- waiting_days：该下跌趋趋势基准日到今天的交易天数（不包含基准日），即在honging_list.csv中，days列的值。
- pred：模型预测的收益率，即holding_list.csv中的target_rate,也是max_up_rate的预测值。
- real: 缺口实际收益率，为空。
##### holding_list.csv
- trade_date：下跌趋势的基准日。
- days：下跌基准日和今天之间的交易天数，即buy_in_list.csv中的waiting_days列的值。
- holding_days：持有天数，等于今天和买入日期之间的间隔天数，以自然日历为基础计算。
- buy_point_base：基准日的收盘价。股价小于等于该价格时，才可以买入。
- target_rate：模型预测收益率，即buy_in_list.csv中的pred，也是max_up_rate的预测值。
- date_in和date_out：买入日期和卖出日期。
- price_in、price_out和price_now：买入价 卖出价和现价。
- amount：资金余额和股票数量。
- cost_fee：交易税费比例，设定值为万分之5。
- profit：利润额，等于(price_now*(1-cost_fee) - price_in*(1+cost_fee)) * amount
- rate_pred和rate_pct：即缺口买入时，系统设定的MIN_PRED_RATE和PRED_RATE_PCT值。
- rate_current：现在的收益率，等于profit / (price_in * amount)
- rate_yearly：年化收益率，等于rate_current * 365 / holding_days，以自然日历天数计算。
