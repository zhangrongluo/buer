### ==OverSold 系统（超跌反弹）==

> #### 超跌反弹定义、数据集构建和刷新
- **OverSold** 系统投资于超跌反弹类型的投资机会，本系统定义的超跌趋势由以下参数来确定：**trade_date，FORWARD_DAYS, BACKWARD_DAYS 和 DOWN_FILTER**。
- 设定某日为基准日 trade_date, FORWARD_DAYS即为基准日向前的时间跨度（含基准日），BACKWARD_DAYS为基准日向后的时间跨度（不含基准日），DOWN_FILTER 为FORWARD_DAYS期间下跌幅度筛选值。
- 在FORWARD_DAYS内取最高价,如果该价格和 trade_date 的收盘价比较,跌幅超过 DOWN_FILTER, **OverSold** 系统将最高价日期和基准日确认为一个超跌趋势。
- 在此基础上，系统判断 trade_date 日后是否有足够的交易天数覆盖BACKWARD_DAYS。如果可以覆盖，则选取BACKWARD_DAYS天数内的最高价，记录 trade_date 的收盘价和该最高价之间的最大涨幅及最高价日期。如果无法覆盖，则略过该步骤（**即最大涨幅及最高价日期字段置为空**）
- **OverSold** 系统将FORWARD_DAYS下跌阶段和BACKWARD_DAYS上涨阶段合并成一条超跌反弹数据，保存在该股票的超跌反弹数据集中。
- 在每个交易日，**OverSold** 系统扫描数据集内无法覆盖BACKWARD_DAYS的数据记录（**即最大涨幅及最高价日期字段为空的行**），检查当日是否能够满足覆盖BACKWARD_DAYS的条件。如果可以覆盖，则补充计算该条数据在BACKWARD_DAYS期间的最大涨幅、最高价日期等数据，并刷新该股票的超跌反弹数据集。
- ==**特别说明：一个超跌趋势往往会产生多条数据记录，这些数据记录在FORWARD_DAYS期间内最高价通常是同一天。这种情况其实就是一个下跌趋势的持续过程，该趋势在某一天产生了一个最高价，在后续交易日内连续下跌（跌幅超过 DOWN_FILTER的计入数据集），直到下跌趋势结束或者超出了FORWARD_DAYS的覆盖天数。**==
- **当一个超跌趋势形成以后，其多条数据记录的基准日会顺延（连续或者不连续），直到该下跌趋势结束。OverSold系统会以最后一条数据记录的基准日为起点，选择一个较小的时间跨度天数作为等待天数waiting_days，以观察该下跌趋势是否已经结束（不会再产生新的超跌数据记录）。如果产生了新的超跌数据记录，则以新产生的数据记录为最后一条重新等待。**
- **OverSold** 系统的交易只能发生在BACKWARD_DAYS天数内，如果在BACKWARD_DAYS实现了模型推理的预期涨幅，则在实现当日卖出，如果在BACKWARD_DAYS天数内没有实现预期涨幅，则在BACKWARD_DAYS天数满后即强制卖出。所以一条超跌反弹数据记录，如果BACKWARD_DAYS天数内相应的指标和数据已经填列（即非空），则该条记录已经完成了交易，只能作为训练集或者测试集。只有基准日后交易天数不能覆盖BACKWARD_DAYS的交易记录才有可能作为交易集。
> #### 参数说明（cons_oversold.py）:
##### OverSold系统参数说明：
- FORWARD_DAYS和DOWN_FILTER：基准日向前交易日的天数（包含基准日）。系统计算该天数内的最大跌幅$$基准日收盘价/FORWARD\_DAYS内最高价-1，$$如果该跌幅小于DOWN_FILTER，则该数据列入该股票的超跌反弹数据集。该参数是以交易日历天数（非自然日历天数）来计算的。
- BACKWARD_DAYS：基准日向后交易日的天数（不包含基准日）。系统计算该天数内的最大涨幅$$BACKWARD\_DAYS内最高价/基准日收盘价-1,$$该涨幅是股票超跌反弹数据集的标签值。该参数是以交易日历天数（非自然日历天数）来计算的。该参数决定了交易集的大小（**基准日后交易天数不能覆盖 BACKWARD_DAYS的所有超跌数据记录即为交易集**）。交易集确定后，剩余部分列入训练集、验证集和测试集。
- MAX_TRADE_DAYS：最大交易天数，即交易记录的BACKWARD_DAYS。该参数是以交易日历天数（非自然日历天数）来计算的。
- MIN_WAITING_DAYS：最小等待天数。当buy_in_list.csv文件中的某股票全部下跌趋势数据最后一条数据记录（按trade_date排序）的waiting_days小于该参数时，推断下跌趋势没有结束，不可买入。该参数是以交易日历天数（非自然日历天数）来计算的。
- WAITING_RATE_PCT：等待买入期间内（**trade_date和今天之间**）最大上涨幅度/预期收益率的比例，如果等待买入期内的最大上涨幅度大于该参数，则不买入。**即在等待买入期间内，该股票已经实现了大部分的预期收益率，为了防范风险，不买入。**
- REST_TRADE_DAYS：剩余交易天数（**等于MAX_TRADE_DAYS - waiting_days**），即买入后最大的持有天数。在股票进入了holding_list.csv后，该参数决定了股票剩余的持有天数。该参数是以交易日历天数（非自然日历天数）来计算的。waiting_days指一个下跌趋推断势终止后，基准日至今天已经发生的交易天数。
- TEST_DATASET_PERCENT：测试数据集占训练数据集+验证集+测试集的比例。
- MIN_PRED_RATE：最低预期收益率。该参数决定了buy_in_list.csv的大小。只用当trade_pred_suffix.csv文件中下跌趋势的预期收益率大于该参数时，才能入选buy_in_list.csv文件。
- PRED_RATE_PCT：预期收益率折扣百分比。该参数和MIN_PRED_RATE一起可以影响卖出的价格(该参数在本系统无实际意义)。
- COST_FEE：交易税费合计，按照申万宏源证券实际费率位万分之 1， 印花税率为万分之5， 综合设定为万分之3。
- MAX_STOCKS：最大持股数，当持股数超过此参数值时，系统不再买入新的股票。
- ONE_TIME_FUNDS：单次买入投入的资金额度。
- MIN_STOCK_PRICE：买入股票的最低价格，低于此价格的股票，系统不会买入。
- MAX_DOWN_LIMIT：即止损线，持有期间跌幅超过该参数值，系统会卖出该股票。
- MAX_BUY_UP_RATE：买入价超过buy_point_base的最大幅度，超过该幅度则不买入。
- SHUFFLE：决定训练时是否打乱数据集。
- PRED_MODELS：用于预测交易集预期收益率的模型数。
- initial_funds：初始投入资金，可以通过trade.change_trade_funds来调整资金额。
- PAUSE：采用TS模式获取实时价格的情况下，获取价格后的时间间隔。
- exception_list：排除列表，股票名称中含有列表中的文字，不会列入买入清单。
- params_list：该字典参数控制了不同数据集的参数设置和下载更新设置，前三个参数（FORWARD_DAYS、BACKWARD_DAYS和DOWN_FILTER）决定数据集的特征，第四个参数（TO_UPDATE）决定是否创建和更新该数据集。
- dataset_to_train：该字典参数控制了训练数据的选择设置，前三个参数（FORWARD_DAYS、BACKWARD_DAYS和DOWN_FILTER）决定选择数据集的特征，第四个参数(TIMES)决定是否训练该数据集或者训练的次数。
- dataset_to_predict_trade：该字典参数控制了交易数据的选择设置，前三个参数（FORWARD_DAYS、BACKWARD_DAYS和DOWN_FILTER）决定数据集的特征，第四个参数(PRED_MODELS)决定是否推测测试集和交易集或则使用多少模型来推测。
> #### 风险来源
##### OverSold 系统风险可能来自于以下方面
- 模型失效。模型在训练集和测试集上较好拟合，在交易集上拟合无效。
- 数据集风险。训练集、验证集、测试集和交易集采用了不同的选择标准，比如交易集使用了排除列表筛选，而训练集、测试集未使用排除列表筛选。
- 执行风险。即在模型发出买入卖出信号后，实际操作和模型存在差异。
- 股票实时价格获取速度。如果股票实时价格获取速度太慢，可能会丧失交易机会。
- 研究不够深入，未能发现下跌趋势行为模式的变化。
- 代码错误。这种风险无处不在，防不胜防。如前次把time.sleep()写成了time.slee(),由于在线程池中运行，没有报错，导致卖出模块失效，原因难寻。比如yyyymmdd型日期格式的处理，在保存过程中会出现string向float的自主转换，这些小问题，如果关注不到，也会造成大麻烦。今后处理日期型字段，将采用yyyymmdd H:m:s格式。比如np.nan、None的区别，也会造成不少问题。
- 多线程运行过程中，可能会发生无法预知的错误。
- **OverSold** 系统没有对指标统一数据格式，比如对于涨跌幅数据，有些文件以0.05 表示 5%，有些文件以 5 表示 5%，务必要注意这个问题。
> #### 风险控制
##### OverSold 模型采用以下风险控制措施
- 通过最大持股个数MAX_STOCKS来控制单个股票最大投资比例。该参数一般要超过60，从而控制单个股票最大持仓比例。
- 通过单次买入资金ONE_TIME_FUNDS来控制单次买入的金额。
- 通过最大持股个数MAX_STOCKS 和 单次买入资金ONE_TIME_FUNDS来控制最大投资金额占可投资金额的比例。
- 通过 waiting_days参数观察下跌趋势是否结束，防止在下跌趋势未结束时买入股票。
- 通过 MAX_BUY_UP_RATE参数，控制买入价格超过买入基准价的最大幅度。
- 通过exception_list强制排除风险股票:退市股票、st股票、pt股票等。
- 通过最低买入价格MIN_STOCK_PRICE来回避股价过低的股票。
- 通过最大下跌幅度MAX_DOWN_LIMIT为股票设置止损线，防止过度下跌。
- 买入时，跌停不买，处于下跌过程不买。
- 卖出时，涨停不卖，处于上涨过程不卖。
- 多线程编程时，采用最佳实践，设置合理的而不是最多的一次线程运行的数目。
- 等待期内的涨幅如果超过了模型设定涨幅的一定比例（系统设定值），系统不会买入该股票。
> #### 几个文件中的指标说明
##### OverSold数据集文件
- trade_date：超跌趋势的基准日。
- max_date_forward：基准日向前FORWARD_DAYS天的最大价格出现的日期。
- max_down_rate：max_date_forward的最高价至基准日收盘价的最大跌幅。
- FORWARD_DAYS：max_date_forward和基准日之间的交易天数（包含基准日）。
- max_date_backward：基准日向后BACKWARD_DAYS天的最大价格出现的日期。
- max_up_rate：max_date_backward的最高价至基准日收盘价的最大涨幅（即数据集的标签值）。
- BACKWARD_DAYS：max_date_backward和基准日之间的交易天数（不包含基准日）。
- last_daily_data：下跌趋势数据集更新的最后日期，下次更新时从该日期开始。
- pct_chg：基准日当日涨跌幅。
- vol_ratio：量比，即今日成交量/昨日成交量。 
- RSI和K指标：即是股票技术分析中的RSI指标和K指标。
- turnover_rate：换手率。
##### buy_in_list.csv
- trade_date：超跌趋势的基准日。
- buy_point_base：基准日的收盘价。股价小于等于该价格时，才可以买入。
- pct_chg：基准日当日涨跌幅。
- max_date_forward：基准日向前FORWARD_DAYS天的最大价格出现的日期。
- max_down_rate：基准日向前FORWARD_DAYS天的最大跌幅。
- FORWARD_DAYS：基准日向前FORWARD_DAYS天的交易天数（包含基准日）。
- waiting_days：等待天数，超跌趋势基准日到今天的交易天数（不包含基准日），在honging_list.csv中，等于days列的值。
- pred：模型预测的收益率，即holding_list.csv中的target_rate,也是max_up_rate的预测值。
- real: 缺口实际收益率，为空。
##### holding_list.csv
- trade_date：超跌趋势的基准日。
- days：超跌基准日和今天之间的交易天数，即buy_in_list.csv中的waiting_days列的值。
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

