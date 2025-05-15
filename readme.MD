#### 程序运行
- 总入口程序为auto_jobs_general.py，汇集了全部模型的自动化运行。输入以下命令，即自动运行全部模型的训练、测试和交易功能。
```bash
python auto_jobs_general.py
```
- 独立模型运行。目前有DownGap和OverSold模型。输入以下命令，即自动运行DownGap模型的训练、测试和交易功能。
```bash
python auto_jobs_downgap.py
```
- 输入以下命令，即自动运行OverSold模型的训练、测试和交易功能。
```bash
python auto_jobs_oversold.py
```
- 分步查看运行过程，可以单独运行notebook文件（dataset_\*.ipynb、model_\*.ipynb、predict_\*.ipynb和trade_\*.ipynb）。

#### 数据集说明