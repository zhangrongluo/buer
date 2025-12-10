#### 程序运行
- 独立模型运行。目前有DownGap和OverSold模型。输入以下命令，即自动运行DownGap模型的训练、测试和交易功能。
```bash
python auto_jobs_downgap.py
```
- 输入以下命令，即自动运行OverSold模型的训练、测试和交易功能。
```bash
python auto_jobs_oversold.py
```
- 分步查看运行过程，可以单独运行notebook文件（dataset_\*.ipynb、model_\*.ipynb、predict_\*.ipynb和trade_\*.ipynb）。

#### 必要的配置:
- Tushare pro 账号10000积分和实时价格查询权限。如没有实时权限，则局部功能会受限。
- Bark APP device key。没有 Bark 推送权限，无法收到交易提醒。
- Pushover app token 和 user key 非必需。