import wx
import pandas as pd
import os
from utils import  calculate_sharpe_ratio

oversold_hd_csv = 'trade/oversold/holding_list.csv'
oversold_indicator_csv = 'trade/oversold/statistic_indicator.csv'
downgap_hd_csv_45 = 'trade/downgap/max_trade_days_45/holding_list.csv'
downgap_indicator_csv_45 = 'trade/downgap/max_trade_days_45/statistic_indicator.csv'
downgap_hd_csv_50 = 'trade/downgap/max_trade_days_50/holding_list.csv'
downgap_indicator_csv_50 = 'trade/downgap/max_trade_days_50/statistic_indicator.csv'
rf_rate = 0.016  # 无风险利率

class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="股票清单和策略指标- Oversold 策略", size=self.get_initial_size())
        
        # 创建主面板
        self.panel = wx.Panel(self)
        
        # 创建布局
        self.create_layout()
        
        # 居中显示
        self.Center()
    
    def get_initial_size(self):
        """获取初始窗口大小(屏幕的75%)"""
        display_size = wx.GetDisplaySize()
        width = int(display_size.width * 0.75)
        height = int(display_size.height * 0.75)
        return (width, height)
    
    def create_layout(self):
        """创建界面布局"""
        # 主水平布局
        main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # 左侧菜单控制区域（180像素宽）
        left_panel = wx.Panel(self.panel, size=(180, -1))
        left_panel.SetBackgroundColour(wx.Colour(245, 245, 245))
        left_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # 左侧顶部标题
        title_label = wx.StaticText(left_panel, label="BUER")
        title_label.SetFont(wx.Font(16, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        
        left_sizer.Add(title_label, 0, wx.TOP|wx.LEFT, 20)
        
        # 菜单项：oversold 策略
        oversold_panel = wx.Panel(left_panel, size=(180, 36))
        oversold_panel.SetBackgroundColour(wx.Colour(245, 245, 245))
        
        # 使用固定位置来实现垂直居中
        oversold_label = wx.StaticText(oversold_panel, label="Oversold 策略", pos=(20, 12))
        oversold_label.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        
        # 绑定鼠标事件
        oversold_panel.Bind(wx.EVT_ENTER_WINDOW, self.on_oversold_enter)
        oversold_panel.Bind(wx.EVT_LEAVE_WINDOW, self.on_oversold_leave)
        oversold_panel.Bind(wx.EVT_LEFT_DOWN, self.on_oversold_click)
        oversold_label.Bind(wx.EVT_ENTER_WINDOW, self.on_oversold_enter)
        oversold_label.Bind(wx.EVT_LEAVE_WINDOW, self.on_oversold_leave)
        oversold_label.Bind(wx.EVT_LEFT_DOWN, self.on_oversold_click)
        
        left_sizer.Add(oversold_panel, 0, wx.TOP|wx.EXPAND, 20)
        
        # 菜单项：Downgap >>> 45策略
        downgap45_panel = wx.Panel(left_panel, size=(180, 36))
        downgap45_panel.SetBackgroundColour(wx.Colour(245, 245, 245))
        
        downgap45_label = wx.StaticText(downgap45_panel, label="Downgap >>> 45策略", pos=(20, 12))
        downgap45_label.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        
        # 绑定鼠标事件
        downgap45_panel.Bind(wx.EVT_ENTER_WINDOW, self.on_downgap45_enter)
        downgap45_panel.Bind(wx.EVT_LEAVE_WINDOW, self.on_downgap45_leave)
        downgap45_panel.Bind(wx.EVT_LEFT_DOWN, self.on_downgap45_click)
        downgap45_label.Bind(wx.EVT_ENTER_WINDOW, self.on_downgap45_enter)
        downgap45_label.Bind(wx.EVT_LEAVE_WINDOW, self.on_downgap45_leave)
        downgap45_label.Bind(wx.EVT_LEFT_DOWN, self.on_downgap45_click)
        
        left_sizer.Add(downgap45_panel, 0, wx.TOP|wx.EXPAND, 20)
        
        # 菜单项：Downgap >>> 50策略
        downgap50_panel = wx.Panel(left_panel, size=(180, 36))
        downgap50_panel.SetBackgroundColour(wx.Colour(245, 245, 245))
        
        downgap50_label = wx.StaticText(downgap50_panel, label="Downgap >>> 50策略", pos=(20, 12))
        downgap50_label.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        
        # 绑定鼠标事件
        downgap50_panel.Bind(wx.EVT_ENTER_WINDOW, self.on_downgap50_enter)
        downgap50_panel.Bind(wx.EVT_LEAVE_WINDOW, self.on_downgap50_leave)
        downgap50_panel.Bind(wx.EVT_LEFT_DOWN, self.on_downgap50_click)
        downgap50_label.Bind(wx.EVT_ENTER_WINDOW, self.on_downgap50_enter)
        downgap50_label.Bind(wx.EVT_LEAVE_WINDOW, self.on_downgap50_leave)
        downgap50_label.Bind(wx.EVT_LEFT_DOWN, self.on_downgap50_click)
        
        left_sizer.Add(downgap50_panel, 0, wx.TOP|wx.EXPAND, 20)
        left_panel.SetSizer(left_sizer)
        
        # 右侧显示区域
        right_panel = wx.Panel(self.panel)
        right_panel.SetBackgroundColour(wx.Colour(255, 255, 255))
        right_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # 上方：股票清单内容显示区域
        self.stock_list_panel = wx.Panel(right_panel)
        self.stock_list_panel.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.stock_list_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # 初始显示内容
        self.stock_list_content = wx.StaticText(self.stock_list_panel, label="点击左侧菜单选择策略")
        self.stock_list_content.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        
        self.stock_list_sizer.Add(self.stock_list_content, 0, wx.ALL|wx.CENTER, 20)
        self.stock_list_panel.SetSizer(self.stock_list_sizer)
        
        # 下方：指标显示区域（固定高度120像素）
        self.indicator_panel = wx.Panel(right_panel, size=(-1, 120))
        self.indicator_panel.SetBackgroundColour(wx.Colour(240, 248, 255))
        
        # 第一行指标：胜率相关（使用固定位置）
        # 持股数量指标
        self.holding_count_label = wx.StaticText(self.indicator_panel, label="持股数量: 0", pos=(10, 20))
        self.holding_count_label.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        
        # 日期胜率指标
        self.win_rate_1_label = wx.StaticText(self.indicator_panel, label="日期胜率1天: --", pos=(145, 20))
        self.win_rate_1_label.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        
        self.win_rate_5_label = wx.StaticText(self.indicator_panel, label="日期胜率5天: --", pos=(285, 20))
        self.win_rate_5_label.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))

        self.win_rate_10_label = wx.StaticText(self.indicator_panel, label="日期胜率10天: --", pos=(425, 20))
        self.win_rate_10_label.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))

        self.win_rate_20_label = wx.StaticText(self.indicator_panel, label="日期胜率20天: --", pos=(565, 20))
        self.win_rate_20_label.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))

        self.win_rate_30_label = wx.StaticText(self.indicator_panel, label="日期胜率30天: --", pos=(705, 20))
        self.win_rate_30_label.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        
        # 第二行指标：数量胜率、Omega指标和信息比例等
        # 数量胜率指标
        self.stock_win_rate_label = wx.StaticText(self.indicator_panel, label="数量胜率: --", pos=(10, 60))
        self.stock_win_rate_label.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        
        # Omega指标
        self.omega_label = wx.StaticText(self.indicator_panel, label="Omega指标: --", pos=(145, 60))
        self.omega_label.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        
        # 夏普比率指标
        self.sharpe_ratio_label = wx.StaticText(self.indicator_panel, label="夏普比率: --", pos=(285, 60))
        self.sharpe_ratio_label.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        
        # 当日收益指标
        self.daily_return_label = wx.StaticText(self.indicator_panel, label="当日收益: --", pos=(425, 60))
        self.daily_return_label.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        
        # 预留第二行的其他两个位置
        # 可以在这里添加更多指标，位置分别是 (565, 60), (705, 60)
        
        # 添加到右侧布局
        right_sizer.Add(self.stock_list_panel, 1, wx.EXPAND|wx.ALL, 0)  # proportion=1，占用剩余空间
        right_sizer.Add(self.indicator_panel, 0, wx.EXPAND|wx.ALL, 0)   # proportion=0，固定高度
        right_panel.SetSizer(right_sizer)
        
        # 添加到主布局
        main_sizer.Add(left_panel, 0, wx.EXPAND|wx.ALL, 0)
        main_sizer.Add(right_panel, 1, wx.EXPAND|wx.ALL, 0) 
        
        self.panel.SetSizer(main_sizer)
        
        # 初始化完成后预先加载oversold策略内容
        self.load_csv_content(oversold_hd_csv, oversold_indicator_csv)
    
    def on_oversold_enter(self, event):
        """鼠标进入oversold菜单项时的处理"""
        panel = event.GetEventObject()
        if isinstance(panel, wx.StaticText):
            panel = panel.GetParent()
        panel.SetBackgroundColour(wx.Colour(220, 220, 220))
        panel.Refresh()
    
    def on_oversold_leave(self, event):
        """鼠标离开oversold菜单项时的处理"""
        panel = event.GetEventObject()
        if isinstance(panel, wx.StaticText):
            panel = panel.GetParent()
        panel.SetBackgroundColour(wx.Colour(245, 245, 245))
        panel.Refresh()
    
    def on_downgap45_enter(self, event):
        """鼠标进入downgap45菜单项时的处理"""
        panel = event.GetEventObject()
        if isinstance(panel, wx.StaticText):
            panel = panel.GetParent()
        panel.SetBackgroundColour(wx.Colour(220, 220, 220))
        panel.Refresh()
    
    def on_downgap45_leave(self, event):
        """鼠标离开downgap45菜单项时的处理"""
        panel = event.GetEventObject()
        if isinstance(panel, wx.StaticText):
            panel = panel.GetParent()
        panel.SetBackgroundColour(wx.Colour(245, 245, 245))
        panel.Refresh()
    
    def on_downgap50_enter(self, event):
        """鼠标进入downgap50菜单项时的处理"""
        panel = event.GetEventObject()
        if isinstance(panel, wx.StaticText):
            panel = panel.GetParent()
        panel.SetBackgroundColour(wx.Colour(220, 220, 220))
        panel.Refresh()
    
    def on_downgap50_leave(self, event):
        """鼠标离开downgap50菜单项时的处理"""
        panel = event.GetEventObject()
        if isinstance(panel, wx.StaticText):
            panel = panel.GetParent()
        panel.SetBackgroundColour(wx.Colour(245, 245, 245))
        panel.Refresh()
    
    def load_csv_content(self, csv_file, indicator_file=None):
        """加载并显示CSV文件内容"""
        try:
            if os.path.exists(csv_file):
                # 读取CSV文件
                df = pd.read_csv(csv_file)
                
                # 清除当前内容
                self.stock_list_sizer.Clear(True)
                
                # 创建网格控件显示CSV内容
                import wx.grid
                grid = wx.grid.Grid(self.stock_list_panel)
                
                # 创建表格
                grid.CreateGrid(len(df), len(df.columns))
                
                # 设置列标题
                for col_idx, col_name in enumerate(df.columns):
                    grid.SetColLabelValue(col_idx, str(col_name))
                
                # 设置数据并格式化
                for row_idx in range(len(df)):
                    for col_idx in range(len(df.columns)):
                        value = df.iloc[row_idx, col_idx]
                        col_name = df.columns[col_idx]
                        
                        # 格式化数值（保留两位小数）
                        if 'date' in col_name.lower() and pd.notna(value):
                            # 日期列特殊处理，格式化为yyyymmdd
                            try:
                                if isinstance(value, (int, float)):
                                    # 如果是数字，转换为整数再转字符串
                                    date_str = str(int(value))
                                    # 只有当长度正好是8位时才认为是有效的yyyymmdd格式
                                    if len(date_str) == 8 and date_str.isdigit():
                                        # 验证是否是合理的日期
                                        year = int(date_str[:4])
                                        month = int(date_str[4:6])
                                        day = int(date_str[6:8])
                                        if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                                            formatted_value = date_str
                                        else:
                                            formatted_value = str(int(value))
                                    else:
                                        formatted_value = str(int(value))
                                elif isinstance(value, str):
                                    # 如果是字符串，去除小数点后格式化
                                    clean_str = value.replace('.0', '').replace('.00', '')
                                    if len(clean_str) == 8 and clean_str.isdigit():
                                        # 验证是否是合理的日期
                                        year = int(clean_str[:4])
                                        month = int(clean_str[4:6])
                                        day = int(clean_str[6:8])
                                        if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                                            formatted_value = clean_str
                                        else:
                                            formatted_value = value
                                    else:
                                        formatted_value = value
                                else:
                                    # 其他类型，直接转字符串，不进行日期解析
                                    formatted_value = str(value)
                            except:
                                formatted_value = str(value) if pd.notna(value) else ""
                        elif pd.api.types.is_numeric_dtype(df[col_name].dtype):
                            if pd.notna(value):
                                if isinstance(value, (int, float)):
                                    # 检查是否是百分比列
                                    if col_name.lower() in ['rate_pred', 'rate_current']:
                                        formatted_value = f"{float(value) * 100:.2f}%"
                                    else:
                                        formatted_value = f"{float(value):.2f}"
                                else:
                                    formatted_value = str(value)
                            else:
                                formatted_value = ""
                        else:
                            formatted_value = str(value) if pd.notna(value) else ""
                        
                        grid.SetCellValue(row_idx, col_idx, formatted_value)
                        
                        # 设置对齐方式
                        if pd.api.types.is_numeric_dtype(df[col_name].dtype) or col_name.lower() == 'status':
                            # 数值列和status列右对齐
                            grid.SetCellAlignment(row_idx, col_idx, wx.ALIGN_RIGHT, wx.ALIGN_CENTER)
                        else:
                            grid.SetCellAlignment(row_idx, col_idx, wx.ALIGN_LEFT, wx.ALIGN_CENTER)
                
                # 设置网格样式
                grid.SetDefaultCellFont(wx.Font(11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))  # 从10号改为11号
                grid.SetColLabelSize(30)  # 列标题高度
                grid.SetRowLabelSize(60)  # 行标题宽度
                grid.SetDefaultRowSize(25)  # 行高度（增加行间距）
                grid.SetDefaultColSize(100)  # 列宽度调整为100像素
                
                # 自动调整列宽
                grid.AutoSizeColumns()
                
                # 设置最小列宽和最大列宽
                for col in range(grid.GetNumberCols()):
                    current_width = grid.GetColSize(col)
                    if current_width < 80:  # 最小列宽调整为80
                        grid.SetColSize(col, 80)
                    elif current_width > 150:  # 最大列宽调整为150
                        grid.SetColSize(col, 150)
                
                # 设置网格线
                grid.SetGridLineColour(wx.Colour(200, 200, 200))
                
                # 设置只读
                grid.EnableEditing(False)
                
                # 添加到布局
                self.stock_list_sizer.Add(grid, 1, wx.EXPAND|wx.ALL, 10)
                
                # 更新持股数量指标（统计status为holding的行数）
                if 'status' in df.columns:
                    holding_count = len(df[df['status'].str.lower() == 'holding'])
                else:
                    holding_count = 0
                self.holding_count_label.SetLabel(f"持股数量: {holding_count}")
                
                # 更新日期胜率指标（读取indicator文件中各种胜率的最后一行）
                win_rate_1_text = "日期胜率1天: --"
                win_rate_5_text = "日期胜率5天: --"
                win_rate_10_text = "日期胜率10天: --"
                win_rate_20_text = "日期胜率20天: --"
                win_rate_30_text = "日期胜率30天: --"
                stock_win_rate_text = "数量胜率: --"
                omega_text = "Omega指标: --"
                sharpe_ratio_text = "夏普比率: --"
                daily_return_text = "当日收益: --"
                
                if indicator_file and os.path.exists(indicator_file):
                    try:
                        indicator_df = pd.read_csv(indicator_file)
                        if len(indicator_df) > 0:
                            if 'win_rate_1' in indicator_df.columns:
                                last_win_rate_1 = indicator_df['win_rate_1'].iloc[-1]
                                win_rate_1_text = f"日期胜率1天: {last_win_rate_1:.2f}%"
                            if 'win_rate_5' in indicator_df.columns:
                                last_win_rate_5 = indicator_df['win_rate_5'].iloc[-1]
                                win_rate_5_text = f"日期胜率5天: {last_win_rate_5:.2f}%"
                            if 'win_rate_10' in indicator_df.columns:
                                last_win_rate_10 = indicator_df['win_rate_10'].iloc[-1]
                                win_rate_10_text = f"日期胜率10天: {last_win_rate_10:.2f}%"
                            if 'win_rate_20' in indicator_df.columns:
                                last_win_rate_20 = indicator_df['win_rate_20'].iloc[-1]
                                win_rate_20_text = f"日期胜率20天: {last_win_rate_20:.2f}%"
                            if 'win_rate_30' in indicator_df.columns:
                                last_win_rate_30 = indicator_df['win_rate_30'].iloc[-1]
                                win_rate_30_text = f"日期胜率30天: {last_win_rate_30:.2f}%"
                            if 'win_rate_stocks' in indicator_df.columns:
                                last_stock_win_rate = indicator_df['win_rate_stocks'].iloc[-1]
                                stock_win_rate_text = f"数量胜率: {last_stock_win_rate:.2f}%"
                            if 'omega_ratio' in indicator_df.columns:
                                last_omega_ratio = indicator_df['omega_ratio'].iloc[-1]
                                omega_text = f"Omega指标: {last_omega_ratio:.4f}"
                            if 'return_ratio' in indicator_df.columns:
                                last_return_ratio = indicator_df['return_ratio'].iloc[-1]
                                daily_return_text = f"当日收益: {last_return_ratio * 100:.2f}%"
                                
                            # 计算夏普比率
                            try:
                                if csv_file == oversold_hd_csv:
                                    sharpe_ratio = calculate_sharpe_ratio('oversold', rf_rate)
                                    sharpe_ratio_text = f"夏普比率: {sharpe_ratio:.4f}"
                                elif csv_file == downgap_hd_csv_45:
                                    sharpe_ratio = calculate_sharpe_ratio('downgap', rf_rate, max_trade_days=45)
                                    sharpe_ratio_text = f"夏普比率: {sharpe_ratio:.4f}"
                                elif csv_file == downgap_hd_csv_50:
                                    sharpe_ratio = calculate_sharpe_ratio('downgap', rf_rate, max_trade_days=50)
                                    sharpe_ratio_text = f"夏普比率: {sharpe_ratio:.4f}"
                            except Exception as sharpe_e:
                                print(f"计算夏普比率出错: {sharpe_e}")
                                sharpe_ratio_text = "夏普比率: --"
                    except Exception as e:
                        print(f"读取指标文件出错: {e}")
                
                self.win_rate_1_label.SetLabel(win_rate_1_text)
                self.win_rate_5_label.SetLabel(win_rate_5_text)
                self.win_rate_10_label.SetLabel(win_rate_10_text)
                self.win_rate_20_label.SetLabel(win_rate_20_text)
                self.win_rate_30_label.SetLabel(win_rate_30_text)
                self.stock_win_rate_label.SetLabel(stock_win_rate_text)
                self.omega_label.SetLabel(omega_text)
                self.sharpe_ratio_label.SetLabel(sharpe_ratio_text)
                self.daily_return_label.SetLabel(daily_return_text)
                
                # 刷新布局
                self.stock_list_panel.Layout()
            else:
                # 文件不存在时显示错误信息
                self.stock_list_sizer.Clear(True)
                error_label = wx.StaticText(self.stock_list_panel, label=f"文件不存在: {csv_file}")
                error_label.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
                self.stock_list_sizer.Add(error_label, 0, wx.ALL|wx.CENTER, 20)
                
                # 重置持股数量为0
                self.holding_count_label.SetLabel("持股数量: 0")
                # 重置所有日期胜率
                self.win_rate_1_label.SetLabel("日期胜率1天: --")
                self.win_rate_5_label.SetLabel("日期胜率5天: --")
                self.win_rate_10_label.SetLabel("日期胜率10天: --")
                self.win_rate_20_label.SetLabel("日期胜率20天: --")
                self.win_rate_30_label.SetLabel("日期胜率30天: --")
                # 重置数量胜率
                self.stock_win_rate_label.SetLabel("数量胜率: --")
                # 重置Omega指标
                self.omega_label.SetLabel("Omega指标: --")
                # 重置夏普比率
                self.sharpe_ratio_label.SetLabel("夏普比率: --")
                # 重置当日收益
                self.daily_return_label.SetLabel("当日收益: --")
                
                self.stock_list_panel.Layout()
        except Exception as e:
            # 发生错误时显示错误信息
            self.stock_list_sizer.Clear(True)
            error_label = wx.StaticText(self.stock_list_panel, label=f"加载文件出错: {str(e)}")
            error_label.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
            self.stock_list_sizer.Add(error_label, 0, wx.ALL|wx.CENTER, 20)
            
            # 重置持股数量为0
            self.holding_count_label.SetLabel("持股数量: 0")
            # 重置所有日期胜率
            self.win_rate_1_label.SetLabel("日期胜率1天: --")
            self.win_rate_5_label.SetLabel("日期胜率5天: --")
            self.win_rate_10_label.SetLabel("日期胜率10天: --")
            self.win_rate_20_label.SetLabel("日期胜率20天: --")
            self.win_rate_30_label.SetLabel("日期胜率30天: --")
            # 重置数量胜率
            self.stock_win_rate_label.SetLabel("数量胜率: --")
            # 重置Omega指标
            self.omega_label.SetLabel("Omega指标: --")
            # 重置夏普比率
            self.sharpe_ratio_label.SetLabel("夏普比率: --")
            # 重置当日收益
            self.daily_return_label.SetLabel("当日收益: --")
            
            self.stock_list_panel.Layout()
    
    def on_oversold_click(self, event):
        """点击oversold策略菜单项时的处理"""
        self.SetTitle("股票清单和策略指标 - Oversold 策略")
        self.load_csv_content(oversold_hd_csv, oversold_indicator_csv)
    
    def on_downgap45_click(self, event):
        """点击downgap45策略菜单项时的处理"""
        self.SetTitle("股票清单和策略指标 - Downgap >>> 45策略")
        self.load_csv_content(downgap_hd_csv_45, downgap_indicator_csv_45)
    
    def on_downgap50_click(self, event):
        """点击downgap50策略菜单项时的处理"""
        self.SetTitle("股票清单和策略指标 - Downgap >>> 50策略")
        self.load_csv_content(downgap_hd_csv_50, downgap_indicator_csv_50)

class StockApp(wx.App):
    def OnInit(self):
        frame = MainFrame()
        frame.Show()
        return True

if __name__ == '__main__':
    app = StockApp()
    app.MainLoop()
