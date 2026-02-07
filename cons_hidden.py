import os
import toml
from pathlib import Path

bark_device_key = os.getenv("BARK_DEVICE_KEY")
tushare_token = os.getenv("TUSHARE_TOKEN")
pushover_user_key = os.getenv("PUSHOVER_USER_KEY")
pushover_app_token = os.getenv("PUSHOVER_APP_TOKEN")

# 通用参数和交易参数配置文件
ROOT = Path(__file__).parent
CONS_GENERAL_TOML = ROOT / 'cons' / 'cons_general.toml'
CONS_DOWNGAP_TOML = ROOT / 'cons' / 'cons_downgap.toml'
CONS_OVERSOLD_TOML = ROOT / 'cons' / 'cons_oversold.toml'

def load_config(toml_path) -> dict:
    """
    ### 加载 TOML 配置文件
    #### param toml_path: TOML 文件路径
    #### return: 配置字典
    """

    with open(toml_path, 'r') as f:
        return toml.load(f)
    
def get_modification_time(toml_path) -> float | None:
    """
    ### 获取 TOML 文件的修改时间
    #### param toml_path: TOML 文件路径
    #### return: 修改时间（秒），如果文件不存在则返回 None
    """
    path = Path(toml_path)
    return path.stat().st_mtime if path.exists() else None

if __name__ == "__main__":
    print("Bark device key:", bark_device_key)
    print("Tushare token:", tushare_token)
    print("Pushover user key:", pushover_user_key)
    print("Pushover app token:", pushover_app_token)
    print("Loaded general config:", load_config(CONS_GENERAL_TOML))
    print("Loaded downgap config:", load_config(CONS_DOWNGAP_TOML))
    print("Loaded oversold config:", load_config(CONS_OVERSOLD_TOML))