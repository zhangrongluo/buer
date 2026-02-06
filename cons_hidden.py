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

def load_config(toml_path):
    """
    ### 加载 TOML 配置文件
    #### param toml_path: TOML 文件路径
    #### return: 配置字典
    """

    with open(toml_path, 'r') as f:
        return toml.load(f)

if __name__ == "__main__":
    print("Bark device key:", bark_device_key)
    print("Tushare token:", tushare_token)
    print("Pushover user key:", pushover_user_key)
    print("Pushover app token:", pushover_app_token)
    print("Loaded general config:", load_config(CONS_GENERAL_TOML))
    print("Loaded general config:", load_config(CONS_DOWNGAP_TOML))
    print("Loaded general config:", load_config(CONS_OVERSOLD_TOML))