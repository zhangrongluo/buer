import os

bark_device_key = os.getenv("BARK_DEVICE_KEY")
tushare_token = os.getenv("TUSHARE_TOKEN")
pushover_user_key = os.getenv("PUSHOVER_USER_KEY")
pushover_app_token = os.getenv("PUSHOVER_APP_TOKEN")

if __name__ == "__main__":
    print("Bark device key:", bark_device_key)
    print("Tushare token:", tushare_token)
    print("Pushover user key:", pushover_user_key)
    print("Pushover app token:", pushover_app_token)
