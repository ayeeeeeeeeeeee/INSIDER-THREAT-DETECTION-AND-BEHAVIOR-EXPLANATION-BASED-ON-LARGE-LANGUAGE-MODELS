
import pandas as pd
import os
from datetime import datetime
import glob

def load_ldap_data(ldap_path):

    all_files = glob.glob(os.path.join(ldap_path, "*.csv"))
    df_list = []  

    for file in all_files:
        try:
            df = pd.read_csv(file)

            filename = os.path.basename(file)
            year_month = filename.replace('.csv', '')
            df['month'] = year_month
            df_list.append(df)
            print(f"加载LDAP文件: {filename}, {len(df)}条记录")
        except Exception as e:
            print(f"加载文件{file}失败: {e}")

    if df_list:
        return pd.concat(df_list, ignore_index=True)
    return pd.DataFrame()

def get_user_role(user_id, ldap_df, timestamp=None):

    if ldap_df.empty:
        return None

    user_data = ldap_df[ldap_df['user_id'] == user_id]
    if user_data.empty:
        return None

    if timestamp and 'year_month' in user_data.columns:

        return user_data.iloc[0].to_dict()

    return user_data.iloc[0].to_dict() if not user_data.empty else None

def parse_datetime(date_str):

    try:
        return pd.to_datetime(date_str)
    except:
        return None

def extract_hour(dt):

    return dt.hour if pd.notnull(dt) else None

def is_weekend(dt):

    return dt.weekday() >= 5 if pd.notnull(dt) else None

def safe_div(a, b):

    return a / b if b != 0 else 0