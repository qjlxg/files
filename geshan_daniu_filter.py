import os
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# ==============================================================================
# 战法名称：隔山打牛（暴力洗盘后精准突击）
# 核心逻辑：
# 1. 寻找“山”：涨停后的次日出现一根巨量阴线（常为假阴），成交量创20日新高。
# 2. 蓄势：随后股价震荡，但绝不跌破这根大阴线的最低点，守住主力底牌。
# 3. 筛选：大阴线后必须有阳线回暖，且之后出现一根极致缩量的阴线（牛）。
# 4. 点火：当股价带量突破那根“缩量阴线”的最高价时，即为隔山打牛的启动买点。
# ==============================================================================

DATA_DIR = 'stock_data'
NAMES_FILE = 'stock_names.csv'
OUTPUT_BASE = 'results'

COL_MAP = {
    '日期': 'date', '开盘': 'open', '最高': 'high', 
    '最低': 'low', '收盘': 'close', '成交量': 'volume', '涨跌幅': 'pct_chg'
}

def check_geshan_daniu_logic(df):
    """
    战法详细校验函数
    """
    if len(df) < 30: return None
    
    close = df['close'].values
    low = df['low'].values
    high = df['high'].values
    vol = df['volume'].values
    pct = df['pct_chg'].values
    open_p = df['open'].values

    # 步骤1：寻找过去15天内的“放量阴（假跌真洗）”标杆
    for i in range(2, 15):
        idx = len(df) - i
        if idx < 20: continue
        
        # 涨停次日 + 20日最大量阴线（这就是“山”）
        if pct[idx-1] > 9.5 and close[idx] < open_p[idx]:
            if vol[idx] == max(vol[idx-20 : idx+1]):
                
                big_yin_idx = idx
                big_yin_low = low[idx]
                
                # 步骤2：确认调整不破大阴线底点（主力护盘底线）
                post_yin_data = df.iloc[big_yin_idx + 1:]
                if any(post_yin_data['low'] < big_yin_low):
                    continue
                
                # 步骤3：寻找中间的阳线和随后的缩量阴（洗盘彻底的信号）
                has_yang = False
                small_yin_idx = -1
                for j in range(big_yin_idx + 1, len(df)):
                    if close[j] > open_p[j]:
                        has_yang = True
                    # 缩量阴线（牛）：量能需小于高量阴线的50%以下
                    if has_yang and close[j] < open_p[j] and vol[j] < vol[big_yin_idx] * 0.5:
                        small_yin_idx = j
                
                # 步骤4：触发买点（今日收盘站上缩量阴的高点，准备打牛）
                if small_yin_idx != -1 and small_yin_idx < len(df) - 1:
                    if close[-1] > high[small_yin_idx]:
                        return "隔山打牛形态确认"
                        
    return None

def process_stock(file_name):
    code = file_name.split('.')[0]
    if code.startswith('30'): return None # 侧重主板
    
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, file_name))
        df = df.rename(columns=COL_MAP)
        if df.empty: return None
        
        if df['pct_chg'].dtype == object:
            df['pct_chg'] = df['pct_chg'].str.replace('%', '').astype(float)
            
        if check_geshan_daniu_logic(df):
            return code
    except:
        return None
    return None

def main():
    if not os.path.exists(NAMES_FILE): return
    names_df = pd.read_csv(NAMES_FILE)
    names_df['code'] = names_df['code'].astype(str).str.zfill(6)
    valid_codes = set(names_df[~names_df['name'].str.contains('ST|st')]['code'])

    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv') and f.split('.')[0] in valid_codes]
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_stock, files))
    
    found_codes = [c for c in results if c is not None]
    
    now = datetime.now()
    month_dir = os.path.join(OUTPUT_BASE, now.strftime('%Y%m'))
    os.makedirs(month_dir, exist_ok=True)
    ts = now.strftime('%Y%m%d_%H%M%S')

    final_df = names_df[names_df['code'].isin(found_codes)]
    file_path = os.path.join(month_dir, f'geshan_daniu_{ts}.csv')
    final_df.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"【隔山打牛】筛选完成，结果已保存至：{file_path}")

if __name__ == '__main__':
    main()
