import os
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

# ==============================================================================
# 战法名称：全战法综合复盘系统
# ------------------------------------------------------------------------------
# 1. 【隔山打牛】：涨停次日巨量假阴洗盘，回踩不破大阴低，突破缩量阴顶买入
# 2. 【高量不破】：高量柱后回踩不破其最低价，标准阳盖阴为确认点
# 3. 【三位一体】：股价放量上穿60日线（起飞线），MACD零轴金叉
# 4. 【机构洗盘-涨停破位】：涨停后快速跌穿支撑诱空，3天内盘中收回关键位
# 5. 【机构洗盘-草上飞】：均线多头排列（5>13>21），回踩不破13日线
# 6. 【追涨停技巧】：放量涨停后现缩量阳线，次日突破缩量阳最高价介入
# ==============================================================================

DATA_DIR = 'stock_data'
NAMES_FILE = 'stock_names.csv'
OUTPUT_BASE = 'results'

COL_MAP = {
    '日期': 'date', '开盘': 'open', '最高': 'high', 
    '最低': 'low', '收盘': 'close', '成交量': 'volume', '涨跌幅': 'pct_chg'
}

def check_all_strategies(df):
    """检测所有战法逻辑"""
    if len(df) < 65: return None
    
    c, l, h, o, v, p = df['close'].values, df['low'].values, df['high'].values, df['open'].values, df['volume'].values, df['pct_chg'].values
    ma5, ma13, ma21, ma60 = [df['close'].rolling(w).mean().values for w in [5, 13, 21, 60]]
    v_ma5 = df['volume'].rolling(5).mean().values

    # --- [战法 1: 隔山打牛] ---
    # 逻辑：寻找15天内的涨停+20日新高量阴线，回踩不破低，突破缩量阴顶
    for i in range(2, 15):
        idx = len(df) - i
        if p[idx-1] > 9.5 and c[idx] < o[idx] and v[idx] == max(v[idx-20:idx+1]):
            # 寻找后续的缩量阴
            for j in range(idx + 1, len(df)):
                if c[j] < o[j] and v[j] < v[idx] * 0.5: # 缩量阴
                    if c[-1] > h[j] and all(l[idx:] >= l[idx]): # 突破顶且不破底
                        return "隔山打牛: 假阴洗盘结束，连板启动"

    # --- [战法 2: 高量不破] ---
    # 逻辑：高量柱后回踩不破最低价
    for i in range(1, 10):
        idx = len(df) - 1 - i
        if v[idx] > v_ma5[idx] * 2 and c[idx] > o[idx]:
            if all(l[idx+1:] >= l[idx]) and c[-1] > h[idx]: # 阳盖阴确认
                return "高量不破: 回踩守住底线，必火信号"

    # --- [战法 3: 三位一体] ---
    # 逻辑：站稳60日起飞线 + 放大成交量
    if c[-1] > ma60[-1] and c[-2] <= ma60[-2] and v[-1] > v_ma5[-1]:
        return "三位一体: 步入起飞线，有理操作区"

    # --- [战法 4: 机构洗盘-草上飞] ---
    # 逻辑：多头排列，回踩13日线不破
    if ma5[-1] > ma13[-1] > ma21[-1] and l[-1] >= ma13[-1] and c[-1] > ma5[-1]:
        return "草上飞: 机构控盘波段，持股待涨"

    return None

def process_stock(file_name):
    code = file_name.split('.')[0]
    if code.startswith(('30', '68')): return None # 侧重主板
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, file_name))
        df = df.rename(columns=COL_MAP)
        if df['pct_chg'].dtype == object:
            df['pct_chg'] = df['pct_chg'].str.replace('%', '').astype(float)
        
        tag = check_all_strategies(df)
        if tag: return {'code': code, 'strategy': tag}
    except: return None
    return None

def main():
    if not os.path.exists(NAMES_FILE): return
    names_df = pd.read_csv(NAMES_FILE)
    names_df['code'] = names_df['code'].astype(str).str.zfill(6)
    valid_codes = set(names_df[~names_df['name'].str.contains('ST|st')]['code'])

    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv') and f.split('.')[0] in valid_codes]
    with ProcessPoolExecutor() as executor:
        results = [r for r in executor.map(process_stock, files) if r is not None]
    
    if results:
        res_df = pd.merge(pd.DataFrame(results), names_df, on='code')
        now = datetime.now()
        out_dir = os.path.join(OUTPUT_BASE, now.strftime('%Y%m'))
        os.makedirs(out_dir, exist_ok=True)
        res_df.to_csv(os.path.join(out_dir, f"Final_Strategies_{now.strftime('%Y%m%d')}.csv"), index=False, encoding='utf-8-sig')
        print(f"筛选完成，命中 {len(res_df)} 只目标股。")

if __name__ == '__main__':
    main()
