import re
import numpy as np
import pandas as pd

RATES = {"TRY": 1.0, "GBP": 42.0, "USD": 33.0, "EUR": 36.0}

def parse_building_age(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().lower()
    if x in ('0', '0.0'):
        return 0.0
    if 'üzeri' in x or 'uzeri' in x:
        m = re.search(r'\d+', x)
        return float(m.group()) + 5.0 if m else 45.0
    if 'arası' in x or 'arasi' in x or '-' in x:
        nums = [float(n) for n in re.findall(r'\d+', x)]
        if len(nums) >= 2:
            return (nums[0] + nums[1]) / 2.0
        elif len(nums) == 1:
            return nums[0]
    try:
        return float(x)
    except:
        return np.nan

def parse_room_count(x):
    if pd.isna(x):
        return np.nan, np.nan, np.nan
    x = str(x).strip()
    if '+' in x:
        parts = x.split('+')
        try:
            r = float(parts[0]) if parts[0] else 0.0
            lr = float(parts[1]) if parts[1] else 0.0
            return r + lr, r, lr
        except:
            pass
    try:
        val = float(x)
        return val, val, 0.0
    except:
        return np.nan, np.nan, np.nan

def parse_floor_no(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    x_lower = x.lower()
    if 'yüksek giriş' in x_lower:
        return 0.5
    if 'giriş' in x_lower or 'zemin' in x_lower or 'bahçe' in x_lower:
        return 0.0
    if 'kot' in x_lower:
        m = re.findall(r'\d+', x)
        return -float(m[0]) if m else -1.0
    if 'asma' in x_lower or 'müstakil' in x_lower or 'komple' in x_lower:
        return 1.0
    if 'çatı' in x_lower:
        return 10.0
    if 've üzeri' in x_lower or 'üzeri' in x_lower:
        m = re.findall(r'\d+', x)
        return float(m[0]) if m else 20.0
    if '-' in x:
        nums = [float(n) for n in re.findall(r'\d+', x)]
        if len(nums) >= 2:
            return (nums[0] + nums[1]) / 2.0
    try:
        return float(x)
    except:
        return np.nan

def parse_total_floor_count(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    x_lower = x.lower()
    if 'arası' in x_lower or '-' in x:
        nums = [float(n) for n in re.findall(r'\d+', x)]
        if len(nums) >= 2:
            return (nums[0] + nums[1]) / 2.0
    m = re.findall(r'\d+\.?\d*', x)
    return float(m[0]) if m else np.nan

def extract_features(df_input):

    df = df_input.copy()

    if 'building_age' in df.columns:
        df['building_age_num'] = df['building_age'].apply(parse_building_age)
    elif 'building_age_num' not in df.columns:
        df['building_age_num'] = np.nan

    if 'room_count' in df.columns:
        room_parsed = df['room_count'].apply(parse_room_count)
        df['room_total'] = [p[0] for p in room_parsed]
        df['rooms'] = [p[1] for p in room_parsed]
        df['living_rooms'] = [p[2] for p in room_parsed]
    else:
        if 'room_total' not in df.columns:
            df['room_total'] = np.nan
        if 'rooms' not in df.columns:
            df['rooms'] = df['room_total']
        if 'living_rooms' not in df.columns:
            df['living_rooms'] = 1.0

    if 'floor_no' in df.columns:
        df['floor_no_num'] = df['floor_no'].apply(parse_floor_no)
    elif 'floor_no_num' not in df.columns:
        df['floor_no_num'] = np.nan

    if 'total_floor_count' in df.columns:
        df['total_floor_count_num'] = df['total_floor_count'].apply(parse_total_floor_count)
    elif 'total_floor_count_num' not in df.columns:
        df['total_floor_count_num'] = np.nan

    if 'address' in df.columns:
        addr_split = df['address'].astype(str).str.split('/', expand=True)
        df['city'] = addr_split[0].str.strip() if 0 in addr_split.columns else 'Unknown'
        df['district'] = addr_split[1].str.strip() if 1 in addr_split.columns else 'Unknown'
    else:
        if 'city' not in df.columns:
            df['city'] = 'Unknown'
        if 'district' not in df.columns:
            df['district'] = 'Unknown'

    safe_tot_floor = df['total_floor_count_num'].replace(0, np.nan)
    df['floor_ratio'] = (df['floor_no_num'] / safe_tot_floor).fillna(0.5).clip(0, 1.5)
    df['is_top_floor'] = (df['floor_no_num'] >= df['total_floor_count_num']).astype(float)
    df['is_ground_floor'] = (df['floor_no_num'] <= 1.0).astype(float)

    safe_rooms = df['room_total'].replace(0, np.nan)
    df['size_per_room'] = (df['size'] / safe_rooms).fillna(35.0).clip(5, 500)

    for col in ['listing_type', 'tom', 'size', 'building_age_num', 'floor_no_num', 
                'total_floor_count_num', 'room_total', 'rooms', 'living_rooms', 
                'floor_ratio', 'is_top_floor', 'is_ground_floor', 'size_per_room']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in ['sub_type', 'heating_type', 'city', 'district']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)

    return df
