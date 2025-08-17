"""
競艇AI予想ツール（研究用）
データ収集から機械学習、予想まで一体化したGUIツール

主な機能:
- 過去10年分の競艇データ自動収集
- 機械学習モデルの訓練・保存・追加学習
- 指定日時・場所でのレース予想
- 学習結果の可視化

注意: このツールは研究目的で作成されています。
      実際の舟券購入には使用しないでください。
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import os
import pickle
import threading
import time
from datetime import datetime, timedelta
import urllib.request
import subprocess
import re
import mojimoji
from lhafile import LhaFile
import warnings
warnings.filterwarnings('ignore')

# 機械学習ライブラリ
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

matplotlib.use('TkAgg')  # Tkinterバックエンドを使用

class KyoteiDataCollector:
    """
    競艇データ収集クラス
    公式サイトからのデータダウンロードと前処理を担当
    """
    
    def __init__(self):
        self.stop_collection_flag = False
        # 競艇場マッピング（場所コード -> 場所名）
        self.place_mapper = {
            1: '桐生', 2: '戸田', 3: '江戸川', 4: '平和島', 5: '多摩川',
            6: '浜名湖', 7: '蒲郡', 8: '常滑', 9: '津', 10: '三国',
            11: '琵琶湖', 12: '住之江', 13: '尼崎', 14: '鳴門', 15: '丸亀',
            16: '児島', 17: '宮島', 18: '徳山', 19: '下関', 20: '若松',
            21: '芦屋', 22: '福岡', 23: '唐津', 24: '大村'
        }
        
    def download_file(self, obj, date):
        """
        競艇公式サイトから圧縮ファイルをダウンロードして展開
        
        Parameters:
        obj (str): 'racelists'(出走表) or 'results'(結果)
        date (str): 'YYYY-MM-DD' 形式の日付
        """
        date = str(pd.to_datetime(date).date())
        ymd = date.replace('-', '')
        
        # ファイル種別による設定（結果ファイルとレースリストファイル）
        S, s = ('K', 'k') if obj == 'results' else ('B', 'b')
        
        # すでにファイルが存在する場合はスキップ
        if os.path.exists(f'downloads/{obj}/{ymd}.txt'):
            return True
            
        # ディレクトリ作成
        os.makedirs(f'downloads/{obj}', exist_ok=True)
        
        try:
            # ダウンロードURL構築
            url_t = f'http://www1.mbrace.or.jp/od2/{S}/'
            url_b = f'{ymd[:-2]}/{s}{ymd[2:]}.lzh'
            
            # 圧縮ファイルダウンロード
            urllib.request.urlretrieve(url_t + url_b, f'downloads/{obj}/{ymd}.lzh')
            
            # LZH形式の展開
            archive = LhaFile(f'downloads/{obj}/{ymd}.lzh')
            d = archive.read(archive.infolist()[0].filename)
            
            # テキストファイルとして保存
            with open(f'downloads/{obj}/{ymd}.txt', 'wb') as f:
                f.write(d)
                
            # 圧縮ファイルを削除
            os.remove(f'downloads/{obj}/{ymd}.lzh')
            
            return True
            
        except Exception as e:
            print(f'データ取得エラー ({date}): {e}')
            return False
    
    def read_file(self, obj, date):
        """
        ダウンロードしたテキストファイルを読み込み、会場ごとに分割
        
        Parameters:
        obj (str): 'racelists' or 'results'
        date (str): 日付
        
        Returns:
        dict: {場所コード: [行データのリスト]}
        """
        date = str(pd.to_datetime(date).date())
        ymd = date.replace('-', '')
        
        try:
            # Shift-JISエンコーディングでファイル読み込み
            with open(f'downloads/{obj}/{ymd}.txt', 'r', encoding='shift-jis') as f:
                lines = [l.strip().replace('\u3000', '') for l in f]
                
            # 全角数字を半角に変換
            lines = [mojimoji.zen_to_han(l, kana=False) for l in lines][1:-1]
            
            # 会場ごとにデータを分割
            lines_by_place = {}
            for l in lines:
                if 'BGN' in l:
                    place_cd = int(l[:-4])
                    place_lines = []
                elif 'END' in l:
                    lines_by_place[place_cd] = place_lines
                else:
                    place_lines.append(l)
                    
            return lines_by_place
            
        except Exception as e:
            print(f'ファイル読み込みエラー ({date}): {e}')
            return {}
    
    def get_racelists(self, date):
        """
        出走表データをDataFrame形式に変換
        
        Parameters:
        date (str): 日付
        
        Returns:
        pandas.DataFrame: 出走表データ
        """
        # カラム定義
        info_cols = ['title', 'day', 'date', 'place_cd', 'place']
        race_cols = ['race_no', 'race_type', 'distance', 'deadline']
        
        # 選手データのキー（6名分）
        keys = ['toban', 'name', 'area', 'class', 'age', 'weight',
                'glob_win', 'glob_in2', 'loc_win', 'loc_in2',
                'motor_no', 'motor_in2', 'boat_no', 'boat_in2']
        racer_cols = [f'{k}_{i}' for k in keys for i in range(1, 7)]
        
        cols = info_cols + race_cols + racer_cols
        stack = []
        
        date = str(pd.to_datetime(date).date())
        
        # 各会場のデータを処理
        for place_cd, lines in self.read_file('racelists', date).items():
            min_lines = 11
            if len(lines) < min_lines:
                continue
                
            # レース情報の抽出
            title = lines[4]
            day = int(re.findall('第(\d)日', lines[6].replace(' ', ''))[0])
            place = self.place_mapper.get(place_cd, f'Unknown_{place_cd}')
            
            info = {k: v for k, v in zip(info_cols, [title, day, date, place_cd, place])}
            
            # レース番号の開始位置を特定
            head_list = []
            race_no = 1
            for i, l in enumerate(lines[min_lines:]):
                if f'{race_no}R' in l:
                    head_list.append(min_lines + i)
                    race_no += 1
                    
            # 各レースのデータを抽出
            for race_no, head in enumerate(head_list, 1):
                try:
                    # レース基本情報
                    race_type = lines[head].split()[1]
                    distance = int(re.findall('H(\d*)m', lines[head])[0])
                    deadline = re.findall('電話投票締切予定(\d*:\d*)', lines[head])[0]
                    
                    # 6艇の選手データを抽出
                    arr = []
                    for l in lines[head + 5: head + 11]:
                        split = re.findall('\d \d{4}.*\d\d\.\d\d', l)
                        if not split:
                            continue
                            
                        split = split[0].split()
                        
                        # 選手情報の解析（正規表現で分離）
                        name_area_class = split[1]
                        character = re.findall('[^\d]*', name_area_class)
                        numbers = re.findall('[\d]*', name_area_class)
                        
                        character = ''.join([e for e in character if e != ''])  # 文字列を結合
                        if len(character) == 7:
                            name = character[:4]  # 名前が4文字
                            area = character[4:6]
                            class1 = character[6:7]
                        elif len(character) == 8:
                            name = character[:5]  # 名前が5文字
                            area = character[5:7]
                            class1 = character[7:8]
                        else:
                            name = character[:3]  # 名前が3文字
                            area = character[3:5]
                            class1 = character[5:6]
                        toban, age, weight, class2 = [e for e in numbers if e != ''][:4] if len([e for e in numbers if e != '']) >= 4 else ['', '', '', '']
                        
                        tmp = [toban, name, area, class1 + class2, age, weight] + split[2:10]
                        if len(tmp) == 14:
                            arr.append(tmp)
                    
                    # 6艇すべてのデータが揃った場合のみ記録
                    if len(arr) == 6:
                        dic = info.copy()
                        dic.update(zip(race_cols, [race_no, race_type, distance, deadline]))
                        dic.update(dict(zip(racer_cols, np.array(arr).T.reshape(-1))))
                        stack.append(dic)
                        
                except (IndexError, ValueError) as e:
                    continue
                    
        if len(stack) > 0:
            df = pd.DataFrame(stack)[cols].dropna()
            return self._convert_racelists_dtypes(df)
        else:
            return None
    
    def get_results(self, date):
        """
        レース結果データをDataFrame形式に変換
        
        Parameters:
        date (str): 日付
        
        Returns:
        pandas.DataFrame: 結果データ
        """
        # レースタイム変換関数
        conv_racetime = lambda x: np.nan if x == '.' else sum([w * float(v) for w, v in zip((60, 1, 1/10), x.split('.'))])
        
        # カラム定義
        info_cols = ['title', 'day', 'date', 'place_cd', 'place']
        race_cols = ['race_no', 'race_type', 'distance']
        
        # 選手結果データのキー
        keys = ['toban', 'name', 'motor_no', 'boat_no', 'ET', 'SC', 'ST', 'RT', 'position']
        racer_cols = [f'{k}_{i}' for k in keys for i in range(1, 7)]
        
        # 舟券結果データ
        res_cols = []
        for k in ('tkt', 'odds', 'poprank'):
            for type_ in ('1t', '1f1', '1f2', '2t', '2f', 'w1', 'w2', 'w3', '3t', '3f'):
                if (k == 'poprank') & (type_ in ('1t', '1f1', '1f2')):
                    pass
                else:
                    res_cols.append(f'{k}_{type_}')
        res_cols.append('win_method')
        
        cols = info_cols + race_cols + racer_cols + res_cols
        stack = []
        
        date = str(pd.to_datetime(date).date())
        
        # 各会場の結果データを処理
        for place_cd, lines in self.read_file('results', date).items():
            min_lines = 26
            if len(lines) < min_lines:
                continue
                
            # レース情報
            title = lines[4]
            day = int(re.findall('第(\d)日', lines[6].replace(' ', ''))[0])
            place = self.place_mapper.get(place_cd, f'Unknown_{place_cd}')
            
            info = {k: v for k, v in zip(info_cols, [title, day, date, place_cd, place])}
            
            # レース開始位置を特定
            head_list = []
            race_no = 1
            for i, l in enumerate(lines[min_lines:]):
                if f'{race_no}R' in l:
                    head_list.append(min_lines + i)
                    race_no += 1
                    
            # 各レース結果の抽出
            for race_no, head in enumerate(head_list, 1):
                try:
                    # レース基本情報
                    race_type = lines[head].split()[1]
                    distance = int(re.findall('H(\d*)m', lines[head])[0])
                    win_method = lines[head + 1].split()[-1]
                    
                    # 舟券結果の抽出
                    _, tkt_1t, pb_1t = lines[head + 10].split()
                    _, tkt_1f1, pb_1f1, tkt_1f2, pb_1f2 = lines[head + 11].split()
                    _, tkt_2t, pb_2t, _, pr_2t = lines[head + 12].split()
                    _, tkt_2f, pb_2f, _, pr_2f = lines[head + 13].split()
                    _, tkt_w1, pb_w1, _, pr_w1 = lines[head + 14].split()
                    tkt_w2, pb_w2, _, pr_w2 = lines[head + 15].split()
                    tkt_w3, pb_w3, _, pr_w3 = lines[head + 16].split()
                    _, tkt_3t, pb_3t, _, pr_3t = lines[head + 17].split()
                    _, tkt_3f, pb_3f, _, pr_3f = lines[head + 18].split()
                    
                    race_vals = [race_no, race_type, distance]
                    res_vals = [
                        tkt_1t, tkt_1f1, tkt_1f2, tkt_2t, tkt_2f,
                        tkt_w1, tkt_w2, tkt_w3, tkt_3t, tkt_3f,
                        pb_1t, pb_1f1, pb_1f2, pb_2t, pb_2f,
                        pb_w1, pb_w2, pb_w3, pb_3t, pb_3f,
                        pr_2t, pr_2f, pr_w1, pr_w2, pr_w3,
                        pr_3t, pr_3f, win_method
                    ]
                    
                    dic = info.copy()
                    dic.update(dict(zip(race_cols, race_vals)))
                    dic.update(dict(zip(res_cols, res_vals)))
                    
                    # オッズを百分率に変換
                    dic = {k: float(v) / 100 if 'odds' in k else v for k, v in dic.items()}
                    
                    # 6艇の成績データを追加
                    for i in range(6):
                        bno, *vals = lines[head + 3 + i].split()[1:10]
                        vals.append(i + 1)  # 着順
                        keys = ['toban', 'name', 'motor_no', 'boat_no', 'ET', 'SC', 'ST', 'RT', 'position']
                        dic.update(zip([f'{k}_{bno}' for k in keys], vals))
                        
                    stack.append(dic)
                    
                except (IndexError, ValueError):
                    continue
                    
        if len(stack) > 0:
            df = pd.DataFrame(stack)[cols].dropna(how='all')
            
            # データ型変換と前処理
            repl_mapper = {'K': np.nan, '.': np.nan}
            for i in range(1, 7):
                df[f'ET_{i}'] = df[f'ET_{i}'].replace(repl_mapper)
                df[f'ST_{i}'] = df[f'ST_{i}'].replace(repl_mapper).str.replace('F', '-').str.replace('L', '1')
                df[f'RT_{i}'] = df[f'RT_{i}'].map(conv_racetime)
                
            # 枠番通りの着順かどうかのフラグ
            waku = np.array([('{}'*6).format(*v) for v in df[[f'SC_{i}' for i in range(1, 7)]].values])
            df['wakunari'] = np.where(waku == '123456', 1, 0)
            
            df = df.replace({'K': np.nan})
            return self._convert_results_dtypes(df)
        else:
            return None
    
    def get_beforeinfo(self, date, place_cd, race_no):
        """
        直前情報（天候、展示結果など）をスクレイピングで取得
        
        Parameters:
        date (str): 日付
        place_cd (int): 場所コード
        race_no (int): レース番号
        
        Returns:
        pandas.Series: 直前情報データ
        """
        try:
            # URL構築
            url_t = 'https://www.boatrace.jp/owpc/pc/race/'
            ymd = str(pd.to_datetime(date)).split()[0].replace('-', '')
            jcd = f'0{place_cd}' if place_cd < 10 else str(place_cd)
            url = f'{url_t}beforeinfo?rno={race_no}&jcd={jcd}&hd={ymd}'
            
            # HTMLを取得して解析
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'lxml')
            
            # 選手の体重とチルト角度を取得
            arr1 = [[tag('td')[4].text, tag('td')[5].text] for tag in soup(class_='is-fs12')]
            arr1 = [[v if v != '\xa0' else '' for v in row] for row in arr1]
            
            # 展示スタートタイミングとコースを取得
            arr2 = [[tag.find(class_=f'table1_boatImage1{k}').text for k in ('Number', 'Time')] 
                   for tag in soup(class_='table1_boatImage1')]
            arr2 = [[v.replace('F', '-') for v in row] for row in arr2]
            arr2 = [row + [i] for i, row in enumerate(arr2, 1)]
            arr2 = pd.DataFrame(arr2).sort_values(by=[0]).values[:, 1:]
            
            # 気象条件を取得
            weather_data = [tag.text for tag in soup(class_='weather1_bodyUnitLabelData')]
            weather = soup(class_='weather1_bodyUnitLabelTitle')[1].text
            wind_direction = int(soup.select_one('p[class*="is-wind"]').attrs['class'][1][7:])
            
            # データフレーム作成
            df = pd.DataFrame(np.concatenate([arr1, arr2], 1), 
                            columns=['ET', 'tilt', 'EST', 'ESC']).replace('L', '1').astype('float')
            
            if len(df) < 6:
                return None
                
            # 結果データの構築
            air_t, wind_v, water_t, wave_h = weather_data
            data = pd.concat([
                pd.Series({'date': date, 'place_cd': place_cd, 'race_no': race_no}),
                pd.Series(df.values.T.reshape(-1), 
                         index=[f'{col}_{i}' for col in df.columns for i in range(1, 7)]),
                pd.Series({
                    'weather': weather, 
                    'air_temp': float(air_t[:-1]),
                    'wind_direction': wind_direction, 
                    'wind_speed': float(wind_v[:-1]),
                    'water_temp': float(water_t[:-1]),
                    'wave_height': float(wave_h[:-2])
                })
            ])
            
            # 展示スタートコースを整数に変換
            for i in range(1, 7):
                data[f'ESC_{i}'] = int(data[f'ESC_{i}'])
                
            return data
            
        except Exception as e:
            print(f'直前情報取得エラー ({date}, {place_cd}, {race_no}): {e}')
            return None
    
    def _convert_racelists_dtypes(self, df):
        """出走表データの型変換"""
        # 数値型変換が必要なカラム
        numeric_cols = ['race_no', 'distance'] + [f'{col}_{i}' for col in 
                       ['age', 'weight', 'glob_win', 'glob_in2', 'loc_win', 'loc_in2', 'motor_in2', 'boat_in2'] 
                       for i in range(1, 7)]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df
    
    def _convert_results_dtypes(self, df):
        """結果データの型変換"""
        # 数値型変換
        numeric_cols = ['race_no', 'distance'] + [f'{col}_{i}' for col in 
                       ['ET', 'ST', 'RT', 'position'] for i in range(1, 7)]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df

class KyoteiMLModel:
    """
    競艇機械学習モデルクラス
    データ前処理、モデル訓練、予想を担当
    """
    
    def __init__(self):
        # 機械学習モデル
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=200, random_state=42, max_depth=6)
        }
        
        # データ前処理用
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.is_trained = False
        
        # 学習データ保存用
        self.training_history = []
        self.model_performance = {}
        
    def create_features(self, race_df, result_df=None, beforeinfo_df=None):
        """
        レースデータから機械学習用の特徴量を作成
        
        Parameters:
        race_df (DataFrame): 出走表データ
        result_df (DataFrame): 結果データ（訓練時のみ）
        beforeinfo_df (DataFrame): 直前情報データ（任意）
        
        Returns:
        DataFrame: 特徴量データ
        """
        features_list = []
        
        # 各レースを処理
        for _, race_row in race_df.iterrows():
            race_features = {}
            
            # レース基本情報
            race_features['date'] = race_row['date']
            race_features['place_cd'] = race_row['place_cd']
            race_features['race_no'] = race_row['race_no']
            race_features['distance'] = race_row.get('distance', 1800)
            
            # 6艇それぞれの特徴量を作成
            for boat_no in range(1, 7):
                boat_features = self._create_boat_features(race_row, boat_no)
                
                # ターゲット変数（結果データがある場合）
                if result_df is not None:
                    # 対応する結果を検索
                    result_mask = (
                        (result_df['date'] == race_row['date']) &
                        (result_df['place_cd'] == race_row['place_cd']) &
                        (result_df['race_no'] == race_row['race_no'])
                    )
                    result_row = result_df[result_mask]
                    
                    if not result_row.empty:
                        result_row = result_row.iloc[0]
                        # 1着かどうかの判定
                        boat_features['is_winner'] = 1 if result_row.get(f'position_{boat_no}', 0) == 1 else 0
                        boat_features['finish_position'] = result_row.get(f'position_{boat_no}', 0)
                    else:
                        boat_features['is_winner'] = 0
                        boat_features['finish_position'] = 0
                
                # 直前情報の追加
                if beforeinfo_df is not None:
                    beforeinfo_mask = (
                        (beforeinfo_df['date'] == race_row['date']) &
                        (beforeinfo_df['place_cd'] == race_row['place_cd']) &
                        (beforeinfo_df['race_no'] == race_row['race_no'])
                    )
                    beforeinfo_row = beforeinfo_df[beforeinfo_mask]
                    
                    if not beforeinfo_row.empty:
                        beforeinfo_row = beforeinfo_row.iloc[0]
                        boat_features.update(self._add_beforeinfo_features(beforeinfo_row, boat_no))
                
                # 艇番を追加
                boat_features['boat_no'] = boat_no
                
                features_list.append(boat_features)
        
        return pd.DataFrame(features_list)
    
    def _create_boat_features(self, race_row, boat_no):
        """
        個別の艇の特徴量を作成
        
        Parameters:
        race_row (Series): レース行データ
        boat_no (int): 艇番
        
        Returns:
        dict: 艇の特徴量
        """
        features = {}
        
        # 選手データ
        features[f'age'] = race_row.get(f'age_{boat_no}', 30)
        features[f'weight'] = race_row.get(f'weight_{boat_no}', 52)
        features[f'glob_win'] = race_row.get(f'glob_win_{boat_no}', 0)
        features[f'glob_in2'] = race_row.get(f'glob_in2_{boat_no}', 0)
        features[f'loc_win'] = race_row.get(f'loc_win_{boat_no}', 0)
        features[f'loc_in2'] = race_row.get(f'loc_in2_{boat_no}', 0)
        
        # モーター・ボートデータ
        features[f'motor_in2'] = race_row.get(f'motor_in2_{boat_no}', 0)
        features[f'boat_in2'] = race_row.get(f'boat_in2_{boat_no}', 0)
        
        # カテゴリカル変数
        features[f'class'] = race_row.get(f'class_{boat_no}', 'B1')
        features[f'area'] = race_row.get(f'area_{boat_no}', '東京')
        
        # 派生特徴量
        features[f'experience'] = max(0, features[f'age'] - 20)  # 経験年数推定
        features[f'win_rate_diff'] = features[f'glob_win'] - features[f'loc_win']  # 全国と当地の差
        features[f'equipment_score'] = (features[f'motor_in2'] + features[f'boat_in2']) / 2  # 機材スコア
        
        # コース別有利度（内側ほど有利）
        course_advantages = [0.55, 0.15, 0.12, 0.10, 0.05, 0.03]
        features[f'course_advantage'] = course_advantages[boat_no - 1]
        
        return features
    
    def _add_beforeinfo_features(self, beforeinfo_row, boat_no):
        """
        直前情報から特徴量を追加
        
        Parameters:
        beforeinfo_row (Series): 直前情報行データ
        boat_no (int): 艇番
        
        Returns:
        dict: 直前情報の特徴量
        """
        features = {}
        
        # 展示タイム、チルト角度、展示スタートタイミング
        features['exhibition_time'] = beforeinfo_row.get(f'ET_{boat_no}', 0)
        features['tilt'] = beforeinfo_row.get(f'tilt_{boat_no}', 0)
        features['exhibition_st'] = beforeinfo_row.get(f'EST_{boat_no}', 0)
        features['exhibition_course'] = beforeinfo_row.get(f'ESC_{boat_no}', boat_no)
        
        # 天候条件
        features['weather'] = beforeinfo_row.get('weather', '晴')
        features['air_temp'] = beforeinfo_row.get('air_temp', 20)
        features['wind_speed'] = beforeinfo_row.get('wind_speed', 0)
        features['wind_direction'] = beforeinfo_row.get('wind_direction', 0)
        features['water_temp'] = beforeinfo_row.get('water_temp', 20)
        features['wave_height'] = beforeinfo_row.get('wave_height', 0)
        
        return features
    
    def preprocess_features(self, df, is_training=True):
        """
        特徴量の前処理（エンコーディング、標準化など）
        
        Parameters:
        df (DataFrame): 特徴量データ
        is_training (bool): 訓練モードかどうか
        
        Returns:
        tuple: (X, y, feature_names) または (X, feature_names)
        """
        df_processed = df.copy()
        
        # カテゴリカル変数のエンコーディング
        categorical_columns = ['class', 'area', 'weather']
        
        for col in categorical_columns:
            if col in df_processed.columns:
                if is_training:
                    # 訓練時は新しいエンコーダーを作成
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                        df_processed[col].fillna('Unknown')
                    )
                else:
                    # 予想時は既存のエンコーダーを使用
                    if col in self.label_encoders:
                        # 未知のカテゴリの処理
                        known_categories = set(self.label_encoders[col].classes_)
                        df_processed[col] = df_processed[col].fillna('Unknown')
                        unknown_mask = ~df_processed[col].isin(known_categories)
                        
                        if unknown_mask.any():
                            # 未知の値を最も頻繁な値に置換
                            most_common = self.label_encoders[col].classes_[0]
                            df_processed.loc[unknown_mask, col] = most_common
                        
                        df_processed[f'{col}_encoded'] = self.label_encoders[col].transform(df_processed[col])
                    else:
                        df_processed[f'{col}_encoded'] = 0
        
        # 特徴量の選択
        feature_columns = [
            'boat_no', 'age', 'weight', 'glob_win', 'glob_in2', 'loc_win', 'loc_in2',
            'motor_in2', 'boat_in2', 'experience', 'win_rate_diff', 'equipment_score',
            'course_advantage', 'class_encoded', 'area_encoded', 'distance'
        ]
        
        # 直前情報の特徴量（利用可能な場合）
        beforeinfo_features = [
            'exhibition_time', 'tilt', 'exhibition_st', 'exhibition_course',
            'weather_encoded', 'air_temp', 'wind_speed', 'wind_direction',
            'water_temp', 'wave_height'
        ]
        
        # 利用可能な特徴量のみを選択
        available_features = [col for col in feature_columns + beforeinfo_features 
                            if col in df_processed.columns]
        
        # 欠損値を平均値で補完
        for col in available_features:
            if df_processed[col].dtype in ['float64', 'int64']:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
            else:
                df_processed[col] = df_processed[col].fillna(0)
        
        self.feature_columns = available_features
        X = df_processed[available_features]
        
        if is_training and 'is_winner' in df_processed.columns:
            y = df_processed['is_winner']
            return X, y, available_features
        else:
            return X, available_features
    
    def train_model(self, X, y, model_name='random_forest'):
        """
        指定されたモデルを訓練
        
        Parameters:
        X (DataFrame): 特徴量
        y (Series): ターゲット変数
        model_name (str): モデル名
        
        Returns:
        dict: 訓練結果
        """
        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 標準化（Gradient Boostingの場合）
        if model_name == 'gradient_boosting':
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            model_X_train = X_train_scaled
            model_X_test = X_test_scaled
        else:
            model_X_train = X_train
            model_X_test = X_test
        
        # モデル訓練
        model = self.models[model_name]
        model.fit(model_X_train, y_train)
        
        # 予測と評価
        y_pred = model.predict(model_X_test)
        y_pred_proba = model.predict_proba(model_X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, model_X_train, y_train, cv=5)
        
        # 結果を記録
        result = {
            'model_name': model_name,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': getattr(model, 'feature_importances_', None),
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_samples': len(X_train)
        }
        
        self.model_performance[model_name] = result
        self.is_trained = True
        
        return result
    
    def predict_race(self, X, model_name='random_forest'):
        """
        レース予想を実行
        
        Parameters:
        X (DataFrame): 特徴量
        model_name (str): 使用するモデル名
        
        Returns:
        numpy.array: 勝利確率
        """
        if not self.is_trained:
            raise ValueError("モデルが訓練されていません")
        
        model = self.models[model_name]
        
        # 標準化（必要な場合）
        if model_name == 'gradient_boosting':
            X_scaled = self.scaler.transform(X)
            probabilities = model.predict_proba(X_scaled)[:, 1]
        else:
            probabilities = model.predict_proba(X)[:, 1]
        
        return probabilities
    
    def save_model(self, filepath):
        """
        訓練済みモデルを保存
        
        Parameters:
        filepath (str): 保存先パス
        """
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained,
            'model_performance': self.model_performance,
            'training_history': self.training_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """
        保存済みモデルを読み込み
        
        Parameters:
        filepath (str): ファイルパス
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = model_data['is_trained']
        self.model_performance = model_data.get('model_performance', {})
        self.training_history = model_data.get('training_history', [])

class KyoteiAIGUI:
    """
    競艇AI予想ツールのGUIクラス
    Tkinterを使用したユーザーインターフェース
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("競艇AI予想ツール（研究用）")
        self.root.geometry("1000x700")
        
        # データ処理とMLモデルのインスタンス
        self.data_collector = KyoteiDataCollector()
        self.ml_model = KyoteiMLModel()
        
        # データ保存用変数
        self.training_data = None
        self.prediction_results = None
        
        # GUI作成
        self.progress_var = tk.DoubleVar()
        self.create_widgets()
        
        # プログレスバー用の変数
        self.progress_var = tk.DoubleVar()
        
    def create_widgets(self):
        """
        GUIウィジェットを作成
        """
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # タブコントロール
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # データ収集タブ
        self.create_data_collection_tab(notebook)
        
        # モデル訓練タブ
        self.create_training_tab(notebook)
        
        # 予想実行タブ
        self.create_prediction_tab(notebook)
        
        # 結果表示タブ
        self.create_results_tab(notebook)
        
        # 設定タブ
        self.create_settings_tab(notebook)
        
        # ステータスバー
        self.create_status_bar(main_frame)
        
        # グリッド設定
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
    
    def create_data_collection_tab(self, notebook):
        """
        データ収集タブの作成
        """
        data_frame = ttk.Frame(notebook, padding="10")
        notebook.add(data_frame, text="データ収集")
        
        # 期間設定
        ttk.Label(data_frame, text="データ収集期間:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        period_frame = ttk.Frame(data_frame)
        period_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(period_frame, text="開始日:").grid(row=0, column=0, padx=5)
        self.start_date_var = tk.StringVar(value="2011-01-01")
        ttk.Entry(period_frame, textvariable=self.start_date_var, width=12).grid(row=0, column=1, padx=5)
        
        ttk.Label(period_frame, text="終了日:").grid(row=0, column=2, padx=5)
        self.end_date_var = tk.StringVar(value="2025-08-15")
        ttk.Entry(period_frame, textvariable=self.end_date_var, width=12).grid(row=0, column=3, padx=5)
        
        # データ種別選択
        data_type_frame = ttk.LabelFrame(data_frame, text="収集データ種別", padding="10")
        data_type_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        self.collect_racelists_var = tk.BooleanVar(value=True)
        self.collect_results_var = tk.BooleanVar(value=True)
        self.collect_beforeinfo_var = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(data_type_frame, text="出走表データ", 
                       variable=self.collect_racelists_var).grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(data_type_frame, text="結果データ", 
                       variable=self.collect_results_var).grid(row=0, column=1, sticky=tk.W)
        ttk.Checkbutton(data_type_frame, text="直前情報（時間がかかります）", 
                       variable=self.collect_beforeinfo_var).grid(row=1, column=0, columnspan=2, sticky=tk.W)
        
        # 実行ボタン
        button_frame = ttk.Frame(data_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="データ収集開始", 
                  command=self.start_data_collection).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="既存データ読み込み", 
                  command=self.load_existing_data).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="データ収集停止"  , 
                  command=self.stop_data_collection).grid(row=0, column=2, padx=5)
        
        # プログレスバー
        self.progress_bar = ttk.Progressbar(data_frame, variable=self.progress_var, 
                                          maximum=100, length=400)
        self.progress_bar.grid(row=4, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        # ログ表示
        log_frame = ttk.LabelFrame(data_frame, text="データ収集ログ", padding="10")
        log_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, width=80)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # グリッド設定
        data_frame.columnconfigure(0, weight=1)
        data_frame.rowconfigure(5, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
    
    def create_training_tab(self, notebook):
        """
        モデル訓練タブの作成
        """
        training_frame = ttk.Frame(notebook, padding="10")
        notebook.add(training_frame, text="モデル訓練")
        
        # モデル選択
        model_frame = ttk.LabelFrame(training_frame, text="モデル設定", padding="10")
        model_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.model_var = tk.StringVar(value="random_forest")
        ttk.Radiobutton(model_frame, text="Random Forest", variable=self.model_var, 
                       value="random_forest").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(model_frame, text="Gradient Boosting", variable=self.model_var, 
                       value="gradient_boosting").grid(row=0, column=1, sticky=tk.W)
        
        # 訓練オプション
        options_frame = ttk.LabelFrame(training_frame, text="訓練オプション", padding="10")
        options_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.use_recent_weight_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="直近データに重みを付ける", 
                       variable=self.use_recent_weight_var).grid(row=0, column=0, sticky=tk.W)
        
        self.include_beforeinfo_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="直前情報を含める", 
                       variable=self.include_beforeinfo_var).grid(row=1, column=0, sticky=tk.W)
        
        # 実行ボタン
        button_frame = ttk.Frame(training_frame)
        button_frame.grid(row=2, column=0, pady=10)
        
        ttk.Button(button_frame, text="新規訓練", 
                  command=self.start_training).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="追加訓練", 
                  command=self.start_additional_training).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="モデル保存", 
                  command=self.save_model).grid(row=0, column=2, padx=5)
        ttk.Button(button_frame, text="モデル読み込み", 
                  command=self.load_model).grid(row=0, column=3, padx=5)
        
        # 訓練結果表示
        result_frame = ttk.LabelFrame(training_frame, text="訓練結果", padding="10")
        result_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        self.training_result_text = scrolledtext.ScrolledText(result_frame, height=15, width=80)
        self.training_result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # グリッド設定
        training_frame.columnconfigure(0, weight=1)
        training_frame.rowconfigure(3, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
    
    def create_prediction_tab(self, notebook):
        """
        予想実行タブの作成
        """
        prediction_frame = ttk.Frame(notebook, padding="10")
        notebook.add(prediction_frame, text="レース予想")
        
        # 予想対象選択
        target_frame = ttk.LabelFrame(prediction_frame, text="予想対象レース", padding="10")
        target_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # 日付選択
        ttk.Label(target_frame, text="日付:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.prediction_date_var = tk.StringVar(value=datetime.now().strftime('%Y-%m-%d'))
        ttk.Entry(target_frame, textvariable=self.prediction_date_var, width=12).grid(row=0, column=1, padx=5)
        
        # 競艇場選択
        ttk.Label(target_frame, text="競艇場:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.place_var = tk.StringVar()
        place_combo = ttk.Combobox(target_frame, textvariable=self.place_var, width=10)
        place_combo['values'] = list(self.data_collector.place_mapper.values())
        place_combo.grid(row=0, column=3, padx=5)
        
        # レース番号選択
        ttk.Label(target_frame, text="レース番号:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.race_no_var = tk.StringVar(value="1")
        race_combo = ttk.Combobox(target_frame, textvariable=self.race_no_var, width=5)
        race_combo['values'] = [str(i) for i in range(1, 13)]
        race_combo.grid(row=1, column=1, padx=5)
        
        # 予想実行ボタン
        ttk.Button(target_frame, text="予想実行", 
                  command=self.start_prediction).grid(row=1, column=2, padx=10)
        
        # 予想結果表示
        result_frame = ttk.LabelFrame(prediction_frame, text="予想結果", padding="10")
        result_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # 結果テーブル
        columns = ('順位', '艇番', '選手名', '勝率予想', '信頼度')
        self.prediction_tree = ttk.Treeview(result_frame, columns=columns, show='headings', height=6)
        
        for col in columns:
            self.prediction_tree.heading(col, text=col)
            self.prediction_tree.column(col, width=100, anchor='center')
        
        self.prediction_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # スクロールバー
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.prediction_tree.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.prediction_tree.configure(yscrollcommand=scrollbar.set)
        
        # 詳細情報表示
        detail_frame = ttk.LabelFrame(prediction_frame, text="詳細情報", padding="10")
        detail_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        self.prediction_detail_text = scrolledtext.ScrolledText(detail_frame, height=8, width=80)
        self.prediction_detail_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # グリッド設定
        prediction_frame.columnconfigure(0, weight=1)
        prediction_frame.rowconfigure(1, weight=1)
        prediction_frame.rowconfigure(2, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(0, weight=1)
    
    def create_results_tab(self, notebook):
        """
        結果表示タブの作成
        """
        results_frame = ttk.Frame(notebook, padding="10")
        notebook.add(results_frame, text="結果・統計")
        
        # モデル性能表示
        performance_frame = ttk.LabelFrame(results_frame, text="モデル性能", padding="10")
        performance_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.performance_text = scrolledtext.ScrolledText(performance_frame, height=8, width=80)
        self.performance_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # グラフ表示エリア
        graph_frame = ttk.LabelFrame(results_frame, text="特徴量重要度", padding="10")
        graph_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # matplotlib用のフレーム
        self.graph_frame = graph_frame
        
        # グリッド設定
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        performance_frame.columnconfigure(0, weight=1)
        graph_frame.columnconfigure(0, weight=1)
    
    def create_settings_tab(self, notebook):
        """
        設定タブの作成
        """
        settings_frame = ttk.Frame(notebook, padding="10")
        notebook.add(settings_frame, text="設定")
        
        # データ保存設定
        data_frame = ttk.LabelFrame(settings_frame, text="データ保存設定", padding="10")
        data_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(data_frame, text="データ保存ディレクトリ:").grid(row=0, column=0, sticky=tk.W)
        self.data_dir_var = tk.StringVar(value="./kyotei_data")
        ttk.Entry(data_frame, textvariable=self.data_dir_var, width=50).grid(row=1, column=0, padx=5)
        ttk.Button(data_frame, text="参照", 
                  command=self.browse_data_directory).grid(row=1, column=1, padx=5)
        
        # モデル保存設定
        model_frame = ttk.LabelFrame(settings_frame, text="モデル保存設定", padding="10")
        model_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(model_frame, text="モデル保存ファイル:").grid(row=0, column=0, sticky=tk.W)
        self.model_file_var = tk.StringVar(value="./kyotei_model.pkl")
        ttk.Entry(model_frame, textvariable=self.model_file_var, width=50).grid(row=1, column=0, padx=5)
        ttk.Button(model_frame, text="参照", 
                  command=self.browse_model_file).grid(row=1, column=1, padx=5)
        
        # APIアクセス設定
        api_frame = ttk.LabelFrame(settings_frame, text="APIアクセス設定", padding="10")
        api_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(api_frame, text="リクエスト間隔（秒）:").grid(row=0, column=0, sticky=tk.W)
        self.request_interval_var = tk.StringVar(value="1.0")
        ttk.Entry(api_frame, textvariable=self.request_interval_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(api_frame, text="タイムアウト（秒）:").grid(row=1, column=0, sticky=tk.W)
        self.timeout_var = tk.StringVar(value="10.0")
        ttk.Entry(api_frame, textvariable=self.timeout_var, width=10).grid(row=1, column=1, padx=5)
        
        # 使用方法とライセンス情報
        info_frame = ttk.LabelFrame(settings_frame, text="使用方法・注意事項", padding="10")
        info_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        info_text = """
競艇AI予想ツール（研究用）

【使用方法】
1. データ収集タブで過去データを収集
2. モデル訓練タブでAIモデルを訓練
3. レース予想タブで実際の予想を実行

【重要な注意事項】
・このツールは研究・学習目的で作成されています
・実際の舟券購入には使用しないでください
・競艇はギャンブルであり、損失のリスクがあります
・データ取得時は競艇公式サイトの利用規約を遵守してください
・過度なアクセスはサーバーに負荷をかけるため控えてください

【免責事項】
・予想の的中を保証するものではありません
・このツールの使用による損失について責任を負いません
・研究目的以外での使用は推奨されません
        """
        
        info_text_widget = scrolledtext.ScrolledText(info_frame, height=12, width=80)
        info_text_widget.insert('1.0', info_text)
        info_text_widget.config(state='disabled')
        info_text_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # グリッド設定
        settings_frame.columnconfigure(0, weight=1)
        settings_frame.rowconfigure(3, weight=1)
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)
    
    def create_status_bar(self, parent):
        """
        ステータスバーの作成
        """
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.status_var = tk.StringVar(value="準備完了")
        ttk.Label(status_frame, textvariable=self.status_var).grid(row=0, column=0, sticky=tk.W)
        
        # 現在時刻表示
        self.time_var = tk.StringVar()
        ttk.Label(status_frame, textvariable=self.time_var).grid(row=0, column=1, sticky=tk.E)
        
        self.update_time()
    
    def update_time(self):
        """
        現在時刻を更新
        """
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.time_var.set(current_time)
        self.root.after(1000, self.update_time)
    
    def log_message(self, message):
        """
        ログメッセージを表示
        
        Parameters:
        message (str): ログメッセージ
        """
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.root.update()
    
    def update_status(self, status):
        """
        ステータスを更新
        
        Parameters:
        status (str): ステータスメッセージ
        """
        self.status_var.set(status)
        self.root.update()
    
    def start_data_collection(self):
        """
        データ収集を開始（別スレッドで実行）
        """
        # フラグの設定
        self.stop_collection_flag = False
        # 入力値の検証
        try:
            start_date = datetime.strptime(self.start_date_var.get(), '%Y-%m-%d')
            end_date = datetime.strptime(self.end_date_var.get(), '%Y-%m-%d')
            
            if start_date >= end_date:
                messagebox.showerror("エラー", "開始日は終了日より前に設定してください")
                return
                
        except ValueError:
            messagebox.showerror("エラー", "日付の形式が正しくありません (YYYY-MM-DD)")
            return
        
        # データ収集を別スレッドで実行
        self.log_text.delete('1.0', tk.END)
        self.log_message("データ収集を開始します...")
        
        thread = threading.Thread(target=self._collect_data_thread)
        thread.daemon = True
        thread.start()

    def stop_data_collection(self):
        self.stop_collection_flag = True
        self.update_status("データ収集停止中...")

    def _collect_data_thread(self):
        """
        データ収集のメインスレッド
        """
        try:
            start_date = datetime.strptime(self.start_date_var.get(), '%Y-%m-%d')
            end_date = datetime.strptime(self.end_date_var.get(), '%Y-%m-%d')
            
            # データ保存ディレクトリを作成
            data_dir = self.data_dir_var.get()
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(f"{data_dir}/racelists", exist_ok=True)
            os.makedirs(f"{data_dir}/results", exist_ok=True)
            
            # 日付リストを作成
            date_list = []
            current_date = start_date
            while current_date <= end_date:
                date_list.append(current_date.strftime('%Y-%m-%d'))
                current_date += timedelta(days=1)
            
            total_dates = len(date_list)
            collected_data = []
            
            self.log_message(f"収集期間: {len(date_list)} 日間")
            
            # 各日のデータを収集
            for i, date in enumerate(date_list):
                if self.stop_collection_flag:
                    self.log_message("データ収集が停止されました")
                    self.update_status("データ収集停止")
                    return
                else:
                    self.update_status(f"データ収集中: {date} ({i+1}/{total_dates})")
                    progress = (i + 1) / total_dates * 100
                    self.progress_var.set(progress)
                    
                    try:
                        # 出走表データの収集
                        if self.collect_racelists_var.get():
                            if self.data_collector.download_file('racelists', date):
                                race_df = self.data_collector.get_racelists(date)
                                if race_df is not None:
                                    race_df.to_csv(f"{data_dir}/racelists/{date.replace('-', '')}.csv", 
                                                    index=False, encoding='utf-8-sig')
                                    self.log_message(f"出走表データ収集完了: {date} ({len(race_df)} レース)")
                                else:
                                    self.log_message(f"出走表データなし: {date}")
                        
                        # 結果データの収集
                        if self.collect_results_var.get():
                            if self.data_collector.download_file('results', date):
                                result_df = self.data_collector.get_results(date)
                                if result_df is not None:
                                    result_df.to_csv(f"{data_dir}/results/{date.replace('-', '')}.csv", 
                                                    index=False, encoding='utf-8-sig')
                                    self.log_message(f"結果データ収集完了: {date} ({len(result_df)} レース)")
                                else:
                                    self.log_message(f"結果データなし: {date}")
                        
                        # 直前情報の収集（時間がかかるため、必要な場合のみ）
                        if self.collect_beforeinfo_var.get():
                            self.log_message(f"直前情報収集開始: {date}")
                            # 実装は省略（非常に時間がかかるため）
                            
                        # リクエスト間隔を空ける
                        time.sleep(float(self.request_interval_var.get()))
                    except Exception as e:
                        self.log_message(f"エラー ({date}): {str(e)}")
                        continue
            
            self.log_message("データ収集が完了しました")
            self.update_status("データ収集完了")
            messagebox.showinfo("完了", "データ収集が完了しました")
            
        except Exception as e:
            self.log_message(f"データ収集エラー: {str(e)}")
            self.update_status("エラー発生")
            messagebox.showerror("エラー", f"データ収集中にエラーが発生しました: {str(e)}")
        
        finally:
            self.progress_var.set(0)
    
    def load_existing_data(self):
        """
        既存のデータを読み込み
        """
        data_dir = self.data_dir_var.get()
        
        if not os.path.exists(data_dir):
            messagebox.showerror("エラー", f"データディレクトリが見つかりません: {data_dir}")
            return
        
        try:
            # 既存のCSVファイルを検索
            racelist_files = []
            result_files = []
            
            racelist_dir = f"{data_dir}/racelists"
            result_dir = f"{data_dir}/results"
            
            if os.path.exists(racelist_dir):
                racelist_files = [f for f in os.listdir(racelist_dir) if f.endswith('.csv')]
            
            if os.path.exists(result_dir):
                result_files = [f for f in os.listdir(result_dir) if f.endswith('.csv')]
            
            self.log_message(f"出走表ファイル: {len(racelist_files)} 個")
            self.log_message(f"結果ファイル: {len(result_files)} 個")
            
            if len(racelist_files) == 0 and len(result_files) == 0:
                messagebox.showwarning("警告", "読み込み可能なデータファイルが見つかりません")
                return
            
            # データを統合
            all_racelists = []
            all_results = []
            
            self.update_status("データ読み込み中...")
            
            # 出走表データの読み込み
            for i, file in enumerate(racelist_files):
                try:
                    df = pd.read_csv(f"{racelist_dir}/{file}", encoding='utf-8-sig')
                    all_racelists.append(df)
                    if i % 100 == 0:  # 進捗表示
                        self.progress_var.set(i / len(racelist_files) * 50)
                        self.root.update()
                except Exception as e:
                    self.log_message(f"ファイル読み込みエラー ({file}): {str(e)}")
            
            # 結果データの読み込み
            for i, file in enumerate(result_files):
                try:
                    df = pd.read_csv(f"{result_dir}/{file}", encoding='utf-8-sig')
                    all_results.append(df)
                    if i % 100 == 0:  # 進捗表示
                        self.progress_var.set(50 + i / len(result_files) * 50)
                        self.root.update()
                except Exception as e:
                    self.log_message(f"ファイル読み込みエラー ({file}): {str(e)}")
            
            # データフレームの統合
            if all_racelists:
                self.racelist_data = pd.concat(all_racelists, ignore_index=True)
                self.log_message(f"出走表データ統合完了: {len(self.racelist_data)} レコード")
            
            if all_results:
                self.result_data = pd.concat(all_results, ignore_index=True)
                self.log_message(f"結果データ統合完了: {len(self.result_data)} レコード")
            
            self.update_status("データ読み込み完了")
            messagebox.showinfo("完了", "既存データの読み込みが完了しました")
            
        except Exception as e:
            self.log_message(f"データ読み込みエラー: {str(e)}")
            messagebox.showerror("エラー", f"データ読み込み中にエラーが発生しました: {str(e)}")
        
        finally:
            self.progress_var.set(0)
    
    def start_training(self):
        """
        新規モデル訓練を開始
        """
        if not hasattr(self, 'racelist_data') or not hasattr(self, 'result_data'):
            messagebox.showerror("エラー", "訓練用データが読み込まれていません。\nまずデータ収集またはデータ読み込みを実行してください。")
            return
        
        self.training_result_text.delete('1.0', tk.END)
        thread = threading.Thread(target=self._train_model_thread, args=(False,))
        thread.daemon = True
        thread.start()
    
    def start_additional_training(self):
        """
        追加モデル訓練を開始
        """
        if not self.ml_model.is_trained:
            messagebox.showerror("エラー", "追加訓練するには、既存のモデルを読み込むか新規訓練を実行してください。")
            return
        
        if not hasattr(self, 'racelist_data') or not hasattr(self, 'result_data'):
            messagebox.showerror("エラー", "追加訓練用データが読み込まれていません。")
            return
        
        thread = threading.Thread(target=self._train_model_thread, args=(True,))
        thread.daemon = True
        thread.start()
    
    def _train_model_thread(self, is_additional=False):
        """
        モデル訓練のメインスレッド
        
        Parameters:
        is_additional (bool): 追加訓練かどうか
        """
        try:
            self.update_status("モデル訓練中...")
            
            # 特徴量作成
            self._append_training_result("特徴量作成中...")
            
            # 直近データへの重み付けを考慮
            if self.use_recent_weight_var.get():
                # 日付でソート
                self.racelist_data['date'] = pd.to_datetime(self.racelist_data['date'])
                self.result_data['date'] = pd.to_datetime(self.result_data['date'])
                
                # 直近3年のデータを重点的に使用
                cutoff_date = datetime.now() - timedelta(days=365 * 3)
                recent_racelist = self.racelist_data[self.racelist_data['date'] >= cutoff_date]
                recent_results = self.result_data[self.result_data['date'] >= cutoff_date]
                
                if len(recent_racelist) > 0:
                    self._append_training_result(f"直近3年のデータを重点使用: {len(recent_racelist)} レース")
                    features_df = self.ml_model.create_features(recent_racelist, recent_results)
                else:
                    self._append_training_result("直近データが不足のため全データを使用")
                    features_df = self.ml_model.create_features(self.racelist_data, self.result_data)
            else:
                features_df = self.ml_model.create_features(self.racelist_data, self.result_data)
            
            if features_df is None or len(features_df) == 0:
                raise ValueError("特徴量の作成に失敗しました")
            
            self._append_training_result(f"特徴量作成完了: {len(features_df)} サンプル")
            
            # 前処理
            X, y, feature_names = self.ml_model.preprocess_features(features_df, is_training=True)
            
            self._append_training_result(f"使用特徴量: {len(feature_names)} 個")
            self._append_training_result(f"特徴量: {', '.join(feature_names[:10])}...")  # 最初の10個のみ表示
            
            # モデル訓練
            model_name = self.model_var.get()
            self._append_training_result(f"\n{model_name} モデルの訓練開始...")
            
            result = self.ml_model.train_model(X, y, model_name)
            
            # 訓練結果の表示
            self._append_training_result(f"\n訓練完了!")
            self._append_training_result(f"精度: {result['accuracy']:.4f}")
            self._append_training_result(f"クロスバリデーション: {result['cv_mean']:.4f} (+/- {result['cv_std'] * 2:.4f})")
            self._append_training_result(f"訓練サンプル数: {result['n_samples']}")
            self._append_training_result(f"訓練日時: {result['training_date']}")
            
            # 特徴量重要度の表示
            if result['feature_importance'] is not None:
                self._append_training_result("\n特徴量重要度 (上位10位):")
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': result['feature_importance']
                }).sort_values('importance', ascending=False)
                
                for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                    self._append_training_result(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
            
            print("モデル訓練完了1")

            # 訓練履歴に追加
            self.ml_model.training_history.append({
                'date': datetime.now(),
                'model': model_name,
                'accuracy': result['accuracy'],
                'samples': result['n_samples'],
                'is_additional': is_additional
            })
            
            self.update_status("モデル訓練完了")
            print("モデル訓練完了2")
            self._update_performance_display()
            print("モデル訓練完了3")
            try:
#                self._plot_feature_importance()
                self.root.after(100, self._plot_feature_importance)
            except Exception as e:
                self._append_training_result(f"グラフ描画エラー: {e}")
            print("モデル訓練完了4")
            messagebox.showinfo("完了", "モデル訓練が完了しました")
            
        except Exception as e:
            self._append_training_result(f"\n訓練エラー: {str(e)}")
            self.update_status("訓練エラー")
            messagebox.showerror("エラー", f"モデル訓練中にエラーが発生しました: {str(e)}")
    
    def _append_training_result(self, text):
        """
        訓練結果テキストに追加
        
        Parameters:
        text (str): 追加するテキスト
        """
        self.training_result_text.insert(tk.END, text + '\n')
        self.training_result_text.see(tk.END)
        self.root.update()
    
    def save_model(self):
        """
        モデルを保存
        """
        if not self.ml_model.is_trained:
            messagebox.showerror("エラー", "保存するモデルがありません。先にモデル訓練を実行してください。")
            return
        
        try:
            filepath = self.model_file_var.get()
            self.ml_model.save_model(filepath)
            messagebox.showinfo("完了", f"モデルを保存しました: {filepath}")
            self.log_message(f"モデル保存完了: {filepath}")
        except Exception as e:
            messagebox.showerror("エラー", f"モデル保存中にエラーが発生しました: {str(e)}")
    
    def load_model(self):
        """
        モデルを読み込み
        """
        try:
            filepath = self.model_file_var.get()
            if not os.path.exists(filepath):
                messagebox.showerror("エラー", f"モデルファイルが見つかりません: {filepath}")
                return
            
            self.ml_model.load_model(filepath)
            messagebox.showinfo("完了", f"モデルを読み込みました: {filepath}")
            self.log_message(f"モデル読み込み完了: {filepath}")
            
            # 性能表示を更新
            self._update_performance_display()
            self._plot_feature_importance()
            
        except Exception as e:
            messagebox.showerror("エラー", f"モデル読み込み中にエラーが発生しました: {str(e)}")
    
    def start_prediction(self):
        """
        レース予想を開始
        """
        if not self.ml_model.is_trained:
            messagebox.showerror("エラー", "予想するには先にモデル訓練またはモデル読み込みを実行してください。")
            return
        
        # 入力値の検証
        try:
            prediction_date = self.prediction_date_var.get()
            datetime.strptime(prediction_date, '%Y-%m-%d')
        except ValueError:
            messagebox.showerror("エラー", "日付の形式が正しくありません (YYYY-MM-DD)")
            return
        
        if not self.place_var.get():
            messagebox.showerror("エラー", "競艇場を選択してください")
            return
        
        try:
            race_no = int(self.race_no_var.get())
            if not 1 <= race_no <= 12:
                raise ValueError()
        except ValueError:
            messagebox.showerror("エラー", "レース番号は1-12の範囲で入力してください")
            return
        
        # 予想実行
        thread = threading.Thread(target=self._predict_race_thread)
        thread.daemon = True
        thread.start()
    
    def _predict_race_thread(self):
        """
        レース予想のメインスレッド
        """
        try:
            self.update_status("予想実行中...")
            
            prediction_date = self.prediction_date_var.get()
            place_name = self.place_var.get()
            race_no = int(self.race_no_var.get())
            
            # 競艇場名から場所コードを取得
            place_cd = None
            for code, name in self.data_collector.place_mapper.items():
                if name == place_name:
                    place_cd = code
                    break
            
            if place_cd is None:
                raise ValueError(f"競艇場が見つかりません: {place_name}")
            
            # 出走表データを取得
            self._append_prediction_detail(f"出走表データ取得中: {prediction_date} {place_name} {race_no}R")
            
            # 実際の予想では、リアルタイムでデータを取得
            # ここではサンプルデータで代替
            race_data = self._create_sample_race_data(prediction_date, place_cd, race_no)
            
            if race_data is None:
                raise ValueError("出走表データを取得できませんでした")
            
            # 特徴量作成
            features_df = self.ml_model.create_features(race_data)
            
            if features_df is None or len(features_df) == 0:
                raise ValueError("特徴量の作成に失敗しました")
            
            # 予想実行
            X, feature_names = self.ml_model.preprocess_features(features_df, is_training=False)
            
            model_name = self.model_var.get()
            probabilities = self.ml_model.predict_race(X, model_name)
            
            # 結果を整理
            results = []
            for i, prob in enumerate(probabilities):
                boat_no = i + 1
                results.append({
                    'boat_no': boat_no,
                    'probability': prob,
                    'racer_name': f'選手{boat_no}',  # 実際のデータでは選手名を取得
                    'confidence': self._calculate_confidence(prob)
                })
            
            # 確率順にソート
            results.sort(key=lambda x: x['probability'], reverse=True)
            
            # 結果表示
            self._display_prediction_results(results)
            self._append_prediction_detail(f"\n予想完了: {prediction_date} {place_name} {race_no}R")
            
            self.update_status("予想完了")
            
        except Exception as e:
            self._append_prediction_detail(f"予想エラー: {str(e)}")
            self.update_status("予想エラー")
            messagebox.showerror("エラー", f"予想実行中にエラーが発生しました: {str(e)}")
    
    def _create_sample_race_data(self, date, place_cd, race_no):
        """
        サンプルレースデータを作成（実際の実装では、リアルタイムでデータを取得）
        
        Parameters:
        date (str): 日付
        place_cd (int): 場所コード
        race_no (int): レース番号
        
        Returns:
        pandas.DataFrame: サンプルレースデータ
        """
        # サンプルデータの作成
        sample_data = {
            'date': [date],
            'place_cd': [place_cd],
            'race_no': [race_no],
            'distance': [1800],
        }
        
        # 6艇のサンプルデータ
        for boat_no in range(1, 7):
            sample_data.update({
                f'age_{boat_no}': [25 + np.random.randint(0, 20)],
                f'weight_{boat_no}': [50 + np.random.randint(0, 6)],
                f'glob_win_{boat_no}': [4.0 + np.random.random() * 4.0],
                f'glob_in2_{boat_no}': [30.0 + np.random.random() * 40.0],
                f'loc_win_{boat_no}': [4.0 + np.random.random() * 4.0],
                f'loc_in2_{boat_no}': [30.0 + np.random.random() * 40.0],
                f'motor_in2_{boat_no}': [25.0 + np.random.random() * 50.0],
                f'boat_in2_{boat_no}': [25.0 + np.random.random() * 50.0],
                f'class_{boat_no}': [np.random.choice(['A1', 'A2', 'B1', 'B2'])],
                f'area_{boat_no}': [np.random.choice(['群馬', '埼玉', '東京', '静岡'])]
            })
        
        return pd.DataFrame(sample_data)
    
    def _calculate_confidence(self, probability):
        """
        予想確率から信頼度を計算
        
        Parameters:
        probability (float): 勝利確率
        
        Returns:
        str: 信頼度
        """
        if probability >= 0.4:
            return "高"
        elif probability >= 0.2:
            return "中"
        else:
            return "低"
    
    def _display_prediction_results(self, results):
        """
        予想結果を表示
        
        Parameters:
        results (list): 予想結果リスト
        """
        # テーブルをクリア
        for item in self.prediction_tree.get_children():
            self.prediction_tree.delete(item)
        
        # 結果を表示
        for rank, result in enumerate(results, 1):
            self.prediction_tree.insert('', 'end', values=(
                rank,
                result['boat_no'],
                result['racer_name'],
                f"{result['probability']:.3f}",
                result['confidence']
            ))
    
    def _append_prediction_detail(self, text):
        """
        予想詳細テキストに追加
        
        Parameters:
        text (str): 追加するテキスト
        """
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.prediction_detail_text.insert(tk.END, f"[{timestamp}] {text}\n")
        self.prediction_detail_text.see(tk.END)
        self.root.update()
    
    def _update_performance_display(self):
        """
        モデル性能表示を更新
        """
        self.performance_text.delete('1.0', tk.END)
        
        if self.ml_model.model_performance:
            self.performance_text.insert(tk.END, "=== モデル性能情報 ===\n\n")
            
            for model_name, performance in self.ml_model.model_performance.items():
                self.performance_text.insert(tk.END, f"モデル: {model_name}\n")
                self.performance_text.insert(tk.END, f"  精度: {performance['accuracy']:.4f}\n")
                self.performance_text.insert(tk.END, f"  CV平均: {performance['cv_mean']:.4f}\n")
                self.performance_text.insert(tk.END, f"  CV標準偏差: {performance['cv_std']:.4f}\n")
                self.performance_text.insert(tk.END, f"  訓練サンプル数: {performance['n_samples']}\n")
                self.performance_text.insert(tk.END, f"  訓練日時: {performance['training_date']}\n\n")
        
        if self.ml_model.training_history:
            self.performance_text.insert(tk.END, "=== 訓練履歴 ===\n\n")
            
            for i, history in enumerate(self.ml_model.training_history[-10:], 1):  # 最新10件
                train_type = "追加訓練" if history['is_additional'] else "新規訓練"
                self.performance_text.insert(tk.END, 
                    f"{i:2d}. {history['date'].strftime('%Y-%m-%d %H:%M')} - "
                    f"{history['model']} ({train_type}) - "
                    f"精度: {history['accuracy']:.4f} - "
                    f"サンプル: {history['samples']}\n")
    
    def _plot_feature_importance(self):
        """
        特徴量重要度をプロット
        """
        print("1. 特徴量重要度プロット開始")
        # 既存のグラフをクリア
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        print("2. グラフフレームのクリア完了")
        if not self.ml_model.model_performance:
            return
        print("3. モデル性能情報が存在するか確認")
        # 最新のモデルの特徴量重要度を取得
        latest_model = list(self.ml_model.model_performance.keys())[-1]
        print(f"4. 最新モデル: {latest_model} の特徴量重要度取得開始")
        importance = self.ml_model.model_performance[latest_model]['feature_importance']
        print("5. 最新モデルの特徴量重要度取得完了")
        if importance is None or len(self.ml_model.feature_columns) == 0:
            return
        print("6. 特徴量重要度が存在するか確認")
        try:
            print("7. 特徴量重要度プロット開始")
            # matplotlib図を作成
            fig, ax = plt.subplots()
            print("8. matplotlib図の作成完了")
            # 重要度データフレーム作成
            importance_df = pd.DataFrame({
                'feature': self.ml_model.feature_columns,
                'importance': importance
            }).sort_values('importance', ascending=True)
            print("9. 特徴量重要度データフレーム作成完了")
            # 上位15個の特徴量のみ表示
            top_features = importance_df.tail(15)
            print("10. 上位15個の特徴量抽出完了")
            # 横棒グラフ
            ax.barh(top_features['feature'], top_features['importance'])
            ax.set_xlabel('重要度')
            ax.set_title(f'特徴量重要度 ({latest_model})')
            ax.grid(True, alpha=0.3)
            print("11. 特徴量重要度グラフの描画完了")
            plt.tight_layout()
            print("12. グラフのレイアウト調整完了")
            # Tkinterに埋め込み
            canvas = FigureCanvasTkAgg(fig, self.graph_frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            print("13. Tkinterにグラフを埋め込み完了")
            self.graph_frame.columnconfigure(0, weight=1)
            self.graph_frame.rowconfigure(0, weight=1)
            print("14. グラフフレームの設定完了")
        except Exception as e:
            print(f"グラフ描画エラー: {e}")
    
    def browse_data_directory(self):
        """
        データディレクトリを選択
        """
        directory = filedialog.askdirectory(initialdir=self.data_dir_var.get())
        if directory:
            self.data_dir_var.set(directory)
    
    def browse_model_file(self):
        """
        モデルファイルを選択
        """
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            initialdir="./",
            initialfile="kyotei_model.pkl"
        )
        if filepath:
            self.model_file_var.set(filepath)

def main():
    """
    メイン関数：アプリケーションを起動
    """
    # 必要なディレクトリを作成
    os.makedirs("downloads", exist_ok=True)
    os.makedirs("downloads/racelists", exist_ok=True)
    os.makedirs("downloads/results", exist_ok=True)
    os.makedirs("kyotei_data", exist_ok=True)
    
    # Tkinterアプリケーションを起動
    root = tk.Tk()
    
    # アプリケーションアイコンの設定（オプション）
    try:
        root.iconbitmap("icon.ico")  # アイコンファイルがある場合
    except:
        pass
    
    # GUIアプリケーションのインスタンス作成
    app = KyoteiAIGUI(root)
    
    # ウィンドウが閉じられる際の処理
    def on_closing():
        if messagebox.askokcancel("終了", "アプリケーションを終了しますか？"):
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # アプリケーション実行
    root.mainloop()

if __name__ == "__main__":
    """
    スクリプトが直接実行された場合のエントリーポイント
    
    使用方法:
    1. 必要なライブラリをインストール:
       pip install pandas numpy scikit-learn matplotlib tkinter requests beautifulsoup4 lhafile mojimoji
       
    2. このスクリプトを実行:
       python kyotei_ai_tool.py
       
    3. GUIが起動するので、以下の手順で使用:
       a) データ収集タブでデータを収集
       b) モデル訓練タブでAIモデルを訓練
       c) レース予想タブで実際の予想を実行
    
    注意事項:
    - このツールは研究目的で作成されています
    - 実際の舟券購入には使用しないでください
    - データ収集時はサーバーへの負荷を考慮し、適切な間隔を設けてください
    - 競艇公式サイトの利用規約を遵守してください
    """
    
    print("=" * 60)
    print("競艇AI予想ツール（研究用）")
    print("=" * 60)
    print("このツールは機械学習の有用性研究のために作成されています。")
    print("実際のギャンブルでの使用は推奨されません。")
    print("")
    print("必要なライブラリ:")
    print("- pandas, numpy, scikit-learn")
    print("- matplotlib, tkinter")
    print("- requests, beautifulsoup4")
    print("- lhafile, mojimoji")
    print("")
    print("使用方法:")
    print("1. データ収集タブで過去データを収集")
    print("2. モデル訓練タブでAIモデルを訓練") 
    print("3. レース予想タブで予想を実行")
    print("")
    print("注意: データ収集には時間がかかる場合があります。")
    print("      サーバーへの負荷を考慮し、適切な間隔を設けています。")
    print("=" * 60)
    print("")
    
    # 必要なライブラリの確認
    missing_libraries = []
    required_libraries = [
        'pandas', 'numpy', 'sklearn', 'matplotlib', 
        'requests', 'bs4', 'lhafile', 'mojimoji'
    ]
    
    for lib in required_libraries:
        try:
            __import__(lib)
        except ImportError:
            missing_libraries.append(lib)
    
    if missing_libraries:
        print("以下のライブラリがインストールされていません:")
        for lib in missing_libraries:
            print(f"  - {lib}")
        print("\n以下のコマンドでインストールしてください:")
        if 'bs4' in missing_libraries:
            print("pip install beautifulsoup4")
            missing_libraries.remove('bs4')
        if 'sklearn' in missing_libraries:
            print("pip install scikit-learn")
            missing_libraries.remove('sklearn')
        for lib in missing_libraries:
            print(f"pip install {lib}")
        print("")
        input("ライブラリをインストールした後、Enterキーを押してください...")
    
    # アプリケーション起動
    try:
        main()
    except KeyboardInterrupt:
        print("\nアプリケーションが中断されました。")
    except Exception as e:
        print(f"\nアプリケーション実行中にエラーが発生しました: {e}")
        input("Enterキーを押して終了...")

# =====================================================================
# 使用方法とサンプルデータについて
# =====================================================================

"""
【詳細な使用方法】

1. データ収集フェーズ
   - 開始日・終了日を設定（10年分推奨: 2014-01-01 ～ 2024-12-31）
   - 出走表データ・結果データにチェック
   - 「データ収集開始」ボタンをクリック
   - 数時間～1日程度の時間がかかります（データ量による）

2. データ読み込み
   - 既にデータを持っている場合は「既存データ読み込み」を使用
   - CSVファイル形式でのデータインポートが可能

3. モデル訓練フェーズ
   - Random ForestまたはGradient Boostingを選択
   - 「直近データに重みを付ける」で最近の成績を重視
   - 「新規訓練」で初回訓練を実行
   - 追加データがある場合は「追加訓練」で既存モデルを改善

4. 予想フェーズ
   - 日付・競艇場・レース番号を指定
   - 「予想実行」で AI が各艇の勝率を計算
   - 結果は確率順で表示されます

【技術的特徴】

- 機械学習アルゴリズム: Random Forest, Gradient Boosting
- 特徴量: 選手成績、モーター性能、コース、天候条件など
- 直近データ重み付け: 過去3年のデータを重点的に学習
- 追加学習: 新しいデータで既存モデルを改善
- モデル永続化: 学習結果をファイルに保存・読み込み可能

【研究での活用】

- 予想精度の分析
- 特徴量重要度の検証
- 異なるアルゴリズムの比較
- データ量と精度の関係調査
- 時系列による予想精度の変化

【免責事項】

このツールは教育・研究目的で作成されており、以下の点にご注意ください：

1. 実際のギャンブルでの使用は推奨されません
2. 予想の的中を保証するものではありません  
3. 経済的損失について責任を負いません
4. データ取得時は利用規約を遵守してください
5. サーバーへの負荷を考慮した適切な使用をお願いします

【ライセンス】

本ソフトウェアはMITライセンスの下で提供されます。
研究・教育目的での使用は自由ですが、商用利用時は注意してください。
"""