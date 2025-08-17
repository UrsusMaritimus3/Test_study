import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import time
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from datetime import datetime, timedelta
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import threading
import re

class BoatRacePrediction:
    def __init__(self):
        """
        競艇予想システムの初期化
        選手個人データを含む高精度予想システム
        """
        # データ保存用のディレクトリ作成
        os.makedirs('data', exist_ok=True)
        
        # 学習済みモデルとスケーラーの保存パス
        self.model_path = 'data/trained_model.pkl'
        self.scaler_path = 'data/feature_scaler.pkl'
        self.racer_stats_path = 'data/racer_stats.csv'
        
        # 競艇場のマッピング（場所番号と名前）
        self.venue_mapping = {
            1: '桐生', 2: '戸田', 3: '江戸川', 4: '平和島', 5: '多摩川', 6: '浜名湖',
            7: '蒲郡', 8: '常滑', 9: '津', 10: '三国', 11: 'びわこ', 12: '住之江',
            13: '尼崎', 14: '鳴門', 15: '丸亀', 16: '児島', 17: '宮島', 18: '徳山',
            19: '下関', 20: '若松', 21: '芦屋', 22: '福岡', 23: '唐津', 24: '大村'
        }
        
        # 機械学習モデルとスケーラー
        self.model = None
        self.scaler = None
        
        # 選手統計データ
        self.racer_stats = {}
        
        # データフレーム
        self.race_data = pd.DataFrame()
        
        # 学習済みモデルとスケーラーの読み込み
        self.load_model_and_scaler()
        self.load_racer_stats()

    def get_race_data(self, venue_code, date, race_num):
        """
        指定された競艇場、日付、レース番号のデータを取得
        
        Args:
            venue_code (int): 競艇場コード (1-24)
            date (str): 日付 (YYYYMMDD形式)
            race_num (int): レース番号 (1-12)
        
        Returns:
            dict: レースデータ
        """
        try:
            # レース結果ページのURL
            url = f"https://www.boatrace.jp/owpc/pc/race/raceresult?rno={race_num}&jcd={venue_code:02d}&hd={date}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # レースデータの初期化
            race_data = {
                'venue_code': venue_code,
                'venue_name': self.venue_mapping.get(venue_code, '不明'),
                'date': date,
                'race_num': race_num,
                'racers': []
            }
            
            # 選手データテーブルを取得
            racer_table = soup.find('table', class_='is-w495')
            if not racer_table:
                print(f"選手データが見つかりません: {url}")
                return None
            
            # 各選手のデータを取得
            rows = racer_table.find_all('tr')[1:]  # ヘッダー行をスキップ
            for i, row in enumerate(rows[:6]):  # 最大6艇
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    # 選手番号を取得
                    racer_num_cell = cells[1]
                    racer_num_link = racer_num_cell.find('a')
                    if racer_num_link:
                        racer_num = racer_num_link.get_text(strip=True)
                    else:
                        racer_num = racer_num_cell.get_text(strip=True)
                    
                    # 選手名を取得
                    racer_name = cells[2].get_text(strip=True) if len(cells) > 2 else '不明'
                    
                    racer_data = {
                        'lane': i + 1,  # 艇番（1-6）
                        'racer_num': racer_num,
                        'racer_name': racer_name
                    }
                    
                    race_data['racers'].append(racer_data)
            
            # 結果データを取得
            result_table = soup.find('table', class_='is-w495')
            if result_table:
                result_rows = result_table.find_all('tr')[1:]
                for i, row in enumerate(result_rows[:6]):
                    cells = row.find_all(['td', 'th'])
                    if i < len(race_data['racers']) and len(cells) > 0:
                        # 着順を取得（結果が存在する場合）
                        finish_pos = cells[0].get_text(strip=True)
                        if finish_pos.isdigit():
                            race_data['racers'][i]['finish_position'] = int(finish_pos)
            
            return race_data
            
        except Exception as e:
            print(f"データ取得エラー: {e}")
            return None

    def get_racer_stats(self, racer_num):
        """
        選手の個人成績データを取得
        
        Args:
            racer_num (str): 選手番号
        
        Returns:
            dict: 選手統計データ
        """
        try:
            # 選手詳細ページのURL
            url = f"https://www.boatrace.jp/owpc/pc/data/racersearch/season?toban={racer_num}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # デフォルトの統計値
            stats = {
                'win_rate': 0.0,      # 1着率
                'quinella_rate': 0.0,  # 2連対率
                'trio_rate': 0.0,     # 3連対率
                'avg_start_time': 0.0, # 平均スタートタイム
                'races_count': 0,     # 出走回数
                'wins': 0,            # 1着回数
                'places': 0,          # 2着回数
                'shows': 0            # 3着回数
            }
            
            # 成績テーブルを探す
            stats_table = soup.find('table', class_='is-w495')
            if stats_table:
                rows = stats_table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        header = cells[0].get_text(strip=True)
                        value_text = cells[1].get_text(strip=True)
                        
                        # 数値の抽出
                        numbers = re.findall(r'[\d.]+', value_text)
                        if numbers:
                            value = float(numbers[0])
                            
                            if '勝率' in header or '1着率' in header:
                                stats['win_rate'] = value
                            elif '2連対率' in header:
                                stats['quinella_rate'] = value
                            elif '3連対率' in header:
                                stats['trio_rate'] = value
                            elif 'スタート' in header and 'タイム' in header:
                                stats['avg_start_time'] = value
                            elif '出走' in header:
                                stats['races_count'] = int(value)
                            elif '1着' in header:
                                stats['wins'] = int(value)
                            elif '2着' in header:
                                stats['places'] = int(value)
                            elif '3着' in header:
                                stats['shows'] = int(value)
            
            return stats
            
        except Exception as e:
            print(f"選手統計データ取得エラー (選手番号: {racer_num}): {e}")
            return {
                'win_rate': 0.0, 'quinella_rate': 0.0, 'trio_rate': 0.0,
                'avg_start_time': 0.0, 'races_count': 0, 'wins': 0,
                'places': 0, 'shows': 0
            }

    def update_racer_stats(self, racer_num):
        """
        選手統計データを更新
        
        Args:
            racer_num (str): 選手番号
        """
        if racer_num not in self.racer_stats or self.should_update_racer_stats(racer_num):
            print(f"選手 {racer_num} の統計データを更新中...")
            stats = self.get_racer_stats(racer_num)
            stats['last_updated'] = datetime.now().strftime('%Y%m%d')
            self.racer_stats[racer_num] = stats
            time.sleep(1)  # APIレート制限対策

    def should_update_racer_stats(self, racer_num):
        """
        選手統計データの更新が必要かどうか判定
        
        Args:
            racer_num (str): 選手番号
        
        Returns:
            bool: 更新が必要かどうか
        """
        if racer_num not in self.racer_stats:
            return True
        
        last_updated = self.racer_stats[racer_num].get('last_updated', '19700101')
        last_date = datetime.strptime(last_updated, '%Y%m%d')
        current_date = datetime.now()
        
        # 1週間以上経過していたら更新
        return (current_date - last_date).days > 7

    def save_racer_stats(self):
        """選手統計データをCSVファイルに保存"""
        if self.racer_stats:
            df_stats = pd.DataFrame.from_dict(self.racer_stats, orient='index')
            df_stats.index.name = 'racer_num'
            df_stats.to_csv(self.racer_stats_path)
            print(f"選手統計データを保存しました: {self.racer_stats_path}")

    def load_racer_stats(self):
        """選手統計データをCSVファイルから読み込み"""
        if os.path.exists(self.racer_stats_path):
            try:
                df_stats = pd.read_csv(self.racer_stats_path, index_col='racer_num')
                self.racer_stats = df_stats.to_dict(orient='index')
                print(f"選手統計データを読み込みました: {len(self.racer_stats)}人")
            except Exception as e:
                print(f"選手統計データ読み込みエラー: {e}")
                self.racer_stats = {}

    def collect_training_data(self, years=3, callback=None):
        """
        学習用データの収集（選手個人データを含む）
        
        Args:
            years (int): 収集する年数（デフォルト3年）
            callback (callable): 進捗報告用コールバック関数
        """
        training_data = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)
        
        total_days = (end_date - start_date).days
        processed_days = 0
        
        print(f"{years}年分のデータ収集を開始...")
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y%m%d')
            
            # 全競艇場のデータを収集
            for venue_code in range(1, 25):  # 1-24の競艇場
                # 各レース（1-12R）のデータを収集
                for race_num in range(1, 13):
                    race_data = self.get_race_data(venue_code, date_str, race_num)
                    if race_data and race_data['racers']:
                        # 各選手の統計データを更新
                        for racer in race_data['racers']:
                            self.update_racer_stats(racer['racer_num'])
                        
                        training_data.append(race_data)
                    
                    time.sleep(0.5)  # レート制限対策
            
            processed_days += 1
            progress = (processed_days / total_days) * 100
            
            if callback:
                callback(f"データ収集中... {processed_days}/{total_days}日 ({progress:.1f}%)")
            
            current_date += timedelta(days=1)
        
        # 選手統計データを保存
        self.save_racer_stats()
        
        print(f"データ収集完了: {len(training_data)}レース")
        return training_data

    def create_features(self, race_data):
        """
        選手個人データを含む特徴量の作成
        
        Args:
            race_data (dict): レースデータ
        
        Returns:
            list: 各艇の特徴量リスト
        """
        features_list = []
        
        for racer in race_data['racers']:
            racer_num = racer['racer_num']
            lane = racer['lane']
            
            # 選手統計データを取得
            stats = self.racer_stats.get(racer_num, {
                'win_rate': 0.0, 'quinella_rate': 0.0, 'trio_rate': 0.0,
                'avg_start_time': 0.0, 'races_count': 0, 'wins': 0,
                'places': 0, 'shows': 0
            })
            
            # 特徴量の作成
            features = [
                race_data['venue_code'],  # 競艇場コード
                lane,                     # 艇番
                stats['win_rate'],        # 1着率
                stats['quinella_rate'],   # 2連対率
                stats['trio_rate'],       # 3連対率
                stats['avg_start_time'],  # 平均スタートタイム
                stats['races_count'],     # 出走回数
                stats['wins'],            # 1着回数
                stats['places'],          # 2着回数
                stats['shows'],           # 3着回数
                # 艇番による傾向（インコース有利など）
                1 if lane == 1 else 0,   # 1号艇フラグ
                1 if lane == 2 else 0,   # 2号艇フラグ
                1 if lane <= 3 else 0,   # インコースフラグ（1-3号艇）
            ]
            
            features_list.append(features)
        
        return features_list

    def train_model(self, training_data, callback=None):
        """
        機械学習モデルの訓練（選手個人データを活用）
        
        Args:
            training_data (list): 学習用データ
            callback (callable): 進捗報告用コールバック関数
        """
        X = []  # 特徴量
        y = []  # ターゲット（着順）
        
        if callback:
            callback("特徴量作成中...")
        
        for i, race_data in enumerate(training_data):
            if 'racers' in race_data:
                features_list = self.create_features(race_data)
                
                for j, features in enumerate(features_list):
                    if j < len(race_data['racers']):
                        racer = race_data['racers'][j]
                        # 着順データが存在する場合のみ学習データに追加
                        if 'finish_position' in racer:
                            X.append(features)
                            # 1着を1、それ以外を0とする二値分類
                            y.append(1 if racer['finish_position'] == 1 else 0)
            
            if callback and i % 100 == 0:
                progress = (i / len(training_data)) * 100
                callback(f"特徴量作成中... {progress:.1f}%")
        
        if len(X) == 0:
            raise ValueError("学習データが不足しています")
        
        # データフレームに変換
        feature_names = [
            'venue_code', 'lane', 'win_rate', 'quinella_rate', 'trio_rate',
            'avg_start_time', 'races_count', 'wins', 'places', 'shows',
            'is_lane_1', 'is_lane_2', 'is_inner_course'
        ]
        
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y)
        
        if callback:
            callback("データの前処理中...")
        
        # 特徴量のスケーリング
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_df)
        
        # 訓練・テストデータの分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_series, test_size=0.2, random_state=42, stratify=y_series
        )
        
        if callback:
            callback("モデル訓練中...")
        
        # ランダムフォレストで学習
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # モデル評価
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"モデル精度: {accuracy:.4f}")
        print("\n分類レポート:")
        print(classification_report(y_test, y_pred))
        
        # 特徴量重要度の表示
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n特徴量重要度:")
        print(feature_importance)
        
        if callback:
            callback(f"訓練完了 - 精度: {accuracy:.4f}")

    def save_model_and_scaler(self):
        """学習済みモデルとスケーラーを保存"""
        if self.model and self.scaler:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print("モデルとスケーラーを保存しました")

    def load_model_and_scaler(self):
        """保存済みのモデルとスケーラーを読み込み"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("学習済みモデルとスケーラーを読み込みました")
                return True
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
        return False

    def predict_race(self, venue_code, date, race_num):
        """
        レース結果の予想（選手個人データを活用）
        
        Args:
            venue_code (int): 競艇場コード
            date (str): 日付 (YYYYMMDD形式)
            race_num (int): レース番号
        
        Returns:
            list: 予想結果（各艇の1着確率）
        """
        if not self.model or not self.scaler:
            return None
        
        # レースデータを取得
        race_data = self.get_race_data(venue_code, date, race_num)
        if not race_data:
            return None
        
        # 選手統計データを更新
        for racer in race_data['racers']:
            self.update_racer_stats(racer['racer_num'])
        
        # 特徴量を作成
        features_list = self.create_features(race_data)
        
        # 予想実行
        predictions = []
        for i, features in enumerate(features_list):
            # 特徴量をスケーリング
            features_scaled = self.scaler.transform([features])
            
            # 1着確率を予想
            win_probability = self.model.predict_proba(features_scaled)[0][1]
            
            racer_info = race_data['racers'][i]
            racer_stats = self.racer_stats.get(racer_info['racer_num'], {})
            
            prediction = {
                'lane': racer_info['lane'],
                'racer_num': racer_info['racer_num'],
                'racer_name': racer_info['racer_name'],
                'win_probability': win_probability,
                'win_rate': racer_stats.get('win_rate', 0.0),
                'quinella_rate': racer_stats.get('quinella_rate', 0.0),
                'trio_rate': racer_stats.get('trio_rate', 0.0)
            }
            
            predictions.append(prediction)
        
        # 1着確率でソート
        predictions.sort(key=lambda x: x['win_probability'], reverse=True)
        
        return predictions

class BoatRacePredictionGUI:
    def __init__(self):
        """GUI アプリケーションの初期化"""
        self.predictor = BoatRacePrediction()
        
        # メインウィンドウの作成
        self.root = tk.Tk()
        self.root.title("競艇予想システム（選手個人データ対応版）")
        self.root.geometry("800x700")
        
        self.create_widgets()
    
    def create_widgets(self):
        """GUI コンポーネントの作成"""
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # タイトル
        title_label = ttk.Label(main_frame, text="競艇予想システム（選手個人データ対応版）", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # データ収集セクション
        data_frame = ttk.LabelFrame(main_frame, text="データ収集・学習", padding="10")
        data_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 年数指定
        ttk.Label(data_frame, text="学習年数:").grid(row=0, column=0, sticky=tk.W)
        self.years_var = tk.StringVar(value="3")
        years_combo = ttk.Combobox(data_frame, textvariable=self.years_var, values=["1", "2", "3", "5"], width=10)
        years_combo.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        
        # データ収集・学習ボタン
        self.collect_button = ttk.Button(data_frame, text="データ収集・学習開始", 
                                        command=self.start_training)
        self.collect_button.grid(row=0, column=2, padx=(20, 0))
        
        # 進捗バー
        self.progress_var = tk.StringVar(value="待機中...")
        ttk.Label(data_frame, textvariable=self.progress_var).grid(row=1, column=0, columnspan=3, pady=(5, 0))
        
        self.progress_bar = ttk.Progressbar(data_frame, mode='indeterminate')
        self.progress_bar.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # 予想セクション
        predict_frame = ttk.LabelFrame(main_frame, text="レース予想", padding="10")
        predict_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 入力フィールド
        ttk.Label(predict_frame, text="競艇場:").grid(row=0, column=0, sticky=tk.W)
        self.venue_var = tk.StringVar()
        venue_combo = ttk.Combobox(predict_frame, textvariable=self.venue_var, 
                                  values=list(self.predictor.venue_mapping.values()), width=15)
        venue_combo.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        
        ttk.Label(predict_frame, text="日付:").grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        self.date_var = tk.StringVar(value=datetime.now().strftime('%Y%m%d'))
        date_entry = ttk.Entry(predict_frame, textvariable=self.date_var, width=10)
        date_entry.grid(row=0, column=3, sticky=tk.W, padx=(5, 0))
        
        ttk.Label(predict_frame, text="レース:").grid(row=1, column=0, sticky=tk.W)
        self.race_var = tk.StringVar(value="1")
        race_combo = ttk.Combobox(predict_frame, textvariable=self.race_var, 
                                 values=[str(i) for i in range(1, 13)], width=10)
        race_combo.grid(row=1, column=1, sticky=tk.W, padx=(5, 0))
        
        # 予想ボタン
        self.predict_button = ttk.Button(predict_frame, text="予想実行", command=self.predict_race)
        self.predict_button.grid(row=1, column=2, padx=(20, 0))
        
        # 結果表示エリア
        result_frame = ttk.LabelFrame(main_frame, text="予想結果", padding="10")
        result_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 結果テキストエリア
        self.result_text = scrolledtext.ScrolledText(result_frame, height=20, width=80)
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # グリッドの重み設定
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
    
    def start_training(self):
        """データ収集・学習の開始（バックグラウンド実行）"""
        def training_thread():
            try:
                self.collect_button.config(state='disabled')
                self.progress_bar.start()
                
                years = int(self.years_var.get())
                
                # データ収集
                def progress_callback(message):
                    self.progress_var.set(message)
                    self.root.update()
                
                training_data = self.predictor.collect_training_data(
                    years, progress_callback
                )
                
                # モデル学習
                progress_callback("モデル学習中...")
                self.predictor.train_model(training_data, progress_callback)
                
                # モデル保存
                self.predictor.save_model_and_scaler()
                
                progress_callback("学習完了！")
                messagebox.showinfo("完了", "データ収集・学習が完了しました！")
                
            except Exception as e:
                messagebox.showerror("エラー", f"学習中にエラーが発生しました: {str(e)}")
                print(f"学習エラー: {e}")
            finally:
                self.progress_bar.stop()
                self.collect_button.config(state='normal')
                self.progress_var.set("待機中...")
        
        # バックグラウンドで実行
        thread = threading.Thread(target=training_thread)
        thread.daemon = True
        thread.start()
    
    def predict_race(self):
        """レース予想の実行"""
        try:
            # 入力値の取得
            venue_name = self.venue_var.get()
            if not venue_name:
                messagebox.showwarning("入力エラー", "競艇場を選択してください")
                return
            
            # 競艇場名からコードを取得
            venue_code = None
            for code, name in self.predictor.venue_mapping.items():
                if name == venue_name:
                    venue_code = code
                    break
            
            if venue_code is None:
                messagebox.showerror("エラー", "競艇場コードが見つかりません")
                return
            
            date = self.date_var.get()
            race_num = int(self.race_var.get())
            
            # 日付形式チェック
            if len(date) != 8 or not date.isdigit():
                messagebox.showwarning("入力エラー", "日付はYYYYMMDD形式で入力してください")
                return
            
            # モデルの存在チェック
            if not self.predictor.model or not self.predictor.scaler:
                messagebox.showwarning("モデルエラー", "先にデータ収集・学習を実行してください")
                return
            
            # 予想実行
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "予想中...\n")
            self.root.update()
            
            predictions = self.predictor.predict_race(venue_code, date, race_num)
            
            if predictions is None:
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "レースデータが取得できませんでした。\n")
                self.result_text.insert(tk.END, "日付、競艇場、レース番号を確認してください。\n")
                return
            
            # 結果表示
            self.result_text.delete(1.0, tk.END)
            
            # ヘッダー情報
            self.result_text.insert(tk.END, f"=== {venue_name} {race_num}R ({date}) 予想結果 ===\n\n")
            
            # 予想順位表示
            self.result_text.insert(tk.END, "【予想順位】\n")
            self.result_text.insert(tk.END, "-" * 80 + "\n")
            self.result_text.insert(tk.END, f"{'順位':<4} {'艇番':<4} {'選手番号':<8} {'選手名':<12} {'1着確率':<8} {'勝率':<6} {'2連率':<6} {'3連率':<6}\n")
            self.result_text.insert(tk.END, "-" * 80 + "\n")
            
            for i, pred in enumerate(predictions, 1):
                self.result_text.insert(tk.END, 
                    f"{i:<4} {pred['lane']:<4} {pred['racer_num']:<8} {pred['racer_name']:<12} "
                    f"{pred['win_probability']:.1%:<8} {pred['win_rate']:<6.2f} "
                    f"{pred['quinella_rate']:<6.2f} {pred['trio_rate']:<6.2f}\n"
                )
            
            # 推奨買い目
            self.result_text.insert(tk.END, "\n" + "=" * 50 + "\n")
            self.result_text.insert(tk.END, "【推奨買い目】\n\n")
            
            # 単勝推奨
            top_pick = predictions[0]
            self.result_text.insert(tk.END, f"◆ 単勝本命: {top_pick['lane']}号艇 ({top_pick['racer_name']})\n")
            self.result_text.insert(tk.END, f"  └ 1着確率: {top_pick['win_probability']:.1%}\n\n")
            
            # 3連単推奨（上位3艇の組み合わせ）
            if len(predictions) >= 3:
                top3 = predictions[:3]
                self.result_text.insert(tk.END, "◆ 3連単推奨:\n")
                for i, first in enumerate(top3[:2]):  # 1着候補は上位2艇
                    remaining = [p for p in top3 if p != first]
                    for j, second in enumerate(remaining):
                        third_candidates = [p for p in remaining if p != second]
                        if third_candidates:
                            third = third_candidates[0]
                            confidence = first['win_probability'] * 0.7 + second['win_probability'] * 0.2 + third['win_probability'] * 0.1
                            self.result_text.insert(tk.END, 
                                f"  {first['lane']}-{second['lane']}-{third['lane']} (信頼度: {confidence:.1%})\n"
                            )
            
            # 選手個人データ詳細
            self.result_text.insert(tk.END, "\n" + "=" * 50 + "\n")
            self.result_text.insert(tk.END, "【選手詳細データ】\n\n")
            
            for pred in predictions:
                racer_stats = self.predictor.racer_stats.get(pred['racer_num'], {})
                self.result_text.insert(tk.END, f"▼ {pred['lane']}号艇: {pred['racer_name']} (選手番号: {pred['racer_num']})\n")
                self.result_text.insert(tk.END, f"  ・勝率: {pred['win_rate']:.2f}%\n")
                self.result_text.insert(tk.END, f"  ・2連対率: {pred['quinella_rate']:.2f}%\n")
                self.result_text.insert(tk.END, f"  ・3連対率: {pred['trio_rate']:.2f}%\n")
                self.result_text.insert(tk.END, f"  ・平均ST: {racer_stats.get('avg_start_time', 0.0):.2f}秒\n")
                self.result_text.insert(tk.END, f"  ・出走数: {racer_stats.get('races_count', 0)}回\n")
                self.result_text.insert(tk.END, f"  ・1着回数: {racer_stats.get('wins', 0)}回\n")
                self.result_text.insert(tk.END, f"  ・AI予想確率: {pred['win_probability']:.1%}\n\n")
            
            # 注意書き
            self.result_text.insert(tk.END, "=" * 50 + "\n")
            self.result_text.insert(tk.END, "※この予想は過去データに基づく統計的予測です。\n")
            self.result_text.insert(tk.END, "※実際の舟券購入は自己責任で行ってください。\n")
            self.result_text.insert(tk.END, "※選手の最新状況や展示情報も併せてご確認ください。\n")
            
        except Exception as e:
            messagebox.showerror("エラー", f"予想中にエラーが発生しました: {str(e)}")
            print(f"予想エラー: {e}")
    
    def run(self):
        """アプリケーションの実行"""
        # 初期状態の設定
        if self.predictor.model and self.predictor.scaler:
            self.progress_var.set("学習済みモデルが読み込まれています")
        else:
            self.progress_var.set("先にデータ収集・学習を実行してください")
        
        self.root.mainloop()

def main():
    """メイン関数"""
    print("競艇予想システム（選手個人データ対応版）を起動中...")
    print("=" * 50)
    print("【機能】")
    print("・選手個人の勝率、2連対率、3連対率などを特徴量に活用")
    print("・過去データに基づく機械学習予想")
    print("・選手統計データの自動取得・更新")
    print("・学習済みモデルの保存・読み込み")
    print("=" * 50)
    
    try:
        app = BoatRacePredictionGUI()
        app.run()
    except Exception as e:
        print(f"アプリケーション起動エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()