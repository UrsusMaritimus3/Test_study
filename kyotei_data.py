import tkinter as tk
from tkinter import ttk

def parse_line(line, byte_split):
    line_bytes = line.encode("shift-jis")
    idx = 0
    values = []
    for b in byte_split:
        part = line_bytes[idx:idx+b]
        values.append(part.decode('shift-jis', errors='ignore'))#.replace('　', '').replace(' ', ''))
        idx += b
    return values

def convert_sex(val):
    return "男" if val == "1" else "女" if val == "2" else val

def convert_birth(year_str, birth_str):
    try:
        era = year_str[0]
        birth = birth_str
        if era == 'S':  # 昭和
            year = 1925 + int(birth[:2])
        elif era == 'H':  # 平成
            year = 1988 + int(birth[:2])
        else:
            return birth_str
        return f"{year}年{birth[2:4]}月{birth[4:6]}日"
    except:
        return birth_str

def format_count(val):
    # 例: "001" → "1回"
    try:
        num = int(val)
        return f"{num}回"
    except:
        return val

def format_win_rate(val):
    # 勝率: 二桁目と三桁目の間に小数点
    if len(val) >= 3:
        if(val[0] == '0'):
            return f"{val[1:2]}.{val[2:]}%"
        return f"{val[:2]}.{val[2:]}"
    return val

def format_fukusho_rate(val):
    # 複勝率: 三桁目と四桁目の間に小数点
    if len(val) >= 4:
        if(val[0] == '0'):
            if(val[1] == '0'):
                return f"{val[2:3]}.{val[3:]}%"  
            return f"{val[1:3]}.{val[3:]}%" 
        return f"{val[:3]}.{val[3:]}"  
    return val

def format_average(val):
    # 平均スタートタイミング・平均スタート順位: 一桁目と二桁目の間に小数点
    if len(val) >= 2:
        return f"{val[0]}.{val[1:]}"
    return val

def format_ability(val):
    # 能力指数: 二桁目と三桁目の間に小数点
    if len(val) >= 3:
        return f"{val[:2]}.{val[2:]}"
    return val

with open("./kyotei_data/fan2410.txt", "r", encoding="shift-jis") as f:
    data = f.read()
    lines = data.split('\n')  # 改行で分割してリストに格納
    byte_split = [4,16,15,4,2,1,6,1,2,3,2,2,4,4,3,3,3,2,2,3,3,4,3,3,3,4,3,3,3,4,3,3,3,4,3,3,3,4,3,3,3,4,3,3,2,2,2,4,4,4,1,8,8,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,3,3,3,3,3,3,2,2,2,2,2,2,2,2,3,3,3,3,3,3,2,2,2,2,2,2,2,2,3,3,3,3,3,3,2,2,2,2,2,2,2,2,3,3,3,3,3,3,2,2,2,2,2,2,2,2,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,6]
    koumoku_name = ["登録番号","選手名(漢字)","選手名(カタカナ)","支部","級","年号","生年月日","性別","年齢","身長","体重","血液型","勝率","複勝率","一着回数","二着回数","出走回数","優出回数","優勝回数","平均スタートタイミング",
                   "1コース進入回数","1コース複勝率","1コース平均スタートタイミング","1コース平均スタート順位",
                   "2コース進入回数","2コース複勝率","2コース平均スタートタイミング","2コース平均スタート順位",
                   "3コース進入回数","3コース複勝率","3コース平均スタートタイミング","3コース平均スタート順位",
                   "4コース進入回数","4コース複勝率","4コース平均スタートタイミング","4コース平均スタート順位",
                   "5コース進入回数","5コース複勝率","5コース平均スタートタイミング","5コース平均スタート順位",
                   "6コース進入回数","6コース複勝率","6コース平均スタートタイミング","6コース平均スタート順位",
                   "前期級","前々期級","前々前期級","前期能力指数","今期能力指数",
                   "年","季","算出期間(自)","算出期間(至)","養成期",
                   "1コース1着回数", "1コース2着回数", "1コース3着回数", "1コース4着回数", "1コース5着回数", "1コース6着回数", 
                   "1コースF回数", "1コースL0回数", "1コースL1回数", "1コースK0回数", "1コースK1回数", 
                   "1コースS0回数", "1コースS1回数", "1コースS2回数",
                   "2コース1着回数", "2コース2着回数", "2コース3着回数", "2コース4着回数", "2コース5着回数", "2コース6着回数", 
                   "2コースF回数", "2コースL0回数", "2コースL1回数", "2コースK0回数", "2コースK1回数",
                   "2コースS0回数", "2コースS1回数", "2コースS2回数",
                   "3コース1着回数", "3コース2着回数", "3コース3着回数", "3コース4着回数", "3コース5着回数", "3コース6着回数",
                   "3コースF回数", "3コースL0回数", "3コースL1回数", "3コースK0回数", "3コースK1回数",
                   "3コースS0回数", "3コースS1回数","3コースS2回数",
                   "4コース1着回数", "4コース2着回数", "4コース3着回数", "4コース4着回数", "4コース5着回数", "4コース6着回数",
                   "4コースF回数", "4コースL0回数", "4コースL1回数", "4コースK0回数", "4コースK1回数",
                   "4コースS0回数", "4コースS1回数","4コースS2回数",
                   "5コース1着回数", "5コース2着回数", "5コース3着回数", "5コース4着回数", "5コース5着回数", "5コース6着回数",
                   "5コースF回数", "5コースL0回数", "5コースL1回数", "5コースK0回数", "5コースK1回数",
                   "5コースS0回数", "5コースS1回数","5コースS2回数",
                   "6コース1着回数", "6コース2着回数", "6コース3着回数", "6コース4着回数", "6コース5着回数", "6コース6着回数",
                   "6コースF回数", "6コースL0回数", "6コースL1回数", "6コースK0回数", "6コースK1回数",
                   "6コースS0回数", "6コースS1回数","6コースS2回数",
                   "コースなしL0回数", "コースなしL1回数", "コースなしK0回数", "コースなしK1回数","出身地"]

# 選手名リスト作成
player_list = []
player_values = []
for line in lines:
    values = parse_line(line, byte_split)
    if len(values) > 1:
        player_list.append(values[1].replace('　', '').replace(' ', ''))
        player_values.append(values)

root = tk.Tk()
root.title("競艇データ表示")

# 検索欄
search_var = tk.StringVar()
search_entry = tk.Entry(root, textvariable=search_var)
search_entry.pack(fill=tk.X, padx=5, pady=5)

# 左側：選手リスト
list_frame = tk.Frame(root)
list_frame.pack(side=tk.LEFT, fill=tk.Y)

listbox = tk.Listbox(list_frame, width=20)
listbox.pack(side=tk.LEFT, fill=tk.Y)

scrollbar_list = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=listbox.yview)
scrollbar_list.pack(side=tk.RIGHT, fill=tk.Y)
listbox.config(yscrollcommand=scrollbar_list.set)

def update_listbox():
    listbox.delete(0, tk.END)
    keyword = search_var.get()#.replace('　', '').replace(' ', '')
    for i, name in enumerate(player_list):
        if keyword in name:
            listbox.insert(tk.END, name)

update_listbox()

def on_search(*args):
    update_listbox()
search_var.trace_add('write', on_search)

# 右側：選手データ（表形式＋スクロール）
data_frame = tk.Frame(root)
data_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

canvas = tk.Canvas(data_frame)
scrollbar = tk.Scrollbar(data_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
canvas.configure(yscrollcommand=scrollbar.set)

info_frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=info_frame, anchor='nw')

def show_player_data(event):
    global info_frame
    info_frame.destroy()
    info_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=info_frame, anchor='nw')
    idx = listbox.curselection()
    if not idx:
        return
    # 検索結果に合わせてインデックスを取得
    keyword = search_var.get()#.replace('　', '').replace(' ', '')
    filtered_indices = [i for i, name in enumerate(player_list) if keyword in name]
    player_idx = filtered_indices[idx[0]]
    values = player_values[player_idx]

    # 基本情報
    info = [
        ("登録番号", values[0]),
        ("選手名", values[1].replace('　', '').replace(' ', '')),
        ("選手名(カタカナ)", values[2].replace('　', '').replace(' ', '')),
        ("支部", values[3]),
        ("級", values[4]),
        ("性別", convert_sex(values[7])),
        ("生年月日", convert_birth(values[5], values[6])),
        ("年齢", f"{values[8]}歳"),
        ("身長", f"{values[9]}cm"),
        ("体重", f"{values[10]}kg"),
        ("血液型", values[11]),
        ("勝率", format_win_rate(values[12])),
        ("複勝率", format_fukusho_rate(values[13])),
        ("一着回数", format_count(values[14])),
        ("二着回数", format_count(values[15])),
        ("出走回数", format_count(values[16])),
        ("優出回数", format_count(values[17])),
        ("優勝回数", format_count(values[18])),
        ("平均スタートタイミング", format_average(values[19])),
        ("前期級", values[44]),
        ("前々期級", values[45]),
        ("前々前期級", values[46]),
        ("前期能力指数", format_ability(values[47])),
        ("今期能力指数", format_ability(values[48])),
        ("年", values[49]),
        ("季", values[50]),
        ("算出期間(自)", values[51]),
        ("算出期間(至)", values[52]),
        ("養成期", values[53]),
    ]

    for label, val in info:
        tk.Label(info_frame, text=f"{label}: {val}", anchor="w", font=("Arial", 10, "bold")).pack(fill=tk.X)

    columns = ["コース", "進入回数", "複勝率", "平均ST", "平均順位", "1着", "2着", "3着", "4着", "5着", "6着", "F", "L0", "L1", "K0", "K1", "S0", "S1", "S2"]
    course_data = []
    base_idx = 20
    course_names = ["1コース", "2コース", "3コース", "4コース", "5コース", "6コース"]
    for c in range(6):
        row = [
            course_names[c],
            format_count(values[base_idx + c*4]),     # 進入回数
            format_fukusho_rate(values[base_idx + c*4 + 1]), # 複勝率
            format_average(values[base_idx + c*4 + 2]),      # 平均ST
            format_average(values[base_idx + c*4 + 3]),      # 平均順位
        ]
        offset = 54 + c*14
        # 着順・F・L・K・S（回数項目は「回」付きに変換）
        row += [format_count(val) if "回数" in koumoku_name[54 + c*14 + i] else val for i, val in enumerate(values[offset:offset+14])]
        # S0,S1,S2はそのまま
        row += values[offset+14:offset+17]
        course_data.append(row)

    tree = ttk.Treeview(info_frame, columns=columns, show="headings", height=7)
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=60, anchor="center")
    for row in course_data:
        tree.insert("", "end", values=row)
    tree.pack(fill=tk.X, pady=10)

    info_frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

listbox.bind('<<ListboxSelect>>', show_player_data)

# 初期表示
if player_list:
    listbox.selection_set(0)
    show_player_data(None)

root.geometry("800x600")
root.mainloop()