import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageTk
import colorsys

# YOLOモデルの読み込み
model = YOLO("runs/train/weights/best.pt")

# 画像の読み込み
img = cv2.imread("images/path_to_image.jpg")
original_img = img.copy()
results = model(img)

# 色相を指定した色に変更する関数
def update_hue(event):
    global img
    
    # マウスクリック位置を取得し、中心点（100, 100）からの相対位置を計算
    x, y = event.x - 100, event.y - 100
    
    # 相対位置から角度（色相環上の位置）を計算
    angle = (np.degrees(np.arctan2(-y, x)) + 360) % 360
    
    # 角度を色相値（0〜180）に変換
    target_hue = int(angle / 2)
    
    # 計算した色相値を保持
    selected_hue.set(target_hue)
    
    # 現在選択されているオブジェクトの名前を取得
    selected_obj = selected_object.get()
    
    # 選択されたオブジェクトの色相を辞書に保存
    if selected_obj in target_hues:
        target_hues[selected_obj] = target_hue
    
    # 画像を更新
    update_image()

# オブジェクトの情報を格納するリスト
objects = []

# 検出されたオブジェクトを処理
for result in results:
    mask_img = np.zeros(img.shape[:2], dtype=np.uint8)
    
    for mask, cls in zip(result.masks.xy, result.boxes.cls.cpu().numpy()):
        cv2.fillPoly(mask_img, [np.array(mask, dtype=np.int32)], 1)
        object_name = model.names[int(cls)]
        objects.append((object_name, mask_img.copy()))
        mask_img.fill(0)

# オブジェクトごとの色相シフト値を保持する辞書
target_hues = {object_name: None for object_name, _ in objects}

# オブジェクトの元の色相を保持する辞書
original_hues = {object_name: None for object_name, _ in objects}

# 初期色相を保存
for object_name, mask_img in objects:
    object_region = cv2.bitwise_and(original_img, original_img, mask=mask_img)
    hsv_image = cv2.cvtColor(object_region, cv2.COLOR_BGR2HSV)
    original_hue = np.median(hsv_image[:, :, 0][mask_img == 1])
    original_hues[object_name] = original_hue

root = tk.Tk()
root.title("色変え")

# 画像の表示
image_label = tk.Label(root)
image_label.grid(row=0, column=0, padx=10, pady=10)

def display_image(image):
    # BGR画像をRGBに変換して表示
    bgr_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(bgr_image)
    pil_image = pil_image.resize((400, 480))
    tk_image = ImageTk.PhotoImage(pil_image)
    image_label.config(image=tk_image)
    image_label.image = tk_image

display_image(img)

# 色相環の作成
canvas = tk.Canvas(root, width=200, height=200)
canvas.grid(row=0, column=1, padx=10, pady=10)

def create_color_wheel(canvas, radius=100):
    # 色相環を描画
    for angle in range(0, 360, 5):
        x0 = radius + radius * np.cos(np.radians(angle))
        y0 = radius - radius * np.sin(np.radians(angle))
        x1 = radius + radius * np.cos(np.radians(angle + 5))
        y1 = radius - radius * np.sin(np.radians(angle + 5))
        color = "#%02x%02x%02x" % tuple(int(c * 255) for c in colorsys.hsv_to_rgb(angle / 360, 1, 1))
        canvas.create_arc((0, 0, 2*radius, 2*radius), start=angle, extent=5, fill=color, outline=color)

create_color_wheel(canvas)

# 色相シフト値を保持
selected_hue = tk.IntVar(value=0)

# マウスイベントのバインディング
canvas.bind("<Button-1>", update_hue)
canvas.bind("<B1-Motion>", update_hue)
canvas.bind("<ButtonRelease-1>", lambda event: None)

# オブジェクト選択ボタンの作成
selected_object = tk.StringVar(value="")

def on_object_button_click(object_name):
    # オブジェクトボタンがクリックされた時の処理
    selected_object.set(object_name)
    update_image()

def reset_object_color(object_name):
    # オブジェクトの色を初期色に戻す処理
    if object_name in original_hues:
        target_hues[object_name] = original_hues[object_name]
    update_image()

# ボタンを配置するフレームの作成
button_frame = ttk.Frame(root)
button_frame.grid(row=1, column=0, columnspan=2, pady=10)

# 検出されたオブジェクトごとにボタンとリセットボタンを作成
for object_name, _ in objects:
    button = ttk.Button(button_frame, text=object_name, command=lambda name=object_name: on_object_button_click(name))
    button.pack(side=tk.LEFT, padx=5)
    reset_button = ttk.Button(button_frame, text=f"Reset {object_name}", command=lambda name=object_name: reset_object_color(name))
    reset_button.pack(side=tk.LEFT, padx=5)

# 画像を更新する関数
def update_image():
    global img
    img = original_img.copy()
    
    # 全てのオブジェクトに対して色相シフトを適用
    for object_name, mask_img in objects:
        if target_hues[object_name] is not None:
            object_region = cv2.bitwise_and(img, img, mask=mask_img)
            hsv_image = cv2.cvtColor(object_region, cv2.COLOR_BGR2HSV)
            hsv_image[:, :, 0] = target_hues[object_name]
            shifted_region = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
            img[mask_img == 1] = shifted_region[mask_img == 1]
    
    display_image(img)

root.mainloop()
