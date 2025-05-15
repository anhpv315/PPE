from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load model
model = YOLO("runs/detect/train/weights/best.pt")


label_map = {
    "person": "Người",
    "ear": "Tai",
    "ear-mufs": "Chụp tai chống ồn",
    "face": "Khuôn mặt",
    "face-guard": "Tấm chắn mặt",
    "face-mask": "Khẩu trang",
    "foot": "Bàn chân",
    "tool": "Dụng cụ",
    "glasses": "Kính",
    "gloves": "Găng tay",
    "helmet": "Mũ bảo hộ",
    "hands": "Bàn tay",
    "head": "Đầu",
    "medical-suit": "Đồ bảo hộ y tế",
    "shoes": "Giày bảo hộ",
    "safety-suit": "Đồ bảo hộ",
    "safety-vest": "Áo phản quang"
}

excluded_labels = {"person", "ear", "face", "foot", "hands", "head"}


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        if image:
            # Lưu ảnh gốc
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)

            # Chạy mô hình
            results = model.predict(source=image_path, save=True, conf=0.25)

            # Lưu ảnh kết quả
            result_image_path = os.path.join(app.config['OUTPUT_FOLDER'], image.filename)
            results[0].save(result_image_path)

            # Lấy các nhãn tiếng Anh từ kết quả mô hình
            english_labels = [model.names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]

            # Lọc và ánh xạ sang tiếng Việt nếu không thuộc nhóm loại trừ
            detected_classes = [
                label_map[label.lower()]
                for label in english_labels
                if label.lower() not in excluded_labels
            ]

            return render_template('index.html',
                                   result_img=url_for('static', filename='outputs/' + image.filename),
                                   detected=detected_classes)

    return render_template('index.html', result_img=None, detected=None)

if __name__ == '__main__':
    app.run(debug=True)
