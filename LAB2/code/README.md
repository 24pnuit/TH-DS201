# DS201 – Lab 2: CNN trên VinaFood21

## Cấu trúc

- `models.py`: LeNet-5, GoogLeNet rút gọn, ResNet-18 và ResNet-50 pretrained.
- `data.py`: đọc VinaFood21, chia train/validation và tạo DataLoader.
- `engine.py`: hàm huấn luyện và đánh giá dùng chung.
- `run.py`: huấn luyện, chọn checkpoint theo macro-F1 validation và đánh giá test một lần.
- `check_shapes.py`: kiểm tra số tham số và kích thước tensor.
- `models/`: các notebook cho từng bài.

## Cài thư viện

```bash
pip install torch torchvision transformers scikit-learn matplotlib numpy pillow
```

## Dữ liệu

Đặt thư mục `VinaFood21` cạnh các file Python, có cấu trúc:

```text
VinaFood21/
  train/<tên lớp>/*.jpg
  test/<tên lớp>/*.jpg
```

Script lấy 10% tập `train` làm validation theo seed 42; tập `test` giữ riêng để đánh giá cuối cùng. Nếu dữ liệu nằm nơi khác, truyền `--data_root`.

## Kiểm tra mô hình

```bash
python check_shapes.py
```

## Chạy huấn luyện

Chạy từ thư mục chứa `run.py`:

```bash
python run.py --model lenet_avg --epochs 30
python run.py --model lenet_max --epochs 30
python run.py --model googlenet --epochs 30
python run.py --model resnet18 --epochs 30 --tag aug
python run.py --model resnet18 --epochs 30 --tag noaug --no_aug
python run.py --model resnet50_fe --epochs 10
python run.py --model resnet50_ft --epochs 10
```

Ví dụ nếu dữ liệu ở nơi khác:

```bash
python run.py --model resnet18 --data_root "D:/duong_dan/VinaFood21" --epochs 30
```

Kết quả được lưu trong `results/<tên_mô_hình>/`: `metrics.json`, `history.csv`, `per_class.csv`, `confusion.npy`, `confusion_matrix.png`, `loss_curve.png` và `best.pt`. Chỉ dùng validation để chọn checkpoint; tập test chỉ được đánh giá sau khi huấn luyện xong.

ResNet-50 cần tải trọng số pretrained từ Hugging Face ở lần chạy đầu tiên và cần kết nối mạng để tải model.
