import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO(r'')
    model.val(data='./data.yaml',
              split='test',
              imgsz=640,
              batch=4,
              save_json=True, # if you need to cal coco metrice
              project='../runs/val',
              name='exp',
              )