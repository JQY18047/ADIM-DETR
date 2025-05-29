import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR



if __name__ == '__main__':
    model = RTDETR(r'')
    model.val(data='./data.yaml',
              split='test',
              imgsz=640,
              batch=4,
              save_json=True,
              project='../runs/val',
              name='exp',
              )