import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR



if __name__ == '__main__':
    model = RTDETR(r'D:\Users\hp\Desktop\资料工具\预训练权重\rtdetr-weights-new\rtdetr-r34.pt') # select your model.pt path
    model.predict(source=r'D:\Users\hp\Desktop\学业\淮大学习\第二学期\数据挖掘\图片\1ef895425a5f92c72076558ae992177b9af7ddbf.png',
                  conf=0.25,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  # visualize=True # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                  )