import torch, torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import matplotlib
import matplotlib.pyplot as plt

import os
import pathlib
import glob

import io
import scipy.misc
import numpy as np
import csv

num_classes = 2

# COCO classes
CLASSES = ['N/A', 'tomato_flower']

# colors for visualization
COLORS = [[0.698, 0.133, 0.133], #firebrick
          #[0.196, 0.804, 0.196], #lime-green
          #[0.576, 0.439, 0.859] #medium purple
]

#file path initialization
result_file_path = 'results'
out_file_name_yoko = 'result_yoko.csv'

# Create the folder if it doesn't exist
if not os.path.exists(result_file_path):
    os.makedirs(result_file_path)

# Create the files if they don't exist
if not os.path.exists(os.path.join(result_file_path, out_file_name_yoko)):
    with open(os.path.join(result_file_path, out_file_name_yoko), 'w'):
        pass

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def filter_bboxes_from_outputs(outputs, size, threshold=0.7):

  # keep only predictions with confidence above threshold
  probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
  keep = probas.max(-1).values > threshold

  probas_to_keep = probas[keep]

  # convert boxes from [0; 1] to image scales
  bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], size)

  return probas_to_keep, bboxes_scaled

#モデルの指定
model = torch.hub.load('facebookresearch/detr',
                       'detr_resnet50',
                       pretrained=False,
                       num_classes=num_classes)

checkpoint = torch.load('checkpoint0199.pth',
                        map_location='cpu')

model.load_state_dict(checkpoint['model'],
                      strict=False)

model.eval()

def plot_finetuned_results(pil_img, prob=None, boxes=None):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    coord_ave = []
    colors = COLORS * 100
    if prob is not None and boxes is not None:
      for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
          ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=2))
          x_ave = (xmin + xmax)  * 0.5
          y_ave = (ymin + ymax)  * 0.5
          coord_ave.append([x_ave, y_ave])

    plt.axis('off')
    plt.show()
    plt.close()

    return pil_img, coord_ave

#メインコード
def run_worflow(my_image, my_model, frame_num):
  # mean-std normalize the input image (batch-size: 1)
  img = transform(my_image).unsqueeze(0)
  print(f'frame_num: {frame_num}')

  # propagate through the model
  outputs = my_model(img)

  #しきい値
  for threshold in [0.9]:

    probas_to_keep, bboxes_scaled = filter_bboxes_from_outputs(outputs, my_image.size, threshold=threshold)

    image_with_boxes, coord = plot_finetuned_results(my_image, probas_to_keep, bboxes_scaled)

     # 必要であれば Save the frame as an image
    #image_with_boxes.save(f'frame_{currentFrame}.png')  # Save the frame with a unique name

    with open(result_file_path + '/' + out_file_name_yoko , 'a',newline='') as f:
      writer = csv.writer(f)
      # Write each coordinate pair (x, y) to separate columns in the same row
      row_data = []
      row_data.append(frame_num)
      for coordinate in coord:
        row_data.append(coordinate)

      writer.writerow(row_data)

#動画入力から座標を抽出する
# choose the file:
cap = cv2.VideoCapture('cut_20221014_yoko_1.mp4')

# returns the frame rate
fps = cap.get(cv2.CAP_PROP_FPS)
print("Frame rate: ", int(fps), "FPS")

currentFrame = 0
while(True):
    # Capture 50 frame-by-frame
    for i in range(3):
      ret, frame = cap.read()

      if not ret: break

    #to check frame data type
    #print(type(frame))

    if not ret: break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    frame_image = Image.fromarray(frame)

    # 座標をresult.csvファイルに記入する
    run_worflow(frame_image ,model, currentFrame)

    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()