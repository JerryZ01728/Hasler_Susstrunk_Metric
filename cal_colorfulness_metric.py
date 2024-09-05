import os
import csv
import cv2
import numpy as np

def colorfulness_metric(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("图像路径无效或图像无法读取")
    
    image = image.astype("float")
    (B, G, R) = cv2.split(image)
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    
    mean_rg = np.mean(rg)
    std_rg = np.std(rg)
    mean_yb = np.mean(yb)
    std_yb = np.std(yb)

    colorfulness = np.sqrt((std_rg ** 2) + (std_yb ** 2)) + 0.3 * np.sqrt((mean_rg ** 2) + (mean_yb ** 2))
    return colorfulness

# image_path = '/home/zhangruijie03/ICAQA/ICAA17K_code/Hot_video_20240801_kvq3.5_Frame_original'
# colorfulness_value = colorfulness_metric(image_path)
# print(f"Colorfulness Metric: {colorfulness_value}")

def cal_colorfulness(folder_path, output_csv):
    image_extensions = ['.jpg', '.jpeg', 'png', 'bmp', 'gif']
    image_data = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(root, file)
                colorfulness = colorfulness_metric(image_path)
                image_data.append([file, colorfulness])
    
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['image', 'Colorfulness_metric'])
        csvwriter.writerows(image_data)

if __name__ == "__main__":
    folder_path = '/home/zhangruijie03/Video_download/chosen_hotvideo_frame'
    output_csv = 'chosen_hotvideo_frame.csv'
    cal_colorfulness(folder_path, output_csv)