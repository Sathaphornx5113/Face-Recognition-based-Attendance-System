import argparse
import time
from tkinter import image_names
from matplotlib import image
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
from xlutils.copy import copy
import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from pprint import pprint
from gspread_dataframe import set_with_dataframe
import time
import datetime
from googleapiclient.discovery import build
import os
import threading
import glob
import os

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

img_counter = 0
def predict():
    global img_counter
    img_name = "Attendance/Attendance-{}.jpg".format(img_counter)
    print(img_counter)
    print("Updating")
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument(
        '-i',
        '--image',
        default=img_name)   
        parser.add_argument(
        '-m',
        '--model_file',
        default='Model/face_recognition.tflite') 
        parser.add_argument(
        '-l',
        '--label_file',
        default='Model/Labels.txt')
        parser.add_argument(
        '--input_mean',
        default=127.5, type=float)
        parser.add_argument(
        '--input_std',
        default=127.5, type=float)
        parser.add_argument(
        '--num_threads', default=None, type=int)
        args = parser.parse_args()



    interpreter = tf.lite.Interpreter(
    model_path=args.model_file, num_threads=args.num_threads)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    floating_model = input_details[0]['dtype'] == np.float32
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    img = Image.open(args.image).resize((width, height))
  
    input_data = np.expand_dims(img, axis=0)
    print(input_data)
    if floating_model:
        input_data = (np.float32(input_data) - args.input_mean) / args.input_std
        interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    top_k = results.argsort()[-1:][::1]
    labels = load_labels(args.label_file)
    for i in top_k:

        print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
        scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
        cerds = ServiceAccountCredentials.from_json_keyfile_name("face-recognition.json", scope)
        client = gspread.authorize(cerds)
        sheet = client.open("face-recognition").worksheet('Sheet1') # เป็นการเปิดไปยังหน้าชีตนั้นๆ
        data = sheet.get_all_values()
        sheet.update_cell(len(data)+1,1,str(datetime.datetime.now()))  
        sheet.update_cell(len(data)+1,2,labels[i]) 
        
    print(img_name)
        
path_to_watch = "Attendance/"
print('Your folder path is"',path_to_watch,'"')
old = os.listdir(path_to_watch)
print(old)

while True:
    
    new = os.listdir(path_to_watch)
    if len(new) > len(old):
        newfile = list(set(new) - set(old))
        print(newfile[0])
        old = new
        extension = os.path.splitext(path_to_watch + "/" + newfile[0])[1]
        if extension == ".jpg":
            predict()
            img_counter += 1
        else:
            continue            
    else:
        continue