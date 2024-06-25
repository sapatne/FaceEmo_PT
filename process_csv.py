import numpy as np
from PIL import Image
import os
import pandas as pd
import csv


image_size = 48
def str_to_int_list(string):
    str_list = string.split(' ')
    int_list = list(map(int, str_list))
    return int_list

def list_to_img(pixels):
    return np.asarray(pixels, dtype=np.uint8).reshape(48, 48)

def read_csv(file_path):
	data = pd.read_csv(file_path)

	# change pixels values from single string to int array
	data['pixels'] = data['pixels'].apply(str_to_int_list)
	data['pixels'] = data['pixels'].apply(list_to_img)

	# change Usage strings to "Testing" and "Validation"
	data.loc[data['Usage']=='PublicTest', 'Usage'] = 'Testing'
	data.loc[data['Usage']=='PrivateTest', 'Usage'] = 'Validation'

	return data

def create_images(csv_data):
	new_csv_data = [["Folder", "Path", "Label"]]
	for i in range(len(csv_data)):
		sample = csv_data.iloc[i]
		emotion = sample.emotion
		pixels = sample.pixels
		usage = sample.Usage
		folder_name = f'./data/{usage}/'

		os.makedirs(folder_name, exist_ok=True)
		fname = f'image_{i}_{emotion}.png'
		f_path = os.path.join(folder_name, fname)
		save_image(pixels, f_path)
		new_csv_data.append([usage, fname, emotion])

	with open(r'data/annotations.csv', mode='w', newline='') as file:
		writer = csv.writer(file)
		writer.writerows(new_csv_data)
		

def save_image(img_data, file_path):
	img = Image.fromarray(img_data, 'L')
	img.save(file_path)
	print(f'Saved image: {file_path}')

def main():
	csv_path = r'./fer2013/fer2013.csv'
	all_data = read_csv(csv_path)
	create_images(all_data)

if __name__ == "__main__":
	main()