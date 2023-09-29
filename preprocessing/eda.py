from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Data directory
data_dir = '../datasets/PlantVillage/Plant_leave_diseases_dataset_without_augmentation/all_data'

# List of all diseases
diseases = os.listdir(data_dir)
print(diseases)

# Number of different diseases
print("Number of total disease classes: {}".format(len(diseases)))

# Extract unique plants in the dataset
plants = []
background = []
for disease in diseases:
    plant = disease.split('__')[0]
    if plant not in plants and plant != 'Background_without_leaves':
        plants.append(plant)

    if plant == 'Background_without_leaves':
        background.append(plant)

print(f"Unique plants are: \n{plants}")
print("Background is: {}".format(background))

# Make dictionary for every plant's disease
plant_diseases_dict = {}
for plant in plants:
    diseases_list = []
    for disease in diseases:
        if disease.split("__")[0] == plant:
            diseases_list.append(disease.split("__")[1][1:])
    
    plant_diseases_dict[plant] = diseases_list

# Count how many diseases are there for each plant
for key in plant_diseases_dict.keys():
    count = 0
    for dis in plant_diseases_dict[key]:
        if dis != 'healthy':
            count += 1
    plant_diseases_dict[key].append(count)

# Plot the number of diseases per plant
x_axis = list(plant_diseases_dict.keys())
y_axis = list(plant_diseases_dict[key][-1] for key in plant_diseases_dict.keys())

plt.bar(x_axis, y_axis)
plt.xlabel('Plant')
plt.ylabel('Number of diseases')
plt.title('Number of diseases per plant')
plt.xticks(range(len(x_axis)), x_axis, rotation=90)
 
plt.savefig("./graphs/num_dis_per_plant.jpg")

# Number of images per disease
nums = {}
for disease in diseases:
    nums[disease] = len(os.listdir(data_dir + disease))
    
# Converting the nums dictionary to pandas dataframe passing index as plant name and number of images as column
img_per_class = pd.DataFrame(nums.values(), index=nums.keys(), columns=["number of images"])

print(img_per_class)

# Plotting number of images available for each disease
index = [n for n in range(len(img_per_class))]
plt.figure(figsize=(20, 10))
plt.bar(index, [n for n in nums.values()], width=0.3)
plt.xlabel('Plants/Diseases', fontsize=10)
plt.ylabel('Number of images per class', fontsize=10)
plt.xticks(index, diseases, fontsize=5, rotation=90)
plt.title('Images per each class of plant disease')

plt.savefig("./graphs/num_img_per_dis.jpg")









