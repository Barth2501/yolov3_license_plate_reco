import os


image_names = os.listdir('./train')
image_names = sorted(image_names) # to make the results reproductibles
for i,image in enumerate(image_names):
    os.rename(os.path.join('./train', image), './train/truck_plate_' + str(i) + '.' + image.split('.')[1])