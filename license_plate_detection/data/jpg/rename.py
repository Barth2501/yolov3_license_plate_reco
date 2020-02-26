import os


image_names = os.listdir('./valid')
image_names = sorted(image_names) # to make the results reproductibles
for i,image in enumerate(image_names):
    os.rename(os.path.join('./valid', image), './valid/valid' + str(i) + '.' + image.split('.')[1])