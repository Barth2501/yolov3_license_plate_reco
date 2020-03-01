Plate Recognition using Faster R-CNN or SSD
============================================
 
This program detects vehicle license plates in images, with either a single detection stage or a double detection stage.  
The models used are either ssd or faster_rcnn depending on the choice made.
The single stage detector, detects plates and plate characters in a single inference stage.  
The double stage detector detects plates in the first inference stage, 
crops the detected plate from the image, passes the cropped plate image to the second inference stage, which detects plate characters.  

The double stage detector uses a single detection model that has been trained to detect plates in full images containing cars/plates, 
and trained to detect plate text in images containing tightly cropped plate images.

##### How the model was trained
 
starting point : ssd_inception_v2_coco_2018_01_28, and faster_rcnn_resnet101_coco_2018_01_28.
Trained after on a large labelled plate image dataset.

##### Directory layout

Images to be analysed must be placed into the image folder. 
ssd and faster_rcnn contain the detection and classification models.
classes contains a simple classes txt file.
Plate functions contains classes used for plate detection.
```
plate detection 
│   │─  classes
│   │─ image
│   │─ faster_rcnn
│   │─  ssd  
│   │─  platefunctions
````
How tu use the trained model
-------------------------
##### predict_images.py

predict_images.py can be runned to analyse and detect plates on the image dataset situated in images. You can chose to use either an ssd or a faster R-CNN for the detection and it Works with either single and double stage prediction.
It prints the detected plate text, and displays the annotated image if being asked.  

Syntax
model: faster_rcnn/frozen_inference_graph.pb or ssd/frozen_inference_graph.pb
pred_stages : either 1 or 2
image_display : boolean

````
python predict_images.py --model faster_rcnn/frozen_inference_graph.pb  --pred_stages 1  --image_display True	
````
Just above, we run a test with a faster_rcnn model, with a single stage prediction and with a display of the annoted resulting image.

````
python predict_images.py --model ssd/frozen_inference_graph.pb  --pred_stages 2  --image_display False	
````
Here, we use an ssd model with a two stage prediction and without any image display.

Your results should look something like this:
````
[INFO] Loading image "image\test.jpg"
    Found:  24g88
[INFO] Loading image "image\test_4.jpg"
    Found:  ex270lae
[INFO] Processed 2 frames in 0.93 seconds. Frame rate: 0.465 Hz
````