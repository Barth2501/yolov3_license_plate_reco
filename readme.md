# Military Truck License Plate detection

This project is to be used for the french army in order to detect a license plate in an image and recognize its license plate.

## How to test this algo

You can type these lines of code to have an exmaple of how the all model works.

Download the weight of the pre trained yolo model:
```
$ cd license_plate_detection/datasets/yolo/
$ wget https://pjreddie.com/media/files/yolov3.weights -O yolov3.weights
$ python convert.py
```

Fine tune the detection model with the current tfrecord files:
```
$ cd ..
$ cd ..
$ cd ..
$ python main.py --mode yolo_fine_tune
```

Apply the trained model to a random image:
```
$ python main.py --mode predict --image ./license_plate_detection/data/jpg/test/truck_plate_3.jpg
```

N.B.: you can also use the other mode like this:

Use the trained model to predict a batch of cropped license plate:
```
$ python main.py --mode yolo_batch_detect
```

Fine tune the recognition model with the current tfrecord file:
```
$ python main.py --mode cnn_fine_tune
```



## Pipeline of the all model

This model has three different steps : 
                            - the license plate location detection inside the image
                            - the character segmentation of the license plate
                            - the character identification

### License plate detection

This step is ensure by a YOLOv3 model, which is pre-trained with the COCO dataset and fine tuned with labelised images of military truck found on the internet.

On this model you can or :
        - change the pre-train weights of the yolo model:

            ```
            $ cd license_plate_detection/datasets/yolo
            $ wget https://pjreddie.com/media/files/yolov3.weights -O yolov3.weights
            $ python convert.py
            ```

            The file as about 250MB so it can take some time to download, and the convert.py file convert the weights into a tf file ready to use.

        - fine tune the yolo model:

            ```
            $ python main.py --mode yolo_fine_tune
            ```

            This command requires that you saved in the right folder the tfrecord file of your labelisation (see the labelisation section if you are not aware of this part). You can specify on other location with the FLAGS of the main.py file.

            So if --mode yolo_fine_tune, you have:
            
            --yolo_train_dataset: path to the train tfrecord file
                (default: './license_plate_detection/data/tfrecord/train_license_plate.tfrecord')
            --yolo_val_dataset: path to the valid tfrecord file
                (default: './license_plate_detection/data/tfrecord/val_license_plate.tfrecord')
            --yolo_pretrained_weights: path to weights file
                (default: './license_plate_detection/datasets/yolo/yolov3.tf')
            --classes: path to classes file
                (default: './license_plate_detection/data/license.plate')
            --yolo_size: resize images to (default: 416)
            --num_classes: number of classes in the model (default: 80)
            --yolo_batch_size: yolo model batch size (default: 8)
            --yolo_learning_rate: learning rate of yolo model (default: 1e-3)
            --yolo_epochs: number of epochs of yolo model is (default: 8)
            

        - batch detect license plate for the next step of the all model:

            ```
            $ python main.py --mode yolo_batch_detect
            ```

            This command will detect the license plate on images from a folder and will save the output on a predefine location. You can change the default settings using FLAGS.
            
            if --mode yolo_batch_detect, you have:
            
            --classes: path to classes file
                (default: './license_plate_detection/data/license.plate')
            --yolo_weights: path to weights file
                (default: './license_plate_detection/checkpoints/yolov3_train_8.tf')       
            --yolo_output: path to output image
                (default: './license_plate_detection/outputs/global_outputs/')
            --yolo_cropped_output: path to output cropped image
                (default: './license_plate_detection/outputs/cropped_outputs/')
            --input_images: path to input images folder
                (default: './license_plate_detection/data/jpg/test/')
            --yolo_size: resize images to (default: 416)
            --num_classes: number of classes in the model (default: 80)
            
            
### Character segmentation

This step is used to separate each character of the license plate. It uses computer vision aspect but still suffers some issues. Next effort will be concentrate at this step to make it more adaptable to any license plate luminosity and shape.

This can be done using the mode segment_only of the main.py file.
```
$ python main.py --mode segment_only
```

The FLAGS for this function are :
            
        --yolo_cropped_output: path to output cropped image
                (default: './license_plate_detection/outputs/cropped_outputs/')
        --seg_output: dir where you want to save the segmented characters
                (default: './character_reco/data/jpg')
        --image: Image you want to predict, to use only if batch detection is disabled 
                (No default value)
        --batch_detection: False 
                (default) if you want to detect only one image

### Character identification

This final step is for the identification character by character. The model I used is a CNN model with two convolutionnal layers. Once again the model is pre-trained on the mnist dataset and then we fine tune it with the labelised character we got from the segmentation phase.

So the mode you can use here is cnn_fine_tune, it allows you to perform the fine tunning of the last layers (second cnn and dense layers). 
```
$ python main.py --mode cnn_fine_tune
```

N.B.: Before training the cnn, you must have created a tfrecord file of the labelised data you had (See the labelisation section)

The FLAGS you can use for this mode are :
            
       --cnn_train_dataset: Path to the train tfrecord dataset
                (default: './character_reco/data/tfrecord/train_character.tfrecord')
       --cnn_val_dataset: Path to the val tfrecord dataset
                (default: './character_reco/data/tfrecord/val_character.tfrecord')
       --cnn_size: The size we want to resize the number 
                (default: 28)
       --cnn_batch_size: Size of the batch 
                (default: 1)
       --cnn_learning_rate: Learning rate 
                (default: 0.001)

## Labelisation

In order to fine tune the models (yolo or cnn), you will have to build tfrecord files of the labelised data.

Personnally, I used labelImg application to perform that, it creates XML file in the folder I selected.
Then I create a csv file form the XML ones and then a tfrecord, you should use this for example with the labelised images of license plate detection:

```
$ cd labelisation
$ python xml_to_csv.py --img_path ./license_plate_detection/data/xml/train/ --save_name ./license_plate_detection/data/csv/train_plate.csv
$ python generate_tfrecord --csv_input ./license_plate_detection/data/csv/train_plate.csv --output_path ./license_plate_detection/data/tfrecord/train_plate.tfrecord --image_dir ./license_plate_detection/data/jpg/train
```


### Prediction

In order to detect where the license plate is on the image, you can use the following flags:

        --classes: path to classes file
            (default: './license_plate_detection/data/license.plate')
        --yolo_weights: path to weights file
            (default: './license_plate_detection/checkpoints/yolov3_train_8.tf')
        --yolo_output: path to output image
            (default: './license_plate_detection/outputs/global_outputs/')
        --yolo_cropped_output: path to output cropped image
            (default: './license_plate_detection/outputs/cropped_outputs/')
        --batch_detection: False if you want to detect only one image
            (default: False)
        --input_images: path to input images folder
            (default: './license_plate_detection/data/jpg/test/')
        --image: Image you want to predict, to use only if batch detection is disabled
            (no default value)
        --yolo_size: resize image to (default: 416)
        --num_classes: number of classes in the model (default: 80)
        
To segment the cropped image, you can use the following flags:

        --yolo_cropped_output: path to output cropped image
            (default: './license_plate_detection/outputs/cropped_outputs/')
        --seg_output: dir where you want to save the segmented characters
            (default: './character_reco/data/jpg')
        --image: Image you want to predict, to use only if batch detection is disabled
            (no default value)
        --batch_detection: False if you want to detect only one image
            (default: False)

        