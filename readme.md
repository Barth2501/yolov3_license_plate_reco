# Military Truck License Plate detection

This project is to be used for the french army in order to detect a license plate in an image and recognize its license plate.

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
            $ cd license_plate_detection/datasets/yolo/
            $ wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
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
                (default: '././license_plate_detection/data/tfrecord/val_license_plate.tfrecord)
            --yolo_pretrained_weights: 

        - batch detect license plate for the next step of the all model:

            ```
            $ python main.py --mode yolo_batch_detect
            ```

            This command will detect the license plate on images from a folder and will save the output on a predefine location. You can change the default settings using FLAGS.

### Character segmentation

This step is used to separate each character of the license plate. It uses computer vision aspect but still suffers some issues. Next effort will be concentrate at this step to make it more adaptable to any license plate luminosity and shape.

This can be done using the mode segment_only of the main.py file.
```
$ python main.py --mode segment_only
```

The FLAGS for this function are :

A TOI DE JOUER PAULINE

### Character identification

This final step is for the identification character by character. The model I used is a CNN model with two convolutionnal layers. Once again the model is pre-trained on the mnist dataset and then we fine tune it with the labelised character we got from the segmentation phase.

So the mode you can use here is cnn_fine_tune, it allows you to perform the fine tunning of the last layers (second cnn and dense layers). 
```
$ python main.py --mode cnn_fine_tune
```

N.B.: Before training the cnn, you must have created a tfrecord file of the labelised data you had (See the labelisation section)

The FLAGS you can use for this mode are :

A TOI DE REJOUER PAULINE

## Labelisation

In order to fine tune the models (yolo or cnn), you will have to build tfrecord files of the labelised data.

Personnally, I used labelImg application to perform that, it creates XML file in the folder I selected.
Then I create a csv file form the XML ones and then a tfrecord, you should use this for example with the labelised images of license plate detection:

```
$ cd labelisation
$ python xml_to_csv.py --img_path ./license_plate_detection/data/xml/train/ --save_name ./license_plate_detection/data/csv/train_plate.csv
$ python generate_tfrecord --csv_input ./license_plate_detection/data/csv/train_plate.csv --output_path ./license_plate_detection/data/tfrecord/train_plate.tfrecord --image_dir ./license_plate_detection/data/jpg/train
```


