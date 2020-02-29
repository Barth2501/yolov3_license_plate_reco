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