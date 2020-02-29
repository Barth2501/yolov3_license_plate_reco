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
            
            if --mode segment_only, you have:
            
            --yolo_cropped_output: path to output cropped image
                (default: './license_plate_detection/outputs/cropped_outputs/')
            --seg_output: dir where you want to save the segmented characters
                (default: './character_reco/data/jpg')
            --image: Image you want to predict, to use only if batch detection is disabled (No default value)
            --batch_detection: False (default) if you want to detect only one image
            
            if --mode cnn_fine_tune, you have:
            
            --cnn_train_dataset: Path to the train tfrecord dataset
                (default: './character_reco/data/tfrecord/train_character.tfrecord')
            --cnn_val_dataset: Path to the val tfrecord dataset
                (default: './character_reco/data/tfrecord/val_character.tfrecord')
            --cnn_size: The size we want to resize the number (default: 28)
            --cnn_batch_size: Size of the batch (default: 1)
            --cnn_learning_rate: Learning rate (default: 0.001)
  
         
            
        - batch detect license plate for the next step of the all model:

            ```
            $ python main.py --mode yolo_batch_detect
            ```

            This command will detect the license plate on images from a folder and will save the output on a predefine location. You can change the default settings using FLAGS. 