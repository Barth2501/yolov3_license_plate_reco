import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment(img, save=False,name=None):
        """
        This method is responsible for licence plate segmentation with histogram of pixel projection approach
        :param img: input image
        :return: list of image, each one contain a digit
        """
        # list that will contains all digits
        caracter_list_image = list()

        # img = cv2.copyMakeBorder(img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=BLACK)
        if len(img.shape)==3:
            # change to gray
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Change to numpy array format
            nb = np.array(gray)
            # get height and weight
            x = gray.shape[1]
            y = gray.shape[0]
        else:
            # Change to numpy array format
            nb = np.array(img)
            # get height and weight
            x = img.shape[1]
            y = img.shape[0]

        # Check if the background is in black or white

        total = np.mean(nb)
        # Binarization
        if total > 255/2:
            # this number are experimental and seems to work well with white background
            nb[nb >= np.quantile(nb, 0.46)] = 255
            nb[nb < np.quantile(nb, 0.46)] = 0
            nb = cv2.bitwise_not(nb)
        else:
            # this number are experimental and seems to work well with black background
            nb[nb > np.quantile(nb, 0.78)] = 255
            nb[nb < np.quantile(nb, 0.78)] = 0

        # compute the sommation
        y_sum = cv2.reduce(nb, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)

        y_sum = y_sum / x

        # convert y_arr to numpy array
        w = np.array(y_sum)

        # convert to zero small details and 1 for needed details
        w[w < 30] = 0
        w[w > 30] = 1

        # Find the zone of interest in the image
        t2 = list()
        f = 0
        ff = w[0]
        for i in range(w.size):
            if w[i] != ff:
                f += 1
                ff = w[i]
                t2.append(i)
        rect_v = np.array(t2)

        # take the appropriate height
        rectv = []
        rectv.append(rect_v[0])
        rectv.append(rect_v[1])
        max = int(rect_v[1]) - int(rect_v[0])
        for i in range(len(rect_v) - 1):
            diff2 = int(rect_v[i + 1]) - int(rect_v[i])

            if diff2 > max:
                rectv[0] = rect_v[i]
                rectv[1] = rect_v[i + 1]
                max = diff2

        # crop the image
        nb = nb[rectv[0]-2:rectv[1]+2,:]
        w = w[rectv[0]-2:rectv[1]+2]

        x_sum = cv2.reduce(nb, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)

        # rotate the vector x_sum
        x_sum = x_sum.transpose()

        # division the result by height and weight
        x_sum = x_sum / y

        # convert x_sum to numpy array
        z = np.array(x_sum)

        # convert to zero small details
        z[z < np.quantile(z,0.3)] = 0
        z[z > np.quantile(z,0.3)] = 1

        # vertical segmentation
        test = z.transpose() * nb

        # horizontal segmentation
        test = w * test

        # Check if the character detection have been done correctly
        cv2.imshow('Binarization of the license plate',test)
        cv2.waitKey(2000)

        # Character detection along the horizontal axis
        f = 0
        ff = z[0]
        t1 = list()
        for i in range(z.size):
            if z[i] != ff:
                f += 1
                ff = z[i]
                t1.append(i)
        rect_h = np.array(t1)

        # extract caracter
        for i in range(len(rect_h) - 1):

            # eliminate slice that can't be a digit, a digit must have width bigger then 8
            diff1 = int(rect_h[i + 1]) - int(rect_h[i])

            if (diff1 > 2) and (z[rect_h[i]] == 1):
                # cutting nb (image) and adding each slice to the list caracter_list_image
                caracter_list_image.append(nb[:, rect_h[i]-1:rect_h[i + 1]+1])

                # draw rectangle on digits
                cv2.rectangle(img, (rect_h[i], rectv[0]), (rect_h[i + 1], rectv[1]), (0, 255, 0), 1)

        # Show segmentation result
        # image = plt.imshow(img)
        # plt.show(image)

        return caracter_list_image

def crop_char_and_save(character, name=None):


    x, y = character.shape
    y_sum = cv2.reduce(character, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    w = y_sum / x

    # convert to zero small details and 1 for needed details
    w[w < 5] = 0
    w[w > 5] = 1

    # Find the zone of interest in the image
    t2 = list()
    f = 0
    ff = w[0]
    for i in range(w.size):
        if w[i] != ff:
            f += 1
            ff = w[i]
            t2.append(i)
    rect_v = np.array(t2)

    # take the appropriate height
    rectv = []
    rectv.append(rect_v[0])
    if len(rect_v)<2:
        return None
        
    rectv.append(rect_v[1])
    maxi = int(rect_v[1]) - int(rect_v[0])
    for i in range(len(rect_v) - 1):
        diff2 = int(rect_v[i + 1]) - int(rect_v[i])
        if diff2 > maxi:
            rectv[0] = rect_v[i]
            rectv[1] = rect_v[i + 1]
            maxi = diff2

    new_char = character[rectv[0]:rectv[1],:]

    # We add a black border to make the image closer to a square
    BLACK = [0, 0, 0]
    img = cv2.copyMakeBorder(new_char, max(0,(y-x)//2)+1, max(0,(y-x)//2)+1, max(0,(x-y)//2)+1, max(0,(x-y)//2)+1, cv2.BORDER_CONSTANT, value=BLACK)
    if name:
        print('image saved in ' + name + '.jpg')
        cv2.imwrite(name + '.jpg', img)

    return img
