def segment(img, save=False,name=None):
        """
        This method is responsible for licence plate segmentation with histogram of pixel projection approach
        :param img: input image
        :return: list of image, each one contain a digit
        """
        # list that will contains all digits
        caracrter_list_image = list()

        # img = crop(img)
        
        # Add black border to the image
        BLACK = [0, 0, 0]
        img = cv2.copyMakeBorder(img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=BLACK)

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
        
        # Binarization
        nb[nb > 150] = 255
        nb[nb < 150] = 0

        # compute the sommation
        x_sum = cv2.reduce(nb, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
        y_sum = cv2.reduce(nb, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)

        # rotate the vector x_sum
        x_sum = x_sum.transpose()

        # division the result by height and weight
        x_sum = x_sum / y
        y_sum = y_sum / x

        # x_arr and y_arr are two vector weight and height to plot histogram projection properly
        x_arr = np.arange(x)
        y_arr = np.arange(y)

        # convert x_sum to numpy array
        z = np.array(x_sum)

        # convert y_arr to numpy array
        w = np.array(y_sum)

        # convert to zero small details
        z[z < 15] = 0
        z[z > 15] = 1

        # convert to zero small details and 1 for needed details
        w[w < 20] = 0
        w[w > 20] = 1

        # vertical segmentation
        test = z.transpose() * nb

        # horizontal segmentation
        test = w * test
        # cv2.imshow('image',test)
        # cv2.waitKey()

        # plot histogram projection result using pyplot
        horizontal = plt.plot(w, y_arr)
        vertical = plt.plot(x_arr ,z)

        # plt.show(horizontal)
        # plt.show(vertical)

        f = 0
        ff = z[0]
        t1 = list()
        t2 = list()
        for i in range(z.size):
            if z[i] != ff:
                f += 1
                ff = z[i]
                t1.append(i)
        rect_h = np.array(t1)

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

        caracter_list_image = []

        # extract caracter
        for i in range(len(rect_h) - 1):

            # eliminate slice that can't be a digit, a digit must have width bigger then 8
            diff1 = int(rect_h[i + 1]) - int(rect_h[i])

            if (diff1 > 4) and (z[rect_h[i]] == 1):
                # cutting nb (image) and adding each slice to the list caracter_list_image
                caracter_list_image.append(nb[int(rectv[0]):int(rectv[1]), rect_h[i]:rect_h[i + 1]])

                # draw rectangle on digits
                cv2.rectangle(img, (rect_h[i], rectv[0]), (rect_h[i + 1], rectv[1]), (0, 255, 0), 1)
                # On prend un peu plus large que juste l'image pour avoir des bords noirs

                if save == True:
                    print('image saved in ../character_reco/data/jpg/output_' + name + '.jpg')
                    cv2.imwrite('../character_reco/data/jpg/output_' + name + '.jpg', nb[int(rectv[0])-2:int(rectv[1])+2, rect_h[i]-2:rect_h[i + 1]+2])

        # # Show segmentation result
        # image = plt.imshow(img)
        # plt.show(image)

        return caracter_list_image