# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import cv2 as cv
import imageio
import glob
import pickle
import pandas as pd
import math
import time
import os
import PyPDF2 as p
import sys
import PyGnuplot as gp
start_time = time.time()

""" Note:  
        Coordinate System: (x, y) or (y, x) but y is always ROW (vertical)
            and x is always COLUMN (horizontal)
        Dataframe collection is in 3D, accessor codes commented out for now
        Using notation 1 = Positive (Yes-eye), 0 = Negative (No-eye)
        Thresholding condition (Positive): feature-value <= Lower Threshold
                                           feature-value >= Upper Threshold
        Special case: If all samples for a feature are negative, then both
            thresholds remain at 0 with gini = infinity
        2880 iterations Adaboosting loop took 3 hours and 12 minutes

    Progress Report:
        Normalization of input images done
        Integral Image generation of input images done
        Building Haar-like Features done
        Apply Features done
        Labeling Features by pre-defined correct metadata done (Training)
        Upper and Lower Thresholds for feature values above and below zero
            computed using minimum gini-value as gini = positive-error^2 +
            negative-error^2 where positive-error = false-negative / 
            number of true-positives and negative-error = false-positive /
            number of true-negatives starting at zero for both thresholds
            and explored outward until minimum ginis found done
        Weight initialization done (Training)
        Adaboosting loops done (Training)
        Strong Classifier's collection of weak classifiers found through
        each iteration of Adaboosting (WORK-IN-PROGRESS)
            ->Need to change the string representation of classifiers
              saved during Adaboosting back to actual classifier type 
"""


def read_metadata(path):
    """ Read BIP bin file of image metadata and reorganize them into 4x1 for each image """
    with open(path, "rb") as f:
        fileContent = np.fromfile(f, dtype=np.uint8)
    reorganized = []
    # Metadata stored in BIP format so re-organizing here into [y, x, width, height]
    for x in range(0, int(len(fileContent)/4)):
        reorganized.append([fileContent[x], fileContent[x+50],
                            fileContent[x+100], fileContent[x+150]])
    return reorganized


def min_max_eye(path):
    """ Find the metadata from all provided correct eye-bounding box from the training set
        Find the minimum starting point and maximum endpoint for all provided bounding boxs
    """
    with open(path, "rb") as f:
        fileContent = np.fromfile(f, dtype=np.uint8)
    # Currently in BIP format, which is useful for extracing min-max as below:
    """ Actually in [horizontal-X*50], [vertical-Y*50], [width*50], [height*50] """
    ret = []
    for x in range(0, 200, 50):
        ret.append(fileContent[x:x+50].min())
        ret.append(fileContent[x:x+50].max())
    # print(ret)
    # print(x_min, x_max, y_min, y_max, width_min, width_max, height_min, height_max)
    return ret

# Import images


def import_image(path, num_images):
    """ Given the path to the folder containing the images, import all images and return them as a list """
    image_list = []
    # filename_list = []
    # for filename in glob.glob(path):
    #     print(filename)
    #     filename_list.append(filename)
    #     image = imageio.imread(filename)
    #     image_list.append(image)
    #     # cv.imshow('image', image)
    #     # cv.waitKey(0)
    #     # cv.destroyAllWindows()
    # print(sorted_image_list)
    for i in range(1, num_images+1):
        filename = path+'training'+str(i)+'.bmp'
        image_list.append(imageio.imread(filename))
    return image_list


def max_normalize(image_list):
    """ normalizing function to 0-1 image, need multiply each normalized pixel by 16 bits (2 byte per pixel) to keep precision """
    normalized_list, count = [], 0
    for im in image_list:
        normalized_image = np.zeros(im.shape)
        # normalized_image = cv.normalize(im, normalized_image, 0, 1, cv.NORM_MINMAX)
        # image_mean = np.mean(im)
        image_max = np.max(im)
        normalized_image = im/image_max    # Normalizing
        # 18 bits to be precise, so losing 2 bits of precision here
        normalized_image *= pow(2, 16)
        normalized_list.append(normalized_image)
        filename = "normalized_images/normalized_" + str(count) + ".txt"
        np.savetxt(filename, normalized_image)
        count += 1
    return normalized_list


def integral_image(image_list):
    """ Generate the integral image which will save computing time later in the algorithm """
    ii_list = []
    count = 0
    for image in image_list:
        ii = np.zeros(image.shape)
        s = np.zeros(image.shape)
        for y in range(len(image)):
            for x in range(len(image[y])):
                s[y][x] = s[y-1][x] + \
                    image[y][x] if (y-1 >= 0) else image[y][x]
                ii[y][x] = ii[y][x-1] + s[y][x] if (x-1 >= 0) else s[y][x]
                # if (x-1 >= 0):
                #     s[x][y] = s[x-1][y] + image[x][y]
                # else:
                #     s[x][y] = image[x][y]
                # if (y-1 >= 0):
                #     ii[x][y] = ii[x][y-1] + s[x][y]
                # else:
                #     ii[x][y] = s[x][y]
        ii_list.append(ii)
        filename = "integral_image/integral_" + str(count) + ".txt"
        np.savetxt(filename, ii)
        count += 1
    return ii_list


class RectangleRegion:
    """ Rectangle that makes up the Haar-features, (x, y) coordinate format where x is horizontal (COL) and y is vertical (ROW) """

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def compute_feature(self, ii):
        """ D + A - (C + B) """
        a = ii[self.y][self.x]  # Row then column
        b = ii[self.y][self.x+self.width]
        c = ii[self.y+self.height][self.x]
        d = ii[self.y+self.height][self.x+self.width]
        val = d + a - (c + b)
        return val


class Feature:
    """ pos_neg 0 for "Other parts of the face" Negative, 1 for "eye part" Positive """

    def __init__(self, start_x, start_y, end_x, end_y, haar_pos, haar_neg, pos_neg=0, feature_type=-1):
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y
        self.haar_pos = haar_pos
        self.haar_neg = haar_neg
        self.pos_neg = pos_neg
        if(feature_type >= 0 and feature_type <= 3):
            self.feature_type = feature_type
        else:
            print("ERROR: Feature's type should always be between 0 and 3 inclusive")

    def __repr__(self):
        return "<Feature of type %s starts(%s, %s) ends(%s, %s) in (x, y) format>" % (self.feature_type, self.start_x, self.start_y, self.end_x, self.end_y)

    def __str__(self):
        return "Feature of type %s starts(%s, %s) ends(%s, %s) in (x, y) format" % (self.feature_type, self.start_x, self.start_y, self.end_x, self.end_y)

    def get_size(self):
        """ Return the starting point and end point in (x, y) """
        return [self.start_x, self.start_y, self.end_x, self.end_y]


def add_value_labels(ax, special, fsize=5, rotate=60, spacing=5):
    """ Function from justfortherec's answer to "Adding value labels on a matplotlib bar chart"
    Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """
    # For each bar: Place a label
    x = 0
    for rect in ax.patches:
        if x in special:
            rect.set_facecolor('r')
        else:
            rect.set_facecolor('b')
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'
        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'
        # Use Y value as label and format number with f decimal place(s)
        label = "{:.3f}".format(y_value)
        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            rotation=rotate,              # Rotate counterclockwise
            fontsize=fsize,
            va=va)                      # Vertically align label differently for positive and negative values.
        x += 1


class ViolaJones:
    """ Strong Classifier """

    def __init__(self, T=10):
        self.T = T
        self.errs = []
        self.alphas = []
        self.clf_indexes = []
        self.clfs = []

    def plot_graphs(self, foldername, X_list, pos_stat):
        """
        Plotting the not-eye and eye bar graphs and save to folder
        Note: Bar plot parameters must be same shape/dimensions!!!
        Vertical axis: the X values (applied feature value)
        Horizontal axis: the image that feature corresponds to! (Which is not in order, unless miracle)
        X-list might or might not be sorted by its feature value
        """
        pdfs, counter = [], 0
        pos_indexes = list(map(lambda ii: ii[0], pos_stat))
        # If sorted:
        # for index, sorted_list in X_list:
        # If NOT sorted:
        for index, sorted_list in X_list:
            print(counter, "Feature Index:\t", index)
            # sorted_index = list(map(lambda y: y[0], sorted_list))
            # sorted_value = list(map(lambda y: y[1], sorted_list))
            # ind = np.arange(len(sorted_index))
            ind = np.arange(len(sorted_list))
            fig, ax = plt.subplots()
            # ax.bar(ind, sorted_value)
            ax.bar(ind, sorted_list)
            ax.set_title('Feature [' + str(index) + ']')
            ax.set_xlabel('Image Index')
            ax.set_ylabel('Feature Value')
            ax.set_xticks(ind)  # y-location of each tick!!!
            ax.set_xticklabels(ind, fontsize=5, rotation=45)

            positive_index_after_sort = []
            for x in range(len(pos_indexes)):
                pos_index = pos_indexes[x]
                if pos_index > index:
                    break
                if pos_index == index:
                    # Find out the new index of the positive indexes after sorting above
                    # positive_index_after_sort = list(
                    #     map(lambda ii: ind.index(ii), pos_stat[x][1]))
                    positive_index_after_sort = pos_stat[x][1]
            add_value_labels(ax, positive_index_after_sort, 5, 'vertical')
            plt.savefig(foldername+'/feature_' +
                        str(index) + '_graph.pdf', bbox_inches='tight')
            pdfs.append(foldername+'/feature_' + str(index) + '_graph.pdf')
            # plt.show()
            plt.close()
            counter += 1
        merger = p.PdfFileMerger()
        for pdf in pdfs:
            merger.append(pdf)
        merger.write(foldername+"/Combined.pdf")
        merger.close()

    def build_features(self, image_shape, minmax):
        height, width = image_shape
        features = []
        # minmax format = [min_x, max_x, min_y, max_y, min_width, max_width, min_height, max_height]
        min_x, max_x, min_y, max_y, min_width, max_width, min_height, max_height = minmax
        print("X: %i->%i, Y: %i->%i, Width: %i->%i, Height: %i->%i" %
              (min_x, max_x, min_y, max_y, min_width, max_width, min_height, max_height))
        # Row
        # for y in range(minmax[0], minmax[1]+1):
        for y in range(min_y, max_y+1):
            # Col
            # for x in range(minmax[2], minmax[3]+1):
            for x in range(min_x, max_x+1):
                # Width
                # for w in range(minmax[4], minmax[5]+1):
                for w in range(min_width, max_width+1):
                    # Height
                    # for h in range(minmax[6], minmax[7]+1):
                    for h in range(min_height, max_height+1):
                        # print('x/y/w/h\t', x, y, w, h)
                        half_width = math.floor(w/2)
                        third_width = math.floor(w/3)
                        half_height = math.floor(h/2)

                        # 2*half_width will never exceed w, at most equals w. Same for half_height and third_width
                        make_up_half_w, make_up_half_h, make_up_third_w = w - \
                            (2*half_width), h - \
                            (2*half_height), w - (3*third_width)

                        # print("x %s, y %s, w %s, h %s" % (x, y, w, h))
                        # print("half_width: %s, 3rd_width %s, half_height %s" % (half_width, third_width, half_height))
                        # # Rectangles: Note that format is [positive], [negative]
                        extra_w, extra_h = 0, 0
                        if(x+w >= width):
                            extra_w = (x+w) - width + 1
                        if(y+h >= height):
                            extra_h = (y+h) - height + 1
                        # Feature A:
                        if(half_width >= 1):
                            # print("\nhalf_width", half_width, "width", w)
                            # print("start point", x, y, "end point", x +
                            #       half_width+half_width-extra_w, y+h-extra_h)
                            # print("extra w, h", extra_w, extra_h)
                            immediate = RectangleRegion(
                                x, y, half_width, h - extra_h)
                            right = RectangleRegion(
                                x+half_width, y, half_width + make_up_half_w - extra_w, h - extra_h)
                            a = Feature(x, y, x+half_width*2+make_up_half_w -
                                        extra_w, y+h-extra_h, [right], [immediate], 0, 0)
                            features.append(a)
                        # print(a)
                        # Feature B:
                        if(half_height >= 1):
                            # if(half_height + make_up_half_h - extra_h) > minmax[7]:
                            #     print(y, "B's height", half_height + make_up_half_h - extra_h)
                            #     print("h", h, "half h", half_height, "makeup", make_up_half_h, "extra h", extra_h)
                            immediate_2 = RectangleRegion(
                                x, y, w-extra_w, half_height)
                            bottom = RectangleRegion(
                                x, y+half_height, w-extra_w, half_height + make_up_half_h - extra_h)
                            b = Feature(x, y, x+w-extra_w, y+half_height*2 +
                                        make_up_half_h-extra_h, [immediate_2], [bottom], 0, 1)
                            features.append(b)
                        # print(b)
                        # Feature C:
                        if(third_width >= 1):
                            immediate_3rd = RectangleRegion(
                                x, y, third_width, h - extra_h)
                            center_3rd = RectangleRegion(
                                x+third_width, y, third_width, h - extra_h)
                            right_3rd = RectangleRegion(
                                x+third_width*2, y, third_width + make_up_third_w - extra_w, h - extra_h)
                            c = Feature(x, y, x+third_width*3+make_up_third_w-extra_w, y+h-extra_h, [center_3rd], [
                                        right_3rd, immediate_3rd], 0, 2)
                            features.append(c)
                        # print(c)
                        # Feature D:
                        if(half_width >= 1 and half_height >= 1):
                            top = RectangleRegion(
                                x, y, half_width, half_height)
                            right = RectangleRegion(
                                x+half_width, y, half_width + make_up_half_w - extra_w, half_height)
                            bottom = RectangleRegion(
                                x, y+half_height, half_width, half_height + make_up_half_h - extra_h)
                            bottom_right = RectangleRegion(
                                x+half_width, y+half_height,  half_width + make_up_half_w - extra_w, half_height + make_up_half_h - extra_h)
                            d = Feature(x, y, x+half_width*2+make_up_half_w-extra_w, y+half_height*2+make_up_half_h-extra_h, [right, bottom], [
                                        top, bottom_right], 0, 3)
                            features.append(d)
                        # print(d)
        return features

    def bigger_box(self, start_end, correct):
        """ Returns true if the provided start_end points/box is bigger than the correct bounding box """
        return (start_end[0] <= correct[0] and
                start_end[1] <= correct[1] and
                start_end[2] >= correct[0]+correct[2] and
                start_end[3] >= correct[1]+correct[3])

    def label_features(self, feature, correct_list):
        """ For each feature compare it to each sample image provided and the correct bounding box and label """
        features = []   # Feature is only build once, so duplicate for each of the images
        for x in range(len(correct_list)):
            features.append(feature)
        stat = []    # For each feature its total stat
        pos_stat = []   # For each feature-> indexes of score corresponding to positive sample
        neg_stat = []   # For each feature-> indexes of score corresponding ot negative samples
        y_list = []  # For each feature, then for each sample
        pos_y_list = []
        for x in range(len(feature)):
            neg, pos = 0, 0
            start_end = feature[x].get_size()
            # print(x, start_end)
            temp_y, temp_pos_y, temp_pos, temp_neg = [], [], [], []
            for y in range(len(correct_list)):
                if(self.bigger_box(start_end, correct_list[y])):    # Positive
                    pos += 1
                    features[y][x].pos_neg = 1
                    temp_y.append(1)
                    temp_pos.append(y)
                else:
                    neg += 1
                    features[y][x].pos_neg = 0
                    temp_y.append(0)
                    temp_neg.append(y)
            y_list.append(temp_y)
            stat.append([pos, neg])
            if temp_pos:    # If not empty
                pos_stat.append([x, temp_pos])
            if temp_neg:    # If not empty
                neg_stat.append([x, temp_neg])
        with open("output/all_features.txt", "w") as f:
            for item in features:
                f.write("%s\n" % item)
        with open("output/y_list.txt", "w") as f:
            for item in y_list:
                f.write("%s\n" % item)
        with open("output/positive_indexes.txt", "w") as f:
            for item in pos_stat:
                f.write("%s\n" % item)
        with open("output/negative_indexes.txt", "w") as f:
            for item in neg_stat:
                f.write("%s\n" % item)
        return features, stat, y_list, pos_stat, neg_stat

    def discard_useless_features(self, features, feature_stat, num_samples):
        """ Features that is completely correct or completely incorrect provides no relevant information, thus discarded """
        useful_features = []
        for x, f in enumerate(features):
            # print(x, "f:", f)
            if(num_samples not in feature_stat[x]):
                useful_features.append([x, f])
                # print("useful_f_haar_pos", f.haar_pos)
        return useful_features

    def apply_features(self, features, ii_list):
        """ Apply each feature to the integral images generated from the samples and save result X list to file """
        X, sorted_X = [], []
        for index, f in enumerate(features):
            positive_regions, negative_regions = f.haar_pos, f.haar_neg
            applied_X = list(map(lambda data: sum([pos.compute_feature(data)
                                                   for pos in positive_regions]) - sum([neg.compute_feature(data)
                                                                                        for neg in negative_regions]), ii_list))
            X.append([index,  applied_X])
            sorted_X.append(
                [index, sorted(list(enumerate(applied_X)), key=lambda ii: ii[1])])
        return X, sorted_X

    def find_gini_thresholds(self, sorted_indexed_X, y_list, pos_stat, neg_stat, features):
        """
        Find lower and upper thresholds both starting from 0 where gini coefficient is minimized
        Definition: Positive if greater than upper threshold or less than lower threshold, Negative otherwise

        Compute the Gini coefficient as ((False-Positive/#-Negative)^2 + (False-Negative/#-Positive)^2) for each
        feature and each sample and select the minimum gini coefficient for a feature by defining
        more than threshold chosen starting at smallest X value to largest X value as 1 for no-eye
        and less than or equal to threshold as 0 for yes-eye

        Toggle: 1 for no-eye (Negative), 0 for yes-eye (Positive)
        Accuracy (Guessed right) is determined as absolute value e(y) = |h(y) - x(y)| where h(y) is the thresholding
        function and x(y) is the actual correctness. All four cases:
        |1 - 1| = 0     True-Negative
        |1 - 0| = 1     False-Positive
        |0 - 1| = 1     False-Negative
        |0 - 0| = 0     True-Positive
        When 1 is positive, 0 is negative
        """
        best_thresholds, classifiers = [], []  # One for each feature
        # x = 0
        # For each feature, sorted_list is [[img_index, feature_value], [],...]
        for index, sorted_list in (sorted_indexed_X):
            """ Splice feature values in half by 0.0, both thresholds start searching at 0+-, feature score does reach 0 quite often """
            below_zero = list(filter(lambda ii: ii[1] < 0, sorted_list))
            below_indexes = list(map(lambda ii: ii[0], below_zero))
            above_zero = list(filter(lambda ii: ii[1] > 0, sorted_list))
            above_indexes = list(map(lambda ii: ii[0], above_zero))
            below_positives, below_negatives, above_positives, above_negatives = 0, 0, 0, 0
            for positive_index, positive_sample_indexes in pos_stat:
                if positive_index > index:
                    break
                if positive_index == index:
                    for ind in below_indexes:
                        if ind in positive_sample_indexes:
                            below_positives += 1
                    for ind in above_indexes:
                        if ind in positive_sample_indexes:
                            above_positives += 1
            for negative_index, negative_sample_indexes in neg_stat:
                if negative_index > index:
                    break
                if negative_index == index:
                    for ind in below_indexes:
                        if ind in negative_sample_indexes:
                            below_negatives += 1
                    for ind in above_indexes:
                        if ind in negative_sample_indexes:
                            above_negatives += 1
            # print("\nIndex %s\nAbove pos-neg:\t%s-%s\nBelow pos-neg:\t%s-%s" %
            #       (index, above_positives, above_negatives, below_positives, below_negatives))

            """ Now we have all the metadata we need on each feature to start searchingfor both thresholds through gini """
            lower_best_threshold_index, lower_best_threshold_value, lower_best_gini = 0, 0, float(
                'inf')
            # Since we start searching the < 0s at closest to 0 downward, order is reversed
            # Trying the value at each below_zero as threshold
            for threshold_index, threshold_value in reversed(below_zero):
                false_positive, false_negative, true_negative, true_positive, error_positive, error_negative = 0, 0, 0, 0, 0, 0
                # Applying current threshold at each below_zero sample, also starting at closest to zero downward:
                for sample_index, sample_value in reversed(below_zero):
                    """ Notation: 1 for positive, 0 for negative """
                    guess = 1 if sample_value <= threshold_value else 0
                    # correctness = abs(guess - y_list[index][sample_index])
                    if guess == 0 and y_list[index][sample_index] == 0:
                        true_negative += 1
                    elif guess == 1 and y_list[index][sample_index] == 1:
                        true_positive += 1
                    elif guess == 1 and y_list[index][sample_index] == 0:
                        false_positive += 1
                    # elif guess == 0 and y_list[index][sample_index] == 1:
                    else:
                        false_negative += 1
                error_positive = false_negative / \
                    below_positives if below_positives > 0 else float('inf')
                error_negative = false_positive / \
                    below_negatives if below_negatives > 0 else float('inf')
                lower_gini = error_positive**2 + error_negative**2
                if lower_gini < lower_best_gini:
                    lower_best_gini = lower_gini
                    lower_best_threshold_index = threshold_index
                    lower_best_threshold_value = threshold_value

            upper_best_threshold_index, upper_best_threshold_value, upper_best_gini = 0, 0, float(
                'inf')
            # Trying the value at each above_zero as threshold
            for threshold_index, threshold_value in above_zero:
                false_positive, false_negative, true_negative, true_positive, error_positive, error_negative = 0, 0, 0, 0, 0, 0
                # Applying current threshold at each below_zero sample, also starting at closest to zero downward:
                for sample_index, sample_value in above_zero:
                    """ Notation: 1 for positive, 0 for negative """
                    guess = 1 if sample_value >= threshold_value else 0
                    # correctness = abs(guess - y_list[index][sample_index])
                    if guess == 0 and y_list[index][sample_index] == 0:
                        true_negative += 1
                    elif guess == 1 and y_list[index][sample_index] == 1:
                        true_positive += 1
                    elif guess == 1 and y_list[index][sample_index] == 0:
                        false_positive += 1
                    # elif guess == 0 and y_list[index][sample_index] == 1:
                    else:
                        false_negative += 1
                error_positive = false_negative / \
                    above_positives if above_positives > 0 else float('inf')
                error_negative = false_positive / \
                    above_negatives if above_negatives > 0 else float('inf')
                upper_gini = error_positive**2 + error_negative**2
                if upper_gini < upper_best_gini:
                    upper_best_gini = upper_gini
                    upper_best_threshold_index = threshold_index
                    upper_best_threshold_value = threshold_value
            best_thresholds.append([index, [lower_best_threshold_index, lower_best_threshold_value, lower_best_gini], [
                                   upper_best_threshold_index, upper_best_threshold_value, upper_best_gini]])
            classifiers.append(WeakClassifier(index, features[index], lower_best_threshold_index,
                                              lower_best_threshold_value, upper_best_threshold_index, upper_best_threshold_value))

            # gini_at_threshold = []
            # total_positive, total_negative = feature_stat[index][0], feature_stat[index][1]
            # # For each threshold its (img_index, feature_value @ image = threshold value)
            # for threshold in sorted_list:
            #     false_positive, false_negative, true_negative, true_positive = 0, 0, 0, 0
            #     error_positive, error_negative = 0, 0
            #     # For each sample to apply threshold to
            #     for sample in sorted_list:
            #         # 1 for Positve, 0 for Negative
            #         # If greater than threshold, guess Positive
            #         guess = 1 if sample[1] > threshold[1] else 0
            #         # AKA guess = 0 if sample[1] <= threshold[1] else 1
            #         if guess == 1 and y_list[index][sample[0]] == 0:
            #             false_positive += 1
            #         elif guess == 0 and y_list[index][sample[0]] == 1:
            #             false_negative += 1
            #         elif guess == 0 and y_list[index][sample[0]] == 0:
            #             true_negative += 1
            #         else:  # elif guess == 1 and y_list[index][sample[0]] == 1:
            #             true_positive += 1
            #     error_positive = false_negative / \
            #         feature_stat[index][0] if false_negative > 0 else 0
            #     error_negative = false_positive / \
            #         feature_stat[index][1] if false_positive > 0 else 0
            #     gini = pow(error_positive, 2) + pow(error_negative, 2)
            #     gini_at_threshold.append(
            #         [index, threshold, gini, (true_negative, true_positive, false_positive, false_negative)])
            # # Best_threshold_value at best_threshold[feat_index][0=feature index, 1=threshold_pair, 2=gini_value, 3=true/false-pos/neg stats][1]
            # min_gini = min(gini_at_threshold, key=lambda ii: ii[2])
            # best_thresholds.append(min_gini)
            # classifiers.append([index, WeakClassifier(
            #     useful_features[x], min_gini[1], min_gini[3][0], min_gini[3][1], min_gini[3][2], min_gini[3][3])])
            # x += 1
        return best_thresholds, classifiers

    def initialize_weights(self, feature_stat, y_list):
        """ Initialize the weights if currently at the first round: """
        weights = []
        for index, pair in enumerate(feature_stat):
            num_positives, num_negatives, row_weights = pair[0], pair[1], []
            for actual in y_list[index]:    # 1 for positive, 0 for negative
                num_sample = len(y_list[index])
                # Special case: When all negative (no-eye) or all positive (yes-eye)
                # Since weights in feature must add up to 1, and only one type exists, do NOT multiply by half as in usual case
                if num_negatives == num_sample or num_positives == num_sample:
                    if actual == 1:
                        row_weights.append(1/num_positives)
                    else:
                        row_weights.append(1/num_negatives)
                else:   # Since total weights in feature must add up to 1, and two types exists, must multiply by half
                    if actual == 1:
                        row_weights.append(1/(2*num_positives))
                    else:
                        row_weights.append(1/(2*num_negatives))
            weights.append([index, row_weights])
        return weights

    def normalize_weights(self, weights):
        """ Normalize the weights as described in step 1 of original paper """
        total_weights = sum(list(map(lambda ii: sum(ii[1]), weights)))
        print("Total weights across all feature and all samples: %s" %
              total_weights)
        # normalized_weights = list(
        #     map(lambda ii: [ii[0], list(map(lambda jj: jj / total_weights, ii[1]))], weights))
        # total_normalized_weights = sum(
        #     list(map(lambda ii: sum(ii[1]), normalized_weights)))
        # max_normalized_weights = max(
        #     list(map(lambda ii: sum(ii[1]), normalized_weights)))
        """ Using nested-for loops is faster by 1.3 seconds with size of 20 """
        normalized_weights, total_normalized_weights, max_normalized_weights = [], 0, 0
        for x in range(len(weights)):
            # index, weight_row = weights[x][0], weights[x][1]
            temp_row, row_sum = [], 0
            for y in range(len(weights[x][1])):
                # weights[x][1][y] = weights[x][1][y] / total_weights
                temp_row.append(weights[x][1][y] / total_weights)
                row_sum += weights[x][1][y] / total_weights
            normalized_weights.append([weights[x][0], temp_row])
            total_normalized_weights += row_sum
            if row_sum > max_normalized_weights:
                max_normalized_weights = row_sum
        return normalized_weights, total_normalized_weights, max_normalized_weights

    def apply_classifiers(self, sorted_X_list, y_list, classifiers, weights, max_normalized_weights):
        """ At each threshold, compare to each feature value and find min error 
        to find the best threshold, which must not repeat previous iteration's findings
        """
        clf_errors, useful_clf_errors, best_error_index, best_error, best_accuracy = [
        ], [], 0, float('inf'), []
        for clf in classifiers:
            error, accuracy = 0, []
            if clf.index != sorted_X_list[clf.index][0]:
                print("ERROR: Classifier index and X_list index mismatched at %s->%s" %
                      (clf.index, sorted_X_list[clf.index][0]))
            for sample_index, sample_value in sorted_X_list[clf.index][1]:
                # Less than threshold = Guess Negative (Guess No-Eye) 0, Keeping consistency with find_gini_threshold
                guess = 1 if sample_value >= clf.upper_threshold_value or sample_value <= clf.lower_threshold_value else 0
                # correctness = 0 if guessed Correctly, 1 if guessed Incorrectly, If matches 0, else absolute value to +1
                correctness = abs(guess - y_list[clf.index][sample_index])
                accuracy.append(correctness)
                if correctness == 1:    # Either False Negative or False Positive
                    # print("weight is", weights[clf.index][1][sample_index])
                    error += weights[clf.index][1][sample_index]
            clf_errors.append([clf.index, error])
            # if clf.index == 0:
            #     print("Total weighted error of this feature adds up to 1.0000000000000004 is", 1.0000000000000004 == error)
            """ Needed to truncate below because the last few decimal places were changed (slightly increased) when passed to here! """
            max_normalized_weights = float('%.10f' % max_normalized_weights)
            # print("Max normalized weight is not truncated to 10 decimal places: %s" %
            #       max_normalized_weights)
            if error < max_normalized_weights:
                useful_clf_errors.append([clf.index, error])
            # if error < min_error and error not in [0, 50]:
            if error < best_error and clf.index not in self.clf_indexes:
                best_error, best_error_index, best_accuracy = error, clf.index, accuracy

        # """ At each threshold, compare to each feature value and find min error """
        # clf_errors, zero_list, counter = [], [], 0
        # for x, clf in classifiers:
        #     error = 0
        #     for y, val in sorted_X_list[counter][1]:
        #         # Less than threshold = Guess Negative (Guess No-Eye) 0, Keeping consistency with find_gini_threshold
        #         guess = 1 if val > clf.threshold[1] else 0
        #         # correctness = 0 if guessed Correctly, 1 if guessed Incorrectly, If matches 0, else absolute value to +1
        #         correctness = abs(guess - y_list[x][y])
        #         if correctness == 1:    # Either False Negative or False Positive
        #             # error += 1 * weights[counter][1]
        #             error += 1
        #     clf_errors.append([x, error])
        #     if error == 0:
        #         zero_list.append(x)
        #     counter += 1
        # with open("output/clf_error.txt", "w") as f:
        #     for item in clf_errors:
        #         f.write("%s\n" % item)

        # # print("zero list", zero_list)
        # sorted_clf_errors = sorted(clf_errors, key=lambda ii: ii[1])
        # to_prune = list(
        #     filter(lambda ii: ii[1] == 0, sorted_clf_errors))   # 2204, 2684
        # # print("len-weights %s len-to-prune %s" % (len(weights), len(to_prune)))
        # num_useful_features = int(len(useful_features) - len(to_prune))
        # # weights = [[row[0], 1/num_useful_features]
        # #            for row in useful_features if row[0] not in zero_list]
        # # with open("output/weights.txt", "w") as f:
        # #     for item in weights:
        # #         f.write("%s\n" % item)
        # pruned_sorted_clf_errors = list(
        #     filter(lambda ii: ii[1] != 0, sorted_clf_errors))
        # # clf_error values still just counter of errors, multiply by corresponding weight
        # weighted_pruned_sorted_clf_errors = list(
        #     map(lambda ii: [ii[0], ii[1]*(1/num_useful_features)], pruned_sorted_clf_errors))
        # # print(weighted_pruned_sorted_clf_errors, pruned_sorted_clf_errors)
        # with open("output/sorted_clf_errors.txt", "w") as f:
        #     for item in weighted_pruned_sorted_clf_errors:
        #         f.write("%s\n" % item)
        # with open("output/sorted_clf_errors_metadata.txt", "w") as f:
        #     for item in weighted_pruned_sorted_clf_errors:
        #         f.write("clf-error %s\t->\t%ix normalized weight\n" %
        #                 (str(item), math.ceil(item[1]*num_useful_features)))
        return clf_errors, useful_clf_errors, best_error_index, best_error, best_accuracy

    def update_weights(self, weights, best_error_index, best_error, best_accuracy):
        """ Update the weight at the end of current round/iteration as described in step 4 of the original paper """
        best_weights, beta = weights[best_error_index][1], 0
        if best_error > 1:
            print(
                "ERROR: Best error should never reach or get close to 1: %s" % best_error)
        elif best_error == 1:
            beta = float('inf')
        else:
            beta = best_error / (1 - best_error)
        print("Beta is %s" % beta)
        for i in range(len(best_weights)):
            best_weights[i] = best_weights[i] * \
                pow(beta, (1 - best_accuracy[i]))
        weights[best_error_index][1] = best_weights
        # print("best weights is", best_weights)
        # print("Updated weight is", weights[best_error_index][1])
        return weights

    # def train(self, image_path: str, metadata_path: str) -> Tuple[List, List, List, List]:
    def train(self, foldername, weights, sorted_X_list, y_list, pos_stat, neg_stat, features):
        """ 
        Doing the actual training procedure here
        """
        updated_weights = []
        for t in range(self.T):
            if t != 0:
                weights = updated_weights
            print("\nROUND %s" % t)
            """ Step 1, Normalize the weights """
            # print("1.) Starting normalizing")
            normalized_weights, total_normalized_weights, max_normalized_weights = self.normalize_weights(
                weights)
            print("Max normalized weight is now %s" % max_normalized_weights)
            print("Total normalized weights is now %s" %
                  total_normalized_weights)
            print("Normalizing Finished")

            """ Step 2 & 3, finding best classifier: """
            # print("2/3.) Starting finding gini-thresholds and calculating errors")
            best_thresholds, classifiers = self.find_gini_thresholds(
                sorted_X_list, y_list, pos_stat, neg_stat, features)

            # Apply the classifier to find its error (epsilon) as min-error of all errors from all samples applied to
            clf_errors, useful_clf_errors, best_error_index, best_error, best_accuracy = self.apply_classifiers(
                sorted_X_list, y_list, classifiers, normalized_weights, max_normalized_weights)
            print("Best error of this round found at index %s with value %s" %
                  (best_error_index, best_error))
            print("With accuracy of %s" % best_accuracy)
            sorted_clf_errors = sorted(clf_errors, key=lambda ii: ii[1])
            sorted_useful_clf_errors = sorted(
                useful_clf_errors, key=lambda ii: ii[1])
            print("Thresholds and errors calculations Finished")

            """ Step 4, Update the weight """
            # print("4.) Starting to update the weights")
            updated_weights = self.update_weights(
                normalized_weights, best_error_index, best_error, best_accuracy)
            print("Updating weights Done")

            """ Step 4.5, Store the best classifier and its alpha
            Keep note of the best classifier(s) already found so they won't be chosen again in future rounds/iterations 
            """
            big_number = sys.float_info.max
            # print("Closest to positive infinit: %s" % big_number)
            # print(big_number == 1.7976931348623157e+308)
            alpha = 0
            if best_error > 1:
                print("ERROR: best error shouldn't even approach 1: %s" %
                      best_error)
                alpha = math.log(1/big_number)
            elif best_error < 0:
                print("ERROR: best error should never drop below 0: %s" %
                      best_error)
                alpha = math.log(big_number)
            elif best_error == 1:
                # alpha = float('-inf')
                alpha = math.log(1/big_number)  # -709.782712893384
            elif best_error == 0:
                # alpha = float('inf')
                alpha = math.log(big_number)    # 709.782712893384
            else:   # 0 < best_error < 1, as it should be
                alpha = math.log((1 - best_error) / best_error)
            print("Alpha is %s" % alpha)
            self.errs.append(best_error)
            self.alphas.append(alpha)
            self.clf_indexes.append(best_error_index)
            self.clfs.append(classifiers[best_error_index])
            if t == (self.T - 1):
                # print("SAVING DATA")
                with open(foldername+"/normalized_weights.txt", "w") as f:
                    for item in normalized_weights:
                        f.write("%s\n" % item)
                with open(foldername+"/best_thresholds.txt", "w") as f:
                    for item in (best_thresholds):
                        f.write("%s\n" % item)
                with open(foldername+"/classifiers.txt", "w") as f:
                    for item in classifiers:
                        f.write("%s\n" % item)
                with open(foldername+"/clf_errors.txt", "w") as f:
                    for item in clf_errors:
                        f.write("%s\n" % item)
                with open(foldername+"/useful_clf_errors.txt", "w") as f:
                    for item in useful_clf_errors:
                        f.write("%s\n" % item)
                with open(foldername+"/sorted_clf_errors.txt", "w") as f:
                    for item in sorted_clf_errors:
                        f.write("%s\n" % item)
                with open(foldername+"/sorted_useful_clf_errors.txt", "w") as f:
                    for item in sorted_useful_clf_errors:
                        f.write("%s\n" % item)
                with open(foldername+"/updated_weights.txt", "w") as f:
                    for item in updated_weights:
                        f.write("%s\n" % item)
                with open(foldername+"/errs.txt", "w") as f:
                    for item in self.errs:
                        f.write("%s\n" % item)
                with open(foldername+"/alphas.txt", "w") as f:
                    for item in self.alphas:
                        f.write("%s\n" % item)
                with open(foldername+"/final_clf_indexes.txt", "w") as f:
                    for item in self.clf_indexes:
                        f.write("%s\n" % item)
                with open(foldername+"/final_clfs.txt", "w") as f:
                    for item in self.clfs:
                        f.write("%s\n" % item)
                print("DATA SAVED")
        print("alpha list:", self.alphas)
        print("WeakClassifier list:", "\n\t".join(str(classifier)
                                                  for classifier in self.clfs))
        return self.clf_indexes, self.alphas, self.errs, self.clfs

        # print(len(classifiers))
        # for x, item in enumerate(classifiers):
        #     print("%s\t%s" % (x, item))
        # with open("output/classifiers.pkl", "wb") as f:
        #     pickle.dump(classifiers, f)

        # Takes about 50 minutes to fun
        # self.plot_graphs("feature_graphs", sorted_X_list, pos_stat)

    # def select_best(self, classifiers, weights, training_data):
    #     """     Out of all the weak_classifiers/features select the best one, accuracy for that feature only across all samples """
    #     x = 0
    #     feature_index, best_clf, best_error, best_accuracy = None, None, float(
    #         'inf'), None
    #     # accuracy is of the feature when applied to each sample images
    #     for clf in classifiers:  # For each of the 2880 weak classifiers
    #         error, accuracy = 0, []
    #         # For each sample feature was applied to:
    #         for data, w, in zip(training_data[x], weights[x]):
    #             # Classification of feature minus the actual correctness of feature
    #             correctness = abs(clf.classify(data[0]) - data[1])
    #             accuracy.append(correctness)
    #             error += w*correctness
    #         error /= len(training_data[x])
    #         if error < best_error:
    #             feature_index, best_clf, best_error, best_accuracy = x, clf, error, accuracy
    #         x += 1
    #     # print("len of best accuracy", len(best_accuracy))
    #     return feature_index, best_clf, best_error, best_accuracy

    def classify(self, integral_image):
        total = 0
        for alpha, clf in zip(self.alphas, self.clfs):
            total += alpha * clf.classify(integral_image)
        return 1 if total >= 0.5 * sum(self.alphas) else 0

    """ Save and Load functions copypasta'd from parandea17/FaceDetection github repo """

    def save(self, filename):
        """
        Saves the classifier to a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename+".pkl", 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        A static method which loads the classifier from a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename+".pkl", 'rb') as f:
            return pickle.load(f)


class WeakClassifier:
    def __init__(self, index, feature, lower_threshold_index, lower_threshold_value, upper_threshold_index, upper_threshold_value):
        self.index = index
        self.feature = feature
        self.lower_threshold_index = lower_threshold_index
        self.lower_threshold_value = lower_threshold_value
        self.upper_threshold_index = upper_threshold_index
        self.upper_threshold_value = upper_threshold_value

    def __repr__(self):
        return "WeakClassifier %s:\n\tLower Threshold @ index %s is %s\n\tUpper Threshold @ index %s is %s" % (self.index, self.lower_threshold_index, self.lower_threshold_value, self.upper_threshold_index, self.upper_threshold_value)

    def __str__(self):
        return "WeakClassifier %s:\n\tLower Threshold @ index %s is %s\n\tUpper Threshold @ index %s is %s" % (self.index, self.lower_threshold_index, self.lower_threshold_value, self.upper_threshold_index, self.upper_threshold_value)

# class WeakClassifier:
#     def __init__(self, feature, threshold, true_negative, true_positive, false_positive, false_negative):
#         self.feature = feature
#         self.threshold = threshold  # (sample index, threshold value)
#         # self.gini = gini
#         # self.alpha = alpha
#         self.true_positive = true_positive
#         self.true_negative = true_negative
#         self.false_positive = false_positive
#         self.false_negative = false_negative
#         # self.polarity = threshold

    # def __repr__(self):
    #     return "<WeakClassifier:\tThreshold %s True_Negative %s True_Positive %s False_Positive %s False Negative %s>" % (self.threshold, self.true_negative, self.true_positive, self.false_positive, self.false_negative)

    # def __str__(self):
    #     return "WeakClassifier: Threshold: %s True_Negative: %s" % (self.threshold, self.true_negative)

    # This is the h(y, f, p, theta) function being calculated
    # def classify(self, y):
    #     def feature(ii): return sum([pos.compute_feature(ii) for pos in self.feature.haar_pos]) - sum(
    #         [neg.compute_feature(ii) for neg in self.feature.haar_neg])
    #     return 1 if (self.polarity * feature(y) < self.polarity * self.threshold) else 0

# Create the tables:
# 1.) 3D table via dictionary holding 2D dataframes: one dataframe for each 50 images/ 2880 features total/ start point, end point, feature type, calculated value
# 2.) 2D dataframe: (50) images/ positive-threshold, negative-threshold
# 3.) 2D dataframe: (50) images/ hit-rate


def create_metadata_table(size):
    # 1.)
    dataframe_collectioon = {}
    column_list = ["start_Y", "start_X", "end_Y",
                   "end_X", "feature_type", "value", "actual"]
    for x in range(0, size):  # 0 to 49th, since we have 50 images in training-set
        framename = "image_" + str(x) + "_data"
        dataframe_collectioon[framename] = pd.DataFrame(columns=column_list)
    # 2.)
    threshold_df = pd.DataFrame(
        columns=['negative_threshold', 'positive_threshold'])
    # print(threshold_df)
    # 3.)
    hit_rate_df = pd.DataFrame(columns=['hit_rate'])
    # print(hit_rate_df)
    return dataframe_collectioon, hit_rate_df, threshold_df


def score_keeping(dataframe_collectioon, X_list, im_feature_label):
    # feature_combined = features[0]+features[1]+features[2]+features[3]
    c = 0
    for key in dataframe_collectioon.keys():
        for x in range(0, len(X_list)):
            row = []
            if (im_feature_label[c][x].feature_type in [0, 1, 2, 3]):
                data = im_feature_label[c][x].get_size() + \
                    [im_feature_label[c][x].feature_type, X_list[x][c],
                     im_feature_label[c][x].pos_neg]
                row = [
                    pd.Series(data, index=dataframe_collectioon.get(key).columns)]
            else:
                # This should never be reached since each feature was initialize as a/b/c/d type
                print("ERROR: There shouldn't be a feature with type not a/b/c/d\t",
                      im_feature_label[c][x].feature_type)
            dataframe_collectioon[key] = dataframe_collectioon[key].append(
                row, ignore_index=True)
        dataframe_collectioon[key].to_csv('applied/df_' + str(c) + '.csv',
                                          sep='|', index=False)
        c += 1
    # print(dataframe_collectioon)
    return dataframe_collectioon


def print_score(dataframe_collectioon):
    for key in dataframe_collectioon.keys():
        print("\n" + "="*40)
        print(key)
        print(dataframe_collectioon[key].columns.tolist())
        print("-"*40)
        print(dataframe_collectioon[key])


# Running here:
# path = "database0/training_set/eye_table.bin"
# minmax = min_max_eye(path)
# print(minmax)
# # correct = read_metadata(path)    # Returns the correct eye-box of each image
# path = 'database0/training_set/*.bmp'
# image_list = import_image(path)
# # num_image = len(image_list)
# normalized_list = max_normalize(image_list)
# # Refactor the data back to before it was multiplied by 2^8
# # for x in range(len(normalized_list)):
# #     # assert np.all(n_list[x] == ret[x]), "Normalized image read not matching with original normalized!"
# #     # Convert back from the multiplication of 2^16
# #     normalized_list[x] /= pow(2, 16)
# ii_list = integral_image(normalized_list)
# # Only needs to be done once since all images have same dimensions
# features = ViolaJones().build_features(ii_list[0].shape, minmax)
# feature_combined = features[0]+features[1]+features[2]+features[3]
# im_feature = []
# for x in range(50):
#     im_feature.append(feature_combined)
# print('Each feature type has:\t', len(features[0]), len(features[1]), len(features[2]), len(features[3]))
# print('For total of\t', len(features[0])+ len(features[1])+ len(features[2])+ len(features[3]))
# with open("features.txt", "wb") as f:
#     pickle.dump(features, f)
# im_feature_label = []
# total_pos = 0
# total_neg = 0
# for x in range(50):
#     print("image ", x)
#     temp = ViolaJones().label_neg_pos_features(features, correct[x])
#     total_pos += temp[2]
#     total_neg += temp[1]
#     im_feature_label.append(temp)   #[2880 features, #neg, #pos] for each image
# print("total neg/pos\t", total_neg, total_pos, "\t", (total_pos*100)/(len(features)*50), "% Positive" )

# ViolaJones().train("database0/training_set/eye_table.bin", ii_list)
# try:
#     os.makedirs("/output")  # Making Folder if not exists
#     # os.makedirs("/feature_graphs")
#     # print("Succeeded!")
# except FileExistsError as e:
#     print(e)
#     pass
image_path, metadata_path, foldername = 'data/database0/training_set/', 'data/database0/training_set/eye_table.bin', "output"
strong_classifier = ViolaJones(2880)
""" Step 0, Finding everything we'll need to run the adaboosting algorithm as described in the viola_jones_2.pdf original document """
print("0.) Starting Prep")
minmax = min_max_eye(metadata_path)
print(minmax)
correct = read_metadata(metadata_path)
image_list = import_image(image_path, 50)
normalized_list = max_normalize(image_list)
# for x in range(len(normalized_list)):
#     # Refactor the data back to before it was multiplied by 2^16
#     normalized_list[x] /= pow(2, 16)
ii_list = integral_image(normalized_list)
features = strong_classifier.build_features(ii_list[0].shape, minmax)
print("Number of features generated is %i" % len(features))
with open(foldername+"/feature_table.txt", "w") as f:
    for item in features:
        f.write("%s\n" % item)
indexed_feature_table = list(enumerate(features))
with open("output/indexed_feature_table.txt", "w") as f:
    for index, item in indexed_feature_table:
        f.write("Index %i->%s\n" % (index, item))
im_feature_label, feature_stat, y_list, pos_stat, neg_stat = strong_classifier.label_features(
    features, correct)
with open(foldername+"/feature_stat.txt", "w") as f:
    for row in feature_stat:
        f.write("%s\n" % row)
X_list, sorted_X_list = strong_classifier.apply_features(
    features, ii_list)   # X_list is already positive_X list because only useful features were passed in
with open(foldername+"/X_list.txt", "w") as f:
    for item in X_list:
        f.write("%s\n" % item)
with open(foldername+"/sorted_X_list.txt", "w") as f:
    for item in sorted_X_list:
        f.write("%s\n" % item)

""" 
Plot the not-sorted feature graphs for verification
Plotting either 2880 sorted or 2880 not-sorted takes about 40+ minutes each
"""
strong_classifier.plot_graphs("not_sorted", X_list, pos_stat)
# temp_list = X_list[0:10]
# strong_classifier.plot_graphs("verify", temp_list, pos_stat)

weights = strong_classifier.initialize_weights(feature_stat, y_list)
with open(foldername+"/weights.txt", "w") as f:
    for item in weights:
        f.write("%s\n" % item)
print("Prep Done")
print("Number of iterations to run is %i" % strong_classifier.T)
# Actually training below, which took 3 hours and 12 minutes
# format = indexes, alphas, errors, weak_classifiers
weak_classifier_list = strong_classifier.train(
    foldername, weights, sorted_X_list, y_list, pos_stat, neg_stat, features)
with open("output/weak_classifier_list.pkl", 'wb') as f:
    pickle.dump(weak_classifier_list, f)
strong_classifier.save(foldername+"/strong_classifier")
# strong_classifier_copy = strong_classifier.load(
#     foldername+"/strong_classifier")

# Since I already ran it once and has the indexes of the final weak classifiers ordered by best errors...
# final_clf_indexes = [line.rstrip('\n')
#                      for line in open("output_old/final_clf_indexes.txt")]
# counter, x = 0, 0
# lower_t, upper_t = [], []
# for index, line in enumerate(open("output_old/final_clfs.txt")):
#     if counter > 2:
#         three = 0
#     line = line.rstrip('\n')
#     if counter == 0:
#         if str(final_clf_indexes[x]) not in line:
#             print("ERROR: final_clf.txt's index not matching ")
#         x += 1
#     elif counter == 1:
#         print()

#     counter += 1


# correct = read_metadata('database0/training_set/eye_table.bin')
# with open("output/correct.txt", "w") as f:
#     for item in correct:
#         f.write("%s\n" % item)
# with open("output/correct_start_end.txt", "w") as f:
#     for item in correct:
#         start_end_point = [item[0], item[1], item[0]+item[2], item[1]+item[3]]
#         f.write("%s\n" % start_end_point)


# feature_table = [line.rstrip('\n') for line in open("output/feature_table.txt")]
# feature_table = []
# empty_line = False
# for line in open("output/feature_table.txt"):
#     if not empty_line:
#         feature_table.append(line.rstrip('\n'))
#     empty_line = not empty_line
# indexed_feature_table = list(enumerate(feature_table))
# with open("output/indexed_feature_table.txt", "w") as f:
#     for index, item in indexed_feature_table:
#         f.write("Index %s->%s\n" % (index, item))

""" Generate Alpha-Error Graph """
# # gp.c('set terminal pdf')
# # gp.c('set output "alpha_error_graph.pdf" ')
# alphas = [float(line.rstrip('\n')) for line in open("output/alphas.txt")]
# errors = [float(line.rstrip('\n')) for line in open("output/errs.txt")]
# betas = list(map(lambda ii: ii / (1 - ii) if ii < 1 else 15, errors))
# sum_alphas, sum_betas, sum_errors = sum(alphas), sum(betas), sum(errors)
# # min_alphas, max_alphas, min_errors, max_errors = min(
# #     alphas), max(alphas), min(errors), max(errors)
# print(sum_alphas, sum_betas, sum_errors)
# normalized_alphas = list(map(lambda ii: ii/sum_alphas, alphas))
# normalized_betas = list(map(lambda ii: ii/sum_betas, betas))
# normalized_errors = list(map(lambda ii: ii/sum_errors, errors))
# # normalized_alphas = list(
# #     map(lambda ii: (ii - min_alphas)/(max_alphas - min_alphas), alphas))
# # normalized_errors = list(
# #     map(lambda ii: (ii - min_errors)/(max_errors - min_errors), errors))
# with open("output/normalized_alphas.txt", "w") as f:
#     for item in normalized_alphas:
#         f.write("%s\n" % item)
# with open("output/normalized_betas.txt", "w") as f:
#     for item in normalized_betas:
#         f.write("%s\n" % item)
# with open("output/normalized_errs.txt", "w") as f:
#     for item in normalized_errors:
#         f.write("%s\n" % item)
# # gp.c('plot \
# #     "output/normalized_alphas.txt" title "alpha" with linespoints, \
# #     "output/normalized_errs.txt" title "error" with linespoints')
# # gp.c('set title "Alpha-Error Graph (Linespoints)" ')
# # gp.c('plot \
# #     "output/normalized_alphas.txt" title "alpha" with linespoints, \
# #     "output/normalized_betas.txt" title "beta" with linespoints')
# # gp.c('set title "Alpha-Beta Graph (Linespoints)" ')
# gp.c('plot \
#     "output/normalized_alphas.txt" title "alpha" with linespoints, \
#     "output/normalized_betas.txt" title "beta" with linespoints, \
#     "output/normalized_errs.txt" title "error" with linespoints ')
# gp.c('set title "Alpha-Beta-Error Graph (Linespoints)" ')
# gp.c('set xlabel "Image Index" ')
# gp.c('set ylabel "Feature Value" ')
# clf_indexes = [line.rstrip('\n')
#                for line in open("output/final_clf_indexes.txt")]
# xtics = 'set xtics add ('
# for index, index_label in enumerate(clf_indexes):
#     xtics += '"' + index_label + '" ' + str(index)
#     if index != (len(clf_indexes) - 1):
#         xtics += ','
# xtics += ') rotate'
# gp.c(xtics)
# # gp.c('save "output/alpha_error.dat" ')
# # gp.c('save "output/alpha_beta.dat" ')
# gp.c('save "output/alpha_beta_error.dat" ')
""" Since alpha-error-graph already generated and saved, just load again """
# gp.c('load "output/alpha_error.dat" ')
# gp.c('load "output/alpha_beta.dat" ')
# gp.c('load "output/alpha_beta_error.dat" ')


# X_list = []
# for x in range(len(features)):  # 4, one for each feature type
#     X, x = ViolaJones().apply_features(features[x], ii_list)     # Applying feature A/B/C/D to integral images
#     print('Total applied feature score:\t', X.shape)  # 2340 y 50, 2340 for each image since there's 2340 feature A
#     np.savetxt("applied/X" + str(x) + ".txt", X)
#     X_list = X_list + (X.tolist())
# X_list, x = ViolaJones().apply_features(features, ii_list)
# print(X_list.shape)
# dataframe_collectioon, hit_rate_df, threshold_df = create_metadata_table(num_image)
# dataframe_collectioon = score_keeping(dataframe_collectioon, X_list, im_feature_label)
# print_score(dataframe_collectioon)

# Start training/Adaboost here

print("--- %s seconds ---" % (time.time() - start_time))


# %%
