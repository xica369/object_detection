#!/usr/bin/env python3

"""Initialize Yolo"""

import tensorflow.keras as K
import numpy as np
import glob
import cv2
import os
import re


class Yolo:
    """class Yolo"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        - model_path is the path to where a Darknet Keras model is stored
        - classes_path is the path to where the list of class names used
          for the Darknet model, listed in order of index, can be found
        - class_t is a float representing the box score threshold for
          the initial filtering step
        - nms_t is a float representing the IOU threshold for non-max
          suppression
        - anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
          containing all of the anchor boxes:
          * outputs is the number of outputs (predictions)
            made by the Darknet model
          * anchor_boxes is the number of anchor boxes used for each prediction
          * 2 => [anchor_box_width, anchor_box_height]
        """

        self.model = K.models.load_model(model_path)

        self.class_names = []
        with open(classes_path, "r") as file:
            for line in file:
                class_name = line.strip()
                self.class_names.append(class_name)

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        public method
        - outputs is a list of numpy.ndarrays containing the predictions from
          the Darknet model for a single image:
          Each output will have the shape:
          (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
            * grid_height & grid_width => the height and width of the grid used
              for the output
            * anchor_boxes => the number of anchor boxes used
            * 4 => (t_x, t_y, t_w, t_h)
            * 1 => box_confidence
            * classes => class probabilities for all classes
        - image_size is a numpy.ndarray containing the image’s original size
          [image_height, image_width]

        Returns a tuple of (boxes, box_confidences, box_class_probs):
        - boxes: a list of numpy.ndarrays of shape
          (grid_height, grid_width, anchor_boxes, 4) containing the processed
          boundary boxes for each output, respectively:
          * 4 => (x1, y1, x2, y2)
          * (x1, y1, x2, y2) should represent the boundary box relative to
            original image
        - box_confidences: a list of numpy.ndarrays of shape
          (grid_height, grid_width, anchor_boxes, 1) containing the box
          confidences for each output, respectively
        - box_class_probs: a list of numpy.ndarrays of shape
          (grid_height, grid_width, anchor_boxes, classes) containing the box’s
          class probabilities for each output, respectively
        """

        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height = image_size[0]
        image_width = image_size[1]

        input_height = self.model.input.shape[2]
        input_width = self.model.input.shape[1]

        for outp in outputs:

            # create list with np.ndarray (grid_h, grid_w, anchor_boxes, 4)
            out = outp[..., :4]
            boxes.append(out)

            # calculate confidences for each output
            box_confidence = 1 / (1 + np.exp(-(outp[..., 4:5])))
            box_confidences.append(box_confidence)

            # calcule class probabilities for each output
            box_class_prob = 1 / (1 + np.exp(-(outp[..., 5:])))
            box_class_probs.append(box_class_prob)

        for iter, box in enumerate(boxes):
            grid_hight = box.shape[0]
            grid_width = box.shape[1]
            anchors_box = box.shape[2]

            # create matrix Cy
            matrix_cy = np.arange(grid_hight).reshape(1, grid_hight)
            matrix_cy = np.repeat(matrix_cy, grid_width, axis=0).T
            matrix_cy = np.repeat(matrix_cy[:, :, np.newaxis],
                                  anchors_box,
                                  axis=2)

            # create matrix Cx
            matrix_cx = np.arange(grid_width).reshape(1, grid_width)
            matrix_cx = np.repeat(matrix_cx, grid_hight, axis=0)
            matrix_cx = np.repeat(matrix_cx[:, :, np.newaxis],
                                  anchors_box,
                                  axis=2)

            # calculate sigmoid to tx and ty
            box[..., :2] = 1 / (1 + np.exp(-(box[..., :2])))

            # calculate bx = sigmoid(tx) + Cx
            box[..., 0] += matrix_cx

            # calculate by = sigmoid(ty) + Cy
            box[..., 1] += matrix_cy

            anchor_width = self.anchors[iter, :, 0]
            anchor_hight = self.anchors[iter, :, 1]

            # calculate e(tw) and e(th)
            box[..., 2:] = np.exp(box[..., 2:])

            # calculate bw = anchor_width * e(tw)
            box[..., 2] *= anchor_width

            # calculate bh = anchor_hight * e(th)
            box[..., 3] *= anchor_hight

            # adjust scale
            box[..., 0] *= image_width / grid_width
            box[..., 1] *= image_height / grid_hight
            box[..., 2] *= image_width / input_width
            box[..., 3] *= image_height / input_height

            # calculate x1 = tx - bw / 2
            box[..., 0] -= box[..., 2] / 2

            # calculate y1 = ty - bh / 2
            box[..., 1] -= box[..., 3] / 2

            # calculate x2 = x1 + bw
            box[..., 2] += box[..., 0]

            # calculate y2 = y1 + bh
            box[..., 3] += box[..., 1]

        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        public method to Filter Boxes
        - boxes: a list of numpy.ndarrays of shape
          (grid_height, grid_width, anchor_boxes, 4) containing the processed
          boundary boxes for each output, respectively
        - box_confidences: a list of numpy.ndarrays of shape
          (grid_height, grid_width, anchor_boxes, 1) containing the processed
          box confidences for each output, respectively
        - box_class_probs: a list of numpy.ndarrays of shape
          (grid_height, grid_width, anchor_boxes, classes) containing the
          processed box class probabilities for each output, respectively

        Returns a tuple of (filtered_boxes, box_classes, box_scores):
        - filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of
          the filtered bounding boxes:
        - box_classes: a numpy.ndarray of shape (?,) containing the class
          number that each box in filtered_boxes predicts, respectively
        - box_scores: a numpy.ndarray of shape (?) containing the box scores
          for each box in filtered_boxes, respectively
        """

        class_t = self.class_t
        filtered_boxes = []
        box_classes = []
        box_scores = []

        scores = []
        for box_confid, box_class_prob in zip(box_confidences,
                                              box_class_probs):
            scores.append(box_confid * box_class_prob)

        for score in scores:

            box_score = score.max(axis=-1)
            box_score = box_score.flatten()
            box_scores.append(box_score)

            box_class = np.argmax(score, axis=-1)
            box_class = box_class.flatten()
            box_classes.append(box_class)

        box_scores = np.concatenate(box_scores, axis=-1)
        box_classes = np.concatenate(box_classes, axis=-1)

        for box in boxes:
            filtered_boxes.append(box.reshape(-1, 4))

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)

        filtering_mask = np.where(box_scores >= self.class_t)

        filtered_boxes = filtered_boxes[filtering_mask]
        box_classes = box_classes[filtering_mask]
        box_scores = box_scores[filtering_mask]

        return (filtered_boxes, box_classes, box_scores)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        public method that applies Non-max Suppression
        - filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of
          the filtered bounding boxes:
        - box_classes: a numpy.ndarray of shape (?,) containing the class
          number for the class that filtered_boxes predicts, respectively
        - box_scores: a numpy.ndarray of shape (?) containing the box scores
          for each box in filtered_boxes, respectively

        Returns a tuple of
        (box_predictions, predicted_box_classes, predicted_box_scores):
        - box_predictions: a numpy.ndarray of shape (?, 4) containing all of
          the predicted bounding boxes ordered by class and box score
        - predicted_box_classes: a numpy.ndarray of shape (?,) containing the
          class number for box_predictions ordered by class and box score,
          respectively
        - predicted_box_scores: a numpy.ndarray of shape (?) containing the
          box scores for box_predictions ordered by class and box score,
          respectively
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for classes in set(box_classes):
            index = np.where(box_classes == classes)

            filtered = filtered_boxes[index]
            scores = box_scores[index]
            classe = box_classes[index]

            x1 = filtered[:, 0]
            x2 = filtered[:, 2]
            y1 = filtered[:, 1]
            y2 = filtered[:, 3]

            keep = []
            area = (x2 - x1) * (y2 - y1)
            index_list = np.flip(scores.argsort(), axis=0)

            while len(index_list) > 0:
                pos1 = index_list[0]
                pos2 = index_list[1:]
                keep.append(pos1)

                xx1 = np.maximum(x1[pos1], x1[pos2])
                yy1 = np.maximum(y1[pos1], y1[pos2])
                xx2 = np.minimum(x2[pos1], x2[pos2])
                yy2 = np.minimum(y2[pos1], y2[pos2])

                height = np.maximum(0.0, yy2 - yy1)
                width = np.maximum(0.0, xx2 - xx1)

                intersection = (width * height)
                union = area[pos1] + area[pos2] - intersection
                iou = intersection / union
                below_threshold = np.where(iou <= self.nms_t)[0]
                index_list = index_list[below_threshold + 1]

            keep = np.array(keep)

            box_predictions.append(filtered[keep])
            predicted_box_classes.append(classe[keep])
            predicted_box_scores.append(scores[keep])

        box_predictions = np.concatenate(box_predictions)
        predicted_box_classes = np.concatenate(predicted_box_classes)
        predicted_box_scores = np.concatenate(predicted_box_scores)

        return (box_predictions, predicted_box_classes, predicted_box_scores)

    @staticmethod
    def load_images(folder_path):
        """
        static method that load images
        - folder_path: a string representing the path to the folder holding all
          the images to load

        Returns a tuple of (images, image_paths):
        - images: a list of images as numpy.ndarrays
        - image_paths: a list of paths to the individual images in images
        """

        images = []

        image_paths = glob.glob(folder_path + "/*")

        for image in image_paths:
            img_read = cv2.imread(image)
            images.append(img_read)

        return (images, image_paths)

    def preprocess_images(self, images):
        """
        Public method that preprocess images
        - images: a list of images as numpy.ndarrays
        Resize the images with inter-cubic interpolation
        Rescale all images to have pixel values in the range [0, 1]

        Returns a tuple of (pimages, image_shapes):
        - pimages: a numpy.ndarray of shape (ni, input_h, input_w, 3)
          containing all of the preprocessed images
          * ni: the number of images that were preprocessed
          * input_h: the input height for the Darknet model Note: this can vary
            by model
          * input_w: the input width for the Darknet model Note: this can vary
            by model
          * 3: number of color channels
        - image_shapes: a numpy.ndarray of shape (ni, 2) containing the
          original height and width of the images
          * 2 => (image_height, image_width)
        """

        input_width = self.model.input.shape[1]
        input_height = self.model.input.shape[2]

        resize = (input_width, input_height)
        image_shapes = []
        pimages = []

        for image in images:
            shape = image.shape[:2]
            image_shapes.append(shape)

            image_resize = cv2.resize(image,
                                      resize,
                                      interpolation=cv2.INTER_CUBIC)

            image_rescaled = image_resize/255

            pimages.append(image_rescaled)

        pimages = np.stack(pimages, axis=0)
        image_shapes = np.stack(image_shapes, axis=0)

        return (pimages, image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Public method that show boxes:

        - image: a numpy.ndarray containing an unprocessed image
        - boxes: a numpy.ndarray containing the boundary boxes for the image
        - box_classes: a numpy.ndarray containing the class indices for each
          box
        - box_scores: a numpy.ndarray containing the box scores for each box
        - file_name: the file path where the original image is stored

        Displays the image with all boundary boxes, class names, and
        box scores (see example below):
        - Boxes should be drawn as with a blue line of thickness 2
        - Class names and box scores should be drawn above each box in red

        If the s key is pressed:
          - The image should be saved in the directory detections, located in
            the current directory
          - If detections does not exist, create it
          - The saved image should have the file name file_name
          - The image window should be closed

        If any key besides s is pressed, the image window should be closed
        without saving
        """

        for iter, box in enumerate(boxes):

            # Draw a Rectangle
            x1 = int(box[0])
            x2 = int(box[2])
            y1 = int(box[3])
            y2 = int(box[1])

            coordinates_top_left = (x1, y1)
            coordinates_lower_right = (x2, y2)
            stroke_color = (255, 0, 0)
            stroke_thickness = 2

            image = cv2.rectangle(image,
                                  coordinates_top_left,
                                  coordinates_lower_right,
                                  stroke_color,
                                  stroke_thickness)

            # Write text on images
            class_name = self.class_names[box_classes[iter]]
            score = box_scores[iter]
            text_written = ("{} {:.2f}".format(class_name, score))
            coordinates_start = (x1, y2 - 5)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_color = (0, 0, 255)
            text_thickness = 1
            type_line = cv2.LINE_AA

            image = cv2.putText(image,
                                text_written,
                                coordinates_start,
                                font,
                                font_scale,
                                text_color,
                                text_thickness,
                                type_line)

        # show image
        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)
        if key == ord("s"):
            if not os.path.exists("../static/detections"):
                os.makedirs("../static/detections")
            os.chdir("detections")
            cv2.imwrite(f'../static/detections/{file_name}', image)
            os.chdir("../")

        cv2.destroyAllWindows()


    def save_image(self, image, boxes, box_classes, box_scores, file_name, path_app="../"):
        """
        Public method that show boxes:

        - image: a numpy.ndarray containing an unprocessed image
        - boxes: a numpy.ndarray containing the boundary boxes for the image
        - box_classes: a numpy.ndarray containing the class indices for each
          box
        - box_scores: a numpy.ndarray containing the box scores for each box
        - file_name: the file path where the original image is stored

        Displays the image with all boundary boxes, class names, and
        box scores (see example below):
        - Boxes should be drawn as with a blue line of thickness 2
        - Class names and box scores should be drawn above each box in red

        If the s key is pressed:
          - The image should be saved in the directory detections, located in
            the current directory
          - If detections does not exist, create it
          - The saved image should have the file name file_name
          - The image window should be closed

        If any key besides s is pressed, the image window should be closed
        without saving
        """

        for iter, box in enumerate(boxes):

            # Draw a Rectangle
            x1 = int(box[0])
            x2 = int(box[2])
            y1 = int(box[3])
            y2 = int(box[1])

            coordinates_top_left = (x1, y1)
            coordinates_lower_right = (x2, y2)
            stroke_color = (255, 0, 0)
            stroke_thickness = 2

            image = cv2.rectangle(image,
                                  coordinates_top_left,
                                  coordinates_lower_right,
                                  stroke_color,
                                  stroke_thickness)

            # Write text on images
            class_name = self.class_names[box_classes[iter]]
            score = box_scores[iter]
            text_written = ("{} {:.2f}".format(class_name, score))
            coordinates_start = (x1, y2 - 5)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_color = (0, 0, 255)
            text_thickness = 1
            type_line = cv2.LINE_AA

            image = cv2.putText(image,
                                text_written,
                                coordinates_start,
                                font,
                                font_scale,
                                text_color,
                                text_thickness,
                                type_line)

        # creates an empty folder
        path_folder_dection = os.path.join(path_app, "static/images/detections")
        os.makedirs(path_folder_dection, exist_ok=True)

        # save image        
        cv2.imwrite(f'{path_folder_dection}/{file_name}', image)

    def predict(self, folder_path="static/images/upload", path_app="../"):
        """
        Public method predict:

        - folder_path: a string representing the path to the folder holding
          all the images to predict

        Displays all images using the show_boxes method

        Returns: a tuple of (predictions, image_paths):
        - predictions: a list of tuples for each image of
          (boxes, box_classes, box_scores)
        - image_paths: a list of image paths corresponding to each
          prediction in predictions
        """
        folder_path = os.path.join(path_app, folder_path)
        images, image_paths = self.load_images(folder_path)
        preprocess_images, image_shapes = self.preprocess_images(images)
        outputs = self.model.predict(preprocess_images)
        predictions = []

        for iter, image in enumerate(images):
            list_outputs = [outputs[0][iter, :, :, :, :],
                            outputs[1][iter, :, :, :, :],
                            outputs[2][iter, :, :, :, :]]

            image_size = np.array([image.shape[0], image.shape[1]])

            boxes, box_confidences, box_class_probs = self.process_outputs(
                list_outputs,
                image_size)

            filtered_boxes, box_classes, box_scores = self.filter_boxes(
                boxes,
                box_confidences,
                box_class_probs)

            b_predict, pred_classes, pred_scores = self.non_max_suppression(
                filtered_boxes,
                box_classes,
                box_scores)

            file_name = re.split(r"/|\\", image_paths[iter])[-1]
            
            predictions.append((b_predict, pred_classes, pred_scores, image, file_name))
            print(file_name)
            self.save_image(image,
                            b_predict,
                            pred_classes,
                            pred_scores,
                            file_name,
                            path_app)

            # self.show_boxes(image,
            #                 b_predict,
            #                 pred_classes,
            #                 pred_scores,
            #                 file_name)

        return (predictions, image_paths)
