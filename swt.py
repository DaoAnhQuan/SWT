# -*- encoding: utf-8 -*-
from __future__ import division
from collections import defaultdict
import hashlib
import math
import os

import numpy as np
import cv2
import scipy.sparse, scipy.spatial

import train_cnn



diagnostics = True


class SWTScrubber(object):
    @classmethod
    def scrub(cls, filepath):
        """
        Apply Stroke-Width Transform to image.
        :param filepath: relative or absolute filepath to source image
        :return: numpy array representing result of transform
        """
        canny, sobelx, sobely, theta = cls._create_derivative(filepath)
        swt,rays = cls._swt(theta, canny, sobelx, sobely)
        swt = SWT_clean(swt,rays)
        contours = cls._connect_components(swt)
        swts, heights, widths, topleft_pts, images,labels = cls._find_letters(swt, contours)
        """word_images = cls._find_words(swts, heights, widths, topleft_pts, images,labels)

        final_mask = np.zeros((len(word_images),swt.shape[0],swt.shape[1]))
        i = 0;
        for word_image in word_images:
            for word in word_image:
                final_mask[i] += word
            i+=1
        return final_mask"""

    @classmethod
    def _create_derivative(cls, filepath):
        img = cv2.imread(filepath,0)
        edges = cv2.Canny(img, 175, 320, apertureSize=3)
        # Create gradient map using Sobel
        sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=-1)
        sobely64f = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=-1)

        theta = np.arctan2(sobely64f, sobelx64f)
        if diagnostics:
            cv2.imwrite('Image\\edges.jpg',edges)
            cv2.imwrite('Image\\sobelx64f.jpg', np.absolute(sobelx64f))
            cv2.imwrite('Image\\sobely64f.jpg', np.absolute(sobely64f))
            # amplify theta for visual inspection
            theta_visible = (theta + np.pi)*255/(2*np.pi)
            cv2.imwrite('Image\\theta.jpg', theta_visible)
        return (edges, sobelx64f, sobely64f, theta)

    @classmethod
    def _swt(self, theta, edges, sobelx64f, sobely64f):
        # create empty image, initialized to infinity
        swt = np.empty(theta.shape)
        swt[:] = np.Infinity
        rays = []

        

        # now iterate over pixels in image, checking Canny to see if we're on an edge.
        # if we are, follow a normal a ray to either the next edge or image border
        # edgesSparse = scipy.sparse.coo_matrix(edges)
        step_x_g = -1 * sobelx64f
        step_y_g = -1 * sobely64f
        mag_g = np.sqrt( step_x_g * step_x_g + step_y_g * step_y_g )
        rows, columnes = mag_g.shape 
        grad_x_g, grad_y_g = np.zeros((rows,columnes)),np.zeros((rows,columnes))
        for x in range(rows):
            for y in range(columnes):
                if mag_g[x,y] != 0:
                    grad_x_g[x,y] = step_x_g[x,y] / mag_g[x,y]
                    grad_y_g[x,y] = step_y_g[x,y] / mag_g[x,y]

        for x in range(1,edges.shape[1]-1):
            for y in range(1,edges.shape[0]-1):
                if edges[y, x] > 0:
                    step_x = step_x_g[y, x]
                    step_y = step_y_g[y, x]
                    mag = mag_g[y, x]
                    grad_x = grad_x_g[y, x]
                    grad_y = grad_y_g[y, x]
                    ray = []
                    ray.append((x, y))
                    prev_x, prev_y, i = x, y, 1
                    while True:
                        i += 1
                        #if grad_x == 0 and grad_y == 0:
                            #print (str(x) + "   " + str(y))
                        cur_x = math.floor(x + grad_x * i)
                        cur_y = math.floor(y + grad_y * i)

                        if cur_x != prev_x or cur_y != prev_y:
                            # we have moved to the next pixel!
                            try:
                                if edges[cur_y, cur_x] > 0:
                                    # found edge,
                                    ray.append((cur_x, cur_y))
                                    theta_point = theta[y, x]
                                    alpha = theta[cur_y, cur_x]
                                    try:
                                        if math.acos(grad_x * -grad_x_g[cur_y, cur_x] + grad_y * -grad_y_g[cur_y, cur_x]) < np.pi/2.0:
                                            thickness = math.sqrt( (cur_x - x) * (cur_x - x) + (cur_y - y) * (cur_y - y) )
                                            for (rp_x, rp_y) in ray:
                                                swt[rp_y, rp_x] = min(thickness, swt[rp_y, rp_x])
                                            rays.append(ray)
                                        break
                                    except ValueError:
                                        break
                                # this is positioned at end to ensure we don't add a point beyond image boundary
                                ray.append((cur_x, cur_y))
                            except IndexError:
                                # reached image boundary
                                break
                            prev_x = cur_x
                            prev_y = cur_y

        # Compute median SWT
        for ray in rays:
            median = np.median([swt[y, x] for (x, y) in ray])
            for (x, y) in ray:
                swt[y, x] = min(median, swt[y, x])
        if diagnostics:
            cv2.imwrite('Image\\swt.jpg', swt * 100)

        return swt,rays

    @classmethod
    def _connect_components(cls, swt):
        # STEP: Compute distinct connected components
        # Implementation of disjoint-set
        class Label(object):
            def __init__(self, value):
                self.value = value
                self.parent = self
                self.rank = 0
            def __eq__(self, other):
                if type(other) is type(self):
                    return self.value == other.value
                else:
                    return False
            def __ne__(self, other):
                return not self.__eq__(other)

        ld = {}

        def MakeSet(x):
            try:
                return ld[x]
            except KeyError:
                item = Label(x)
                ld[x] = item
                return item

        def Find(item):
            # item = ld[x]
            if item.parent != item:
                item.parent = Find(item.parent)
            return item.parent

        def Union(x, y):
            """
            :param x:
            :param y:
            :return: root node of new union tree
            """
            x_root = Find(x)
            y_root = Find(y)
            if x_root == y_root:
                return x_root

            if x_root.rank < y_root.rank:
                x_root.parent = y_root
                return y_root
            elif x_root.rank > y_root.rank:
                y_root.parent = x_root
                return x_root
            else:
                y_root.parent = x_root
                x_root.rank += 1
                return x_root

        # apply Connected Component algorithm, comparing SWT values.
        # components with a SWT ratio less extreme than 1:3 are assumed to be
        # connected. Apply twice, once for each ray direction/orientation, to
        # allow for dark-on-light and light-on-dark texts
        trees = {}
        # Assumption: we'll never have more than 65535-1 unique components
        label_map = np.zeros(shape=swt.shape, dtype=np.uint16)
        next_label = 1
        # First Pass, raster scan-style
        swt_ratio_threshold = 3.0
        for y in range(swt.shape[0]):
            for x in range(swt.shape[1]):
                sw_point = swt[y, x]
                if sw_point < np.Infinity and sw_point > 0:
                    neighbors = [(y, x-1),   # west
                                 (y-1, x-1), # northwest
                                 (y-1, x),   # north
                                 (y-1, x+1)] # northeast
                    connected_neighbors = None
                    neighborvals = []

                    for neighbor in neighbors:
                        # west
                        try:
                            sw_n = swt[neighbor]
                            label_n = label_map[neighbor]
                        except IndexError:
                            continue
                        if label_n > 0 and sw_n / sw_point < swt_ratio_threshold and sw_point / sw_n < swt_ratio_threshold:
                            neighborvals.append(label_n)
                            if connected_neighbors:
                                connected_neighbors = Union(connected_neighbors, MakeSet(label_n))
                            else:
                                connected_neighbors = MakeSet(label_n)

                    if not connected_neighbors:
                        # We don't see any connections to North/West
                        trees[next_label] = (MakeSet(next_label))
                        label_map[y, x] = next_label
                        next_label += 1
                    else:
                        # We have at least one connection to North/West
                        label_map[y, x] = min(neighborvals)
                        # For each neighbor, make note that their respective connected_neighbors are connected
                        # for label in connected_neighbors. @todo: do I need to loop at all neighbor trees?
                        trees[connected_neighbors.value] = Union(trees[connected_neighbors.value], connected_neighbors)

        # Second pass. re-base all labeling with representative label for each connected tree
        contours = defaultdict(list)
        for y in range(swt.shape[0]):
            for x in range(swt.shape[1]):
                if label_map[y, x] > 0:
                    item = ld[label_map[y, x]]
                    common_label = Find(item).value
                    label_map[y, x] = common_label
                    contours[common_label].append([x, y])
        return contours

    @classmethod
    def _find_letters(cls, swt, contours):
        # STEP: Discard shapes that are probably not letters
        swts = []
        heights = []
        widths = []
        topleft_pts = []
        images = []
        labels = []
        
        for label,contour in contours.items():
            #layer = make_layer(contours,swt)
            #(nz_y, nz_x) = np.nonzero(layer)
            contour = np.array(contour,dtype = 'int32')
            mean,variance = calculate_variance(contour,swt)
            #if variance > 0.5*mean:
                #continue
            nz_y = contour[:,1]
            nz_x = contour[:,0]
            east, west, south, north = max(nz_x), min(nz_x), max(nz_y), min(nz_y)
            width, height = east - west, south - north

            if width < 4 or height < 4:
                print('qua hep')
                continue

            if width / height > 20 or height / width > 20:
                print('qua dai or qua det')
                continue

            pxld = pixel_density(contour,width,height)
            if pxld < 30 or pxld > 90:
                print('mat do anh'+str(pxld))
                
                continue

            diameter = math.sqrt(width * width + height * height)
            median_swt = np.median(swt[(nz_y, nz_x)])
            if diameter / median_swt > 20 or diameter/median_swt < 1.5:
                print('duong cheo')
                continue

            """if width / swt.shape[1] > 0.4 or height / swt.shape[0] > 0.4:
                #print('e4')
                continue"""

            layer = make_layer(contour,swt)
            if diagnostics:
                #print(label)
                resized_image = SWTScrubber.change_size((28,28),east, west, south, north,layer)
                #train_cnn.main(resized_image)
                cv2.imwrite('Image\\layer'+ str(label) +'.jpg', resized_image)
                

            # we use log_base_2 so we can do linear distance comparison later using k-d tree
            # ie, if log2(x) - log2(y) > 1, we know that x > 2*y
            # Assumption: we've eliminated anything with median_swt == 1
            swts.append([math.log(median_swt, 2)])
            heights.append([math.log(height, 2)])
            topleft_pts.append(np.asarray([north, west]))
            widths.append(width)
            images.append(layer)
            labels.append(label)
        for label in labels:
            print(label)
        return swts, heights, widths, topleft_pts, images,labels

    @classmethod
    def _find_words(cls, swts, heights, widths, topleft_pts, images,labels):
        # Find all shape pairs that have similar median stroke widths
        
        swt_tree = scipy.spatial.KDTree(np.asarray(swts))
        stp = swt_tree.query_pairs(1)

        # Find all shape pairs that have similar heights
        height_tree = scipy.spatial.KDTree(np.asarray(heights))
        htp = height_tree.query_pairs(1)

        # Intersection of valid pairings
        isect = htp.intersection(stp)

        chains = []
        pairs = []
        pair_angles = []
        for pair in isect:
            left = pair[0]
            right = pair[1]
            widest = max(widths[left], widths[right])
            #distance = np.linalg.norm(topleft_pts[left] - topleft_pts[right])
            delta_yx = topleft_pts[left] - topleft_pts[right]
            distance = abs(delta_yx[1])
            if distance < widest*1.5:
                
                angle = np.arctan2(delta_yx[0], delta_yx[1])
                #if angle < 0:
                    #angle += np.pi

                pairs.append(pair)
                pair_angles.append(np.asarray([angle]))

        angle_tree = scipy.spatial.KDTree(np.asarray(pair_angles))
        atp = angle_tree.query_pairs(np.pi/12)

        for pair_idx in atp:
            pair_a = pairs[pair_idx[0]]
            pair_b = pairs[pair_idx[1]]
            left_a = pair_a[0]
            right_a = pair_a[1]
            left_b = pair_b[0]
            right_b = pair_b[1]

            # @todo - this is O(n^2) or similar, extremely naive. Use a search tree.
            added = False
            for chain in chains:
                if left_a in chain:
                    chain.add(right_a)
                    added = True
                elif right_a in chain:
                    chain.add(left_a)
                    added = True
            if not added:
                chains.append(set([left_a, right_a]))
            added = False
            for chain in chains:
                if left_b in chain:
                    chain.add(right_b)
                    added = True
                elif right_b in chain:
                    chain.add(left_b)
                    added = True
            if not added:
                chains.append(set([left_b, right_b]))

        word_images = []
        i = 0
        for chain in [c for c in chains]:
            word_images.append([])
            #print('word ' +str(i))
            for idx in chain:
                #print(idx)
                word_images[i].append(images[idx])
            i += 1
        """word_images = []
        word = []
        word_list = []
        while pairs:
            pair = pairs[0]
            left = pair[0]
            right = pair[1]
            word.append(left,right)
            for part in pairs:
                if """
        

        return word_images
    
    def change_size(size,east, west, south, north,layer):
        letter = layer[north:south+1,west:east+1]*255
        dim = (size[0]-8,size[1]-8)
        resized = cv2.resize(letter,dim,interpolation = cv2.INTER_AREA)
        result = np.zeros(size)
        result[4:24,4:24]=resized
        return result

def calculate_variance(contour,swt):
    values = []
    length = len(contour)
    for i in range(length):
        values.append(swt[contour[i,1],contour[i,0]])
    mean = np.mean(values)
    variance = np.var(values)
    return mean,variance
    
def make_layer(contour,swt):
    layer = np.zeros(swt.shape)
    length = len(contour)
    for i in range(length):
        layer[contour[i,1],contour[i,0]]=1
    return layer

def SWT_clean(image,rays):
    #x,y = image.shape
    swt_clean = np.copy(image)
    """for i in range(x):
        for j in range(y):
            if image[i,j] > 0 and image[i,j] < np.Infinity:
                neighbors = [(i-1,j+1),(i-1,j),(i-1,j-1),(i,j+1),(i,j-1),(i,j+1),(i+1,j+1),(i+1,j),(i+1,j-1)]
                count = 0
                nonzero = [image[i,j]]
                for neighbor in neighbors:
                    try:
                        if image[neighbor] > 0 and image[neighbor] < np.Infinity:
                            count +=1
                            nonzero.append(image[neighbor])
                    except IndexError:
                        continue
                median = np.median(nonzero)
                if count > 3:
                    for neighbor in neighbors:
                        try:
                            if swt_clean[neighbor] == np.Infinity or swt_clean[neighbor] == 0:
                                swt_clean[neighbor] = median
                        except IndexError:
                            continue """
    for ray in rays:
        for (j,i) in ray[1:len(ray)-2]:
            neighbors = [(i-1,j+1),(i-1,j),(i-1,j-1),(i,j+1),(i,j-1),(i,j+1),(i+1,j+1),(i+1,j),(i+1,j-1)]
            for neighbor in neighbors:
                try:
                    if swt_clean[neighbor] == np.Infinity or swt_clean[neighbor] == 0:
                        swt_clean[neighbor] = swt_clean[i,j]
                except IndexError:
                    continue
    cv2.imwrite('Image\\clean.jpg',swt_clean*100)
    return swt_clean

def pixel_density(contour,width,height):
    no_pixel = len(contour)
    return no_pixel/((width+1)*(height+1))*100
            
#final_mask = SWTScrubber.scrub('wallstreetsmd.jpeg')
final_mask = SWTScrubber.scrub('Data\\test54.jpg')
# final_mask = cv2.GaussianBlur(final_mask, (1, 3), 0)
# cv2.GaussianBlur(sobelx64f, (3, 3), 0)
"""i = 0
for image in final_mask:
    cv2.imwrite('Image\\final'+str(i)+'.jpg', final_mask[i] * 255)
    i+=1"""
