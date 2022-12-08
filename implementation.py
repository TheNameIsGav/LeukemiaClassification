from dis import dis
import os
from cv2 import IMREAD_COLOR
import numpy as np
import cv2
import random
from sklearn_extra.cluster import KMedoids
from datetime import datetime
import matplotlib.pylab as plt
import math



#https://stackoverflow.com/questions/15408522/rgb-to-xyz-and-lab-colours-conversion
def bgr_2_lab(bgr_color, illuminant):
    #Convert to XYZ first
    (b, g, r) = bgr_color

    var_r = r / 255
    var_g = g / 255
    var_b = b / 255

    gamma = lambda x : ((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92

    var_r = gamma(var_r) * 100
    var_g = gamma(var_g) * 100
    var_b = gamma(var_b) * 100



    X = var_r * 0.4124 + var_g * 0.3576 + var_b * 0.1805
    Y = var_r * 0.2126 + var_g * 0.7152 + var_b * 0.0722
    Z = var_r * 0.0193 + var_g * 0.1192 + var_b * 0.9505

    ref_X, ref_Y, ref_Z = illuminant

    var_x = X / ref_X
    var_y = Y / ref_Y
    var_z = Z / ref_Z

    delta = lambda x: ((x ** (1/3)) if x > 0.008856 else (7.787 * x) + (16/116))

    var_x = delta(var_x)
    var_y = delta(var_y)
    var_z = delta(var_z)

    CIE_L = (116 * var_y) - 16
    CIE_a = 500 * (var_x - var_y)
    CIE_b = 200 * (var_y - var_z)

    return (CIE_L, CIE_a, CIE_b)

"""
Clusters images based on input
"""
def cluster(image):

    originalWidth = image.shape[1]
    originalHeight = image.shape[0]
    # print(f'Original (X: {originalWidth}, Y: {originalHeight})')

    scale_percent = 45 # percent of original size
    scaledWidth = int(originalWidth * scale_percent / 100)
    scaledHeight = int(originalHeight * scale_percent / 100)
    dim = (scaledHeight, scaledWidth)

    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # cv2.imshow("Resized: ", resized)

    # print(f'Modified (X: {scaledWidth}, Y: {scaledHeight})')

    vectorized_image = np.reshape(resized, (scaledWidth*scaledHeight, 3))

    # print("Testing")

    #Perform image segmentation using K-Medoids algorithm
    #cluster == medoid
    K = 4

    print(f"\t\tKMedoids Start: {datetime.now()}")
    KMobj = KMedoids(n_clusters=K, method='alternate', init='heuristic').fit(vectorized_image)
    print(f"\t\tKMedoids Ending: {datetime.now()}")
    labels = KMobj.labels_

    unq_lab = set(labels)
    #Generate Color's for the plots
    colors_plot = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unq_lab))]

    luminosity = []
    for k, col in zip(unq_lab, colors_plot):
        class_member_mask = labels == k
        xy = vectorized_image[class_member_mask]
        luminosity.append(xy)
        plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="white", markersize=10);

    #Setup clusters of luminosity levels
    cluster_0 = luminosity[0]
    cluster_1 = luminosity[1]
    cluster_2 = luminosity[2]
    cluster_3 = luminosity[3]
    #print(KMobj.cluster_centers_[:, 0], KMobj.cluster_centers_[:, 1])

    #Once we have the clusters, we can loop through them to find the images we want
    cluster_0_image = np.zeros((scaledWidth, scaledHeight, 3))
    cluster_1_image = np.zeros((scaledWidth, scaledHeight, 3))
    cluster_2_image = np.zeros((scaledWidth, scaledHeight, 3))
    cluster_3_image = np.zeros((scaledWidth, scaledHeight, 3))

    firstTime = datetime.now()

    print(f"\t\tCluster 0 Image Start Time: {datetime.now()}")
    cluster_0_start = datetime.now()
    #Go through the RGB image and match each pixel to the cluster
    for x in range(0, scaledWidth):
        for y in range(0, scaledHeight):
            [r, g, b] = resized[x][y]

            for slice in cluster_0:
                [a, d, e] = slice
                if r == a and g == d and e == b:
                    cluster_0_image[x][y] = [r, g, b]
    cluster_0_end = datetime.now()
    print(f"\t\tCluster 0 duration: {cluster_0_end - cluster_0_start}")

    # print(f"\t\tCluster 1 Image Start Time: {datetime.now()}")
    # cluster_1_start = datetime.now()
    # #Go through the RGB image and match each pixel to the cluster
    # for x in range(0, scaledWidth):
    #     for y in range(0, scaledHeight):
    #         [r, g, b] = resized[x][y]

    #         for slice in cluster_1:
    #             [a, d, e] = slice
    #             if r == a and g == d and e == b:
    #                 cluster_1_image[x][y] = [r, g, b]
    # cluster_1_end = datetime.now()
    # print(f"\t\tCluster 1 duration: {cluster_1_end - cluster_1_start}")

    # Removed for Efficiency
    # print(f"Cluster 2 Image Start Time: {datetime.now()}")
    # cluster_2_start = datetime.now()
    # #Go through the RGB image and match each pixel to the cluster
    # for x in range(0, scaledWidth):
    #     for y in range(0, scaledHeight):
    #         [r, g, b] = resized[x][y]

    #         for slice in cluster_2:
    #             [a, d, e] = slice
    #             if r == a and g == d and e == b:
    #                 cluster_2_image[x][y] = [r, g, b]
    # cluster_2_end = datetime.now()
    # print(f"Cluster 2 duration: {cluster_2_end - cluster_2_start}")

    # print(f"Cluster 3 Image Start Time: {datetime.now()}")
    # cluster_3_start = datetime.now()
    # #Go through the RGB image and match each pixel to the cluster
    # for x in range(0, scaledWidth):
    #     for y in range(0, scaledHeight):
    #         [r, g, b] = resized[x][y]

    #         for slice in cluster_3:
    #             [a, d, e] = slice
    #             if r == a and g == d and e == b:
    #                 cluster_3_image[x][y] = [r, g, b]
    # cluster_3_end = datetime.now()
    # print(f"Cluster 3 duration: {cluster_3_end - cluster_3_start}")

    finalTime = datetime.now()
    print(f"\t\tElapsed Clustering time: {finalTime-firstTime}")

    # cv2.imshow("Cluster 0: ", cluster_0_image) #Background Cluster
    # cv2.imshow("Cluster 1: ", cluster_1_image) #Cytoplasm (Blue is WBC, white is RBC)
    # cv2.imshow("Cluster 2: ", cluster_2_image) #Not sure
    # cv2.imshow("Cluster 3: ", cluster_3_image) #Nucleus of the WBCs

    # plt.plot(KMobj.cluster_centers_[:, 0], KMobj.cluster_centers_[:, 1], "o", markerfacecolor="orange", markeredgecolor="k", markersize=10);
    # plt.title("KMedoids Clustering", fontsize=14);
    # plt.show()

    return (cluster_0_image, cluster_1_image, cluster_2_image, cluster_3_image)

"""
Returns a plot of the white blood cell segmentation
"""
def cytoplasm(image):
    contrasty_image = image
    alpha = .75 #Alpha and Beta can change for different things
    beta = 1
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                contrasty_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    ret, otsu_threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
    fig = plt.figure(figsize=(10, 7))
    rows = 4
    colums = 3

#CV2 is BGR, matplotlib is RGB
    # fig.add_subplot(rows, colums, 1)
    # plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.title('Grayscale')

    # fig.add_subplot(rows, colums, 2)
    # plt.imshow(cv2.cvtColor(equ, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.title('Equalized Histogram')

    # fig.add_subplot(rows, colums, 3)
    # # plt.imshow(cv2.cvtColor(otsu_threshold, cv2.COLOR_BGR2RGB))
    # # plt.axis('off')
    # # plt.title('Otsu\'s Threshold Complement')

    erosion_kernal = np.ones((3, 3), np.uint8)
    a_opened = cv2.morphologyEx(otsu_threshold, cv2.MORPH_OPEN, erosion_kernal, iterations=3)
    # fig.add_subplot(rows, colums, 4)
    # plt.imshow(cv2.cvtColor(a_opened, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.title("Eroded Image")

    sure_bg = cv2.dilate(a_opened, erosion_kernal, iterations=4)
    # fig.add_subplot(rows, colums, 5)
    # plt.imshow(cv2.cvtColor(sure_bg, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.title("Sure Background")

    dist_transform = cv2.distanceTransform(a_opened, cv2.DIST_L2, 5)
    # fig.add_subplot(rows, colums, 6)
    # plt.imshow(cv2.cvtColor(dist_transform, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.title("Distance Transform")


    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # fig.add_subplot(rows, colums, 7)
    # plt.imshow(cv2.cvtColor(sure_fg, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.title("Sure Forground")

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # fig.add_subplot(rows, colums, 8)
    # plt.imshow(cv2.cvtColor(unknown, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.title("Unknown")

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==255] = 0
    # fig.add_subplot(rows, colums, 9)
    # plt.imshow(markers)
    # plt.axis('off')
    # plt.title("Markers")

    markers = cv2.watershed(image, markers)
    # fig.add_subplot(rows, colums, 10)
    # plt.imshow(markers)
    # plt.axis('off')
    # plt.title("Marker Image after segmentation")

    image[markers == -1] = [255, 0, 0]
    # fig.add_subplot(rows, colums, 11)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.title("Final")

    #cv2.imwrite("segemented_wbc_cytoplasm.jpg", image

    #plt.show()

    return markers


"""
Returns an image of the red blood cell cytoplasm
"""
def separate_rbcs(image, wbc_mask, all_cytoplasmic_mask):
    img = cv2.cvtColor(np.uint8(image), cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, inverted = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    gray_mask = cv2.cvtColor(cv2.cvtColor(np.uint8(wbc_mask), cv2.COLOR_Lab2BGR), cv2.COLOR_BGR2GRAY)
    ret, gray_mask = cv2.threshold(gray_mask, 50, 255, cv2.THRESH_BINARY_INV)

    (width, height) = gray_mask.shape
    rbcs_and_wbc_cytoplasm = np.zeros((gray_mask.shape))
    for x in range(width):
        for y in range(height):
            r = gray_mask[x][y]
            m = inverted[x][y]
            if r == 255 and m == 255:
                rbcs_and_wbc_cytoplasm[x][y] = 0
            else:
                rbcs_and_wbc_cytoplasm[x][y] = inverted[x][y]
    
    #Extract white blood cell cytoplasm from the entire cytoplasmic content and grayscale it
    (width, height, depth) = all_cytoplasmic_mask.shape
    rbcs = np.zeros((rbcs_and_wbc_cytoplasm.shape))
    for x in range(width):
        for y in range(height):
            gray = rbcs_and_wbc_cytoplasm[x][y]
            (l, a, b) = all_cytoplasmic_mask[x][y]
            rbcs[x][y] = b

    return rbcs
    
"""
Given cluster 0, extract all cells from it with white being cells and dark being background
"""
def get_all_cells(cell_extraction):
    cell_extraction = cell_extraction.astype(np.uint8)
    bgr = cv2.cvtColor(cell_extraction, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    (width, height) = gray.shape
    output = np.zeros((gray.shape))
    for x in range(width):
        for y in range(height):
            v = gray[x][y]
            if v > 60:
                output[x][y] = 0
            else:
                output[x][y] = 255
    return output

"""
Doesn't work - If it doesn't work then why am I using it D:
"""
def extract_wbcs(all_cells, wbc):
    (width, height) = all_cells.shape
    resized = cv2.resize(wbc, (height, width), interpolation=cv2.INTER_AREA)

    output = np.zeros((all_cells.shape), np.uint8)
    for x in range(width):
        for y in range(height):
            v = all_cells[x][y]
            b = resized[x][y]
            if b > 1 and v == 255:
                output[x][y] = 0
            else:
                output[x][y] = all_cells[x][y]

    # erosion_kernal = np.ones((2, 2), np.uint8)
    # a_opened = cv2.morphologyEx(output, cv2.MORPH_OPEN, erosion_kernal, iterations=3)

    a_opened = output.astype(np.uint8)
    a_cells = all_cells.astype(np.uint8)
    
    wbcs = np.zeros((a_cells.shape))
    for x in range(width):
        for y in range(height):
            v = a_cells[x][y]
            m = a_opened[x][y]
            if v == 255 and m == 0:
                wbcs[x][y] = 255
            else:
                wbcs[x][y] = 0

    # wbcs = cv2.morphologyEx(wbcs, cv2.MORPH_OPEN, erosion_kernal, iterations=3)
    return wbcs

"""
Extract cytoplasm from the cluster of white blood cells
"""
def wbc_cytoplasm_calculating(cluster_1):
    (width, height, depth) = cluster_1.shape

    output = np.zeros((cluster_1.shape), np.uint8)
    for x in range(width):
        for y in range(height):
            (l, a, b) = cluster_1[x][y]
            output[x][y] = b

    return output

"""
Compute Features for a given image of binary threshold white blood cells
"""
def feature_extraction(binary_wbcs, filename):
    (width, height) = binary_wbcs.shape
    try:
        os.mkdir(filename, 0o666)
    except OSError as error:
        print(f"caught error {error}")
    gray_img = binary_wbcs.astype(np.uint8)#cv2.cvtColor(binary_wbcs, cv2.COLOR_BGR2GRAY)
    
    erosion_kernal = np.ones((2, 2), np.uint8)
    eroded = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, erosion_kernal, iterations=3)

    # Apply the Component analysis function
    analysis = cv2.connectedComponentsWithStats(eroded, 4, cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis
    
    # Initialize a new image to
    # store all the output components
    output = np.zeros(eroded.shape, dtype="uint8")
    
    file = open(f"{filename}\\data.json", "w")
    file.write("{")
    # Loop through each component
    comp_int = 0
    for i in range(1, totalLabels):
    
        # Area of the component
        area = values[i, cv2.CC_STAT_AREA]
        
        if (area > 200):
            file.write(f"\"{comp_int}\"" + " : {\n")
            # Create a new image for bounding boxes
            
            new_img = np.zeros((width, height, 3), np.uint8)
            for x in range(width):
                for y in range(height):
                    v = binary_wbcs[x][y]
                    new_img[x][y] = (v, v, v)
            
            # Now extract the coordinate points
            x1 = values[i, cv2.CC_STAT_LEFT]
            y1 = values[i, cv2.CC_STAT_TOP]
            w = values[i, cv2.CC_STAT_WIDTH]
            h = values[i, cv2.CC_STAT_HEIGHT]
            
            # Coordinate of the bounding box
            pt1 = (x1, y1)
            pt2 = (x1+ w, y1+ h)
            (X, Y) = centroid[i]
            center = (int(X), int(Y))

            #Eccentricity Calculation - Recommended refinement for more precise axis detection
            axis_one = h#(int(Y) - int(h/2)) * 2
            axis_two = w#(int(X) - int(w/2)) * 2
            major_axis = axis_two if axis_two > axis_one else axis_one
            minor_axis = axis_two if axis_two < axis_one else axis_one
            eccentricity = major_axis / minor_axis

            
            #green if > .6
            #red if < .6
            green = (0, 255, 0)
            red = (0, 0, 255)
            draw_color = green if eccentricity > .6 else red
            file.write(f"\t\"eccentricity\" : {eccentricity},\n")

            #Form Factor
            perimeter = ( 2 * 3.14 * math.sqrt( ( (major_axis* major_axis) + (minor_axis * minor_axis) ) / 2 ))
            form_factor = (4 * math.pi * area) / (perimeter * perimeter)
            file.write(f"\t\"form_factor\" : {form_factor},\n")

            #Major Axis Length
            major_axis_length = major_axis
            file.write(f"\t\"major_axis_length\" : {major_axis_length},\n")

            #Area is uhhhh, just area
            file.write(f"\t\"area\" : {area},\n")

            # Bounding boxes for each component
            cv2.line(new_img, center, (int(X) - int(w/2), int(Y)), red, 1)
            cv2.line(new_img, center, (int(X), int(Y) - int(h/2)), green, 1)
            cv2.rectangle(new_img, pt1, pt2, draw_color, 1)
            #cv2.circle(new_img, center, 4, draw_color, -1)

            #cv2.line(output, center, (int(X) - int(w/2), int(Y)), red, 1)
            #cv2.line(output, center, (int(X), int(Y) - int(h/2)), green, 1)
            #cv2.rectangle(output, pt1, pt2, draw_color, 1)
            # cv2.circle(output, center, 4, draw_color, -1)
            cv2.putText(new_img, str(comp_int), center, cv2.FONT_HERSHEY_PLAIN , .5, green)
            
            comp_int += 1
    
            # Create a new array to show individual component
            component = np.zeros(gray_img.shape, dtype="uint8")
            componentMask = (label_ids == i).astype("uint8") * 255

            # Apply the mask using the bitwise operator
            component = cv2.bitwise_or(component,componentMask)
            output = cv2.bitwise_or(output, componentMask)

            #Centroid - Aka center mass
            containing_x = []
            containing_y = []
            for u in range(0, gray_img.shape[0]):
                for v in range(0, gray_img.shape[1]):
                    if component[u][v] == 255:
                        containing_x.append(u)
                        containing_y.append(v)
            centroid_x = sum(containing_x) / area
            centroid_y = sum(containing_y) / area

            centroid_variation = (centroid_x - centroid_y) / centroid_x * 100
            file.write(f"\t\"centroid_variation\" : {centroid_variation},\n")

            #Solidity
            first = 0
            second = 0

            for x in range(0, len(containing_x)-1):
                for y in range(1, len(containing_y)):
                    first += (x * y)
            first += (containing_x[-1] * containing_y[0])

            for x in range(1, len(containing_x)):
                for y in range(0, len(containing_y)-1):
                    second = second + (y * x)
            second += (containing_y[-1] * containing_x[0])
            convex_area = .5 * (first - second)
            solidity = area / convex_area
            file.write(f"\t\"solidity\" : {solidity},\n")
            

            #Equivalent Diameter
            eq_dia = 2 * math.sqrt(area * math.pi)
            file.write(f"\t\"eq_dia\" : {eq_dia},\n")

            #Perimeter Distance
            file.write(f"\t\"perimeter\" : {perimeter},\n")

            #roudness
            roudness = 4 * math.pi * area / math.pow(perimeter, 2) * area
            file.write(f"\t\"roudness\" : {roudness},\n")

            #radius
            radius = eq_dia / 2
            file.write(f"\t\"radius\" : {radius},\n")

            #Cytoplasmic area and computation
            
            #Minor Axis Length
            minor_axis_length = minor_axis
            file.write(f"\t\"minor_axis_length\" : {minor_axis_length}\n")
            file.write("},\n")
            #cv2.add(output, new_img)

            # Show the final images
            # cv2.imshow("Image", new_img)
            cv2.imwrite(f"{filename}\{comp_int-1}.jpg", new_img)
            # cv2.imshow("Individual Component", component)
            # cv2.imshow("Filtered Components", output)
            # cv2.waitKey(0)
    cv2.imwrite(f"{filename}\\reference_data.jpg", output)
    file.write("}")
    file.close()


#Standard Illuminant 10 degrees- http://www.easyrgb.com/en/math.php
Illu_C = (97.285,	100.000,	116.145) #Best Baseline
Illu_D65  = (94.811,	100.000,	107.304) #Could be useful, slightly chaotic
Illu_F3  = (108.968,	100.000,	51.965) #Good for partitioning out stained cells
Illu_F7  = (95.792,	100.000,	107.687) #Second best baseline

#Read every file in the database, and compute outputs and put them in the database
database_directory = os.getcwd() + "\ModifiedDatabase\chunk9"
output_directory = os.getcwd() + "\Output"
clusters_directory = output_directory + "\Clusters"
data_directory = output_directory + "\Data"
markers_directory = output_directory + "\Markers"
cytoplasm_directory = output_directory + "\Cytoplasm"
extracted_cells_directory = output_directory + "\Extracted_Cells"
wbc_directory = output_directory + "\WhiteBloodCells"
blob_directory = output_directory + "\Blobs"

first_time = datetime.now()

for filename in os.listdir(database_directory):
    print("Beginning", filename)
    start_time = datetime.now()
    image = cv2.imread(database_directory + "\\" + filename, IMREAD_COLOR)
    filename = filename[:-4]
    #Image Pre-processing into LAB color space
    (height, width, depth) = image.shape
    #Converts to LAB color space - technically this results in a blue background instead of red
    lab_imgC = np.zeros((height, width, depth))
    # lab_imgD65 = np.zeros((height, width, depth))
    # lab_imgF3 = np.zeros((height, width, depth))
    # lab_imgF7 = np.zeros((height, width, depth))
    for y in range(0, height):
        for x in range(0, width):
            lab_imgC[y][x] = bgr_2_lab(image[y][x], Illu_C)
            # lab_imgD65[y][x] = bgr_2_lab(im2[y][x], Illu_D65)
            # lab_imgF3[y][x] = bgr_2_lab(im2[y][x], Illu_F3)
            # lab_imgF7[y][x] = bgr_2_lab(im2[y][x], Illu_F7)
    print(f"\tBeginning clustering @ {datetime.now()}")
    (c0, c1, c2, c3) = cluster(lab_imgC)
    #c0_reshaped = c0.reshape(int(width/2), -1)
    #np.savetxt(f"{clusters_directory}\{filename}_cluster0.txt", c0_reshaped)
    #c1_reshaped = c1.reshape(int(width/2), -1)
    #np.savetxt(f"{clusters_directory}\{filename}_cluster1.txt", c1_reshaped)

    original_c0 = cv2.resize(c0, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"{clusters_directory}\{filename}_cluster0.jpg", original_c0)
    #original_c1 = cv2.resize(c1, (width, height), interpolation=cv2.INTER_AREA)
    #cv2.imwrite(f"{clusters_directory}\{filename}_cluster1.jpg", original_c1)
    #original_c2 = cv2.resize(c2, (width, height), interpolation=cv2.INTER_AREA)
    #original_c3 = cv2.resize(c3, (width, height), interpolation=cv2.INTER_AREA)
    # print(f"KMedoids Start: {datetime.now()}")
    #cluster_1_end = datetime.now()
    #print(f"\t\tCluster 1 duration: {cluster_1_end - cluster_1_start}")

    print(f"\tPost Processing Starting @ {datetime.now()}")
    markers = cytoplasm(image)
    cv2.imwrite(f"{markers_directory}\{filename}_markers.jpg", markers)
    all_cells = get_all_cells(original_c0)
    cv2.imwrite(f"{extracted_cells_directory}\{filename}_cells.jpg", all_cells)
    wbcs = extract_wbcs(all_cells, markers)
    cv2.imwrite(f"{wbc_directory}\{filename}_wbcs.jpg", wbcs)
    #wbc_cytoplasm = wbc_cytoplasm_calculating(original_c1)
    #cv2.imwrite(f"{cytoplasm_directory}\{filename}_wbc_cytoplasm.jpg", wbc_cytoplasm)
    feature_extraction(wbcs, f"{blob_directory}\{filename}")

    print(f"{filename} duration: {datetime.now() - start_time}")

final_time = datetime.now()

print(f"Overall duration: {final_time - first_time}")
# if False:
#     c0_reshaped = c0.reshape(146, -1)
#     np.savetxt("c0.txt", c0_reshaped)
#     c1_reshaped = c1.reshape(146, -1)
#     np.savetxt("c1.txt", c1_reshaped)
#     c2_reshaped = c2.reshape(146, -1)
#     np.savetxt("c2.txt", c2_reshaped)
#     c3_reshaped = c3.reshape(146, -1)
#     np.savetxt("c3.txt", c3_reshaped)

# loaded_arr = np.loadtxt("c0.txt")
# original_c0 = loaded_arr.reshape(loaded_arr.shape[0], loaded_arr.shape[1] // 3, 3)
# loaded_arr = np.loadtxt("c1.txt")
# original_c1 = loaded_arr.reshape(loaded_arr.shape[0], loaded_arr.shape[1] // 3, 3)
# loaded_arr = np.loadtxt("c2.txt")
# original_c2 = loaded_arr.reshape(loaded_arr.shape[0], loaded_arr.shape[1] // 3, 3)
# loaded_arr = np.loadtxt("c3.txt")
# original_c3 = loaded_arr.reshape(loaded_arr.shape[0], loaded_arr.shape[1] // 3, 3)

# cv2.imshow("c0: ", original_c0) #Purple background, black cells
# cv2.imshow("c1: ", original_c1) #
# cv2.imshow("c2: ", original_c2)
# cv2.imshow("c3: ", original_c3)

