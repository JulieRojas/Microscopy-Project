
# coding: utf-8

# Live-cell imaging countings 

import mahotas as mh
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (8.0, 8.0)
def count_frame(green_fr, red_fr):

    frame = green_fr + '05.jpg'
    image = mh.imread(frame, as_grey = True)

    ### Mad2-neonGreen Cell Segmentation

    # First thing is to find cells and find their position.
    # To do this, I used (a lot) the mahotas library and tutorial.
    # Links:
    #
    # https://mahotas.readthedocs.io/en/latest/labeled.html
    #
    # https://github.com/luispedro/python-image-tutorial/blob/master/Segmenting%20cell%20images%20(fluorescent%20microscopy).ipynb


    # https://mahotas.readthedocs.io/en/latest/labeled.html

    image_gaus = mh.gaussian_filter(image, 4)
    mean = image_gaus.mean()
    bin_image = image_gaus > mean*1.2

    thresholded = image_gaus > 9 # Other thresholding style
    labeled, nr_objects = mh.label(thresholded)


    # The next steps are to separate touching cells

    sigma = 10
    imageG = mh.gaussian_filter(image.astype(float), sigma)
    maxima = mh.regmax(mh.stretch(imageG))
    maxima = mh.dilate(maxima, np.ones((5,5)))
    maxima,nb_cell= mh.label(maxima)

    dist = mh.distance(thresholded)
    # https://mahotas.readthedocs.io/en/latest/distance.html
    dist = 255 - mh.stretch(dist)
    watershed = mh.cwatershed(dist, maxima)

    # Take out background
    watershed *= thresholded

    # Take out cells touching borders
    watershed = mh.labeled.remove_bordering(watershed)

    sizes = mh.labeled.labeled_size(watershed)
    watershed = watershed.astype(np.intc)
    # remove too small and too big regions
    filtered_max = mh.labeled.remove_regions_where(watershed, sizes > 3500)
    filtered = mh.labeled.remove_regions_where(filtered_max, sizes < 440)
    relabeled, n_left = mh.labeled.relabel(filtered)

    # The bboxes contains the position of each cell
    bboxes = mh.labeled.bbox(relabeled, as_slice=False)

    ##################################################
    # Counting of SPBs and Mad2 clusters for each cell and through all time points
    ##################################################

    # Now that we found were the cells are, we report that information for each time points (separated jpg pictures with ascending number) to follow their progression through meiosis.
    # However, a big problem is that cells are moving quit a lot...
    # So we increase a bit the cell frame and recenter them.


    ############################
    #Mad2-GFP clusters detection
    ############################

    # Countings of SPBs and Mad2 clusters in each frame, and each cell

    # We store the number of detected SPBs and Mad2 clusters for each time point (list) and each cell (list of list)

    SPB = []
    clusters = []
    margin = 5
    margC = 10
    for n in range(1,len(bboxes)): # for each cell, 0 is background so we start at 1
        SPBcount = []
        cluster_count = []
        for m in range(5, 28): # for each frame
            if m < 10:
                red = red_fr + '0' + str(m) + '.jpg'
                green = green_fr + '0' + str(m) + '.jpg'
            else:
                red = red_fr + str(m) + '.jpg'
                green = green_fr + str(m) + '.jpg'

            image_red =  mh.imread(red, as_grey = True)
            image_green = mh.imread(green, as_grey = True)

            # increase space around cell to cope with cell moving (margin)
            if bboxes[n][0] < margin:
                a = 0
            else:
                a = bboxes[n][0] - margin

            if bboxes[n][1] > (bboxes[0][1] + margin):
                b = bboxes[0][1]
            else:
                b = bboxes[n][1] + margin

            if bboxes[n][2] < margin:
                c = 0
            else:
                c = bboxes[n][2] - margin

            if bboxes[n][3] > (bboxes[0][3] + margin):
                d = bboxes[0][3]
            else:
                d = bboxes[n][3] + margin

            cell_red = image_red[a:b, c:d]
            cell_green = image_green[a:b, c:d]

            image_gaus = mh.gaussian_filter(cell_green, 4)
            #thresholded = image_gaus > 22
            mean = image_gaus.mean()
            thresholded = image_gaus > mean*2

            # Detect were is the cell to recenter it
            labeled, nr_objects = mh.label(thresholded)
            sigma = 5.0
            imageG = mh.gaussian_filter(cell_green.astype(float), sigma)
            maxima = mh.regmax(mh.stretch(imageG))
            maxima = mh.dilate(maxima, np.ones((5,5)))
            maxima,_= mh.label(maxima)
            dist = mh.distance(thresholded)
            dist = 255 - mh.stretch(dist) # What does this 255 correspond to?
            watershed = mh.cwatershed(dist, maxima)
            watershed *= thresholded
            watershed = mh.labeled.remove_bordering(watershed)
            relab_cell, n_left = mh.labeled.relabel(watershed)
            loc = mh.labeled.bbox(relab_cell, as_slice=False) # Carefull, index 0 is always background
            # Add some margin (margC)
            a = loc[0][1]
            b = 0
            c = loc[0][3]
            d = 0
            if len(loc) == 1:
                a = loc[0][0]
                b = loc[0][1]
                c = loc[0][2]
                d = loc[0][3]
            if len(loc) == 2:
                a = loc[1][0]
                b = loc[1][1]
                c = loc[1][2]
                d = loc[1][3]
            else:
                for m in range(1,len(loc)):
                    if loc[m][0] < a:
                        a = loc[m][0]
                    if loc[m][1] > b:
                        b = loc[m][1]
                    if loc[m][2] < c:
                        c = loc[m][2]
                    if loc[m][3] > d:
                        d = loc[m][3]

            if a < margC:
                a = 0
            else:
                a -= margC

            if b > (loc[0][1] + margC):
                b = loc[0][1]
            else:
                b += margC

            if c < margC:
                c = 0
            else:
                c -= margC

            if d > (loc[0][3] + margC):
                d = loc[0][3]
            else:
                d += margC

            nucl = cell_red[a:b,c:d]
            nucl_Thr = nucl > 30
            spbs, nb_spb = mh.label(nucl_Thr)
            maxima = mh.regmax(mh.stretch(nucl))
            #maxima = mh.dilate(maxima, np.ones((5,5)))
            #axs[i].imshow(mh.as_rgb(np.maximum(maxima, cell_red), cell_red, cell_red > 20))

            # Now counting mad2-neonGreen clusters
            Green = cell_green[a:b,c:d]
            green_gaus = mh.gaussian_filter(Green, 1.5)
            mean = green_gaus.mean()
            bin_green = green_gaus > mean*5
            cluster, nb_clust = mh.label(bin_green)
            cluster_count.append(nb_clust)

            # To separate touching SPBs, works well!
            maxima,_= mh.label(maxima)
            dist = mh.distance(nucl_Thr)
            dist = 255 - mh.stretch(dist)
            red_water= mh.cwatershed(dist, maxima)
            red_water *= nucl_Thr
            relab_red, spb_sep = mh.labeled.relabel(red_water)
            SPBcount.append(spb_sep)

        SPB.append(SPBcount)
        clusters.append(cluster_count)

    ######################
    #Remove abnormal cells
    ######################

    # We shall remove cells that do not progress (stay with one SPBs), dead cells (typically brighter, they have high number of "SPBs").

    # Additionally, cells start with 1 SPBs, then get 2, then 4. When cells are too close from each other, I might wrongly count SPBs belonging to neighbouring cells. These results in aberrant SPBs counting. These cells are also removed.

    # This code clean up a bit SPBs data according to what's before and after

    clean_SPB = []

    for x in range(len(SPB)):
        spb = SPB[x]
        new_spb = [1]
        for y in range(1,len(spb)-1):
            s = 0
            if spb[y+1] <= 1 and new_spb[y-1] <= 1:
                s = 1
            elif spb[y+1] == 2 and new_spb[y-1] == 2:
                s = 2
            elif spb[y+1] == 4 and new_spb[y-1] == 4:
                s = 4
            elif spb[y] == 2 and spb[y+1] >= 2:
                s = 2
            elif spb[y+1] > 2 and new_spb[y-1] > 2:
                s = 4
            elif spb[y] > 2 and new_spb[y-1] > 2:
                s = 4
            elif spb[y] >= 4 or spb[y] > 2:
                s = 4
            elif spb[y] < 2:
                s = 1
            else:
                s = 2

            new_spb.append(s)
        clean_SPB.append(new_spb)

    # Now that I did what I could for cleaning up SPBs,
    # I need to exclude cells that still don't behave

    # Meaning, cells that never enter meiosis (2 SPBs)
    # Plus, x has to be <= x+1

    # Boolean, if cells is fine: True, else: False
    Good_cells = []
    for x in range(len(clean_SPB)):
        cell = clean_SPB[x]
        good = True
        for y in range(len(cell)-1):
            if cell[y] > cell[y+1]:
                good = False
            if cell[y] == 1 and cell[y+1] == 4:
                good = False
            if max(cell) == 1:
                good = False
        Good_cells.append(good)

    good_cells_spb = []
    Good_Green = []
    for x in range(len(clean_SPB)):
        if Good_cells[x] == True:
            good_cells_spb.append(clean_SPB[x])
            Good_Green.append(clusters[x])

    return(good_cells_spb, Good_Green)

two_spbs = [0] * 22
four_spbs = [0] * 22
mad2_clust = [0] * 22

sync_spbs_cells = []
sync_mad2_cells = []

for fr in range(1,11):
    GREEN =  'mad2ng_ndt80AR/' + str(fr) + 'green'
    RED = 'mad2ng_ndt80AR/' + str(fr) + 'red'
    spbs_l, mad2nG_l = count_frame(GREEN, RED)

    # For un-synchronized countings, I just have to add this up
    for x in range(len(spbs_l)):
        cell = spbs_l[x]
        clust = mad2nG_l[x]
        for y in range(len(cell)):
            if clust[y] > 0:
                mad2_clust[y] += 1
            if cell[y] == 2:
                two_spbs[y] += 1
            if cell[y] == 4:
                four_spbs[y] += 1

    # FOR IN SILICO SYNCHRONIZATION

    for x in range(len(spbs_l)):
        cell = spbs_l[x]
        clust = mad2nG_l[x]
        sync_spbs = [0] * (len(cell)+3)
        sync_Green = [0] * (len(cell)+3)
        y = 0
        while cell[y] == 1:
            y += 1
            if y > len(cell):
                break
        for z in range(y, len(cell)):
            sync_spbs[z - y + 3] = cell[z]  # The +3 is there for aesthetic region on the graph
            sync_Green[z - y + 3] = clust[z]
        sync_spbs_cells.append(sync_spbs)
        sync_mad2_cells.append(sync_Green)

    # Because of synchronizations the four SPBs drop at some point (when list is over) but it does not make sense,
    # so once we have 4 spbs, we keep it at 4



###########
# Plotting without in silico synchronization
###########

m = len(two_spbs)*10
x_axis = list(range(0,m, 10))

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(x_axis, two_spbs,color='r',label='two SPBss')
ax.plot(x_axis, four_spbs,color='k',label='four SPBss')
ax.plot(x_axis, mad2_clust,color='g',label='mad2 cluster')
plt.xlabel("time (min)")
plt.ylabel("# cells")
plt.title("Time course ...")
ax.legend()
plt.savefig('Not_synchronized.png')
plt.clf()


########################## 
#In silico synchronization 
##########################
 
# I now want to synchronize, in silico, my countings to the appearance of 2 SPBs.

sync_spbs_corrected = []
for x in range(len(sync_spbs_cells)):
    cell = sync_spbs_cells[x]
    for y in range(1, len(cell)):
        if cell[y - 1] == 4:
            cell[y] = 4
    sync_spbs_corrected.append(cell)

two_sync = [0] * len(sync_spbs_corrected[0])
four_sync = [0] * len(sync_spbs_corrected[0])
mad2_sync = [0] * len(sync_spbs_corrected[0])
for x in range(len(sync_spbs_corrected)):
    cell = sync_spbs_corrected[x]
    clust = sync_mad2_cells[x]
    for y in range(len(cell)):
        if clust[y] > 0:
            mad2_sync[y] += 1
        if cell[y] == 2:
            two_sync[y] += 1
        if cell[y] == 4:
            four_sync[y] += 1

# Now the plotting of from in silico synchronized culture!
m = len(two_sync)*10
x = list(range(0,m, 10))
for n in range(len(x)):
    x[n] = x[n] - 20

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(x, two_sync, color='r',label='two SPBss')
ax.plot(x, four_sync, color='k',label='four SPBss')
ax.plot(x, mad2_sync, color='g',label='mad2 cluster')
plt.xlabel("time (min)")
plt.ylabel("# cells")
plt.title("Time course ... Synchronized")
ax.legend()
plt.savefig('synchronized.png')

# Looks very similar to manual counting :)
