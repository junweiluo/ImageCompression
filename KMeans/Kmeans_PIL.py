from PIL import Image
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# import original image, get the size of the image
im_original = Image.open('Resource/sample.JPG')
(x, y) =im_original.size  # Get the width and hight of the image 

# get image data, reshape to list of pixels of RGB, and scale between 0 and 1
im_data = np.array(im_original.getdata())
im_data_transformed = im_data.reshape(x * y, int(im_data.size/x/y)) /255

# select 16 colors to represent all pixels
km = KMeans(
    n_clusters=16, init='random',
    n_init=10, max_iter=100, 
    tol=1e-04, random_state=0
)
km.fit_predict(im_data_transformed)
# km.labels_ as assigments for each pixel, km.clueter_centers_ as selected colors

# transform centroids to 255 scale
centroids = (km.cluster_centers_* 255).astype("uint8")


# change to DataFrames
df_centroids = pd.DataFrame(centroids)

df_pixels = pd.DataFrame(km.labels_)
df_pixels["Index"] = df_pixels.index # add index as a column for sorting

# merge two DataFrames, sort by original pixel order, keep the RGB fields only
df_combined = df_pixels.merge(df_centroids, how = "inner", left_on = 0, right_index = True )
df_combined_sort = df_combined.sort_values(by = ["Index"])
df_combined_sort_select = df_combined_sort.iloc[:,3:]

# transform back to image shape, form image, save impage
list2 = np.reshape(df_combined_sort_select.values,(y,x,int(im_data.size/x/y)))
im_compressed = Image.fromarray(list2, 'RGB')
im_compressed.save('KMeans/sample_PIL_compressed.JPG')

# show original and compressed images 
im_original.show()
im_compressed.show()