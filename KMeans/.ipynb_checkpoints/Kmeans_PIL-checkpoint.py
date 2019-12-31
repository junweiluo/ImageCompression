from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

im = Image.open('Resource/sample.JPG')
(x, y) =im.size  # Get the width and hight of the image 
im.show()
print(x, y)

im_data = np.array(im.getdata())
im_data_transformed = im_data.reshape(x * y, int(im_data.size/x/y)) /255
#print(im_data_transformed)

km = KMeans(
    n_clusters=16, init='random',
    n_init=10, max_iter=100, 
    tol=1e-04, random_state=0
)
km.fit_predict(im_data_transformed)
# km.labels_ as assigments for each pixel, km.clueter_centers_ as selected colors

# transform centroids to 255 scale
centroids = (km.cluster_centers_* 255).astype("uint8")

# fill each pixel with selected centroids
list_ = []
for label in km.labels_:
    list_.append(centroids[label])

# transform back to image shape
list2 = np.reshape(list_,(y,x,int(im_data.size/x/y))
# list2 = np.array(np.reshape(list_,()))
print(list2)

im_compressed = Image.fromarray(list2, 'RGB')
im_compressed.save('KMeans/sample_PIL_compressed.JPG')
im_compressed.show()