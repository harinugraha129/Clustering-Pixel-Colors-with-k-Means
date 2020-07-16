import argparse
import numpy as np
import cv2

parser = argparse.ArgumentParser(description="Start a clustering k-Means pixel image")
parser.add_argument("-i", "--image", help="image path", type=str)
parser.add_argument("-o","--output", help="image output", type=str, default="result.jpg")
parser.add_argument("-k", "--k_value", default="2",  help="k value of k-Means", type=int)
args = parser.parse_args()

# input image
img = cv2.imread(args.image)
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = args.k_value
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

# save result image
path = "output/"+args.output
cv2.imwrite(path,res2)

cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
