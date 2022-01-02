import argparse
import cv2


path_to_image = './testowa.jpg'

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", default=path_to_image,
	help="path to input directory of images")
ap.add_argument("-t", "--threshold", type=float, default=100.0,
	help="focus measures that fall below this value will be considered 'blurry'")
args = vars(ap.parse_args())


image = cv2.imread(args["images"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
focus_measure = cv2.Laplacian(gray, cv2.CV_64F).var()
text = "Not Blurry"
if focus_measure < args["threshold"]:
	text = "Blurry"
cv2.putText(image, "{}: {:.2f}".format(text, focus_measure), (10, 30),
	cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
cv2.imshow("Image", image)
key = cv2.waitKey(0)



