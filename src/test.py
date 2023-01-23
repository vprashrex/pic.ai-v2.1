# no_face 
import cv2
import enhance_filter
import os
import anime_filter

''' img = cv2.imread("./images/img.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
res_main = anime_filter.FaceDetect()
res_main = res_main.detect(img)
out = cv2.cvtColor(res_main,cv2.COLOR_RGB2BGR)
cv2.imwrite("out.png",out) '''

img = cv2.imread("out2.png")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
res_main = enhance_filter.Enhance()
res_main = res_main.upcunet_full(img)
res_main = cv2.cvtColor(res_main,cv2.COLOR_RGB2BGR)
cv2.imwrite("out3.png",res_main)




''' x,y = res_main.gfpgan_upcunet(img)
x_s.append(x)
y_s.append(y) '''

''' print(sum(x_s)/len(x_s))
print(sum(y_s)/len(y_s)) '''



''' img = cv2.imread("out2.png")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
res_main = arcane_filter.FaceDetect()
res_main = res_main.detect(img)
res_main = cv2.cvtColor(res_main,cv2.COLOR_RGB2BGR)
cv2.imwrite("out3.png",res_main) '''

''' res_main = enhance_filter.Enhance()
res_main = res_main.upcunet_full("pic_ai - 2023-01-16T195704.989.png")
cv2.imwrite("res.png",res_main) '''