import cv2
import numpy as np
import onnxruntime

class Arcane:
    def __init__(self):
        self.means = [0.485,0.456,0.406]
        self.stds = [0.229,0.224,0.225]
    
    def preprocess(self,img):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = img/255.0
        img -= self.means
        img /= self.stds
        img = np.transpose(img,(2,0,1))
        img = img[None,:,:,:]
        return img
    
    def postprocess(self,res):

        res = (np.squeeze(res))
        res = np.transpose(res,(1,2,0))
        res *= self.stds
        res += self.means
        res = res.clip(0,255)
        res = np.uint8((res*255.0).round())
        res = cv2.cvtColor(res,cv2.COLOR_RGB2BGR)
        return res
    
    def image(self,im):
        #(self.h,self.w) = im.shape[:2]
        im = self.preprocess(im)
        im = im.astype(np.float16)
        #session = onnxruntime.InferenceSession("./weights/arcane_gan.onnx",None)
        session = onnxruntime.InferenceSession("./weights/arcaneganv2.onnx",None)
        input_name = session.get_inputs()[0].name
        res = session.run(None,{input_name:im})
        res = res[0]
        res = self.postprocess(res)
        return res

if __name__ == '__main__':
    res_main = Arcane()
    img = cv2.imread("face4.png")
    img = cv2.resize(img,(512,512),cv2.INTER_AREA)
    out = res_main.image(img)
    cv2.imwrite("res2.png",out)