import cv2
import numpy as np
import onnxruntime


class Bg_remove:

    def __init__(self):
        self.session = onnxruntime.InferenceSession("./weights/7999_iter2.onnx",None)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.bg_classes = [0,14,18,16,17]

    def image(self,img):
        (h,w) = img.shape[:2]

        img = img.astype(np.float32)
        img = img/127.5 - 1.0
        img = np.transpose(img,(2,0,1))
        img = np.expand_dims(img,axis=0)

        out = self.session.run([self.output_name],{self.input_name:img})[0]
        out = np.stack([out != cls for cls in self.bg_classes],axis=0).all(axis=0)

        out = out.squeeze(0).clip(-1,1)
        out = np.transpose(out,(1,2,0))
        out = cv2.resize(out,(w,h),cv2.INTER_AREA)

        out = np.float32(out)
        return out


class Animate:
    def preprocess(self,img):
        #img = img.astype(np.float32)
        #
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #
        img = img/127.5 - 1.0
        img = np.transpose(img,(2,0,1))
        img = np.expand_dims(img,axis=0)
        return img.astype(np.float16)

    def postprocess(self,res):
        res = res.squeeze(0).clip(-1,1)*0.5+0.5
        res = np.transpose(res,(1,2,0))
        res = np.uint8((res*255.0).round())
        
        return res

    def image(self,im):
        (self.h,self.w) = im.shape[:2]
        im = self.preprocess(im)
        session = onnxruntime.InferenceSession("./weights/anime_gan.onnx",None)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        res = session.run([output_name],{input_name:im})
        res = res[0]
        res = self.postprocess(res)
        res = cv2.resize(res,(self.w,self.h),cv2.INTER_AREA)
        return res