import cv2
import numpy as np
import onnxruntime

class BGRemove:
    def resize_long(self,im, long_size=224, interpolation=cv2.INTER_LINEAR):
        value = max(im.shape[0], im.shape[1])
        scale = float(long_size) / float(value)
        resized_width = int(round(im.shape[1] * scale))
        resized_height = int(round(im.shape[0] * scale))
        im = cv2.resize(im, (resized_width, resized_height),interpolation=interpolation)
        return im

    def process_original_im(self,im,im_size=768):
        im_h,im_w = im.shape[:2]
        if max(im_h,im_w) < im_size or min(im_h,im_w):
            if im_w >= im_h:
                im_rh = im_size
                im_rw = int(im_w/im_h * im_size)
            elif im_w < im_h:
                im_rw = im_size
                im_rh = int(im_h/im_w * im_size)
        else:
            im_rh = im_h
            im_rw = im_w
        im = cv2.resize(im,(im_rw,im_rh),cv2.INTER_AREA)
        return im

    def preprocess(self, im):
        self.im = im
        self.height,self.width = self.im.shape[:2]
        im = self.resize_long(im,768)
        im = cv2.resize(im,(512,512), interpolation=cv2.INTER_LINEAR)
        im = (im-127.5)/127.5
        im = np.transpose(im)
        im = np.swapaxes(im, 1, 2)
        im = np.expand_dims(im, axis=0).astype('float32')
        return im

    def postprocess(self,mask_data):
        mask_data = (np.squeeze(mask_data[0]))
        matte = np.dstack([mask_data]*4)
        im = self.process_original_im(self.im)
        (im_h,im_w) = im.shape[:2]
        rgba_img = cv2.cvtColor(im,cv2.COLOR_RGB2RGBA)
        bg = np.full(rgba_img.shape,(0,0,0,0),dtype=np.uint8)
        matte = cv2.resize(matte,(im_w,im_h),cv2.INTER_AREA)
        res = matte * rgba_img + (1-matte) * bg
        return res

    def image(self,im):        
        im = self.preprocess(im)
        session = onnxruntime.InferenceSession('./weights/pic_ai_v1-3.onnx', None)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: im})
        result = self.postprocess(result)
        return result


class ColorSplash:
    def resize_long(self,im, long_size=224, interpolation=cv2.INTER_LINEAR):
        value = max(im.shape[0], im.shape[1])
        scale = float(long_size) / float(value)
        resized_width = int(round(im.shape[1] * scale))
        resized_height = int(round(im.shape[0] * scale))
        im = cv2.resize(im, (resized_width, resized_height),interpolation=interpolation)
        return im

    def process_original_im(self,im,im_size=768):
        im_h,im_w = im.shape[:2]
        if max(im_h,im_w) < im_size or min(im_h,im_w):
            if im_w >= im_h:
                im_rh = im_size
                im_rw = int(im_w/im_h * im_size)
            elif im_w < im_h:
                im_rw = im_size
                im_rh = int(im_h/im_w * im_size)
        else:
            im_rh = im_h
            im_rw = im_w
        im = cv2.resize(im,(im_rw,im_rh),cv2.INTER_AREA)
        return im

    def preprocess(self, im):
        self.im = im
        self.height,self.width = self.im.shape[:2]
        im = self.resize_long(im,768)
        im = cv2.resize(im,(512,512), interpolation=cv2.INTER_LINEAR)
        im = (im-127.5)/127.5
        im = np.transpose(im)
        im = np.swapaxes(im, 1, 2)
        im = np.expand_dims(im, axis=0).astype('float32')
        return im

    def postprocess(self,mask_data):
        mask_data = (np.squeeze(mask_data[0]))
        matte = np.dstack([mask_data]*3)
        im = self.process_original_im(self.im)
        (im_h,im_w) = im.shape[:2]
        bg_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        bg_gray = bg_gray[:,:,None]
        bg_gray = np.dstack([bg_gray]*3)
        matte = cv2.resize(matte,(im_w,im_h),cv2.INTER_AREA)
        res = matte * im + (1-matte) * bg_gray
        return res
        

    def image(self,im):        
        im = self.preprocess(im)
        session = onnxruntime.InferenceSession('./weights/pic_ai_v1-3.onnx', None)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: im})
        result = self.postprocess(result)
        return result


if __name__ == '__main__':
    img = cv2.imread("6.jpg")
    res = ColorSplash()
    res.image(img)
