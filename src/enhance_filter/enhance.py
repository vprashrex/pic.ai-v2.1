import cv2
import numpy
import numpy as np
import onnxruntime
from .config import cfg_mnet as cfg
from .align_faces import warp_and_crop_face,get_reference_facial_points
from itertools import product as product
from math import ceil

class GFPGAN:
    def __init__(self):
        
        self.session = onnxruntime.InferenceSession("./weights/gfpgan.onnx")
        self.input_name = self.session.get_inputs()[0].name
        _,self.net_input_channels,self.net_input_height,self.net_input_width = self.session.get_inputs()[0].shape
        self.net_output_count = len(self.session.get_outputs())

    def preprocess(self,img):
        img = img.astype(np.float32)/255.0
        '''
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        '''
        img[:,:,0] = (img[:,:,0]-0.5)/0.5
        img[:,:,1] = (img[:,:,1]-0.5)/0.5
        img[:,:,2] = (img[:,:,2]-0.5)/0.5
        img = np.float32(img[np.newaxis,:,:,:])
        img = img.transpose(0,3,1,2)
        return img
    
    def post_process(self,out):
        
        out = out.clip(-1,1)
        out = (out + 1)/2
        out = out.transpose(1,2,0)
        
        '''
        out = cv2.cvtColor(out,cv2.COLOR_RGB2BGR)
        '''
        out = (out * 255.0).round()
        out = np.uint8(out)
        return out
    
    def enhance(self,img):
        img = self.preprocess(img)
        input = self.session.get_inputs()[0].name
        res = self.session.run(None,{input:img})[0][0]
        res = self.post_process(res)
        return res

class RealESRGAN:
    def __init__(self):
        self.session = onnxruntime.InferenceSession("./weights/realesrgan.onnx")
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self,im):
        img = im.astype(np.float32)/255.0
        img = np.transpose(img,(2,0,1))
        img = img[None,:,:,:]
        return img

    def postprocess(self,im):
        im = im.squeeze(0).clip(0,1)
        im = np.transpose(im,(1,2,0))
        im = np.uint8((im*255.0).round())
        return im
    
    def enhance(self,im):
        im = self.preprocess(im)
        res = self.session.run(None,{self.input_name:im})[0]
        res = self.postprocess(res)
        return res

class Upcunet:
    def __init__(self):
        self.session = onnxruntime.InferenceSession("./weights/upcunet.onnx")
        self.input_name = self.session.get_inputs()[0].name
    
    def preprocess(self,img):
        img = img/255.0
        img = np.transpose(img,(2,0,1))
        img = img[None,:,:,:]
        img = img.astype(np.float16)
        return img

    def postprocess(self,img):
        img = img.squeeze(0).clip(0,1)
        img = np.transpose(img,(1,2,0))
        img = np.uint8((img*255.0).round())
        return img
    
    def enhance(self,img):
        img = self.preprocess(img)
        res = self.session.run(None,{self.input_name:img})[0]
        res = self.postprocess(res)
        return res


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


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        ouput = numpy.array(anchors)
        output = np.reshape(ouput,(-1,4))

        return output


class ONNXModel(object):
    def __init__(self, onnx_path):
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)


    def get_output_name(self, onnx_session):
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_tensor):
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_tensor
        return input_feed

    def forward(self, image_tensor):
        input_feed = self.get_input_feed(self.input_name, image_tensor)
        loc, conf, landms = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return loc, conf, landms


class Enhance:
    def decode(self,loc, priors, variances):
        boxes = np.concatenate((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def decode_landm(self,pre, priors, variances):
        landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                            ), axis=1)
        return landms
    
    def py_cpu_nms(self,dets, thresh):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def get_scale_factor(self,im_h, im_w, ref_size):
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w
        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        x_scale_factor = im_rw / im_w
        y_scale_factor = im_rh / im_h
        return x_scale_factor, y_scale_factor

    def preprocess(self,img_raw):
        if isinstance(img_raw,str):
            img_raw = cv2.imread(img_raw, cv2.IMREAD_COLOR)

        _img = np.float32(img_raw)

        x_size = 320
        y_size = 320
        im_shape = _img.shape

        resize_x = round(float(x_size) / float(im_shape[1]), 6)
        resize_y = round(float(y_size) / float(im_shape[0]), 6)

        img = cv2.resize(_img, None, None, fx=resize_x, fy=resize_y, interpolation=cv2.INTER_LINEAR)
        (self.h,self.w) = img.shape[:2]
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis, :]
        return img_raw, img


    def bg_remove(self,img):
        res_main = Bg_remove()
        res_main = res_main.image(img)
        return res_main

    def face_enhance(self,img):
        res_main = GFPGAN()
        res_main = res_main.enhance(img)
        return res_main
    
    #######################
    def realesr_full(self,img):
        if isinstance(img,str):
            img = cv2.imread(img)
        (h,w) = img.shape[:2]
        res_main = RealESRGAN()
        x,y = self.get_scale_factor(h,w,512)
        img = cv2.resize(img,None,fx=x,fy=y,interpolation=cv2.INTER_AREA)
        res_main = res_main.enhance(img)
        res_main = cv2.resize(res_main,(w,h),interpolation=cv2.INTER_AREA)
        return res_main
    ###########################

    ###### UPCUNET ############
    def upcunet_full(self,img):
        if isinstance(img,str):
            img = cv2.imread(img)
        (h,w) = img.shape[:2]
        res_main = Upcunet()
        x,y = self.get_scale_factor(h,w,512)
        img = cv2.resize(img,None,fx=x,fy=y,interpolation=cv2.INTER_AREA)
        res_main = res_main.enhance(img)
        res_main = cv2.resize(res_main,(w,h),cv2.INTER_AREA)
        return res_main
    ###########################

    def gfpgan_upcunet(self,img):
        worker = ONNXModel("./weights/face_detect.onnx")
        img_raw, img = self.preprocess(img)
        (self.h,self.w) = img_raw.shape[:2]

        loc,conf,landms = worker.forward(img)

        scale = numpy.array((img_raw.shape[1], img_raw.shape[0], img_raw.shape[1], img_raw.shape[0]))
        scale1 = numpy.array([img_raw.shape[1], img_raw.shape[0], img_raw.shape[1], img_raw.shape[0],
                            img_raw.shape[1], img_raw.shape[0], img_raw.shape[1], img_raw.shape[0],
                            img_raw.shape[1], img_raw.shape[0]])

        scores = conf.squeeze(0)[:,1]
        priorbox = PriorBox(cfg, image_size=(320,320))
        priors = priorbox.forward()

        boxes = self.decode(loc.squeeze(0), priors, cfg['variance'])
        landms = self.decode_landm(landms.squeeze(0), priors, cfg['variance'])
        boxes = (boxes * scale) / 1
        landms = landms * scale1 / 1
        
        inds = np.where(scores > 0.6)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        order = scores.argsort()[::-1][:5000]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = self.py_cpu_nms(dets, 0.4)
        dets = dets[keep, :]
        landms = landms[keep]

        dets = dets[:750, :]
        landms = landms[:750, :]

        landms = landms.reshape((-1, 5, 2))
        landms = landms.transpose((0, 2, 1))
        landms = landms.reshape(-1, 10, )
        
        in_size = 512
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        default_square = True
        kernel = np.array((
            [0.0625, 0.125, 0.0625],
            [0.125, 0.25, 0.125],
            [0.0625, 0.125, 0.0625]), dtype="float32")
        threshold = 0.9
        reference_5pts = get_reference_facial_points((in_size,in_size), inner_padding_factor, outer_padding, default_square)
        full_mask = np.zeros((self.h,self.w), dtype=np.float32)
        full_img = np.zeros(img_raw.shape,dtype=np.uint8)
        
        if len(dets) == 0:
            out = self.upcunet_full(img_raw)
            return out
        
        else:
            
            for i, (faceb, facial5points) in enumerate(zip(dets, landms)):
                
                fh, fw = (faceb[3]-faceb[1]), (faceb[2]-faceb[0])
                facial5points = np.reshape(facial5points, (2, 5))
                of, tfm_inv = warp_and_crop_face(img_raw, facial5points, reference_pts=reference_5pts, crop_size=(in_size,in_size))

                
                if i<1:
                    x = round(float(fw)/float(self.w),6)
                    y = round(float(fh)/float(self.h),6)

                    if x<0.2 and y<0.2:
                        break
                    else:
                        ef = self.face_enhance(of)
                else:
                    break
                
                facial5points = np.reshape(facial5points, (2, 5))
                of, tfm_inv = warp_and_crop_face(img_raw, facial5points, reference_pts=reference_5pts, crop_size=(in_size,in_size))

                ef = self.face_enhance(of)
                tmp_mask = self.bg_remove(of)
                tmp_mask = self.mask_postprocess(tmp_mask)
                tmp_mask = cv2.warpAffine(tmp_mask,tfm_inv,(self.w,self.h),flags=3)
                
                if min(fh,fw) < 100:
                    ef = cv2.filter2D(ef,-1,self.kernel)
                
                ef = cv2.addWeighted(ef,1,of,1.-1,0.0)
                tmp_img = cv2.warpAffine(ef,tfm_inv,(self.w,self.h),flags=3)

                mask = tmp_mask - full_mask
                full_mask[np.where(mask>0)] = tmp_mask[np.where(mask>0)]
                full_img[np.where(mask>0)] = tmp_img[np.where(mask>0)]

            bg_img = img_raw
            full_mask = full_mask[:,:,np.newaxis]
            out = cv2.convertScaleAbs(full_img * full_mask + (1-full_mask) * bg_img)
            out = self.upcunet_full(out)
            return out


    def mask_postprocess(self,mask,thres=26):
        mask[:thres,:] = 0;mask[-thres:,:] = 0
        mask[:,:thres] = 0;mask[:,-thres:] = 0
        mask = cv2.GaussianBlur(mask,(101,101),4)
        mask = cv2.GaussianBlur(mask,(101,101),4)
        return mask.astype(np.float32) 

    def detect(self,img):
        worker = ONNXModel("./weights/face_detect.onnx")
        img_raw, img = self.preprocess(img)
        (self.h,self.w) = img_raw.shape[:2]

        loc,conf,landms = worker.forward(img)

        scale = numpy.array((img_raw.shape[1], img_raw.shape[0], img_raw.shape[1], img_raw.shape[0]))
        scale1 = numpy.array([img_raw.shape[1], img_raw.shape[0], img_raw.shape[1], img_raw.shape[0],
                            img_raw.shape[1], img_raw.shape[0], img_raw.shape[1], img_raw.shape[0],
                            img_raw.shape[1], img_raw.shape[0]])

        scores = conf.squeeze(0)[:,1]
        priorbox = PriorBox(cfg, image_size=(320,320))
        priors = priorbox.forward()

        boxes = self.decode(loc.squeeze(0), priors, cfg['variance'])
        landms = self.decode_landm(landms.squeeze(0), priors, cfg['variance'])
        boxes = (boxes * scale) / 1
        landms = landms * scale1 / 1
        
        inds = np.where(scores > 0.6)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        order = scores.argsort()[::-1][:5000]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = self.py_cpu_nms(dets, 0.4)
        dets = dets[keep, :]
        landms = landms[keep]

        dets = dets[:750, :]
        landms = landms[:750, :]

        landms = landms.reshape((-1, 5, 2))
        landms = landms.transpose((0, 2, 1))
        landms = landms.reshape(-1, 10, )
        
        in_size = 512
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        default_square = True
        kernel = np.array((
            [0.0625, 0.125, 0.0625],
            [0.125, 0.25, 0.125],
            [0.0625, 0.125, 0.0625]), dtype="float32")
        threshold = 0.9
        reference_5pts = get_reference_facial_points((in_size,in_size), inner_padding_factor, outer_padding, default_square)
        full_mask = np.zeros((self.h,self.w), dtype=np.float32)
        full_img = np.zeros(img_raw.shape,dtype=np.uint8)
        
        if len(dets) == 0:
            out = self.upcunet_full(img_raw)
            return out
        
        else:
            for i, (faceb, facial5points) in enumerate(zip(dets, landms)):
                
                fh, fw = (faceb[3]-faceb[1]), (faceb[2]-faceb[0])

                facial5points = np.reshape(facial5points, (2, 5))
                of, tfm_inv = warp_and_crop_face(img_raw, facial5points, reference_pts=reference_5pts, crop_size=(in_size,in_size))
                
                x = round(float(fw)/float(self.w),6)
                y = round(float(fh)/float(self.h),6)

                if i<1:
                    if x<0.2 and y<0.2:
                        out = self.upcunet_full(img_raw)
                        return out
                    else:
                        ef = self.face_enhance(of) 
                else:
                    break

                tmp_mask = self.bg_remove(of)
                tmp_mask = self.mask_postprocess(tmp_mask)
                tmp_mask = cv2.warpAffine(tmp_mask,tfm_inv,(self.w,self.h),flags=3)
                
                if min(fh,fw) < 100:
                    ef = cv2.filter2D(ef,-1,self.kernel)
                
                ef = cv2.addWeighted(ef,1,of,1.-1,0.0)
                tmp_img = cv2.warpAffine(ef,tfm_inv,(self.w,self.h),flags=3)

                mask = tmp_mask - full_mask
                full_mask[np.where(mask>0)] = tmp_mask[np.where(mask>0)]
                full_img[np.where(mask>0)] = tmp_img[np.where(mask>0)]

            bg_img = img_raw
            full_mask = full_mask[:,:,np.newaxis]
            out = cv2.convertScaleAbs(full_img * full_mask + (1-full_mask) * bg_img)
            return out

if __name__ == '__main__':
    det = Enhance("../weights")
    im_name = "gabriel-silverio-u3WmDyKGsrY-unsplash.jpg"
    out = det.detect("D:/PPM-130/mobilenetv2/PhotoMatte_250/image/{}".format(im_name))
    cv2.imwrite("out23.png",out)