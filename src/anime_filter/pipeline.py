import cv2
from . import inference
import numpy
import numpy as np
import onnxruntime
from .config import cfg_mnet as cfg
from .align_faces import warp_and_crop_face,get_reference_facial_points
from itertools import product as product
from math import ceil


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


class FaceDetect:
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
        img_raw = cv2.cvtColor(img_raw,cv2.COLOR_RGB2BGR)
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

    def portrait2anime(self,img):
        res_main = inference.Animate()
        res_main = res_main.image(img)
        return res_main

    def portrait2anime_full(self,img):
        res_main = inference.Animate()
        x,y = self.get_scale_factor(self.h,self.w,512)
        img = cv2.resize(img,None,fx=x,fy=y,interpolation=cv2.INTER_AREA)
        res_main = res_main.image(img)
        res_main = cv2.resize(res_main,(self.w,self.h),cv2.INTER_AREA)
        return res_main

    def bg_remove(self,img):
        res_main = inference.Bg_remove()
        res_main = res_main.image(img)
        return res_main

    def detect(self,img):
        worker = ONNXModel("./weights/face_detect.onnx")
        img_raw, img = self.preprocess(img)
        (self.h,self.w) = img_raw.shape[:2]

        loc,conf,landms = worker.forward(img)
        #loc = torch.Tensor(loc)

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

        for i, (faceb, facial5points) in enumerate(zip(dets, landms)):
            if faceb[4]<threshold: continue
            fh, fw = (faceb[3]-faceb[1]), (faceb[2]-faceb[0])

            facial5points = np.reshape(facial5points, (2, 5))

            of, tfm_inv = warp_and_crop_face(img_raw, facial5points, reference_pts=reference_5pts, crop_size=(in_size,in_size))
            
            if i<1:
                x = round(float(fw)/float(self.w),6)
                y = round(float(fh)/float(self.h),6)
                if x<0.2 and y<0.2:
                    break
                else:
                    ef = self.portrait2anime(of)
            else:
                break


            tmp_mask = self.bg_remove(of)
            tmp_mask = cv2.warpAffine(tmp_mask, tfm_inv, (self.w,self.h), flags=3)

            if min(fh, fw)<100: # gaussian filter for small faces
                ef = cv2.filter2D(ef, -1, self.kernel)

            ef = cv2.addWeighted(ef,1,of,1.-1,0.0)
            tmp_img = cv2.warpAffine(ef, tfm_inv, (self.w,self.h), flags=3)

            mask = tmp_mask - full_mask
            full_mask[np.where(mask>0)] = tmp_mask[np.where(mask>0)]
            full_img[np.where(mask>0)] = tmp_img[np.where(mask>0)]

        full_mask = full_mask[:, :, np.newaxis]
        img = self.portrait2anime_full(img_raw)
        out = cv2.convertScaleAbs(full_img * full_mask + (1-full_mask) * img)
        return out

if __name__ == '__main__':
    det = FaceDetect()
    im_name = "gabriel-silverio-u3WmDyKGsrY-unsplash.jpg"
    out = det.detect("D:/PPM-130/mobilenetv2/PhotoMatte_250/image/{}".format(im_name))
    cv2.imwrite("out23.png",out)