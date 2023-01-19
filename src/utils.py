import base64
from PIL import Image
import io
import numpy as np
import inference
from imageio import imread
import arcane_filter
import anime_filter
import enhance_filter


def enhance_filt(img,opt):
    res_main = enhance_filter.Enhance()
    if opt == "realesr_full":
        res_main = res_main.realesr_full(img)
    elif opt == "fbcnn":
        res_main = res_main.gfpgan_upcunet(img)
    else:
        res_main = res_main.detect(img)
    return res_main

def anime_filt(img):
    res_main = anime_filter.FaceDetect()
    res_main = res_main.detect(img)
    res_main = np.uint8(res_main)
    return res_main

def arcane_filt(img):
    res_main = arcane_filter.FaceDetect()
    res_main = res_main.detect(img)
    res_main = np.uint8(res_main)
    return res_main

def bg_remove(img):
    res_main = inference.BGRemove()
    res_main = res_main.image(img)
    res_main = np.uint8(res_main)
    return res_main

def bg_grayscale(img):
    res_main = inference.ColorSplash()
    res_main = res_main.image(img)
    res_main = np.uint8(res_main)
    return res_main


def encode_array_to_base64(image_array):
    with io.BytesIO() as output_bytes:
        PIL_image = Image.fromarray(
            _convert(image_array, np.uint8, force_copy=False))
        PIL_image.save(output_bytes, 'PNG')
        bytes_data = output_bytes.getvalue()
    base64_str = str(base64.b64encode(bytes_data), 'utf-8')
    return base64_str

def encode_to_base64(f, type='image', ext=None, header=True, prnt=False):
    encoded_str = base64.b64encode(f.read())
    base64_str = str(encoded_str, 'utf-8')
    if not header:
        return base64_str
    if ext is None:
        ext = f.split('.')[-1]
    if prnt:
        print("data:" + type + "/" + ext + ";base64," + base64_str)

    return "data:" + type + "/" + ext + ";base64," + base64_str


def base64_to_img(file):
    s = base64.b64encode(file.file.read())
    s = str(s,'utf-8')
    encoded_b64 = "data:" + 'image' + "/" + 'png' + ";base64," + s
    content = encoded_b64.split(';')[1]
    img_encoded = content.split(',')[1]
    img = imread(io.BytesIO(base64.b64decode(img_encoded)))
    return img

def _convert(image, dtype, force_copy=False, uniform=False):
    dtype_range = {bool: (False, True),
                   np.bool_: (False, True),
                   np.bool8: (False, True),
                   float: (-1, 1),
                   np.float_: (-1, 1),
                   np.float16: (-1, 1),
                   np.float32: (-1, 1),
                   np.float64: (-1, 1)}

    def _dtype_itemsize(itemsize, *dtypes):
        return next(dt for dt in dtypes if np.dtype(dt).itemsize >= itemsize)

    def _dtype_bits(kind, bits, itemsize=1):
        s = next(i for i in (itemsize, ) + (2, 4, 8) if
                 bits < (i * 8) or (bits == (i * 8) and kind == 'u'))

        return np.dtype(kind + str(s))

    def _scale(a, n, m, copy=True):
        kind = a.dtype.kind
        if n > m and a.max() < 2 ** m:
            mnew = int(np.ceil(m / 2) * 2)
            if mnew > m:
                dtype = "int{}".format(mnew)
            else:
                dtype = "uint{}".format(mnew)
            n = int(np.ceil(n / 2) * 2)
            return a.astype(_dtype_bits(kind, m))
        elif n == m:
            return a.copy() if copy else a
        elif n > m:
            # downscale with precision loss
            if copy:
                b = np.empty(a.shape, _dtype_bits(kind, m))
                np.floor_divide(a, 2**(n - m), out=b, dtype=a.dtype,
                                casting='unsafe')
                return b
            else:
                a //= 2**(n - m)
                return a
        elif m % n == 0:
            # exact upscale to a multiple of `n` bits
            if copy:
                b = np.empty(a.shape, _dtype_bits(kind, m))
                np.multiply(a, (2**m - 1) // (2**n - 1), out=b, dtype=b.dtype)
                return b
            else:
                a = a.astype(_dtype_bits(
                    kind, m, a.dtype.itemsize), copy=False)
                a *= (2**m - 1) // (2**n - 1)
                return a
        else:
            # upscale to a multiple of `n` bits,
            # then downscale with precision loss
            o = (m // n + 1) * n
            if copy:
                b = np.empty(a.shape, _dtype_bits(kind, o))
                np.multiply(a, (2**o - 1) // (2**n - 1), out=b, dtype=b.dtype)
                b //= 2**(o - m)
                return b
            else:
                a = a.astype(_dtype_bits(
                    kind, o, a.dtype.itemsize), copy=False)
                a *= (2**o - 1) // (2**n - 1)
                a //= 2**(o - m)
                return a

    image = np.asarray(image)
    dtypeobj_in = image.dtype
    if dtype is np.floating:
        dtypeobj_out = np.dtype('float64')
    else:
        dtypeobj_out = np.dtype(dtype)
    dtype_in = dtypeobj_in.type
    dtype_out = dtypeobj_out.type
    kind_in = dtypeobj_in.kind
    kind_out = dtypeobj_out.kind
    itemsize_in = dtypeobj_in.itemsize
    itemsize_out = dtypeobj_out.itemsize
    if np.issubdtype(dtype_in, np.obj2sctype(dtype)):
        if force_copy:
            image = image.copy()
        return image

    if kind_in in 'ui':
        imin_in = np.iinfo(dtype_in).min
        imax_in = np.iinfo(dtype_in).max
    if kind_out in 'ui':
        imin_out = np.iinfo(dtype_out).min
        imax_out = np.iinfo(dtype_out).max

    # any -> binary
    if kind_out == 'b':
        return image > dtype_in(dtype_range[dtype_in][1] / 2)

    # binary -> any
    if kind_in == 'b':
        result = image.astype(dtype_out)
        if kind_out != 'f':
            result *= dtype_out(dtype_range[dtype_out][1])
        return result

    # float -> any
    if kind_in == 'f':
        if kind_out == 'f':
            # float -> float
            return image.astype(dtype_out)

        if np.min(image) < -1.0 or np.max(image) > 1.0:
            raise ValueError("Images of type float must be between -1 and 1.")
        # floating point -> integer
        # use float type that can represent output integer type
        computation_type = _dtype_itemsize(itemsize_out, dtype_in,
                                           np.float32, np.float64)

        if not uniform:
            if kind_out == 'u':
                image_out = np.multiply(image, imax_out,
                                        dtype=computation_type)
            else:
                image_out = np.multiply(image, (imax_out - imin_out) / 2,
                                        dtype=computation_type)
                image_out -= 1.0 / 2.
            np.rint(image_out, out=image_out)
            np.clip(image_out, imin_out, imax_out, out=image_out)
        elif kind_out == 'u':
            image_out = np.multiply(image, imax_out + 1,
                                    dtype=computation_type)
            np.clip(image_out, 0, imax_out, out=image_out)
        else:
            image_out = np.multiply(image, (imax_out - imin_out + 1.0) / 2.0,
                                    dtype=computation_type)
            np.floor(image_out, out=image_out)
            np.clip(image_out, imin_out, imax_out, out=image_out)
        return image_out.astype(dtype_out)

    # signed/unsigned int -> float
    if kind_out == 'f':
        # use float type that can exactly represent input integers
        computation_type = _dtype_itemsize(itemsize_in, dtype_out,
                                           np.float32, np.float64)

        if kind_in == 'u':
            # using np.divide or np.multiply doesn't copy the data
            # until the computation time
            image = np.multiply(image, 1. / imax_in,
                                dtype=computation_type)
            # DirectX uses this conversion also for signed ints
            # if imin_in:
            #     np.maximum(image, -1.0, out=image)
        else:
            image = np.add(image, 0.5, dtype=computation_type)
            image *= 2 / (imax_in - imin_in)

        return np.asarray(image, dtype_out)

    # unsigned int -> signed/unsigned int
    if kind_in == 'u':
        if kind_out == 'i':
            # unsigned int -> signed int
            image = _scale(image, 8 * itemsize_in, 8 * itemsize_out - 1)
            return image.view(dtype_out)
        else:
            # unsigned int -> unsigned int
            return _scale(image, 8 * itemsize_in, 8 * itemsize_out)

    # signed int -> unsigned int
    if kind_out == 'u':
        image = _scale(image, 8 * itemsize_in - 1, 8 * itemsize_out)
        result = np.empty(image.shape, dtype_out)
        np.maximum(image, 0, out=result, dtype=image.dtype, casting='unsafe')
        return result

    # signed int -> signed int
    if itemsize_in > itemsize_out:
        return _scale(image, 8 * itemsize_in - 1, 8 * itemsize_out - 1)

    image = image.astype(_dtype_bits('i', itemsize_out * 8))
    image -= imin_in
    image = _scale(image, 8 * itemsize_in, 8 * itemsize_out, copy=False)
    image += imin_out
    return image.astype(dtype_out)