from cython.cimports.libc.stdlib import malloc
from cython.cimports.libc.string import strcpy, strlen
import cv2, cython, math, shutil, bisect, os, time, sys, warnings
from typing import List, Optional
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import multiprocessing as mp
import queue
from tqdm import tqdm
from queue import Queue

_NUMBER_OF_COLOR_CHANNELS = 3

class FlowEstimator(nn.Module):
    """Small-receptive field predictor for computing the flow between two images.

    This is used to compute the residual flow fields in PyramidFlowEstimator.

    Note that while the number of 3x3 convolutions & filters to apply is
    configurable, two extra 1x1 convolutions are appended to extract the flow in
    the end.

    Attributes:
      name: The name of the layer
      num_convs: Number of 3x3 convolutions to apply
      num_filters: Number of filters in each 3x3 convolution
    """

    def __init__(self, in_channels: cython.int, num_convs: cython.int, num_filters: cython.int):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning
            )
        super(FlowEstimator, self).__init__()
        i: cython.int
        self._convs = nn.ModuleList()
        for i in range(num_convs):
            self._convs.append(Conv2d(in_channels=in_channels, out_channels=num_filters, size=3))
            in_channels = num_filters
        self._convs.append(Conv2d(in_channels, num_filters // 2, size=1))
        in_channels = num_filters // 2
        # For the final convolution, we want no activation at all to predict the
        # optical flow vector values. We have done extensive testing on explicitly
        # bounding these values using sigmoid, but it turned out that having no
        # activation gives better results.
        self._convs.append(Conv2d(in_channels, 2, size=1, activation=None))

    def forward(self, features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
        """Estimates optical flow between two images.

        Args:
          features_a: per pixel feature vectors for image A (B x H x W x C)
          features_b: per pixel feature vectors for image B (B x H x W x C)

        Returns:
          A tensor with optical flow from A to B
        """
        net = torch.cat([features_a, features_b], dim=1)
        for conv in self._convs:
            net = conv(net)
        return net

class PyramidFlowEstimator(nn.Module):
    """Predicts optical flow by coarse-to-fine refinement.
    """

    def __init__(self, filters: cython.int = 64,
                 flow_convs: tuple = (3, 3, 3, 3),
                 flow_filters: tuple = (32, 64, 128, 256)):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning
            )
        super(PyramidFlowEstimator, self).__init__()
        i: cython.int
        in_channels: cython.int
        in_channels = filters << 1
        predictors = []
        for i in range(len(flow_convs)):
            predictors.append(
                FlowEstimator(
                    in_channels=in_channels,
                    num_convs=flow_convs[i],
                    num_filters=flow_filters[i]))
            in_channels += filters << (i + 2)
        self._predictor = predictors[-1]
        self._predictors = nn.ModuleList(predictors[:-1][::-1])

    def forward(self, feature_pyramid_a: List[torch.Tensor],
                feature_pyramid_b: List[torch.Tensor]) -> List[torch.Tensor]:
        """Estimates residual flow pyramids between two image pyramids.

        Each image pyramid is represented as a list of tensors in fine-to-coarse
        order. Each individual image is represented as a tensor where each pixel is
        a vector of image features.

        util.flow_pyramid_synthesis can be used to convert the residual flow
        pyramid returned by this method into a flow pyramid, where each level
        encodes the flow instead of a residual correction.

        Args:
          feature_pyramid_a: image pyramid as a list in fine-to-coarse order
          feature_pyramid_b: image pyramid as a list in fine-to-coarse order

        Returns:
          List of flow tensors, in fine-to-coarse order, each level encoding the
          difference against the bilinearly upsampled version from the coarser
          level. The coarsest flow tensor, e.g. the last element in the array is the
          'DC-term', e.g. not a residual (alternatively you can think of it being a
          residual against zero).
        """
        levels: cython.int
        i: cython.int
        levels = len(feature_pyramid_a)
        v = self._predictor(feature_pyramid_a[-1], feature_pyramid_b[-1])
        residuals = [v]
        for i in range(levels - 2, len(self._predictors) - 1, -1):
            # Upsamples the flow to match the current pyramid level. Also, scales the
            # magnitude by two to reflect the new size.
            level_size = feature_pyramid_a[i].shape[2:4]
            v = F.interpolate(2 * v, size=level_size, mode='bilinear')
            # Warp feature_pyramid_b[i] image based on the current flow estimate.
            warped = warp(feature_pyramid_b[i], v)
            # Estimate the residual flow between pyramid_a[i] and warped image:
            v_residual = self._predictor(feature_pyramid_a[i], warped)
            residuals.insert(0, v_residual)
            v = v_residual + v

        for k, predictor in enumerate(self._predictors):
            i = len(self._predictors) - 1 - k
            # Upsamples the flow to match the current pyramid level. Also, scales the
            # magnitude by two to reflect the new size.
            level_size = feature_pyramid_a[i].shape[2:4]
            v = F.interpolate(2 * v, size=level_size, mode='bilinear')
            # Warp feature_pyramid_b[i] image based on the current flow estimate.
            warped = warp(feature_pyramid_b[i], v)
            # Estimate the residual flow between pyramid_a[i] and warped image:
            v_residual = predictor(feature_pyramid_a[i], warped)
            residuals.insert(0, v_residual)
            v = v_residual + v
        return residuals


class SubTreeExtractor(nn.Module):
    """Extracts a hierarchical set of features from an image.

    This is a conventional, hierarchical image feature extractor, that extracts
    [k, k*2, k*4... ] filters for the image pyramid where k=options.sub_levels.
    Each level is followed by average pooling.
    """

    def __init__(self, in_channels: cython.int=3, channels: cython.int=64, n_layers: cython.int=4):
        i: cython.int
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning
            )
        super().__init__()
        convs = []
        for i in range(n_layers):
            convs.append(nn.Sequential(
                Conv2d(in_channels, (channels << i), 3),
                Conv2d((channels << i), (channels << i), 3)
            ))
            in_channels = channels << i
        self.convs = nn.ModuleList(convs)

    def forward(self, image: torch.Tensor, n: cython.int) -> List[torch.Tensor]:
        """Extracts a pyramid of features from the image.

        Args:
          image: TORCH.Tensor with shape BATCH_SIZE x HEIGHT x WIDTH x CHANNELS.
          n: number of pyramid levels to extract. This can be less or equal to
           options.sub_levels given in the __init__.
        Returns:
          The pyramid of features, starting from the finest level. Each element
          contains the output after the last convolution on the corresponding
          pyramid level.
        """
        i: cython.int
        head = image
        pyramid = []
        for i, layer in enumerate(self.convs):
            head = layer(head)
            pyramid.append(head)
            if i < n - 1:
                head = F.avg_pool2d(head, kernel_size=2, stride=2)
        return pyramid

class FeatureExtractor(nn.Module):
    """Extracts features from an image pyramid using a cascaded architecture.
    """

    def __init__(self, in_channels: cython.int=3, channels: cython.int=64, sub_levels: cython.int=4):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning
            )
        super().__init__()
        self.extract_sublevels = SubTreeExtractor(in_channels, channels, sub_levels)
        self.sub_levels = sub_levels

    def forward(self, image_pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
        """Extracts a cascaded feature pyramid.

        Args:
          image_pyramid: Image pyramid as a list, starting from the finest level.
        Returns:
          A pyramid of cascaded features.
        """
        i: cython.int
        j: cython.int
        sub_pyramids: List[List[torch.Tensor]] = []
        for i in range(len(image_pyramid)):
            # At each level of the image pyramid, creates a sub_pyramid of features
            # with 'sub_levels' pyramid levels, re-using the same SubTreeExtractor.
            # We use the same instance since we want to share the weights.
            #
            # However, we cap the depth of the sub_pyramid so we don't create features
            # that are beyond the coarsest level of the cascaded feature pyramid we
            # want to generate.
            capped_sub_levels = min(len(image_pyramid) - i, self.sub_levels)
            sub_pyramids.append(self.extract_sublevels(image_pyramid[i], capped_sub_levels))
        # Below we generate the cascades of features on each level of the feature
        # pyramid. Assuming sub_levels=3, The layout of the features will be
        # as shown in the example on file documentation above.
        feature_pyramid: List[torch.Tensor] = []
        for i in range(len(image_pyramid)):
            features = sub_pyramids[i][0]
            for j in range(1, self.sub_levels):
                if j <= i:
                    features = torch.cat([features, sub_pyramids[i - j][j]], dim=1)
            feature_pyramid.append(features)
        return feature_pyramid

def get_channels_at_level(level: cython.int, filters):
    n_images: cython.int
    channels: cython.int
    flows: cython.int
    n_images = 2
    channels = _NUMBER_OF_COLOR_CHANNELS
    flows = 2
    return (sum(filters << i for i in range(level)) + channels + flows) * n_images

class Fusion(nn.Module):
    """The decoder."""

    def __init__(self, n_layers: cython.int=4, specialized_layers: cython.int=3, filters: cython.int=64):
        """
        Args:
            m: specialized levels
        """
        increase: cython.int
        i: cython.int
        in_channels: cython.int
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning
            )
        super().__init__()

        # The final convolution that outputs RGB:
        self.output_conv = nn.Conv2d(filters, 3, kernel_size=1)

        # Each item 'convs[i]' will contain the list of convolutions to be applied
        # for pyramid level 'i'.
        self.convs = nn.ModuleList()

        # Create the convolutions. Roughly following the feature extractor, we
        # double the number of filters when the resolution halves, but only up to
        # the specialized_levels, after which we use the same number of filters on
        # all levels.
        #
        # We create the convs in fine-to-coarse order, so that the array index
        # for the convs will correspond to our normal indexing (0=finest level).
        # in_channels: tuple = (128, 202, 256, 522, 512, 1162, 1930, 2442)

        in_channels = get_channels_at_level(n_layers, filters)
        increase = 0
        for i in range(n_layers)[::-1]:
            num_filters = (filters << i) if i < specialized_layers else (filters << specialized_layers)
            convs = nn.ModuleList([
                Conv2d(in_channels, num_filters, size=2, activation=None),
                Conv2d(in_channels + (increase or num_filters), num_filters, size=3),
                Conv2d(num_filters, num_filters, size=3)]
            )
            self.convs.append(convs)
            in_channels = num_filters
            increase = get_channels_at_level(i, filters) - num_filters // 2

    def forward(self, pyramid: List[torch.Tensor]) -> torch.Tensor:
        """Runs the fusion module.

        Args:
          pyramid: The input feature pyramid as list of tensors. Each tensor being
            in (B x H x W x C) format, with finest level tensor first.

        Returns:
          A batch of RGB images.
        Raises:
          ValueError, if len(pyramid) != config.fusion_pyramid_levels as provided in
            the constructor.
        """
        
        k: cython.int
        
        # As a slight difference to a conventional decoder (e.g. U-net), we don't
        # apply any extra convolutions to the coarsest level, but just pass it
        # to finer levels for concatenation. This choice has not been thoroughly
        # evaluated, but is motivated by the educated guess that the fusion part
        # probably does not need large spatial context, because at this point the
        # features are spatially aligned by the preceding warp.
        net = pyramid[-1]

        # Loop starting from the 2nd coarsest level:
        # for i in reversed(range(0, len(pyramid) - 1)):
        for k, layers in enumerate(self.convs):
            i = len(self.convs) - 1 - k
            # Resize the tensor from coarser level to match for concatenation.
            level_size = pyramid[i].shape[2:4]
            net = F.interpolate(net, size=level_size, mode='nearest')
            net = layers[0](net)
            net = torch.cat([pyramid[i], net], dim=1)
            net = layers[1](net)
            net = layers[2](net)
        net = self.output_conv(net)
        return net

def pad_batch(batch, align):
    width: cython.int
    height: cython.int
    height_to_pad: cython.int
    width_to_pad: cython.int
    align: cython.int
    # Omitted: batch, crop_region
    
    height, width = batch.shape[1:3]
    height_to_pad = (align - height % align) if height % align != 0 else 0
    width_to_pad = (align - width % align) if width % align != 0 else 0

    crop_region = [height_to_pad >> 1, width_to_pad >> 1, height + (height_to_pad >> 1), width + (width_to_pad >> 1)]
    batch = np.pad(batch, ((0, 0), (height_to_pad >> 1, height_to_pad - (height_to_pad >> 1)),
                           (width_to_pad >> 1, width_to_pad - (width_to_pad >> 1)), (0, 0)), mode='constant')
    return batch, crop_region

def load_image(img1, align: cython.int=64):
    #image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) / np.float32(255)
    #image = cv2.cvtColor(cv2.imencode('.png', img1)[1], cv2.COLOR_BGR2RGB).astype(np.float32) / np.float32(255)
    #image = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB).astype(np.float32) / np.float32(255)
    #image = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB).astype(np.float32) / np.float32(255)

    image = img1.astype(np.float32) / np.float32(255)
    image_batch, crop_region = pad_batch(np.expand_dims(image, axis=0), align)
    return image_batch, crop_region

def build_image_pyramid(image: torch.Tensor, pyramid_levels: cython.int = 3) -> List[torch.Tensor]:
    """Builds an image pyramid from a given image.

    The original image is included in the pyramid and the rest are generated by
    successively halving the resolution.

    Args:
      image: the input image.
      options: film_net options object

    Returns:
      A list of images starting from the finest with options.pyramid_levels items
    """
    i: cython.int
    pyramid = []
    for i in range(pyramid_levels):
        pyramid.append(image)
        if i < pyramid_levels - 1:
            image = F.avg_pool2d(image, 2, 2)
    return pyramid

def warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward warps the image using the given flow.

    Specifically, the output pixel in batch b, at position x, y will be computed
    as follows:
      (flowed_y, flowed_x) = (y+flow[b, y, x, 1], x+flow[b, y, x, 0])
      output[b, y, x] = bilinear_lookup(image, b, flowed_y, flowed_x)

    Note that the flow vectors are expected as [x, y], e.g. x in position 0 and
    y in position 1.

    Args:
      image: An image with shape BxHxWxC.
      flow: A flow with shape BxHxWx2, with the two channels denoting the relative
        offset in order: (dx, dy).
    Returns:
      A warped image.
    """
    
    ls1: cython.float
    ls2: cython.float
    
    flow = -flow.flip(1)

    dtype = flow.dtype
    device = flow.device

    # warped = tfa_image.dense_image_warp(image, flow)
    # Same as above but with pytorch
    ls1 = 1 - 1 / flow.shape[3]
    ls2 = 1 - 1 / flow.shape[2]

    normalized_flow2 = flow.permute(0, 2, 3, 1) / torch.tensor(
        [flow.shape[2] * .5, flow.shape[3] * .5], dtype=dtype, device=device)[None, None, None]
    normalized_flow2 = torch.stack([
        torch.linspace(-ls1, ls1, flow.shape[3], dtype=dtype, device=device)[None, None, :] - normalized_flow2[..., 1],
        torch.linspace(-ls2, ls2, flow.shape[2], dtype=dtype, device=device)[None, :, None] - normalized_flow2[..., 0],
    ], dim=3)

    warped = F.grid_sample(image, normalized_flow2,
                           mode='bilinear', padding_mode='border', align_corners=False)
    return warped.reshape(image.shape)

def multiply_pyramid(pyramid: List[torch.Tensor],
                     scalar: torch.Tensor) -> List[torch.Tensor]:
    """Multiplies all image batches in the pyramid by a batch of scalars.

    Args:
      pyramid: Pyramid of image batches.
      scalar: Batch of scalars.

    Returns:
      An image pyramid with all images multiplied by the scalar.
    """
    # To multiply each image with its corresponding scalar, we first transpose
    # the batch of images from BxHxWxC-format to CxHxWxB. This can then be
    # multiplied with a batch of scalars, then we transpose back to the standard
    # BxHxWxC form.
    return [image * scalar[..., None, None] for image in pyramid]

def flow_pyramid_synthesis(
        residual_pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
    """Converts a residual flow pyramid into a flow pyramid."""
    flow = residual_pyramid[-1]
    flow_pyramid: List[torch.Tensor] = [flow]
    for residual_flow in residual_pyramid[:-1][::-1]:
        level_size = residual_flow.shape[2:4]
        flow = F.interpolate(2 * flow, size=level_size, mode='bilinear')
        flow = residual_flow + flow
        flow_pyramid.insert(0, flow)
    return flow_pyramid

def pyramid_warp(feature_pyramid: List[torch.Tensor],
                 flow_pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
    """Warps the feature pyramid using the flow pyramid.

    Args:
      feature_pyramid: feature pyramid starting from the finest level.
      flow_pyramid: flow fields, starting from the finest level.

    Returns:
      Reverse warped feature pyramid.
    """
    warped_feature_pyramid = []
    for features, flow in zip(feature_pyramid, flow_pyramid):
        warped_feature_pyramid.append(warp(features, flow))
    return warped_feature_pyramid

def concatenate_pyramids(pyramid1: List[torch.Tensor],
                         pyramid2: List[torch.Tensor]) -> List[torch.Tensor]:
    """Concatenates each pyramid level together in the channel dimension."""
    result = []
    for features1, features2 in zip(pyramid1, pyramid2):
        result.append(torch.cat([features1, features2], dim=1))
    return result

class Conv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, size, activation: Optional[str] = 'relu'):
        assert activation in (None, 'relu')
        super().__init__(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=size,
                padding='same' if size % 2 else 0)
        )
        self.size = size
        self.activation = nn.LeakyReLU(.2) if activation == 'relu' else None

    def forward(self, x):
        if not self.size % 2:
            x = F.pad(x, (0, 1, 0, 1))
        y = self[0](x)
        if self.activation is not None:
            y = self.activation(y)
        return y

class filmIntLib:
    model                       = None
    model_path: str             = ''
    model_loaded: cython.bint   = False
    
    def __init__(self, model_path='', gpu=False, half=False):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning
            )
        super().__init__()
        if model_path != '':
            self.load_model(model_path, gpu, half)

    
    def load_model(self, model_path: str = '', gpu=False, half=False, keep_model_loaded=False):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning
            )
        if self.model_loaded:
            if keep_model_loaded and (self.model_path == model_path):
                return True
            else:
                del self.model
        self.model_path     = model_path
        self.model          = torch.jit.load(model_path, map_location='cpu')
        self.model.eval()
        if not half:
            self.model.float()
        if gpu and torch.cuda.is_available():
            if half:
                self.model = self.model.half()
            else:
                self.model.float()
            self.model = self.model.cuda()
        self.model_loaded = True
        return True
    
    def eval_model(self, model_path: str = '', gpu=False, half=False):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning
            )
        if not self.model_loaded:
            return False
        self.model_path     = model_path
        self.model          = torch.jit.load(model_path, map_location='cpu')
        self.model.eval()
        if not half:
            self.model.float()
        if gpu and torch.cuda.is_available():
            if half:
                self.model = self.model.half()
            else:
                self.model.float()
            self.model = self.model.cuda()
        return True

    def inference(self, model_path, img1, img2, gpu: cython.bint = True, inter_frames: cython.int = 1, half: cython.bint = False, keep_model_loaded: cython.bint = False, doProcessFrames: cython.bint = False):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning
            )
        if img1 is None:
            raise Exception("First image for inference was not supplied.")
        if img2 is None:
            raise Exception("Second image for inference was not supplied.")
        self.load_model(model_path, gpu, half, keep_model_loaded)
        
        img_batch_1, crop_region_1 = load_image(img1)
        img_batch_2, crop_region_2 = load_image(img2)

        img_batch_1 = torch.from_numpy(img_batch_1).permute(0, 3, 1, 2)
        img_batch_2 = torch.from_numpy(img_batch_2).permute(0, 3, 1, 2)
        
        results = [
            img_batch_1,
            img_batch_2
        ]

        idxes = [0, inter_frames + 1]
        remains = list(range(1, inter_frames + 1))

        splits = torch.linspace(0, 1, inter_frames + 2)
        
        do_cuda = (gpu and torch.cuda.is_available())
        
        for _ in range(len(remains)):
            starts = splits[idxes[:-1]]
            ends = splits[idxes[1:]]
            distances = ((splits[None, remains] - starts[:, None]) / (ends[:, None] - starts[:, None]) - .5).abs()
            matrix = torch.argmin(distances).item()
            start_i, step = np.unravel_index(matrix, distances.shape)
            end_i = start_i + 1

            x0 = results[start_i]
            x1 = results[end_i]

            if do_cuda:
                if half:
                    x0 = x0.half()
                    x1 = x1.half()
                x0 = x0.cuda()
                x1 = x1.cuda()

            dt = x0.new_full((1, 1), (splits[remains[step]] - splits[idxes[start_i]])) / (splits[idxes[end_i]] - splits[idxes[start_i]])

            with torch.no_grad():
                prediction = self.model(x0, x1, dt)
            insert_position = bisect.bisect_left(idxes, remains[step])
            idxes.insert(insert_position, remains[step])
            results.insert(insert_position, prediction.clamp(0, 1).cpu().float())
            del remains[step]
        y1, x1, y2, x2 = crop_region_1
        if doProcessFrames:
            frames = [(tensor[0] * 255).byte().flip(0).permute(1, 2, 0).numpy()[y1:y2, x1:x2].copy() for tensor in results]
            results = frames
        return (results, y1, x1, y2, x2)
    
    def inference2(
            self,
            model_path:             str,
            img1,
            img2,
            gpu:                    cython.bint = True,
            inter_frames:           cython.int = 1,
            half:                   cython.bint = False,
            keep_model_loaded:      cython.bint = False,
            doProcessFrames:        cython.bint = False,
            permute_already_done:   cython.bint = False
        ):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning
            )
        if img1 is None:
            raise Exception("First image for inference was not supplied.")
        if img2 is None:
            raise Exception("Second image for inference was not supplied.")
        self.load_model(model_path, gpu, half, keep_model_loaded)
        img_batch_1:    np.ndarray
        img_batch_2:    np.ndarray
        crop_region_1:  list            = []
        crop_region_2:  list            = []
        results:        list            = []
        idxes:          list            = []
        remains:        list            = []
        do_cuda:        cython.bint     = False
        remains_len:    cython.int      = 0
        start_i:        cython.int      = 0
        end_i:          cython.int      = 0
        step:           cython.int      = 0
        splits: torch.Tensor
        
        img_batch_1, crop_region_1 = img1
        img_batch_2, crop_region_2 = img2
        
        if not permute_already_done:
            img_batch_1 = torch.from_numpy(img_batch_1).cpu().permute(0, 3, 1, 2)
            img_batch_2 = torch.from_numpy(img_batch_2).cpu().permute(0, 3, 1, 2)
        
        results     = [img_batch_1, img_batch_2]
        idxes       = [0, inter_frames + 1]
        remains     = list(range(1, inter_frames + 1))
        splits      = torch.linspace(0, 1, inter_frames + 2)
        do_cuda     = (gpu and torch.cuda.is_available())
        remains_len = len(remains)
        
        for _ in range(remains_len):
            starts          = splits[idxes[:-1]]
            ends            = splits[idxes[1:]]
            distances       = ((splits[None, remains] - starts[:, None]) / (ends[:, None] - starts[:, None]) - .5).abs()
            matrix          = torch.argmin(distances).item()
            start_i, step   = np.unravel_index(matrix, distances.shape)
            end_i           = start_i + 1
            x0              = results[start_i]
            x1              = results[end_i]
            if do_cuda:
                if half:
                    x0 = x0.half()
                    x1 = x1.half()
                x0 = x0.cuda()
                x1 = x1.cuda()
            dt = x0.new_full((1, 1), (splits[remains[step]] - splits[idxes[start_i]])) / (splits[idxes[end_i]] - splits[idxes[start_i]])
            with torch.no_grad():
                prediction = self.model(x0, x1, dt)
            insert_position = bisect.bisect_left(idxes, remains[step])
            idxes.insert(insert_position, remains[step])
            results.insert(insert_position, prediction.clamp(0, 1).cpu().float())
            del remains[step]
        y1, x1, y2, x2 = crop_region_1
        if doProcessFrames:
            frames = [(tensor[0] * 255).byte().flip(0).permute(1, 2, 0).numpy()[y1:y2, x1:x2].copy() for tensor in results]
            results = frames
        return [results, y1, x1, y2, x2]


    def batch_inference(self, model_path, imgs, gpu, inter_frames, half, just_generated_frames=False, doProcessFrames=False, keep_model_loaded=False):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning
            )
        global_results = []
        
        img1 = None
        img2 = None
        
        first_run = True
        second_run = True
        first_inference = True
        
        self.load_model(model_path, gpu, half, keep_model_loaded)
        
        do_cuda = (gpu and torch.cuda.is_available())
        
        for img in imgs:
            if first_run:
                img1 = img
                first_run = False
                continue
            if second_run:
                second_run = False
                img2 = img
            else:
                img1 = img2
                img2 = img
            
            img_batch_1, crop_region_1 = load_image(img1)
            img_batch_2, crop_region_2 = load_image(img2)

            img_batch_1 = torch.from_numpy(img_batch_1).permute(0, 3, 1, 2)
            img_batch_2 = torch.from_numpy(img_batch_2).permute(0, 3, 1, 2)
            
            results = [
                img_batch_1,
                img_batch_2
            ]

            idxes = [0, inter_frames + 1]
            remains = list(range(1, inter_frames + 1))

            splits = torch.linspace(0, 1, inter_frames + 2)
            
            
            for _ in range(len(remains)):
                
                starts = splits[idxes[:-1]]
                ends = splits[idxes[1:]]
                
                distances = ((splits[None, remains] - starts[:, None]) / (ends[:, None] - starts[:, None]) - .5).abs()
                matrix = torch.argmin(distances).item()
                start_i, step = np.unravel_index(matrix, distances.shape)
                end_i = start_i + 1

                x0 = results[start_i]
                x1 = results[end_i]

                if do_cuda:
                    if half:
                        x0 = x0.half()
                        x1 = x1.half()
                    x0 = x0.cuda()
                    x1 = x1.cuda()

                dt = x0.new_full((1, 1), (splits[remains[step]] - splits[idxes[start_i]])) / (splits[idxes[end_i]] - splits[idxes[start_i]])

                with torch.no_grad():
                    prediction = self.model(x0, x1, dt)
                insert_position = bisect.bisect_left(idxes, remains[step])
                idxes.insert(insert_position, remains[step])
                results.insert(insert_position, prediction.clamp(0, 1).cpu().float())
                del remains[step]
            y1, x1, y2, x2 = crop_region_1
            if doProcessFrames:
                frames = [(tensor[0] * 255).byte().flip(0).permute(1, 2, 0).numpy()[y1:y2, x1:x2].copy() for tensor in results]
                results = frames
            if just_generated_frames:
                results = results[1:len(results)-1]
            global_results.append((results, y1, x1, y2, x2))
            first_inference = False
        return global_results

def perform_inference_two_images(img1, img2, frames_to_generate: cython.int=0, frames_to_remove: cython.int=0):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning
        )
    outputFrames: list = []
    outputFrames.append(img1)
    outputFrames.append(img1)# for testing
    outputFrames.append(img2)
    return outputFrames
