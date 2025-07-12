import numpy as np
import torch
import torchvision.transforms.functional as TF
from scipy.fftpack import dct, idct
from scipy.ndimage import zoom
from PIL import Image
import random

class DataLoader:
    def __init__(self, data, labels, batch_size, shuffle=False, jpeg=False, dynamicQ=False, Q=[100], subsampling="none", transform=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory=True
        self.jpeg = jpeg
        self.Q = Q if jpeg else [100]
        # transformations are really slow, try FFCV
        self.transform = transform
        self.dynamicQ = dynamicQ
        self.subsampling = subsampling
        self.labels = torch.LongTensor(labels).to(self.device)
        self.zero_count = [0,0,0]
        self.exit_count = 0
        self.data = []
        for qf in self.Q:
            self.data.append(self.process_image(data, qf))
        self.indices = torch.arange(len(data))
        if self.shuffle:
            self.indices = self.indices[torch.randperm(len(self.indices))]

    def __len__(self):
        return len(self.data) // self.batch_size

    def __iter__(self):
        self.current_index = 0
        if self.shuffle:
            self.indices = self.indices[torch.randperm(len(self.indices))]
        return self

    def __next__(self):
        if self.current_index >= len(self.indices):
            raise StopIteration

        start = self.current_index
        end = start + self.batch_size
        batch_indices = self.indices[start:end]
        self.current_index = end

        tuple_data =  list(zip(*[d[batch_indices] for d in self.data]))
        transformed_data= []
        for t in tuple_data:
            transformed_t = []
            for img in t:
                # pil = TF.to_pil_image(img)

                # if self.transform is not None:
                #     image_tensor = self.transform(pil)
                # else:
                #     image_tensor = TF.pil_to_tensor(pil).to(torch.float32)

                # transformed_t.append(image_tensor) 
                transformed_t.append(img) 
            transformed_data.append(torch.stack(transformed_t, dim=0))

        return_data = torch.stack(transformed_data, dim=0).to(self.device)
        return return_data, self.labels[batch_indices]

    def process_image(self, data, qf):
        self.luminance_table = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype='float32')


        self.chrominance_table = np.array([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
        ], dtype='float32')

        #scaling as defined by libjpeg
        scale_factor = 5000/qf if qf < 50 else 200 - qf*2
        self.scaled_lum_table = (self.luminance_table * scale_factor + 50) / 100
        self.scaled_chrom_table = (self.chrominance_table * scale_factor + 50) / 100
        self.scaled_lum_table = np.clip(self.scaled_lum_table, 1, 255).astype(np.uint8)
        self.scaled_chrom_table = np.clip(self.scaled_chrom_table, 1, 255).astype(np.uint8)

        compressed_data = []
        for idx, image in enumerate(data):

            image = np.rot90(image.reshape((32,32,3),order='F'), k=-1)
            if self.jpeg and qf != 100:
                image = self.rgb_to_ycbcr(image)
                image = self.jpeg_compression(image)
            image = np.transpose(image, (2, 0, 1))
            tensor = torch.tensor(image.copy(), dtype=torch.float32)
            compressed_data.append(tensor)
        return torch.stack(compressed_data).to(self.device)

    def rgb_to_ycbcr(self, image):
        transform_matrix = np.array([
            [0.299, 0.587, 0.114],
            [-0.1687, -0.3313, 0.5],
            [0.5, -0.4187, -0.0813]
        ])
        ycbcr = np.dot(image, transform_matrix.T)
        ycbcr[:, :, [1, 2]] = ycbcr[:, :, [1, 2]] + 128
        return ycbcr.astype('float32')

    def ycbcr_to_rgb(self, image):
        transform_matrix = np.array([
            [1, 0, 1.402],
            [1, -0.344136, -0.714136],
            [1, 1.772, 0]
        ])
        image[:, :, [1, 2]] = image[:, :, [1, 2]] - 128
        rgb = np.dot(image, transform_matrix.T)
        return np.clip(rgb, 0, 255).astype('uint8')

    def subsample_chroma(self, ycbcr):
        y = ycbcr[:, :, 0]
        cb = ycbcr[:, :, 1]
        cr = ycbcr[:, :, 2]

        if (self.subsampling == "4:2:2"):
            cb = cb[:, ::2]
            cr = cr[:, ::2]
        elif (self.subsampling =="4:2:0"):
            cb = cb[::2, ::2]
            cr = cr[::2, ::2]
        return y, cb, cr

    def upsample_chroma(self, y, cb, cr, order=0):
        if (self.subsampling =="4:2:2"):
            cb = zoom(cb, (1, 2), order=order)
            cr = zoom(cr, (1, 2), order=order)
        elif (self.subsampling == "4:2:0"):
            cb = zoom(cb, (2, 2), order=order)
            cr = zoom(cb, (2, 2), order=order)

        return np.stack([y, cb, cr], axis=2)

    def block_process_channel(self, channel, block_size, quant_table, channel_type):
        height, width = channel.shape
        processed_channel = np.zeros_like(channel)
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                block_height = min(block_size, height - i)
                block_width = min(block_size, width - j)
                block = channel[i:i+block_height, j:j+block_width]
                padded_block = self.pad_block(block, block_size)
                dct_block = self.dct_2d(padded_block.astype('float32'))
                quantized_block = self.quantize(dct_block, quant_table)
                # self.zero_count[channel_type] += np.count_nonzero(quantized_block == 0)
                idct_block = self.idct_2d(quantized_block)
                processed_channel[i:i+block_height, j:j+block_width] = idct_block[:block_height, :block_width]
                np.set_printoptions(suppress=True, formatter={'int': '{:d}'.format})


        return processed_channel

    def pad_block(self, block, block_size):
        padded_block = np.zeros((block_size, block_size))
        padded_block[:block.shape[0], :block.shape[1]] = block
        padded_block[block.shape[0]:, :] = np.mean(block)
        padded_block[:, block.shape[1]:] = np.mean(block)
        return padded_block

    def dct_2d(self, block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    def idct_2d(self, block):
            return idct(idct(block.T, norm='ortho').T, norm='ortho')

    def quantize(self, block, quant_table):
        return np.round(block / quant_table)

    def jpeg_compression(self, image, block_size=8):
        y, cb, cr = self.subsample_chroma(image)
        channels = [y, cb, cr]
        compressed_channels = []
        for index, channel in enumerate(channels):
            if index == 0:    
                quant_table = self.scaled_lum_table
            else: 
                quant_table = self.scaled_chrom_table
            compressed_channel = self.block_process_channel(channel, block_size, quant_table, index)
            if index == 0:
                processed_channel = np.clip(compressed_channel, 0, 255).astype('float32')
            else:
                processed_channel = np.clip(compressed_channel, 16, 240).astype('float32')
            compressed_channels.append(processed_channel)
        
        y, cb, cr = compressed_channels
        ycbcr = self.upsample_chroma(y, cb, cr)
        rgb = self.ycbcr_to_rgb(ycbcr)
        return rgb