import numpy as np
import torch
from scipy.fftpack import dct, idct
from scipy.ndimage import zoom
from PIL import Image
import random

class DataLoader:
    def __init__(self, data, labels, batch_size, shuffle=False, jpeg=False, dynamicQ = False, Q=90, subsampling="none", train_ratio=1):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory=True
        self.jpeg = jpeg
        self.Q = Q
        self.dynamicQ = dynamicQ
        self.subsampling = subsampling
        self.train_ratio = train_ratio
        self.labels = torch.LongTensor(labels).to(self.device)
        self.zero_count = [0,0,0]
        self.exit_count = 0
        self.data = self.process_image(data)
        self.indices = torch.arange(len(self.data))
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
        # train ratio will over shoot the batch size so we can reduce to correct batch size
        end = start + int(self.batch_size / self.train_ratio)
        batch_indices = self.indices[start:end]
        self.current_index = end

        return_data = self.data[batch_indices]
        return_labels = self.labels[batch_indices]
        # extract original batch size, effectively skipping some data for correct train ratio
        return return_data[:self.batch_size], return_labels[:self.batch_size]

    def process_image(self, data):
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
        scale_factor = 5000/self.Q if self.Q < 50 else 200 - self.Q*2
        self.scaled_lum_table = (self.luminance_table * scale_factor + 50) / 100
        self.scaled_chrom_table = (self.chrominance_table * scale_factor + 50) / 100
        self.scaled_lum_table = np.clip(self.scaled_lum_table, 1, 255).astype(np.uint8)
        self.scaled_chrom_table = np.clip(self.scaled_chrom_table, 1, 255).astype(np.uint8)

        compressed_data = []
        for idx, image in enumerate(data):

            image = np.rot90(image.reshape((32,32,3),order='F'), k=-1)
            if self.jpeg:
                image = self.rgb_to_ycbcr(image)
                # cifar10_classes = [
                #     "airplane", "automobile", "bird", "cat", "deer",
                #     "dog", "frog", "horse", "ship", "truck"
                # ]
                # print(cifar10_classes[self.labels[idx].item()])
                image = self.jpeg_compression(image)
            image = np.transpose(image, (2, 0, 1))
            tensor = torch.tensor(image.copy(), dtype=torch.float32)
            compressed_data.append(tensor)
        # chrom = 16
        # if self.subsample_chroma == "4:2:2":
        #     chrom = 8
        # elif self.subsample_chroma == "4:2:0":
        #     chrom = 4
        # print(f"{self.Q}: avg_y_0={self.zero_count[0]/(50000*16)}, avg_cb_0={self.zero_count[1]/(50000*chrom)}, avg_cr_0={self.zero_count[2]/(50000*chrom)}")
        # exit()
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
        # image = Image.fromarray(rgb, 'RGB')
        # image.save('output.bmp')
        # self.exit_count += 1
        # if self.exit_count == 4:
        #     exit()
        return rgb