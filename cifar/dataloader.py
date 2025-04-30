import numpy as np
import torch
from scipy.fftpack import dct, idct
from PIL import Image

class DataLoader:
    def __init__(self, data, labels, batch_size, shuffle=False, jpeg=False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory=True
        self.jpeg = jpeg
        self.labels = torch.LongTensor(labels).to(self.device)
        self.data = self.process_image(data)
        self.indices = torch.arange(len(self.data))
        if self.shuffle:
            self.indices = self.indices[torch.randperm(len(self.indices))]

    def __len__(self):
        return len(self.data) // self.batch_size

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= len(self.indices):
            raise StopIteration
        start = self.current_index
        end = start + self.batch_size
        batch_indices = self.indices[start:end]
        self.current_index = end
        return self.data[batch_indices], self.labels[batch_indices]

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
        self.luminance_table = self.luminance_table * 0.058

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
        self.chrominance_table = self.chrominance_table * 0.058
        compressed_data = []
        for image in data:
            image = np.rot90(image.reshape((32,32,3),order='F'), k=-1)
            if self.jpeg:
                image = self.rgb_to_ycbcr(image)
                image = np.transpose(image, (2, 0, 1))
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

    def block_process_channel(self, channel, block_size, quant_table):
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
                idct_block = self.idct_2d(quantized_block)
                processed_channel[i:i+block_height, j:j+block_width] = idct_block[:block_height, :block_width]

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
        compressed_channels = []
        for index, channel in enumerate(image):
            if index == 0:    
                quant_table = self.luminance_table
            else: 
                quant_table = self.chrominance_table
            compressed_channel = self.block_process_channel(channel, block_size, quant_table)
            if index == 0:
                processed_channel = np.clip(compressed_channel, 0, 255).astype('float32')
            else:
                processed_channel = np.clip(compressed_channel, 16, 240).astype('float32')
            compressed_channels.append(processed_channel)

        image = np.stack(compressed_channels, axis=-1)
        image = self.ycbcr_to_rgb(image)
        return image