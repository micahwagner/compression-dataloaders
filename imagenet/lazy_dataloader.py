import numpy as np
import cv2
import torch
from scipy.fftpack import dct, idct
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

class Dataset:
    def __init__(self, order, path, labels, jpeg=False):
        self.order = order
        self.path = path
        self.jpeg = jpeg
        self.labels = labels
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

    def __len__(self):
        return len(self.order)

    def __getitem__(self, index):
        img_path = f"{self.path}{self.order[index]}"
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image = image[16:-16, 16:-16]
        image = np.array(image)
        if self.jpeg:
            image = self.process_image(image)
        else:
            image = np.transpose(image, (2, 0, 1))
        return image, self.labels[index]
    
    def process_image(self, data):
        image = self.rgb_to_ycbcr(data)
        image = np.transpose(image, (2, 0, 1))
        image = self.jpeg_compression(image)
        image = np.transpose(image, (2, 0, 1))
        return image

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

class DataLoader:
    def __init__(self, dataset, batch_size, num_workers=4, shuffle=False, pin_memory=True):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.indices = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(self.indices)

    def _worker(self, indices, queue):
        try:
            for idx in indices:
                item = self.dataset[idx]
                queue.put(item)
        finally:
            queue.put("__done__")

    def __iter__(self):
        if self.num_workers > 0: # multi-thread
            # create worker procs with chunks of indices to get
            chunk_size = int(np.ceil(len(self.indices) / self.num_workers))
            chunks = [self.indices[i:i+chunk_size] for i in range(0, len(self.indices), chunk_size)]

            ctx = mp.get_context("spawn")
            queue = ctx.Queue()
            processes = []
            for i, chunk in enumerate(chunks):
                p = ctx.Process(target=self._worker, args=(chunk, queue))
                p.start()
                processes.append(p)
            
            # get batches from dataset
            batch_data = []
            batch_labels = []
            num_done = 0
            total_expected_done = len(processes)
            while True:
                item = queue.get()
                if isinstance(item, str) and item == "__done__":
                    num_done += 1
                    if num_done == total_expected_done:
                        break
                    continue
                data = torch.tensor(item[0], dtype=torch.float32)
                label = torch.tensor(item[1], dtype=torch.long)
                batch_data.append(data)
                batch_labels.append(label)
                if len(batch_data) == self.batch_size:
                    batch_data = torch.stack(batch_data)
                    batch_labels = torch.stack(batch_labels)
                    if self.pin_memory:
                        batch_data = batch_data.pin_memory()
                        batch_labels = batch_labels.pin_memory()
                    yield batch_data.to(self.device, non_blocking=self.pin_memory), batch_labels.to(self.device, non_blocking=self.pin_memory)
                    batch_data = []
                    batch_labels = []

            if batch_data:
                batch_data = torch.stack(batch_data)
                batch_labels = torch.stack(batch_labels)
                if self.pin_memory:
                    batch_data = batch_data.pin_memory()
                    batch_labels = batch_labels.pin_memory()
                yield batch_data.to(self.device, non_blocking=self.pin_memory), batch_labels.to(self.device, non_blocking=self.pin_memory)

            # close processes after finish
            for p in processes:
                p.join()

        else: # single thread
            batch_data = []
            batch_labels = []
            for idx in self.indices:
                data, label = self.dataset[idx]
                data = torch.tensor(item[0], dtype=torch.float32)
                label = torch.tensor(item[1], dtype=torch.long)
                batch_data.append(data)
                batch_labels.append(label)
                if len(batch_data) == self.batch_size:
                    batch_data = torch.stack(batch_data)
                    batch_labels = torch.stack(batch_labels)
                    if self.pin_memory:
                        batch_data = batch_data.pin_memory()
                        batch_labels = batch_labels.pin_memory()
                    yield batch_data.to(self.device, non_blocking=self.pin_memory), batch_labels.to(self.device, non_blocking=self.pin_memory)
                    batch_data = []
                    batch_labels = []

    def __len__(self):
        return len(self.dataset) // self.batch_size