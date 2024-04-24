import numpy as np

def convolution(input_tensor, filters, stride=1, padding=0):
    input_height, input_width = input_tensor.shape[2:]
    filter_height, filter_width = filters.shape[2:]

    output_height = (input_height - filter_height + 2 * padding) // stride + 1
    output_width = (input_width - filter_width + 2 * padding) // stride + 1
    
    output = np.zeros((len(filters), output_height, output_width))
    
    if padding > 0:
        padded_imgs = []
        for img in input_tensor:
            padded_channels = []
            for channel in img:
                padded_channel = np.pad(channel, padding, mode='constant')
                padded_channels.append([padded_channel])

            padded_imgs.append(np.concatenate(padded_channels))
        input_tensor = np.concatenate([padded_imgs])
    
    for i in range(len(filters)):
        for y in range(0, output_height):
            for x in range(0, output_width):
                output[i, y, x] = np.sum(input_tensor[:, :, y*stride:y*stride+filter_height, x*stride:x*stride+filter_width] * filters[i])

    return output


if __name__ == "__main__":
    input_tensor = np.array([[[[1, 5, 1, 2],
                            [2, 3, 2, 3],
                            [1, 2, 1, 2],
                            [2, 3, 7, 3]],
                            
                            [[1, 4, 5, 2],
                            [4, 2, 1, 3],
                            [0, 1, 1, 0],
                            [-2, 4, -1, -3]]]])

    filter_weights = np.array([[[[1, 0],
                                [0, 1]],
                                [[1, 2],
                                [0, 1]]],
                               
                               [[[0, 1],
                                 [1, 0]],
                                [[2, 1],
                                 [0, 1]]]])

    result = convolution(input_tensor, filter_weights, stride=2, padding=1)
    print("Result:")
    print(result)