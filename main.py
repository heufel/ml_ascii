from scipy.ndimage import convolve
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os, sys, logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel("FATAL")

def imgToGreyscale(img: np.ndarray, weights = [0.2126, 0.7152, 0.0722]):
    grey = np.zeros(img.shape[:2])
    for i in range(3):
        grey += img[:,:,i] * weights[i]
    return grey

def sobel(img):
    img = imgToGreyscale(img)
    xSobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    ySobel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    xEdges = convolve(img, xSobel, mode="constant")
    yEdges = convolve(img, ySobel, mode="constant")
    return np.sqrt(np.pow(xEdges, 2) + np.pow(yEdges, 2))

def split(img, shape: tuple[int, int]):
    xStep = img.shape[0]/shape[0]
    yStep = img.shape[1]/shape[1]
    splits = []
    for i in range(shape[0]):
        xsplit = []
        for j in range(shape[1]):
            xsplit.append(img[int(i*xStep):int((i+1)*xStep), int(j*yStep):int((j+1)*yStep)])
        splits.append(xsplit)
    return splits

def avgSectionColor(img, dims):
    pil_img = Image.fromarray((img*255).astype(np.uint8))
    pil_img = pil_img.resize((dims[1], dims[0])) #i don't know why this only works like this, it should be the same because resize takes (width, heigh) 
    avgColor = np.asarray(pil_img)
    return avgColor.astype(np.float32)/255

def thresholdSectionAlpha(img, min_avg_alpha) -> bool:
    checks = []
    for i in range(len(img)):
        xchecks = []
        for j in range(len(img[i])):
            section = img[i][j]
            xchecks.append(section.T[3].mean() >= min_avg_alpha)
        checks.append(xchecks)
    return checks

def thresholdSectionEdgeStrength(gradient, min_gradient):
    checks = []
    for i in range(len(gradient)):
        xchecks = []
        for j in range(len(gradient[i])):
            section = gradient[i][j]
            xchecks.append(np.max(section) >= min_gradient)
        checks.append(xchecks)
    return checks

def crop_center(img, crop_width, crop_height):
    img_width, img_height = img.size
    return img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def outputChars(edges, model, class_names):
    scaled_edges = []
    for edge in edges:
        width = len(edge)
        height = len(edge[0])
        edge = Image.fromarray(edge)
        edge = crop_center(edge, min(width, height), min(width, height))
        edge = edge.resize((64, 64))
        scaled_edges.append(np.asarray(edge)[:, :, np.newaxis])
    scaled_edges = np.expand_dims(scaled_edges, 0)
    dataset = tf.data.Dataset.from_tensor_slices(scaled_edges)
    predictions = model.predict(dataset)
    return [chr(int(class_names[prediction])) for prediction in np.argmax(predictions, 1)]

def imgToAscii(path: str, shape: tuple[int,int],  model, class_names, fillChar = "#"):
    img = plt.imread(path)
    avg_color = avgSectionColor(img, shape)
    edges = sobel(img)
    edges = np.where(edges >= 0.8, edges, 0)
    split_img = split(img, shape)
    split_edges = split(edges, shape)
    renderEdge = thresholdSectionEdgeStrength(split_edges, 0.8)
    renderSpace = np.logical_not(np.logical_or(thresholdSectionAlpha(split_img, 0.4), renderEdge))
    plt.imsave("test_imgs/avg.png", avg_color)
    lines = []
    processQueue = []
    processQueueIndices = []
    for i in range(len(split_img)):
        chars = []
        for j in range(len(split_img[i])):
            colorChar = '\033[38;2;{};{};{}m'.format(*[int(col * 255) if not np.isnan(col)  else 0 for col in avg_color[i][j][0:3]])
            if renderEdge[i][j]:
                processQueue.append(split_edges[i][j])
                processQueueIndices.append([i,j])
            elif renderSpace[i][j]:
                colorChar = (' ')
            else:
                colorChar += (fillChar)
                
            chars.append(colorChar)
        
        lines.append(chars)
    predictions = outputChars(processQueue, model, class_names)
    for i,char in enumerate(predictions):
        x, y = processQueueIndices[i]
        lines[x][y] += char
    return lines

if __name__ == "__main__":
    path=sys.argv[1]
    width=int(sys.argv[2])
    height=int(sys.argv[3])
    model = tf.keras.models.load_model("ascii_reader.keras")
    with open("class_names.txt", "r") as f:
        class_names = f.readlines()
    
    asciiList = imgToAscii(path, (width, height), model, class_names)
    asciiStr = '\n'.join([''.join(line) for line in asciiList if ''.join(line).replace(" ", '') != ''])+'\033[0m'

    if len(sys.argv) > 4:
        output = open(sys.argv[4], 'w')
    else:
        output = sys.stdout
    output.write(asciiStr)
    output.close()

