import numpy as np
import cv2
import math
import operator
import heapq
import matplotlib.pyplot as plt
from PIL import Image
from binarytree import Node
from skimage.measure import compare_mse as mse
from skimage import img_as_float
from skimage.measure import compare_ssim as ssim
import datetime

import os
import cProfile

#Create an image class that holds image information so don't have to get calc image size everytime

# how to decrease the size of the matrix
def rgb2ycbcr(im):
    xform = np.array([[0.299, 0.587, 0.114],
                      [-0.1687, -0.3313, 0.5],
                      [0.5, -0.4187, -0.0813]])  #Create np array of convertion coeffcients
    ycbcr = im.dot(xform.T) # Find the dot product of the image and YCbCr matrix
    ycbcr[:,:,[1,2]] += 128 # Add 128 to the chrominance layers
    return np.float64(ycbcr)  # Return the converted image

def ycbcr2rgb(im): # edit variable names
    xform = np.array([[1, 0, 1.402],
                      [1, -0.34414, -.71414],
                      [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    #np.putmask(rgb, rgb > 255, 255)
    #np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

# Print arrays to validate output
def PrintImageArray2File(im, filename):
    # print array to file to checkc if buffered
    width, height, channels = im.shape
    string = ""
    file = open(filename, 'w')
    #
    #print("h: " + str(height) + " w: " + str(width))
    for x in range(width):
        for y in range(height):
            #string += str(float("%0.2f"%im[x,y, 0])) + ","
            string += str(int(im[x, y, 1])) + ","
            #string += str(im[x, y, 0]) + ","
            #print("y : " + str(y) + " x: " + str(x) + " value: " + str(im[y,x, 1]))
        string += '\n'
    file.write(string)
    file.close()

# Used for validating outputs and saving computation on DCT calculations
def PrintStream(stream, filename, csv):
    stringStream = ""
    file = open(filename, 'w')
    if csv == True:
        for i in range(len(stream)):
            stringStream += str(stream[i]) + ","
    else:
        for i in range(len(stream)):
            stringStream += str(stream[i])

    file.write(stringStream)
    file.close()

def SubsampleChrominance(im_ycbcr, samplingRatio):
    #4:2:0
    height, width, channels = im_ycbcr.shape
    Y = im_ycbcr[:,:,0]
    Cb = np.zeros((int(width / 2), int(height / 2)))
    Cr = np.zeros((int(width / 2), int(height / 2)))

    for i in range(int(width/2)):
        for j in range(int(height/2)):
            # subsample blue chrominance layer
            #print("i: " + str(i) + " j: " + str(j))
            avg = (im_ycbcr[(i * 2), (j * 2), 1] + im_ycbcr[(i * 2) + 1, (j * 2), 1] \
                  + im_ycbcr[(i * 2), (j * 2) + 1, 1] + im_ycbcr[(i * 2 ) + 1, (j * 2) + 1, 1]) / 4
            Cb[i, j] = int(avg)
            # subsample red chrominance layer
            avg = (im_ycbcr[(i * 2), (j * 2), 2] + im_ycbcr[(i * 2) + 1, (j * 2), 2] \
                   + im_ycbcr[(i * 2), (j * 2) + 1, 2] + im_ycbcr[(i * 2) + 1, (j * 2) + 1, 2]) / 4
            Cr[i, j] = int(avg)
    print("Subsample complete")

    YCbCrList = [Y, Cb, Cr]
    filenames = ['YlayerO.txt', 'CblayerO.txt', 'CrlayerO.txt']
    PrintYCbCrList(YCbCrList, filenames)
    return YCbCrList

def ResampleChromiance():
    # Increase chromiance layers
    x = 1

def ReduceChrominance(im_ycbcr):
    width, height, channels = im_ycbcr.shape
    for i in range(width):
        for j in range(height):
            #for k in range(2):
            im_ycbcr[i, j, 1] = im_ycbcr[i, j, 1] / 2  #down scaling by a factor of 2
            im_ycbcr[i, j, 2] = im_ycbcr[i, j, 2] / 2
    return im_ycbcr

def UpsampleChrominance(imList):
    width, height, channels = im_ycbcr.shape
    for i in range(width):
        for j in range(height):
            #for k in range(2):
            imList[1][i, j] = imList[1][i, j] * 2  #down scaling by a factor of 2
            imList[2][i, j] = imList[2][i, j] * 2
    return imList

def PadImage(im, blockSize):
    width, height, channels = im.shape

    ph = 0
    pw = 0
    # calculate addition pixel height
    r = height % blockSize
    if r != 0:
        ph = blockSize - r
    # calculate addition pixel width
    r = width % blockSize
    if r != 0:
        pw = blockSize - r

    resizedIm = im
    if (ph != 0 and pw != 0):
        #resizedIm = np.zeros((height + ph, width + pw, 3))  # create nparray with size of image + buffers
        resizedIm = np.zeros((width + pw, height + ph, 3))  # create nparray with size of image + buffers
        resizedIm[:im.shape[0], :im.shape[1], :im.shape[2]] = im
    return resizedIm

# Implement CUDA on DCT Block
# implement fast DCT 9 multiplications 1 addition
def BlockDCT(blockSize, channel, blockX , blockY):
    B = np.zeros((blockSize, blockSize))
    sumDCT = 0
    for u in range(blockSize):
        for v in range(blockSize):
            B[u, v] = 0
            for x in range(blockSize):    # Loop through blockY pixel values
                for y in range(blockSize):   # Loop through blockX pixel values
                    B[u,v] += channel[(blockSize * blockX) + x, (blockSize * blockY) + y] * \
                              math.cos(math.pi / blockSize * (x+0.5) * u) * math.cos(math.pi / blockSize * (y+0.5) * v)

    return B

def CreateDCTMatrix(QM, YCbCrList, blockSize):
    #height, width, channels = im.shape
    stream = []
    DCC = [] # Direct current components
    ImageList = []
    ImageList2 = []
    DC = 0
    count = 0
    previousDC = 0
    DPCM = 0
    channels = 3 # change to get value from image using .shape
    #for channel in range(channels): # loop through YCbCr channels
    #for channel in YCbCrList:
    for c in range(channels):
        width, height = YCbCrList[c].shape
        Test = np.zeros((width, height))
        Test2 = np.zeros((width, height))
        for blockX in range(int(width/blockSize)): # loop through number of blocks in X
            for blockY in range(int(height/blockSize)):  # loop through number of blocks in Y
                #print("BX: " + str(blockX) + " BY: " + str(blockY))
                DCT = BlockDCT(blockSize, YCbCrList[c], blockX, blockY)

                QDCT = QuantiseBlock(QM, DCT, blockSize)    # Quantise the DCT block

                #B7 = DequantiseBlock(QM, QDCT, blockSize)
                #print(B7)

                # find DPCM difference - make into function
                DC = QDCT[0, 0] # DC coefficient
                DPCM = DC
                if (blockX != 0 and blockY != 0):
                    DPCM = DC - previousDC
                    previousDC = DPCM
                QDCT[0, 0] = DPCM


                # zigzag scan the remaining values
                stream += ZigzagEncoding(QDCT, blockSize)   # Zigzag scan the QDCT block and append to zigzag stream
                # Copies QDCT block into FullQDCT(holds all QDCT blocks) (used for debugging reasons)
                for x in range(blockSize):
                   for y in range(blockSize):
                        Test[x + (blockSize * blockX), y + (blockSize * blockY)] = QDCT[x, y] # error here
                        Test2[x + (blockSize * blockX), y + (blockSize * blockY)] = DCT[x, y]  # error here

        ImageList.append(Test)
        ImageList2.append(Test2)
       # print("channel: " + str(channel))
    # get each process individually
    PrintYCbCrList(ImageList2[0], 'YlayerDCT.txt')
    PrintYCbCrList(ImageList2[1], 'CblayerDCT.txt')
    PrintYCbCrList(ImageList2[2], 'CrlayerDCT.txt')

    PrintYCbCrList(ImageList[0], 'YlayerQDCT.txt')
    PrintYCbCrList(ImageList[1], 'CblayerQDCT.txt')
    PrintYCbCrList(ImageList[2], 'CrlayerQDCT.txt')


    print("QDCT Matrix and zigzag scan complete")


    return stream

def equaliseImage(im):
    # subtract 128 from each value because dct is designed to work on pixel values ranging from -128 to 127
    width, height, channels = im.shape
    M = np.zeros((width, height, channels))
    for x in range(width):
        for y in range(height):
            for c in range(channels):
                M[x, y, c] = np.float64(im[x, y, c]) - np.float64(128)
    print("Image Equalisation complete")
    return M

def UnnormaliseImage(im, blockSize):
    M = np.zeros((blockSize, blockSize))
    for x in range(blockSize):
        for y in range(blockSize):
                M[x, y] = np.float64(im[x, y]) + np.float64(128)
    return M

def QuantiseBlock(QM, B, blockSize):
    A = np.zeros((blockSize, blockSize))

    for i in range(blockSize):
        for j in range(blockSize):
            A[i, j] = math.ceil(B[i, j] / QM[i, j])
    return A

def DequantiseBlock(QM, B, blockSize):
    A = np.zeros((blockSize, blockSize))

    for i in range(blockSize):
        for j in range(blockSize):
            A[i, j] = math.ceil(B[i, j] * QM[i, j])
            #print(str(B[i, j]) + " * " + str(QM[i, j]) + " = " + str(A[i, j]))
    return A

def CreateQuantizationMatrix(quality, blockSize):

    #Read in from a file
    QM = [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]]
    QM = np.array(QM)

    if (quality < 50):
        S = 5000/quality
    else:
        S = 200 - (2*quality)

    for i in range(blockSize):
        for j in range(blockSize):
            QM[i, j] = math.floor((S * QM[i, j] + 50) / 100)
            if QM[i, j] == 0:
                QM[i, j] = 1
    return QM

def ZigzagEncoding(QDCT, blockSize):

    stream = []
    QDCT = np.array(QDCT)
    for elementNum in range(0, blockSize * blockSize):
        stream.append(int(QDCT[ZigzagScan(elementNum, blockSize)]))
    return stream

# matrix to zigzag string
# store DC values into seperate array they will be delta encoded
def ZigzagScan(elementNum, blockSize):
    # upper side of interval
    if elementNum >= blockSize * (blockSize + 1) // 2:
        i, j = ZigzagScan(blockSize * blockSize - 1 - elementNum, blockSize)
        return blockSize - 1 - i, blockSize - 1 - j
    # lower side of interval
    i = int((np.sqrt(1 + 8 * elementNum) - 1) / 2)
    j = elementNum - i * (i + 1) // 2
    return (j, i - j) if i & 1 else (i - j, j)

# zigzag string to matrix
def ZigzagCreateMatrix(i, j, blockSize):
    # upper side of interval
    if i + j >= blockSize:
        return blockSize * blockSize - 1 - ZigzagCreateMatrix(blockSize - 1 - i, blockSize - 1 - j, blockSize)
    # lower side of interval
    k = (i + j) * (i + j + 1) // 2
    return k + i if (i + j) & 1 else k + j

def DPCM(DCC):
    x = 1

#def cmp(a, b):
#    return (a > b) - (a < b)

class Node:
    left = None
    right = None
    data = None
    freq = 0

    def __init__(self, data, freq):
        self.data = data
        self.freq = freq

    def __lt__(self, value):
        return self.freq < value.freq

    def __eq__(self, other):
        if (other == None):
            return False
        if (not isinstance(other, Node)):
            return False
        return self.freq == other.freq

    def SetChildren(self, left, right):
        self.left = left
        self.right = right

    def GetLeft(self):
        return self.left

    def GetRight(self):
        return self.right

    def GetFreq(self):
        return self.freq

class Tree:
    heap = []
    codes = {}
    reverseMap = {}
    root = Node(0, 0)

def MakeTree(fqList):
    HuffTree = Tree()
    newNode = Node(0, 0)

    # make heap
    for i in range(len(fqList)):  # construct nodes for each value and freq and push onto a heap
        node = Node(fqList[i][0], int(fqList[i][1]))
        heapq.heappush(HuffTree.heap, node)                     #

    # merge nodes
    while (len(HuffTree.heap) > 1):
        node1 = heapq.heappop(HuffTree.heap)  # Pop the 2 least frequent values
        node2 = heapq.heappop(HuffTree.heap)

        newNode = Node(None, node1.GetFreq() + node2.GetFreq()) #Sum the freqs and the two nodes
        newNode.SetChildren(node1, node2)                       #Set the two nodes to be the children to the newNode
        heapq.heappush(HuffTree.heap, newNode)                  #push newNode back onto the heap

    HuffTree.root = newNode
    return HuffTree


def MakeCodes(HuffTree, root, code):
    if (root == None):  # If node is a leaf node
        return

    if (root.data != None): # If node has a data component
        HuffTree.codes[root.data] = code    #look up key (root.data) and make the value = codes
        HuffTree.reverseMap[code] = root.data   # inverse of codes - reverse map used to decode binary stream
        return

    # if left node is a leaf node then method will return and take right node if possible
    # if neither then recurse up stack
    MakeCodes(HuffTree, root.left, code + "0")  # append 0 to code when taking left node
    MakeCodes(HuffTree, root.right, code + "1") # append 1 to code when taking right node

def CreateBinaryStream(HuffTree, stream):
    # Make code dictionary
    #root = heapq.heappop(HuffTree.heap)
    code = ""
    MakeCodes(HuffTree, HuffTree.root, code)     # Traverse tree to

    print("# of codes: " + str(len(HuffTree.codes)))
    print("Huffman tree hash table:")
    print(HuffTree.codes)
    print(" ")
    # Convert stream to codes
    binaryStream = []
    for value in stream:                                    #loop through all key:value pairs
        binaryStream.append(int(HuffTree.codes.get(value)))      #look up value in hash table and append corrosponding code

    print("Binary Stream constructed")
    return binaryStream

def ConstructHuffmanTree(stream):

    # create hash table that accumulates the frequency of each value
    fqHashTable = {}
    for value in stream:
        if value not in fqHashTable:
            fqHashTable.update({value: 1})
        else:
            fqHashTable[value] += 1

    # sort hash table by frequency
    fqList = sorted(fqHashTable.items(), key=operator.itemgetter(1))

    print(fqList)

    # construct huffman tree based on node freq
    HuffTree = MakeTree(fqList)
    print("HuffTree Created")
    return HuffTree

# traverse tree because new user will not have the tree with node values
# very slow
def GetNodeData(root, huffmanBinaryStream):
    code = ""
    QDCTValueStream = []
    for bit in huffmanBinaryStream:
        code += bit
        if code in HuffTree.reverseMap:
            QDCTValueStream.append(int(HuffTree.reverseMap[code]))

    return QDCTValueStream

def RunLengthEncoding(stream):
    zeroCount = 0
    for value in stream:
        if (value == 0):
            zeroCount += 1
        else:
            if zeroCount > 0:
                # code
                zeroCount = 0

def DecodeStream(HuffTree, huffmanBinaryStream):
    code = ""
    QDCTValueStream = []
    i = 0
    for bit in huffmanBinaryStream:
        code += str(bit)
        #i = i + 1
        #print(str(i) + " / " + str(len(huffmanBinaryStream)))
        if code in HuffTree.reverseMap:
            QDCTValueStream.append(HuffTree.reverseMap[code])
            code = ""
    return QDCTValueStream

def ReconstructQDCTArray(QDCTList, QM, blockSize, width, height, channel):

    YCbCrList = []
    IQDCTList = []
    IDCTList = []
    previousDC = 0
    ImageList = []
    for c in range(int(channel)):
        B5 = np.zeros((width, height))
        B6 = np.zeros((width, height))
        B7 = np.zeros((width, height))

        for blockX in range(int(width/blockSize)): # loop through number of blocks in X
            for blockY in range(int(height/blockSize)):  # loop through number of blocks in Y

                # re-zigzag data and place into a QDCT block
                B1 = np.zeros((blockSize, blockSize))

                for x in range(blockSize):
                    for y in range(blockSize):
                        B1[x, y] = QDCTList[ZigzagCreateMatrix(x, y, blockSize)]
                QDCTList = QDCTList[(blockSize * blockSize):]

                #place DC coefficient back into block
                DC = B1[0, 0]
                if (blockY != 0 and blockX != 0):
                    B1[0, 0] = DC + previousDC
                    previousDC = DC

                # Dequantize the block
                B2 = DequantiseBlock(QM, B1, blockSize) # Dequantise block
                B3 = IDCT(B2)                           # Perform inverse DCT on block
                B4 = UnnormaliseImage(B3, blockSize)    # unormalise image

                for x in range(blockSize):  # perhaps index doesn't work?
                    for y in range(blockSize):
                        B5[x + (blockSize * blockX), y + (blockSize * blockY)] = B2[x, y]  # error here
                        B6[x + (blockSize * blockX), y + (blockSize * blockY)] = B3[x, y]  # error here
                        B7[x + (blockSize * blockX), y + (blockSize * blockY)] = B4[x, y]  # error here

        YCbCrList.append(B7)
        IDCTList.append(B6)
        IQDCTList.append(B5)

    #print(IQDCTList[0][0, 0])
    PrintYCbCrList(IQDCTList[0], 'YlayerIQDCT.txt')
    PrintYCbCrList(IQDCTList[1], 'CblayerIQDCT.txt')
    PrintYCbCrList(IQDCTList[2], 'CrlayerIQDCT.txt')

    PrintYCbCrList(IDCTList[0], 'YlayerIDCT.txt')
    PrintYCbCrList(IDCTList[1], 'CblayerIDCT.txt')
    PrintYCbCrList(IDCTList[2], 'CrlayerIDCT.txt')

    return YCbCrList
    #PrintImageArray2File(B4, 'FullQCT2.csv')


def IDCT(B1):
    B2 = np.zeros((blockSize, blockSize))
    for u in range(blockSize):
        for v in range(blockSize):
            B2[u][v] = 0.25 * B1[0][0]
            for i in range(1, blockSize):
                B2[u][v] += 0.5 * B1[i][0]
            for j in range(1, blockSize):
                B2[u][v] += 0.5 * B1[0][j]

            for i in range(1, blockSize):
                for j in range(1, blockSize):
                    B2[u][v] += B1[i][j] * math.cos((math.pi/blockSize) * (u + 0.5) * (i)) \
                                * math.cos((math.pi/blockSize) * (v + 0.5) * (j))


            B2[u][v] *= (2/blockSize) * (2/blockSize)
    #print(B2)
    return B2

def PrintTree(HuffTree):
    x = 1

def PrintYCbCrList(A, filename):
    # print array to file to checkc if buffered
    width, height = A.shape
    string = ""
    file = open(filename, 'w')
    for x in range(width):
        for y in range(height):
            string += str(int(A[x, y])) + ","
        string += '\n'
    file.write(string)
    file.close()

def mseA(im1, im2): # calculate the MSE
    mse = np.mean((im1 - im2) ** 2)
    return mse

def psnr(img1, img2): # calculate the PSNR
    mse = np.mean( (img1 - img2) ** 2 )
    return 20 * math.log10(255 / math.sqrt(mse))

def msnr(imOutput, mse): # calculate the MSNR
    squaredMean = np.mean(np.power(imOutput, 2))
    msnr = squaredMean / math.sqrt(mse)
    return msnr


# Read in image
im = cv2.imread('building2.png')
quality = 90# Quantization matrix scaled to specified quality

# Select block size
blockSize = 8;

startFinal = 0
startFinal = datetime.datetime.now()

# Convert bmp to YCbCr
im_ycbcr = rgb2ycbcr(im)
#PrintImageArray2File(im_ycbcr, 'YUVImage.txt')
print("image colour space convertion completed")

PrintYCbCrList(im_ycbcr[:,:,0], 'YlayerOrig.txt')
PrintYCbCrList(im_ycbcr[:,:,1], 'CblayerOrig.txt')
PrintYCbCrList(im_ycbcr[:,:,2], 'CrlayerOrig.txt')

#down sample chrominance
im_ycbcr = ReduceChrominance(im_ycbcr)

# Equalise image
M = equaliseImage(im_ycbcr)  # Centre between -127 and 128

# Add buffer around image so blocks are equal
M = PadImage(M, blockSize)
print("Padding image complete")

YCbCrList = [M[:,:,0], M[:,:,1], M[:,:,2]]

PrintYCbCrList(YCbCrList[0], 'YlayerPre.txt')
PrintYCbCrList(YCbCrList[1], 'CblayerPre.txt')
PrintYCbCrList(YCbCrList[2], 'CrlayerPre.txt')

QM = CreateQuantizationMatrix(quality, blockSize)  # Create Quantization matrix

# Create Quantized AC values and remove DC values int seperate streams
start = datetime.datetime.now()
stream = CreateDCTMatrix(QM, YCbCrList, blockSize)                            # (UNCOMMENT)
finish = datetime.datetime.now()
time = finish - start
print("Scan creation time: " + str(time))
PrintStream(stream, 'UncompressedValueStream.txt', True) # print stream of QDCT values

########
# Create Run length encoding
#RunLengthEncoding(stream)


HuffTree = ConstructHuffmanTree(stream) # construct the huffman tree
#PrintTree(HuffTree)

#cProfile.run('HuffTree = ConstructHuffmanTree(stream)')    # UNCOMMENT
start = datetime.datetime.now()
binaryStream = CreateBinaryStream(HuffTree, stream)               # (UNCOMMENT)
finish = datetime.datetime.now()
time = finish - start
print("Binary Stream: " + str(time))

#cProfile.run('binaryStream = CreateBinaryStream(HuffTree, stream)')    # UNCOMMENT
PrintStream(binaryStream, 'binaryStreamTest.txt', False)           #print binary stream to file

# Load data in from file including data -  data like hufftree and block size
# DECODER - method stubs for values
start = datetime.datetime.now()
QDCTValueStream = DecodeStream(HuffTree, binaryStream)
finish = datetime.datetime.now()
time = finish - start
print("Decode binary stream into QDCT values - time: " + str(time))

PrintStream(QDCTValueStream, 'CompressedValueStreamTest.txt', True)
print("Decode stream complete")

#print(QDCTValueStream)
width, height, channels = im.shape
start = datetime.datetime.now()
YCbCrList = ReconstructQDCTArray(QDCTValueStream, QM, blockSize, width, height, channels) #QDCTValueStream = QDCTList
finish = datetime.datetime.now()
time = finish - start
print("Recostruct image from value stream - time: " + str(time))
print("Reconstruct image complete")

#PrintYCbCrList(YCbCrList[0], 'YlayerF.txt')
#PrintYCbCrList(YCbCrList[1], 'CblayerF.txt')
#PrintYCbCrList(YCbCrList[2], 'CrlayerF.txt')


YCbCrList = UpsampleChrominance(YCbCrList) #upsample chrominance

PrintYCbCrList(YCbCrList[0], 'YlayerF2.txt')
PrintYCbCrList(YCbCrList[1], 'CblayerF2.txt')
PrintYCbCrList(YCbCrList[2], 'CrlayerF2.txt')

# create output image array
width, height, channel = im.shape
imOutput = np.zeros((width, height, channel))

# lazy implementation to turn YCbCr list into np array
for c in range(channel):
    for i in range(width):
        for j in range(height):
                imOutput[i, j, c] = YCbCrList[c][i, j]

w, h, c = imOutput.shape
imOutput = ycbcr2rgb(imOutput)

FinalFinal = datetime.datetime.now()
finalTime = FinalFinal - startFinal

string = ""

file = open('OutputBlayer.txt', 'w')
for x in range(128):
    for y in range(128):
        string += str(int(imOutput[x, y, 2])) + ","
    string += '\n'
file.write(string)
file.close()

mse = mseA(im, imOutput)
print("Computation time:           " + str(finalTime))
print("Compression Ratio:          " + str( (w * h * (c * 8)) / len(binaryStream) ) + ":1")
print("Mean squared error:         " + str(mse))
print("Mean Signal to Noise Ratio: " + str(msnr(imOutput, mse)))
print("Peak signal to Noise Ratio: " + str(psnr(im, imOutput)))

cv2.imshow('Uncompressed JPG', imOutput)
cv2.imwrite('CompressedImage.bmp', imOutput)
cv2.waitKey()

