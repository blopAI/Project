def buildHeatmapImage(vertexPositions, size=340, vertexSize=4, sigma=0, power=0.625):
    x = vertexPositions[:, 0]
    y = vertexPositions[:, 2]  
    z = vertexPositions[:, 1]  

    normX = (x - x.min()) / (x.max() - x.min())
    normY = (y - y.min()) / (y.max() - y.min())

    normZ = gaussian_filter((z - z.min()) / (z.max() - z.min()), sigma=sigma)

    normZ = np.power(normZ, power)

    pixelX = (normX * (size - 1)).astype(int)
    pixelY = (normY * (size - 1)).astype(int)

    imgArray = np.zeros((size, size, 3), dtype=np.uint8)

    halfVertexSize = vertexSize // 2
    
    for i in range(len(pixelX)):
        Xstart = max(0, pixelX[i] - halfVertexSize)
        Xend = min(size, pixelX[i] + halfVertexSize + 1)
        yStart = max(0, pixelY[i] - halfVertexSize)
        yEnd = min(size, pixelY[i] + halfVertexSize + 1)

        color = plt.cm.jet(normZ[i])
        rgbColor = tuple((np.array(color[:3]) * 255).astype(int))
        imgArray[yStart:yEnd, Xstart:Xend] = rgbColor

    img = Image.fromarray(imgArray, mode='RGB')

    return img
