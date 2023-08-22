def makeLabelAndSave(vertexPositions):
    img = buildGrayscaleImage(vertexPositions)

    outputDir = "/Users/lukaknez/Desktop/Projekt/Outputs"

    imageName = 'Image'
    imageType = '.jpg'
    path = os.path.join(outputDir, f'{imageName}{imageType}')

    i = 1
    while os.path.exists(path):
        path = os.path.join(outputDir, f'{imageName}{i}{imageType}')
        i += 1

    img.save(path)
    img.show()
