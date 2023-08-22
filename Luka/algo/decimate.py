def findNeighbours(vertex):
    #triangulacijo nardimo na x,y osi
    tri = Delaunay(vertex[:, :2])
    
    neighbours = [[] for _ in vertex]
    # loopamo skozi vse sosede ki so 
    for triangle in tri.simplices:
        for i in range(3):
            neighbours[triangle[i]].extend(triangle[j] for j in range(3) if j != i)
            #print(neighbours)
    return neighbours

def decimateVertexes(vertices, threshold):
    
    neighbors = findNeighbours(vertices)
    
    #izracunamo povprecno razadljo vertexa
    avgDistance = np.zeros(vertices.shape[0])

    for i, vertex in enumerate(vertices):
        if len(neighbors[i]) == 0:
            print(i, " Vertex nima sosedov.")
            pass
        else:
            #izracunamo razdaljo med tem vozliscem in med sosednjimi vozlisci
            # vrnemo razdaljo za to vozlisce
            distances = np.linalg.norm(vertices[neighbors[i]] - vertex, axis=1)
            print("distances: ", distances)
            if np.all(distances == 0):
                print(i, "Vertex enaka lokacija kot sosedi.")
            else:
                avgDistance[i] = np.mean(distances)

    avgDistance = (avgDistance - avgDistance.min()) / (avgDistance.max() - avgDistance.min())
    
    print("Povprecna razdalja (min, max, mean):", avgDistance.min(), avgDistance.max(), avgDistance.mean())
    
    return vertices[avgDistance >= threshold]
