import numpy as np
import tkinter as tk
import random
import math
import matplotlib.pyplot as plt
from collections import deque
import networkx as nx
from networkx.algorithms import bipartite


random.seed(54)


class Trikotnik:
    def __init__(self, i1: int, i2: int, i3: int, vn: np.ndarray) -> None:
        self.vertex_i = np.asarray([i1, i2, i3])
        self.vn = vn
        self.points_i = list()
        self.points_c = list()
    def add_point(self, pt_i, pt_c):
        self.points_i.append(pt_i)
        self.points_c.append(pt_c)
    def has_point(self, p):
        if self.vertex_i.any() == p:
            return True
        else:
            return False


class Tri: # mogoc bom mogo dodat se vidited
    def __init__(self, v1, v2, v3, vn) -> None:
        self.vertices = np.asarray([v1, v2, v3])
        self.vn = vn
        self.neighbors = list()
        self.outside = list()
    def add_neighbor(self, f):
        self.neighbors.append(f)
    def is_neighbor(self, n):
        ver = n.vertices
        v = self.vertices
        e1n = sorted(np.asarray([ver[0], ver[1]]))
        e2n = sorted(np.asarray([ver[0], ver[2]]))
        e3n = sorted(np.asarray([ver[1], ver[2]]))
        e1s = sorted(np.asarray([v[0], v[1]]))
        e2s = sorted(np.asarray([v[0], v[2]]))
        e3s = sorted(np.asarray([v[1], v[2]]))
        if e1n == e1s or e1n == e2s or e1n == e3s or e2n == e1s or e2n == e2s or e2n == e3s or e3n == e1s or e3n == e2s or e3n == e3s:
            return True
        else:
            return False
    def add_out(self, i):
        self.outside.append(i)
    def out_empty(self):
        if len(self.outside) == 0:
            return True
        else:
            return False


def plot_tetra(ax, pts, tetra):
    for i in range(len(tetra)-1):
        for j in range(i, len(tetra)):
            ax.plot3D([pts[tetra[i], 0], pts[tetra[j], 0]], [pts[tetra[i], 1], pts[tetra[j], 1]], [pts[tetra[i], 2], pts[tetra[j], 2]], 'blue')


def t(p, q, r):
    x = p-q
    return np.dot(r-q, x)/np.dot(x, x)


def dist_p_l_3d(p, q, r):
    return np.linalg.norm(t(p, q, r)*(p-q)+q-r)


def dist_3d(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)


def vector(odstejemo_to: np.ndarray, odstejemo_od: np.ndarray) -> np.ndarray:
    vec = np.asarray([odstejemo_od[0] - odstejemo_to[0], odstejemo_od[1] - odstejemo_to[1], odstejemo_od[2] - odstejemo_to[2]])
    return vec


def norm(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> np.ndarray:
    v1 = vector(p1, p2)
    v2 = vector(p1, p3)
    vn = np.cross(v1, v2)
    v3 = vector(p1, p4)
    o = np.dot(vn, v3)

    if o > 0:
        vn = vn * -1

    return vn


def generate_points(n: int) -> np.ndarray: # dodaj se preverjanje duplicateov
    pts = np.ndarray((n, 3), dtype=object)
    existing = np.empty((n), dtype=object)
    existing.fill(np.asarray([0, 0, 0]))

    i = 0
    while i < n:
        x = round(random.gauss(50, 50))
        y = round(random.gauss(50, 50))
        z = round(random.gauss(50, 50))
        
        pt = np.asarray([x, y, z])
        for entry in existing:
            if pt.all() == entry.all():
                i -= 1
                break
        existing[i] = pt

        pts[i] = pt
        #pts[i,0] = x
        #pts[i,1] = y
        #pts[i,2] = z

        i += 1

    return pts


def quickhull_3d(pts: np.ndarray) -> np.ndarray:
    hl = list()
    stack = deque()
    stck = deque()

    # iskanje zacetnega tetraedra
    min_x = np.argmin(pts[:,0])
    max_x = np.argmax(pts[:,0])
    min_y = np.argmin(pts[:,1])
    max_y = np.argmax(pts[:,1])
    min_z = np.argmin(pts[:,2])
    max_z = np.argmax(pts[:,2])

    extremes = np.full((6), [min_x, max_x, min_y, max_y, min_z, max_z])

    # prvi dve tocki tetraedra
    tetra = [0, 0]
    max_dist = 0
    for i in range(5):
        for j in range(i+1, 6):
            p1 = pts[extremes[i],:]
            p2 = pts[extremes[j],:]
            dist = dist_3d(p1, p2)
            if dist > max_dist:  
                max_dist = dist
                tetra[0] = extremes[i]
                tetra[1] = extremes[j]
    
    # tretja tocka tetraedra
    max_dist_p_l = 0
    point_3 = 0
    for i in range(6):
        if i != tetra[0] or i != tetra[1]:
            dist = dist_p_l_3d(pts[tetra[0],:], pts[tetra[1],:], pts[extremes[i],:])
            if dist > max_dist_p_l:
                max_dist_p_l = dist
                point_3 = extremes[i]
    tetra.append(point_3)

    # cetrta tocka tetraedra
    v1 = vector(pts[tetra[0],:], pts[tetra[1],:])
    v2 = vector(pts[tetra[0],:], pts[tetra[2],:])
    vn = np.cross(v1, v2)
    max_dot = 0
    point_4 = 0
    ind = 0
    for p in pts:
        vp = vector(pts[tetra[0],:], p)
        dot_prod = abs(np.dot(vn, vp))
        if dot_prod > max_dot:
            max_dot = dot_prod
            point_4 = ind
        ind += 1
    tetra.append(point_4)

    #hl.append(tetra[0])
    #hl.append(tetra[1])
    #hl.append(tetra[2])
    #hl.append(tetra[3])

    # na sklad dodamo tocke iz tetraedra
    pt0 = pts[tetra[0],:]
    pt1 = pts[tetra[1],:]
    pt2 = pts[tetra[2],:]
    pt3 = pts[tetra[3],:]
    stack.append(Trikotnik(tetra[0], tetra[1], tetra[2], norm(pt0, pt1, pt2, pt3)))
    stack.append(Trikotnik(tetra[0], tetra[1], tetra[3], norm(pt0, pt1, pt3, pt2)))
    stack.append(Trikotnik(tetra[0], tetra[2], tetra[3], norm(pt0, pt2, pt3, pt1)))
    stack.append(Trikotnik(tetra[0], tetra[1], tetra[2], norm(pt1, pt2, pt3, pt0)))

    '''
    # v hl dodajamo nove tocke dokler sklad ni prazen
    max_dot_tri_pt = -1000000
    max_pt = None
    max_pt_i = 0
    i = 0
    pos_pts_dict = dict()
    group_a = list()
    group_b = list()
    # tocka
    while len(stack) != 0:
        tri = stack.pop()
        no_positive = True
        for pt in pts:
            vp = vector(pts[tri.vertex_i[0],:], pt)
            dot = np.dot(vp, tri.vn) # > 0 - zunaj, = 0 - na, < 0 - znotraj
            if dot > max_dot_tri_pt:
                max_dot_tri_pt = dot
                max_pt = pt
                max_pt_i = i
            if dot > 0: # koncamo z to iteracijo while zanke (vem da ta continue deluje samo na for, mogoce treba popravit)
                hl.append(i)
                tri.add_point(i, pt)
                if i in pos_pts_dict:
                    pos_pts_dict[tri.vertex_i[0]] += 1
                    pos_pts_dict[tri.vertex_i[1]] += 1
                    pos_pts_dict[tri.vertex_i[2]] += 1
                else:
                    pos_pts_dict[tri.vertex_i[0]] = 1
                    pos_pts_dict[tri.vertex_i[1]] = 1
                    pos_pts_dict[tri.vertex_i[2]] = 1
                no_positive = False
            i += 1
        if no_positive:
            if len(stack) == 0:
                i1 = tri.vertex_i[0]
                i2 = tri.vertex_i[1]
                i3 = tri.vertex_i[2]
                stack.append(Trikotnik(i1, i2, max_pt_i, norm(pts[i1,:], pts[i2,:], max_pt, pts[i3,:])))
                stack.append(Trikotnik(i1, i3, max_pt_i, norm(pts[i1,:], pts[i3,:], max_pt, pts[i2,:])))
                stack.append(Trikotnik(i2, i3, max_pt_i, norm(pts[i2,:], pts[i3,:], max_pt, pts[i1,:])))
            continue
    
    # delitev v grupo a in b
    for k in pos_pts_dict:
        if pos_pts_dict[k] > 2:
            group_b.append(pos_pts_dict[k])
        else:
            group_a.append(pos_pts_dict[k])
    
    for point in group_b:
        if point in hl:
            hl.remove(point)
            # odstranimo robove

    # Točke b) odstranimo z izbočene lupine. Prav tako odstranimo vse robove, ki imajo katero od krajišč v taki točki. (ka?)
    '''
    '''
    # nov poskus
    g = nx.Graph()
    stck.append(Tri(1, tetra[0], tetra[1], tetra[2], norm(pt0, pt1, pt2, pt3)))
    stck.append(Tri(2, tetra[0], tetra[1], tetra[3], norm(pt0, pt1, pt3, pt2)))
    stck.append(Tri(3, tetra[0], tetra[2], tetra[3], norm(pt0, pt2, pt3, pt1)))
    stck.append(Tri(4, tetra[1], tetra[2], tetra[3], norm(pt1, pt2, pt3, pt0)))
    g.add_nodes_from([str(stck[0].name), str(stck[1].name), str(stck[2].name), str(stck[3].name)], bipartite=0)
    g.add_nodes_from(['B1','B2','B3'],bipartite=1)
    g.add_nodes_from(['B4','B5','B6'],bipartite=2)
    
    nx.draw_networkx(g, pos = nx.drawing.layout.bipartite_layout(g, [1, 2, 3, 4], scale=2), width = 2)
    '''

    #facet - triangle, ridge - edge, simplex 2d - trikotnik

    stck.append(Tri(tetra[0], tetra[1], tetra[2], norm(pt0, pt1, pt2, pt3)))
    stck.append(Tri(tetra[0], tetra[1], tetra[3], norm(pt0, pt1, pt3, pt2)))
    stck.append(Tri(tetra[0], tetra[2], tetra[3], norm(pt0, pt2, pt3, pt1)))
    stck.append(Tri(tetra[1], tetra[2], tetra[3], norm(pt1, pt2, pt3, pt0)))

    unassigned = list()
    for i in range(pts.shape[0]):
        unassigned.append(i)
    unassigned.remove(tetra[0])
    unassigned.remove(tetra[1])
    unassigned.remove(tetra[2])
    unassigned.remove(tetra[3])

    for tri in stck:
        for i in unassigned:
            vp = vector(pts[tri.vertices[0],:], pts[i,:])
            dot = np.dot(vp, tri.vn) # > 0 - zunaj, = 0 - na, < 0 - znotraj
            if dot > 0:
                tri.add_out([i, dot])
    for tri in stck:
        if not tri.out_empty():
            max_out = 0
            max_i = 0
            for pt in tri.outside:
                if pt[1] > max_out:
                    max_out = pt[1]
                    max_i = pt[0]
            visible_set = [tri]
            for facet in visible_set:
                for neighbor in facet.neighbors: # mogoce morem pazit da niso visited
                    vp = vector(pts[neighbor.vertices[0],:], pts[max_i,:])
                    dot = np.dot(vp, tri.vn)
                    if dot > 0:
                        visible_set.append(neighbor)
            # the boundary of V is the set of horizon ridges H
            hl.append(visible_set[-1].vertices[0])
            hl.append(visible_set[-1].vertices[1])
            hl.append(visible_set[-1].vertices[0])
            hl.append(visible_set[-1].vertices[2])
            hl.append(visible_set[-1].vertices[1])
            hl.append(visible_set[-1].vertices[2])
            new_facets = list()
            for i in range(0, len(hl), 2):
                edge = [pts[hl[i],:], pts[hl[i+1],:]]
                new_facet = Tri(hl[i], hl[i+1], max_i, norm(pts[hl[i],:], pts[hl[i+1],:], pts[max_i,:], pt0))
                #unassigned.remove(max_i)
                for f in stck:
                    if new_facet.is_neighbor(f):
                        new_facet.add_neighbor(f)
                new_facets.append(new_facet)
            for facet in new_facets:
                for facet_v in visible_set:
                    for pt in facet_v.outside:
                        if pt in unassigned:
                            vp = vector(pts[facet.vertices[0],:], pts[pt,:])
                            dot = np.dot(vp, facet.vn)
                            if dot > 0:
                                facet.add_out(pt)
            visible_set.clear()



    return hl, tetra


'''
root = tk.Tk()
root.geometry('1200x800')
root.configure(bg='gray')

pts_num = tk.StringVar()
pts = None
hull = None


def generate_btn():
    n = int(pts_num.get())
    pts = generate_points(n)
    pts_num.set('')
    #return pts


def quickhull_btn(pts):
    hull = quickhull_3d(pts)
    print(pts)
    return hull


canvas = tk.Canvas(root, width=1200, height=700, background='black')
canvas.pack()

textbox = tk.Entry(root)
textbox.pack()

button_gen = tk.Button(root, text="Generiraj Tocke", command=generate_btn) # spremeni u lambda pa dej v funkcije argumente pa return
button_gen.pack()
pts = button_gen.invoke()
print(pts)

button_alg = tk.Button(root, text="Quickhull 3D", command= lambda: quickhull_btn(pts))
button_alg.pack()

root.mainloop()
'''

if __name__ == '__main__':
    pts = generate_points(100)
    #print(pts)
    hull, tetra = quickhull_3d(pts)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #for i in range(0, len(hull), 2): # za izris tetraedra range(0, len(hull)-1, 1) drugace pa range(0, len(hull), 2)
    #    ax.plot3D([pts[hull[i], 0], pts[hull[i+1], 0]], [pts[hull[i], 1], pts[hull[i+1], 1]], [pts[hull[i], 2], pts[hull[i+1], 2]], 'red')
    plot_tetra(ax, pts, tetra)
    ax.scatter3D(pts[:,0], pts[:,1], pts[:,2], c=np.linspace(0, pts.shape[0], num=pts.shape[0], endpoint=False), cmap='Greens')
    plt.show()
