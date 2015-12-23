 # -*- coding: utf-8 -*-
"""client to do clustering with dbscan """

 # -*- coding: utf-8 -*-

"""client process to perform dbscan clustering algorithm """


from multiprocessing import Process
from multiprocessing import Queue

import cv2

import numpy as np

from sklearn.cluster import DBSCAN


class ClusteringProcess(Process):

    #Process gérant l'execution de DBSCAN
    def __init__ ( self,
                   idT,
                   status,
                   imglist,
                   outqueue,
                   config):

        self.idT = idT
        self.classif = DBSCAN(
                eps=config.eps,
                min_samples=config.min_samples)

        self.status = status
        self.imglist = imglist
        self.outqueue = outqueue
        self.config = config
        Process.__init__(self)

    def run (self):
        while 1:
            try:
                a2 = self.imglist.get(True, 0.2)
            except Exception as e:
                if self.imglist.empty():
                    continue
                else:
                    print 'exception!'
                    raise e
            else:
                self.status = 0
                a2sum=a2.sum()
                if a2sum > self.config.erodeseuil:
                    print 'old', a2sum
                    it = int(np.floor(a2sum / self.config.erodeseuil * 0.33))
                    it = 1 if not it else it
                    a2 = cv2.erode(a2,
                                   self.config.kernelerode,
                                   iterations=it)
                    a2sum = a2.sum()
                    a2 = cv2.morphologyEx(a2,
                                          cv2.MORPH_OPEN,
                                          self.config.kernel)
                    print 'new', a2sum
                M = np.transpose(np.nonzero(a2))
                try:
                    cluster = dbscan(M, self.classif)
                except Exception as e:
                    print 'clustering err', e
                    continue

                self.status=1

                if cluster:
                    shape=procedure(cluster)
                    shapes=[]
                    for sh in shape:
                        x, y, w, h = sh
                        rectangle = a2.T[x:x+w, y+h:y].astype('float32')

                        shapes.append((sh, np.sum(rectangle)))
                        try:
                            self.outqueue.put(shapes)
                        except Exception as e:
                            print e


def dbscan(M, classif):
    """dbscan algorithm"""
    print 'cl started'
    classif.fit(M)
    labels = classif.labels_.astype('int32')
    cluster = [[] for i in range(max(labels)+1)]
    print 'nombre cluster:',len(cluster)
    for i in range(len(M)):
        if labels[i] > -1:
            cluster[labels[i]].append(M[i])
    return [np.asarray(cl) for cl in cluster]

def procedure(cluster):
    #transformation d'un cluster en rectangle
    shape = []
    for M in cluster:
        amin, amax, omin, omax = (M.T[0].min(),
                                  M.T[0].max(),
                                  M.T[1].min(),
                                  M.T[1].max())
        shape.append((omin,
                      amax,
                      omax - omin,
                      amin - amax))
    return shape
