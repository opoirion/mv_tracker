import unittest


class TestPackages(unittest.TestCase):
    """ """

    def test_1_numpy_package(self):
        """test that the numpy package is correctly installed """
        import numpy as np
        self.assertTrue(np)
        self.assertTrue(np.random.random((10,10)).any())

    def test_2_scipy_package(self):
        """test that the scipy package is correctly installed """
        import scipy as sp
        self.assertTrue(sp)

    def test_3_sklearn_and_dbscan_package(self):
        """
        test that the sklearn package is correctly installed
        and that DBSCAN works
        """
        import sklearn
        self.assertTrue(sklearn)

        from sklearn.cluster import DBSCAN
        import numpy as np

        dbscan = DBSCAN()

        self.assertTrue(dbscan.fit_predict(np.random.random((10,10))).any())

    def test_4_cv2(self):
        """test that cv2 can be loaded and that an image can be extracted from the cam """
        import cv2
        self.assertTrue(cv2)
        self.assertTrue(cv2.WINDOW_AUTOSIZE)

        cam  =  cv2.VideoCapture(0)
        s, img  =  cam.read()
        self.assertTrue(s)
        self.assertTrue(img.any())




if __name__ == "__main__":
    unittest.main()
