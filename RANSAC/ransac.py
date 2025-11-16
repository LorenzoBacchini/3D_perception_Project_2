import random
import math
import open3d as o3d
from tqdm import tqdm
import numpy as np

class RANSAC:
    def __init__(self, point_cloud, max_iterations,            
                    distance_ratio_threshold):
        self.point_cloud = point_cloud
        self.max_iterations = max_iterations
        self.distance_ratio_threshold = distance_ratio_threshold

    def _ransac_algorithm(self, max_iterations, distance_ratio_threshold):
        inliers_result = []
        iteration_counter = 1
        for i in tqdm(range(max_iterations), desc="Segmenting lidar points", unit="iterations"):
            iteration_counter = iteration_counter + 1
            max_iterations -= 1
            # Add 3 random indexes
            random.seed()
            inliers = np.zeros((len(self.point_cloud),))
            it = 0
            indexes = []
            while it < 3:
                random_index = random.randint(0, len(self.point_cloud)-1)
                if inliers[random_index] == True:
                    it = it - 1
                    continue
                indexes.append(random_index)
                inliers[random_index] = True
                it = it + 1

            x1, y1, z1, _ = self.point_cloud[indexes[0]]
            x2, y2, z2, _ = self.point_cloud[indexes[1]]
            x3, y3, z3, _ = self.point_cloud[indexes[2]]
            # Plane Equation --> ax + by + cz + d = 0
            # Value of Constants for inlier plane
            a = (y2 - y1)*(z3 - z1) - (z2 - z1)*(y3 - y1)
            b = (z2 - z1)*(x3 - x1) - (x2 - x1)*(z3 - z1)
            c = (x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1)
            d = -(a*x1 + b*y1 + c*z1)
            plane_lenght = max(0.1, math.sqrt(a*a + b*b + c*c))

            for i, point in enumerate(self.point_cloud):
                index = i
                # Skip iteration if point matches the randomly generated inlier point
                if index in indexes:
                    continue

                x, y, z, _ = point

                # Calculate the distance of the point to the inlier plane
                distance = math.fabs(a*x + b*y + c*z + d)/plane_lenght
                # Add the point as inlier, if within the threshold distancec ratio
                inliers[index] = (distance <= distance_ratio_threshold)
            # Update the set for retaining the maximum number of inlier points
            if np.sum(inliers) > np.sum(inliers_result):
                inliers_result = []
                inliers_result = inliers

        return inliers_result
