from mrjob.job import MRJob
import math

class MRKmeans(MRJob):

    def configure_args(self):
        super(MRKmeans, self).configure_args()
        self.add_file_arg("--centroids", help="External file with centroid values.")


    def read_centroids(self):
        """
        Read the initialized centroid values from a file.
        """
        centroids = []

        with open (self.options.centroids, "r") as f:
            for line in f:
                x, y = line.split(',')
                centroids.append((float(x), float(y)))
        return centroids

    def nearest_centroid(self, datum, centroids):
        """
        Compute the nearest centroid, return the index.
        """
        min_distance = math.inf
        for c in centroids:
            dist = math.sqrt(sum([(a-b)**2 for a, b in zip(datum, c)]))
            if dist < min_distance:
                min_distance = dist
                cluster_index = centroids.index(c)
        return cluster_index
        

    def mapper(self, _, line):
        # Read line
        x, y = line.split(',')

        # Initialize centroid values
        centroids = self.read_centroids()

        # Assign cluster to each datapoint
        cluster_index = self.nearest_centroid((float(x), float(y)), centroids)

        yield cluster_index, (float(x), float(y))

    def combiner(self, cluster_index, datums):
        x_values = y_values = 0
        for i, datum in enumerate(datums):
            x_values += datum[0]
            y_values += datum[1]

        n = i+1

        yield cluster_index, (x_values, y_values, n)
    

    def reducer(self, cluster_index, values):
        total_x = total_y = n = 0

        for v in values:
            total_x += v[0]
            total_y += v[1]
            n += v[2]

        x_centroid = total_x/n
        y_centroid = total_y/n

        yield (f"Centroid {cluster_index} coordinates: ", (x_centroid, y_centroid))


if __name__ == "__main__":
    MRKmeans.run()
