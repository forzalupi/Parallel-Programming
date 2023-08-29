from mrjob.job import MRJob
from mrjob.step import MRStep
import math
import statistics

class SummaryStatistics(MRJob):

    def configure_args(self):
        super(SummaryStatistics, self).configure_args()
        self.add_passthru_arg('--min', type=float, default=3.0, help='Minimum value')
        self.add_passthru_arg('--max', type=float, default=8.0, help='Maximum value')
    
    def mapper(self, _, line):
        _, key, value = line.split('\t')
        yield key, (float(value), float(value)**2)
    
    def combiner(self, key, values):
        # Initialize variables
        total = square_total = 0
        min_val = math.inf
        max_val = -math.inf
        bins = 10
        # Calculate bin range and width
        bin_range = self.options.max-self.options.min
        bin_width = bin_range/bins
        # Create list to store counts
        histogram = [0]*bins
        # Create list to store values for each group
        group_values = []

        # Unpack all 
        for i, (value,square_value) in enumerate(values):
            # Add to total and square total
            total += value
            square_total += square_value
            # Check min and max value
            if value < min_val:
                min_val = value
            if value > max_val:
                max_val = value
                # Add to bin counts if value is in the range
            if self.options.min <= value < self.options.max:
                bin_index = math.floor((value-self.options.min) / bin_width)
                histogram[bin_index] += 1
            # Append value
            group_values.append(value)
        # Median
        median = statistics.median(group_values)

        # Total rows
        n = i+1

        yield key, (total, square_total, n, min_val, max_val, histogram, median)
        
    def reducer(self, _, values):
        # Initialize
        total = square_total = n = 0
        min_val = math.inf
        max_val = -math.inf
        bins = 10
        histogram = [0]*bins
        medians = []
        
        # Unpack
        for v in values:
            total += v[0]
            square_total += v[1]
            n += v[2]
            if v[3] < min_val:
                min_val = v[3]
            if v[4] > max_val:
                max_val = v[4]
            histogram = [x+y for x, y in zip(histogram, v[5])]
            medians.append(v[6])

        # Approximate the median of all the groups median
        median = statistics.median(medians)

        yield None, (total, square_total, n, min_val, max_val, histogram, median)

    def find_statistics(self, _, values):
        # Initialize
        total = square_total = n = 0
        min_val = math.inf
        max_val = -math.inf
        bins = 10
        histogram = [0]*bins
        medians = []
        for v in values:
            total += v[0]
            square_total += v[1]
            n += v[2]
            if v[3] < min_val:
                min_val = v[3]
            if v[4] > max_val:
                max_val = v[4]
            histogram = [x+y for x, y in zip(histogram, v[5])]
            medians.append(v[6])
        
        # Calculate mean       
        mean = total / n
        # Calculate std
        variance = (square_total / n) - mean**2
        std = variance**0.5

        # Approximate the median of all the groups median
        median = statistics.median(medians)

        yield ("Mean", mean)
        yield ("Standard Deviation", std)
        yield ("Minimum Value", min_val)
        yield ("Maximum Value", max_val)
        yield ("Bin counts", histogram)
        yield ("Median", median)
    
    def steps(self):
        return [
            MRStep(mapper=self.mapper, combiner=self.combiner, reducer=self.reducer),
            MRStep(reducer=self.find_statistics)
        ]

if __name__ == "__main__":
    SummaryStatistics.run()