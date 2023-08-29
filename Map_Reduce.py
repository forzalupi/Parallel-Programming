from mrjob.job import MRJob
from mrjob.step import MRStep
import math

class SummaryStatistics(MRJob):
    
    def mapper(self, _, line):
        _, key, value = line.split('\t')
        yield key, (float(value), float(value)**2)
    
    def combiner(self, key, values):
        # Initialize variables
        total = square_total = 0
        min_val = math.inf
        max_val = -math.inf

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
        # Total rows
        n = i+1

        yield key, (total, square_total, n, min_val, max_val)
        
    def reducer(self, _, values):
        # Initialize
        total = square_total = n = 0
        min_val = math.inf
        max_val = -math.inf
        
        # Unpack
        for v in values:
            total += v[0]
            square_total += v[1]
            n += v[2]
            if v[3] < min_val:
                min_val = v[3]
            if v[4] > max_val:
                max_val = v[4]

        yield None, (total, square_total, n, min_val, max_val)

    def find_statistics(self, _, values):
        # Initialize
        total = square_total = n = 0
        min_val = math.inf
        max_val = -math.inf

        for v in values:
            total += v[0]
            square_total += v[1]
            n += v[2]
            if v[3] < min_val:
                min_val = v[3]
            if v[4] > max_val:
                max_val = v[4]
        
        # Calculate mean       
        mean = total / n
        # Calculate std
        variance = (square_total / n) - mean**2
        std = variance**0.5

        yield ("Mean", mean)
        yield ("Standard Deviation", std)
        yield ("Minimum Value", min_val)
        yield ("Maximum Value", max_val)
    
    def steps(self):
        return [
            MRStep(mapper=self.mapper, combiner=self.combiner, reducer=self.reducer),
            MRStep(reducer=self.find_statistics)
        ]

if __name__ == "__main__":
    SummaryStatistics.run()