import findspark
findspark.init()
from pyspark import SparkContext
import argparse

def summary_statistics(num_workers, data_path):
    sc = SparkContext(master = f"local[{num_workers}]")

    dist_file = sc.textFile(data_path)

    # Create RDD with each row being a key, value, value**2 pair.
    key_tuples = dist_file.map(lambda l: l.split('\t')) \
        .map(lambda t: (int(t[1]), (float(t[2]), float(t[2])**2)))

    grouped_key_tuples = key_tuples.reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1]))

    summed_values = grouped_key_tuples.map(lambda x: x[1]).reduce(lambda a, b: (a[0]+b[0], a[1]+b[1]))
    total = summed_values[0]
    square_total = summed_values[1]

    n = key_tuples.count()

    # Calculate mean       
    mean = total / n
    # Calculate std
    variance = (square_total / n) - mean**2
    std = variance**0.5

    # Min, max value
    min_val = key_tuples.min(lambda x: x[1])[1][0]
    max_val = key_tuples.max(lambda x: x[1])[1][0]


    print(f"""
    Mean: {mean},
    std: {std},
    Max value: {max_val},
    Min val: {min_val}
    """)

    sc.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = \
                                    'Spark summary statistics')
    parser.add_argument('--workers','-w', default=1,type=int,
                        help='Number of parallel processes')
    parser.add_argument('--file', '-f', default="/data/2023-DAT470-DIT065/data-assignment-3-1M.dat", type=str, help='Data file path')
    args = parser.parse_args()

    summary_statistics(args.workers, args.file)