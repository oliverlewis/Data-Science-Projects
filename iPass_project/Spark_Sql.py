import findspark
findspark.init('/home/oliver/Documents/spark-2.0.0-bin-hadoop2.7')
# findspark.add_packages(['org.apache.spark:spark-streaming-kafka-0-8-assembly_2.11:2.0.0-preview'])
from pyspark import SparkContext
# from pyspark.streaming.kafka import KafkaUtils
# from pyspark.streaming import StreamingContext

from pyspark.sql import SQLContext, Row


num_of_users = 1000
sc = SparkContext()
sqlContext = SQLContext(sc)

# Load a text file and convert each line to a Row.
lines = sc.textFile('user_scan_list_transform_'+str(num_of_users)+'.csv')
for name in lines.collect():
    print name