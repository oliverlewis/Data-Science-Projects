import json
import ast
import os
from scipy.spatial.distance import euclidean
# from kafka import KafkaConsumer
#
# consumer1 = KafkaConsumer('test_5', bootstrap_servers=['172.17.0.1:9092'])
#
# for message in consumer1:
#     print message

import findspark
findspark.init('/home/oliver/Documents/spark-2.0.0-bin-hadoop2.7')
findspark.add_packages(['org.apache.spark:spark-streaming-kafka-0-8-assembly_2.11:2.0.0-preview'])
from pyspark import SparkContext
from pyspark.streaming.kafka import KafkaUtils
from pyspark.streaming import StreamingContext

import pandas as pd

offsetRanges = []

wifi_hotspots = [{
    'name': '2WIRE413',
    'mac_add': '28:16:2e:a4:c4:41',
    'lat_lng': (333, 333)
},
{
    'name': 'ATTUuVi3A2',
    'mac_add': '78:96:84:6e:6f:a0',
    'lat_lng': (333, 666)
},
{
    'name': '2WIRE869',
    'mac_add': '00:1f:b3:d7:94:11',
    'lat_lng': (666, 500)
}]


def flatten_json(rdd):
    result = []
    row_sl = ast.literal_eval(rdd['sl'])
    row_gp = ast.literal_eval(rdd['gp'])
    for dct in row_sl:
        dct['user_lt'] = row_gp['lt']
        dct['user_lg'] = row_gp['lg']
        result.append((dct['s'], dct))
    return result



def calculate_accuracy(lt_lg, column):
    wifi_list = wifi_hotspots_bc.value
    for wifi in wifi_list:
        if column in wifi['name']:
            lat_lng_true = wifi['lat_lng']
    return round((300-euclidean(lat_lng_true, lt_lg))*100 / 300, 2)


def calc_max_min(rdd):
    min_max_dict = {}


    df = pd.DataFrame(list(rdd))
    min_calc_lt = df.groupby(['s'])['user_lt'].min()[0]
    max_calc_lt = df.groupby(['s'])['user_lt'].max()[0]
    min_calc_lg = df.groupby(['s'])['user_lg'].min()[0]
    max_calc_lg = df.groupby(['s'])['user_lg'].max()[0]
    lt_lg = (min_calc_lt+(max_calc_lt- min_calc_lt)/2, min_calc_lg+(max_calc_lg- min_calc_lg)/2)
    accuracy = calculate_accuracy(lt_lg, df['s'][0])
    min_max_dict[df['s'][0]] = {'lt': (min_calc_lt, max_calc_lt), 'lg': (min_calc_lg, max_calc_lg), 'center': lt_lg, 'accuracy': accuracy}
    return min_max_dict

def change_values(new_values, old_values):
    # print type(new_values), old_values
    if old_values is not None and new_values is not None and len(new_values) > 0:
        # print
        new_lg_min, new_lg_max = new_values[0].values()[0]['lg']
        new_lt_min, new_lt_max = new_values[0].values()[0]['lt']
        old_lg_min, old_lg_max = old_values[0].values()[0]['lg']
        old_lt_min, old_lt_max = old_values[0].values()[0]['lt']

        # print new_lg_min, new_lg_max, old_lt_max, old_lt_min
        new_lg_min = new_lg_min if new_lg_min < old_lg_min else old_lg_min
        new_lg_max = new_lg_max if new_lg_max > old_lg_max else old_lg_max
        new_lt_min = new_lt_min if new_lt_min < old_lt_min else old_lt_min
        new_lt_max = new_lt_max if new_lt_max > old_lt_max else old_lt_max
        lt_lg = (new_lt_min+(new_lt_max - new_lt_min)/2, new_lg_min+(new_lg_max - new_lg_min)/2)
        accuracy = calculate_accuracy(lt_lg, new_values[0].keys()[0])
        return [{new_values[0].keys()[0]: {'lt': (new_lt_min, new_lt_max), 'lg': (new_lg_min, new_lg_max), 'center': lt_lg, 'accuracy': accuracy}}]
    else:
        return new_values



sc = SparkContext()
ssc = StreamingContext(sc, 1)
ssc.checkpoint(os.getcwd())
wifi_hotspots_bc = sc.broadcast(wifi_hotspots)
directKafkaStream = KafkaUtils.createDirectStream(ssc, ['test_5'], {'bootstrap.servers': '172.17.0.1:9092'})
# directKafkaStream.transform(storeOffsetRanges).foreachRDD(printOffsetRanges)
# kafkaStrStream = directKafkaStream.map(lambda (offRan, (k, v)) :  str(offRan._fromOffset) + " " + str(offRan._untilOffset) + " " + str(k) + " " + str(v))
# kafkaStrStream.pprint()
parsed = directKafkaStream.map(lambda (key, value): json.loads(value))
# parsed.pprint()
flatten = parsed.flatMap(flatten_json)
groups = flatten.groupByKeyAndWindow(3, slideDuration=3)
groups_list = groups.mapValues(calc_max_min)
update_previous_state = groups_list.updateStateByKey(change_values)
update_previous_state.pprint()

ssc.start()
ssc.awaitTermination()
# # 172.17.0.1
# # https://mvnrepository.com/artifact/org.apache.spark/spark-streaming-kafka-0-8-assembly_2.10