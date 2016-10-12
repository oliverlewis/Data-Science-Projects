import json
from time import sleep

import pandas as pd
from datetime import time

from kafka import KafkaProducer


df = pd.read_csv('user_scan_list_5000.csv')
# producer = KafkaProducer(bootstrap_servers=['172.17.0.1:9092'])
producer = KafkaProducer(value_serializer=lambda m: json.dumps(m).encode('ascii'), bootstrap_servers=['172.17.0.1:9092'])

# print len(df)

# producer.send('test_5', key=b'foo', value=b'bar')
# print "sent"
for data in df.iterrows():
    producer.send('test_5', dict(data[1]))
    sleep(0.01
          )
#     print "run"
producer.flush()


