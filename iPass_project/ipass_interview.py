import json
import random
from cdecimal import ROUND_UP
from random import randrange
import ast
import pandas as pd
from decimal import Decimal
from scipy.spatial.distance import euclidean

import copy
import numpy as np


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


def build_virtual_env():
    env = np.zeros((1000, 1000), dtype=object)
    for wifi in wifi_hotspots:
        env[wifi['lat_lng'][0]][wifi['lat_lng'][1]] = wifi['name']
    return env


def make_user(env):
    user = {}

    u_lt = randrange(len(env))
    u_lg = randrange(len(env[0]))
    user['gp'] = {
        'lt': u_lt,
        'lg': u_lg
    }
    scan_list = []
    for wifi in wifi_hotspots:
        (lt, lg) = wifi['lat_lng']
        # (lt, lg) = np.where(env == wifi['name'])
        if euclidean((u_lt, u_lg), (lt, lg)) < 300:
            data = {
                's': wifi['name'],
                'b': wifi['mac_add']
            }
            scan_list.append(data)
    user['sl'] = scan_list
    return user
    # print euclidean(a,b)


def make_random_data(scan_list):
    new_list = copy.deepcopy(scan_list)
    # sl = new_list['sl']
    Decimal(str(16.2)).quantize(Decimal('.01'), rounding=ROUND_UP)
    new_list['gp']['lg'] += round(random.uniform(-0.9, -0.1), 6)
    new_list['gp']['lt'] += round(random.uniform(0.1, 0.9), 6)
    new_ss_list = map(lambda k: int(k['ss']) + round(random.uniform(-10, 10), 0), new_list['sl'])
    for i in xrange(len(new_ss_list)):
        new_list['sl'][i]['ss'] = new_ss_list[i]
    # new_list['sl']['ss'] += round(random.uniform(0.1, 0.9), 6)

    return new_list
    # gp['lg'] = gp['lg'] + random.uniform(-122.27, -122.28)

data_sets = \
    {"scans": [{"sl": [{"mn": "2Wire Inc", "b": "28:16:2e:a4:c4:41", "s": "2WIRE413", "cn": 2, "ss": -81,
                        "st": "[WPA-PSK-CCMP+TKIP][WPA2-PSK-CCMP+TKIP][ESS]", "h": 0, "fc": "F:2417.0:MHz"},
                       {"mn": "ARRIS Group, Inc.", "b": "78:96:84:6e:6f:a0", "s": "ATTUuVi3A2", "cn": 11, "ss": -90,
                        "st": "[WPA-PSK-CCMP+TKIP][WPA2-PSK-CCMP+TKIP][ESS]", "h": 0, "fc": "F:2462.0:MHz"},
                       {"mn": "2Wire Inc", "b": "f8:18:97:ff:05:f2", "s": "ATT5yKz2zA", "cn": 1, "ss": -85,
                        "st": "[WPA-PSK-CCMP+TKIP][WPA2-PSK-CCMP+TKIP][WPS][ESS]", "h": 0, "fc": "F:2412.0:MHz"},
                       {"mn": "2Wire Inc", "b": "00:1f:b3:d7:94:11", "s": "2WIRE869", "cn": 6, "ss": -88,
                        "st": "[WPA-PSK-TKIP][ESS]", "h": 0, "fc": "F:2437.0:MHz"},
                       {"mn": "ARRIS Group, Inc.", "b": "d0:39:b3:4c:b2:00", "s": "ATTnjXcsys", "cn": 11, "ss": -87,
                        "st": "[WPA-PSK-CCMP+TKIP][WPA2-PSK-CCMP+TKIP][WPS][ESS]", "h": 0, "fc": "F:2462.0:MHz"},
                       {"mn": "Mediabridge Products, LLC.", "b": "14:35:8b:11:b5:80", "s": "FlowerBud", "cn": 3,
                        "ss": -89, "st": "[WPA-PSK-CCMP][ESS]", "h": 0, "fc": "F:2422.0:MHz"},
                       {"mn": "Cisco-Linksys, LLC", "b": "58:6d:8f:17:53:4a", "s": "chewy-guest", "cn": 11, "ss": -92,
                        "st": "[ESS]", "h": 0, "fc": "F:2462.0:MHz"},
                       {"mn": "ASUSTek COMPUTER INC.", "b": "38:2c:4a:5c:0c:c9", "s": "GreatWall_Guest1", "cn": 6,
                        "ss": -89, "st": "[WPA2-PSK-CCMP][ESS]", "h": 0, "fc": "F:2437.0:MHz"},
                       {"mn": "ARRIS Group, Inc.", "b": "1c:1b:68:5c:d2:70", "s": "ATT6d4R4S5", "cn": 1, "ss": -90,
                        "st": "[WPA-PSK-CCMP+TKIP][WPA2-PSK-CCMP+TKIP][ESS]", "h": 0, "fc": "F:2412.0:MHz"},
                       {"mn": "ARRIS Group, Inc.", "b": "94:87:7c:3b:d9:d0", "s": "HOME-D9D2", "cn": 6, "ss": -91,
                        "st": "[WPA-PSK-CCMP+TKIP][WPA2-PSK-CCMP+TKIP][WPS][ESS]", "h": 0, "fc": "F:2437.0:MHz"},
                       {"mn": "ARRIS Group, Inc.", "b": "3c:36:e4:f9:9f:40", "s": "ATT464B5n2", "cn": 1, "ss": -90,
                        "st": "[WPA-PSK-CCMP+TKIP][WPA2-PSK-CCMP+TKIP][ESS]", "h": 0, "fc": "F:2412.0:MHz"},
                       {"mn": "ARRIS Group, Inc.", "b": "96:87:7c:3b:d9:d0", "s": "xfinitywifi", "cn": 6, "ss": -90,
                        "st": "[ESS]", "h": 0, "fc": "F:2437.0:MHz"}], "t": 1471619928086,
                "gp": {"t": 1471619898224, "s": "fused", "lg": -122.27535, "a": 34, "_byGeo.isoCode": "US",
                       "lt": 37.54704, "_byGeo.country": "United States"}}]}


def transform_data():
    df = pd.read_csv('user_scan_list_'+str(num_of_users)+'.csv')
    result = []
    for idx, row in df.iterrows():
        row_sl = ast.literal_eval(row['sl'])
        row_gp = ast.literal_eval(row['gp'])
        for dct in row_sl:
            dct['user_lt'] = row_gp['lt']
            dct['user_lg'] = row_gp['lg']
            result.append(dct)
    del df

    df = pd.DataFrame(result)
    # print df
    df.to_csv('user_scan_list_transform_'+str(num_of_users)+'.csv', index=False)


def calculate_accuracy(lt_lg, column_list):
    lat_lng_true = []
    result = []
    for column in column_list:
        for wifi in wifi_hotspots:
            if column in wifi['name']:
                lat_lng_true.append(wifi['lat_lng'])
    # result.append(lt_lg)
    for i in xrange(len(lat_lng_true)):
        result.append((column_list[i], round((300-euclidean(lat_lng_true[i], lt_lg[i]))*100 / 300, 2), lt_lg[i], lat_lng_true[i]))
    return result


def create_data():
    env = build_virtual_env()
    scans = []

    scan_dict = dict()
    for i in xrange(num_of_users):
        scans.append(make_user(env))
    scan_dict['scans'] = scans
    # print scan_dict
    df = pd.DataFrame(scan_dict['scans'])
    df = df[df['sl'].str.len() > 0]
    df.to_csv('user_scan_list_'+str(num_of_users)+'.csv', index=False)


def run_algorithm():
    df = pd.read_csv('user_scan_list_transform_'+str(num_of_users)+'.csv')
    lt_calc = (df.groupby(['s'])['user_lt'].max() - df.groupby(['s'])['user_lt'].min())/2 + df.groupby(['s'])['user_lt'].min()
    lg_calc = (df.groupby(['s'])['user_lg'].max() - df.groupby(['s'])['user_lg'].min())/2 + df.groupby(['s'])['user_lg'].min()
    lt_lg = zip(lt_calc, lg_calc)
    # pd.DataFrame(calculate_accuracy(lt_lg, lg_calc.keys())).to_csv("user_scan_list_result_"+str(num_of_users)+'.csv')
    # print calculate_accuracy(lt_lg, lg_calc.keys())
    df = pd.DataFrame(calculate_accuracy(lt_lg, lg_calc.keys()))
    df = df.rename(columns={0: 'Wifi-name', 1: 'Accuracy', 2: 'Calc location', 3: 'True location'})
    df.to_csv('user_scan_list_output_'+str(num_of_users)+'.csv', index = False)


if __name__ == '__main__':

    num_of_users = 1000

    # create_data()

    # transform_data()

    run_algorithm()
