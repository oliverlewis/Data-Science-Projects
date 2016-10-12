import pandas as pd
import numpy as np

import boto


conn = boto.connect_s3(
        aws_access_key_id = access_key,
        aws_secret_access_key = secret_key
        )

bucket = conn.get_bucket('')
keys = bucket.list('')


def get_all_files():
    all_files = []
    for file_key in keys:
        if len(file_key.name[str(file_key.name).rfind("/")+1:]) > 0:
            all_files.append(file_key.name)

    return all_files

def user_csv():
    df = pd.read_csv("files/users.csv", names=['user_id', 'dtype_appversion', 'ts_created'], encoding='utf-8')
    return df


def convert_test_cohort_to_dt(file_path_list):
    for file_path in file_path_list:
        df = pd.read_csv(file_path, names=['test_cohort', 'event name', 'user_id', 'whisper_id', 'extra_information', 'time_generated', 'dtype_appversion', 'ts_created'], encoding='utf-8')
        t_generated_dt_df = pd.DataFrame(pd.to_datetime(df['time_generated'], unit='ms'))
        ts_created_dt_df = pd.DataFrame(pd.to_datetime(df['ts_created'], unit='us'))
        t_generated_dt_df = t_generated_dt_df.rename(columns={'time_generated': 'time_generated_dt'})
        ts_created_dt_df = ts_created_dt_df.rename(columns={'ts_created': 'ts_created_dt'})
        df = pd.concat([df, t_generated_dt_df, ts_created_dt_df], axis=1, join_axes=[df.index])
        df[['test_cohort', 'event name', 'user_id', 'whisper_id', 'extra_information', 'time_generated', 'time_generated_dt', 'dtype_appversion', 'ts_created', 'ts_created_dt']].to_csv(file_path+".csv", index=False)

def events_csv():
    df = pd.read_csv("files/events.csv", names=['test_cohort', 'event name', 'user_id', 'whisper_id', 'extra_information', 'time_generated'], encoding='utf-8')
    data_group = df.groupby(['test_cohort'])
    # return [k for k,v in data_group]
    for k,v in data_group:
        dt = pd.to_datetime(v['time_generated'], unit='ms')
        dt_df = pd.DataFrame(dt)
        dt_df = dt_df.rename(columns={'time_generated': 'dt_converted'})
        v = pd.concat([v, dt_df], axis=1, join_axes=[v.index])
        # print v.head(n=10)
        v[['test_cohort', 'event name', 'user_id', 'whisper_id', 'extra_information', 'dt_converted']].to_csv("output_files/"+str(k)+".csv", index=False)


def deep_analysis_events():
    df = pd.read_csv("files/events.csv", names=['test_cohort', 'event_name', 'user_id', 'whisper_id', 'extra_information', 'time_generated'], encoding='utf-8')
    # cohort_A_bool = df['test_cohort'] == "A"
    # cohort_A = df[cohort_A_bool]
    # dt = pd.to_datetime(df['time_generated'], unit='ms')
    # dt_df = pd.DataFrame(dt)
    # dt_df = dt_df.rename(columns={'time_generated': 'dt_converted'})
    # df = pd.concat([df, dt_df], axis=1, join_axes=[df.index])
    by_test_cohort = df.groupby(['test_cohort'])
    # by_test_cohort = by_test_cohort.filter(lambda k: k['test_cohort'] == 'A')
    print by_test_cohort['event_name'].describe()
    by_user_id_event_name = df.groupby(['test_cohort','event_name'])
    preprocessed_df = pd.DataFrame(by_user_id_event_name['event_name'].agg(['count']))
    print by_test_cohort['event_name'].agg([np.std])

def convert_user_ts_microsecond__created():
    user_df = user_csv()
    df = pd.to_datetime(user_df['ts_created'], unit='us')
    ts_converted_df = pd.DataFrame(df)
    ts_converted_df = ts_converted_df.rename(columns={'ts_created': 'ts_convert'})
    # print list(ts_converted_df.columns.values)
    v = pd.concat([user_df, ts_converted_df], axis=1, join_axes=[user_df.index])
    # print list(v.columns.values)
    v[['user_id', 'dtype_appversion', 'ts_convert']].to_csv("output_files/users.csv", index=False)

def count_len(test_cohorts):
    for k in test_cohorts:
        df = pd.read_csv("output_files/"+k, names=['test_cohort', 'event name', 'user_id', 'whisper_id', 'extra_information', 'time_generated'], encoding='utf-8', index_col=False)
        print len(df)


def download_file(file_entry):
    key = bucket.get_key(file_entry)
    print "Downloading file: "+file_entry[file_entry.rfind("/")+1:]
    print "output_files/"+str(file_entry).rsplit("/")[3]+"/"+str(file_entry).rsplit("/")[4]
    key.get_contents_to_filename("output_files/"+str(file_entry).rsplit("/")[3]+"/"+str(file_entry).rsplit("/")[4])

if __name__ == '__main__':
    test_cohorts = ['A', 'B', 'C', 'D', 'E', 'F']
    # user_df = user_csv()
    # events_df = events_csv()
    # count_len(test_cohorts)
    # convert_user_ts_microsecond__created()
    # convert_test_cohort_to_dt(["output_files/"+str(entry).rsplit("/")[3]+"/"+str(entry).rsplit("/")[4] for entry in get_all_files()])
    # match_users_events(user_df, events_df)
    # for file_entry in get_all_files():
    #     download_file(file_entry)
    deep_analysis_events()