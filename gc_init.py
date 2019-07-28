from sqlalchemy import create_engine
import dotenv


n_rows = 277396.0 #Unique IP
sample_size = 150
def get_cursor():

    awsqlKey = dotenv.get_key('.env', 'awsqlKey')

    from psycopg2 import connect
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    params = {
        'host': 'project3.czq1askywdkq.us-west-2.rds.amazonaws.com',
        'user': 'sculla',
        'password': awsqlKey,
        'port': 5432
    }

    connection = connect(**params, dbname='sculla')
    connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    #connion_string = 'postgresql://localhost:5432/sculla'
    #engine = create_engine(connion_string, echo=True, isolation_level='AUTOCOMMIT')
    # cursor = engine.conn()
    # cursor = cursor.connion.cursor()
    cursor = connection.cursor()
    cursor.execute('SET search_path TO project3;') # for console afterwards
    return cursor

def ip_sample(cursor, name):


    cursor.execute(f'SELECT DISTINCT ip FROM project3.phone_data TABLESAMPLE SYSTEM '
                   f'(({sample_size}) / {n_rows}) LIMIT {sample_size};\n')

    ip_addr = cursor.fetchall()
    for ip in ip_addr[:1]:
        print(ip[0], 'starting')
        cursor.execute(f'INSERT INTO project3.phone_data{name} '
                       f'SELECT * FROM project3.phone_data '
                       f'WHERE ip = {ip[0]};\n'
                       )
        print(ip[0], 'done')


def new_sample_table(cursor,name):

    # cursor.execute('drop table project3.phone_data_sample2;')

    cursor.execute(
        f'CREATE TABLE project3.phone_data{name} ('
            f'index integer NOT NULL,'
            f'app integer NOT NULL,'
            f'ip integer NOT NULL,'
            f'device integer NOT NULL,'
            f'os integer NOT NULL,'
            f'channel integer NOT NULL,'
            f'click_time timestamp without time zone NOT NULL,'
            f'attributed_time timestamp without time zone,'
            f'is_attributed integer NOT NULL'
            f');\n'
    )
    cursor.execute(
        f'alter table project3.phone_data{name} '
        f'add constraint phone_data_{name}_pk '
        f'primary key (index);\n')

def main(cursor, name):
    columns = ['app','device','os','channel']


    tab = f'phone_data{name}'

    for column in columns:
        cursor.execute(f'SELECT DISTINCT {column} FROM project3.{tab}')
        unique_values = cursor.fetchall()

        for value in unique_values:
            val = value[0]
            try:

                cursor.execute(f'ALTER TABLE project3.{tab} '
                               f'ADD COLUMN {column}_{val} INT;\n')

                # f.write(f'ALTER TABLE project3.{tab} '
                #                f'ADD COLUMN {column}_{val} INT;\n')


                cursor.execute(f'UPDATE project3.{tab} '
                               f'SET {column}_{val} = 1 '
                               f'WHERE {column} = {val};\n')

                cursor.execute(f'UPDATE project3.{tab} '
                               f'SET {column}_{val} = 0 '
                               f'WHERE {column} <> {val};\n')

                # f.write(f'UPDATE project3.{tab} '
                #                f'SET {column}_{val} = 1 '
                #                f'WHERE {column} = {val};\n')
            except: #TODO column already exists
                pass
    cursor.close()
    f.close()


    #TODO FEATURE ENG
    # user_features = ['user_total_orders','user_avg_cartsize','user_total_products','user_avg_days_since_prior_order']
    # df_user_features = (df_order_products_prior.groupby(['user_id'],as_index=False)
    #                                            .agg(OrderedDict(
    #                                                    [('order_id',['nunique', (lambda x: x.shape[0] / x.nunique())]),
    #                                                     ('product_id','nunique'),
    #                                                     ('days_since_prior_order','mean')])))
    # df_user_features.columns = ['user_id'] + user_features
    # df_user_features.head()

if __name__ == '__main__':
    cursor = get_cursor()
    name = ''
    #new_sample_table(cursor, name)
    #ip_sample(cursor, name)
    main(cursor, name)
['ip','app','device,os','channel','click_time','attributed_time','is_attributed']