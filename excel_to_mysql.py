import pandas as pd
from sqlalchemy import create_engine
import pymysql
from pymysql import IntegrityError
import uuid
import os


class Data_Mysql():

    def __init__(self, MYSQL_DB=None, MYSQL_HOST=None,
                 MYSQL_USER=None, MYSQL_PASSWORD=None, mult=False, port=3306, connect_timeout=10):

        self.host = MYSQL_HOST
        self.user = MYSQL_USER
        self.password = MYSQL_PASSWORD
        self.db = MYSQL_DB
        self.port = port
        self.connect_timeout = connect_timeout

        def setMultMode():##��ָ���������ݿ⣬�������϶����ݿ���в�ѯ
            self.db = None

        mult and setMultMode()
        self.conn = None
        self.cursor = None

    def __enter__(self):
        return self

    def __exit__(self, type, value, trace):
        try:
            self.close()
        except:
            pass

    def read(self, table, column='*', LIMIT=' '):

        if not self.conn:
            self.connect()
        sql = "select " + str(column) + " from " + str(table) + ' ' + LIMIT
        print ('sql = ', sql)
        df = pd.read_sql(sql, self.conn)

        self.close()
        return df

    def write(self, sql, values):
        if not self.conn:
            self.connect()
        print(sql)
        print(values)
        sta = self.cursor.execute(sql, values)
        self.conn.commit()
        return sta

    def connect(self):
        self.conn = pymysql.connect(host=self.host, user=self.user,
                                    password=self.password, db=self.db,
                                    charset='utf8', use_unicode=True, port=self.port,
                                    connect_timeout=self.connect_timeout)
        self.cursor = self.conn.cursor()

    '''��ָ�������ݿ��д�����,����������Լ���csv�ļ�����'''

    def create_table(self, table_name, csv_filename):
        engine = create_engine(
            str(r'mysql+pymysql://%s:' + '%s' + '@%s/%s?charset=utf8') % (self.user, self.password, self.host, self.db))
        try:
            data = pd.read_csv(csv_filename, sep=',', encoding='utf-8')  # , sep='\t'
            data.to_sql(table_name, con=engine, if_exists='append', index=False)
        except Exception as e:
            print(e)

    def close(self):
        self.conn.close()


def list_path_file(file_dir):
    """
    ��ȡ·���µ�ȫ���ļ���
    """
    filter_file_type = ['csv', 'xls', 'xlsx']
    if not os.path.isdir(file_dir):
        raise FileExistsError(" dir {} not exist".format(file_dir))
    files = os.listdir(file_dir)
    for each_file in files:
        if each_file.splist(".")[-1] not in filter_file_type:
            files.remove(each_file)
    return files


if __name__ == "__main__":
    """
    ��ȡĳһ�ļ����µ�ȫ��excel��csv�ļ�������mysql ���ݿ���
    ���޸ĵ�һ�д�����ļ��е�ַ��
    ���޸����ݿ�������Ϣ��
    """
    file_path = "C:\\Users\\linfengde\\Desktop\\������Ŀ\\�����ɷ�2019��10��15��10��47��41��.xls"
    readMysql = Data_Mysql(MYSQL_DB='middleware', MYSQL_HOST='localhost', MYSQL_PASSWORD='root',
                           MYSQL_USER='root')

    files = list_path_file(file_path)
    for filename in files:
        table_name = filename.split(".")[0]
        file_path = file_path + "/" + filename
        readMysql.create_table('table_name', file_path)