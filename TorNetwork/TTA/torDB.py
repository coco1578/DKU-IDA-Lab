import pymysql


class DBSearch:

    def __init__(self):
        self.config = self._set_config()
        self._set_connection_with_db()

    def _set_connection_with_db(self):
        self.conn = pymysql.connect(host=self.config['db_host'],
                                    user=self.config['db_user'],
                                    password=self.config['db_passwd'],
                                    db=self.config['db_name'],
                                    charset='utf8')
        self.curs = self.conn.cursor()

        return self

    def _set_config(self):
        import configparser
        config = configparser.ConfigParser()
        config.read('config.ini')
        config = config['DB']

        return config

    def get_url_query(self, table_name, class_number):
        sql = 'select SITE_URL from %s where CLASS_NO=%d' % (table_name, class_number)
        self.curs.execute(sql)

        rows = self.curs.fetchall()
        url = rows[0][0]
        contents = rows[0][1]

        return url, contents