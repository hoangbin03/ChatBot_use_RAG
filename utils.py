import pymysql

def get_db_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='',
        database='chatbot_DB',
        port=3307
    )
