from sqlalchemy import create_engine  # sqlalchemy es una librería que permite conectarse y trabajar con bases de datos de manera más abstracta y flexible. 'create_engine' permite crear una conexión a bases de datos SQL de diferentes tipos, como MySQL, PostgreSQL, SQLite, etc.
import pymysql

def connect_to_mysql():
    connection = pymysql.connect(
    host='localhost',
    user='root',
    password='AlumnaAdalab'
)

    # Crear un cursor
    cursor = connection.cursor()

    # Crear una base de datos de ejemplo
    cursor.execute("CREATE DATABASE IF NOT EXISTS Spanish_Issues_DB")
    print("Base de Datos creada exitosamente.")

    return connection





def load_data_to_mysql(df1, df2,df3):

    engine = create_engine('mysql+pymysql://root:AlumnaAdalab@127.0.0.1/Spanish_Issues_DB') #este tipo de conexión se usa cuando vamos a trabajar con pandas

    # Insertar datos desde los DataFrames en MySQL
    df1.to_sql('IPV', con=engine, if_exists='append', index=False)
    df2.to_sql('Rentas', con=engine, if_exists='append', index=False)
    df3.to_sql('Precio_vivienda', con=engine, if_exists='append', index=False)


    #con= conexion, if_exists=agregar los datos a la tabla si ya existe, index=false no inclu

    print("Datos cargados exitosamente en MySQL.")
    engine.dispose()  # Cerrar la conexión al motor de la base de datos
    return engine


def drop_table(tabla):
    connection = pymysql.connect(
    host='localhost',
    user='root',
    password='AlumnaAdalab'
)

    # Crear un cursor
    cursor = connection.cursor()

    # Crear una base de datos de ejemplo
    cursor.execute(f"DROP TABLE IF EXISTS {tabla}")
    print("Base de Datos borrada exitosamente.")

    return connection