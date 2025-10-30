# Archivo principal para tratar csv, excel o json.
#Analisis y ETL
#Autora: Gema Antón

import Functions_EDA as fn
import mysql.connector
from mysql.connector import errorcode
from sqlalchemy import create_engine,FLOAT, VARCHAR, INTEGER, DATE, SmallInteger
from sqlalchemy.sql.sqltypes import String

#Lectura de ficheros
archivo = 'spain_clean.csv'
df = fn.leer_df(archivo)

fn.EDA(df)


engine = create_engine(
    "mysql+mysqlconnector://root:AlumnaAdalab@127.0.0.1/IDEALISTA",
    connect_args={
        "auth_plugin": "mysql_native_password"})


df.to_sql("Datos_idealista", con=engine, if_exists="replace", index=False)
print


