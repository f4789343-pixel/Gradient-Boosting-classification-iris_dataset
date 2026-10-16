import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,Field
from psycopg_pool import ConnectionPool
from dotenv import load_dotenv
import psycopg
import os

load_dotenv()
password = os.getenv('password') or os.getenv('POSTGRESS_PASSWORD')
db_host = os.getenv('DB_HOST','localhost')
db_port = int(os.getenv('DB_PORT', "5432"))
model = joblib.load(filename='Gradient_boosting_iris.joblib')

class InputData(BaseModel):
   sepal_length: float=Field(ge=4.3,le=7.9)
   sepal_width: float=Field(ge=2.0,le=4.4)
   petal_length: float=Field(ge=0.1,le=7.9)
   petal_width: float=Field(ge=0.1,le=4.4)
pool = ConnectionPool(f'host={db_host} ' 
                      f'port={db_port} ' 
                      'dbname=iris_gr '
                      'user=postgres ' 
                      f'password={password}')
@asynccontextmanager
async def lifespan(app: FastAPI):
   yield
   pool.close()
app = FastAPI()
@app.post('/predict')
def prediction(data:InputData):
  try:
   features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
   predictions = model.predict(features)[0]
   with pool.connection() as conn:
    with conn.cursor() as cursor:
      cursor.execute("""
              INSERT INTO prediction_history
              (sepal_length,sepal_width,petal_length,petal_width,prediction)
              VALUES(%s, %s, %s, %s, %s)
              """,(data.sepal_length,data.sepal_width,data.petal_length, data.petal_width,predictions))
    conn.commit()
   return {'prediction':float(predictions)}
  except psycopg.OperationalError:
    raise HTTPException(status_code=503,detail="Database currently unavailable")
class PredictionHistory(BaseModel):
   id: int
   sepal_length: float
   sepal_width: float
   petal_length: float
   petal_width: float
   prediction:  int
class PredictionsResponse(BaseModel):
    predictions: list[PredictionHistory]
@app.get('/predictions',response_model=PredictionsResponse)
def get_prediction():
   try:
    with pool.connection() as conn:
     with conn.cursor() as cursor:
      cursor.execute('SELECT * FROM prediction_history')
      rows = cursor.fetchall()
      columns = [desc.name for desc in cursor.description]
    predictions = [dict(zip(columns,row)) for row in rows]
    return {'predictions':predictions}
   except psycopg.OperationalError:
     raise HTTPException(status_code=503, detail="database currently unavailable")

@app.get("/health")
def health_check():
  try:
    with pool.connection() as conn:
      with conn.cursor() as cursor:
        cursor.execute("SELECT 1")
    return {"api": "healthy", "detabase": "healthy"}
  except psycopg.OperationalError:
    raise HTTPException(status_code=503, detail={"api":"healthy", "database": "unavailable"})



