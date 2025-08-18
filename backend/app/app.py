from fastapi import FastAPI
from backend.app import test_postgres_connection
from backend.app import test_influx_connection, write_sample_point

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/test/postgres")
def test_postgres():
    return {"postgres_connected": test_postgres_connection()}

@app.get("/test/influx")
def test_influx():
    ok = write_sample_point()
    return {"influx_connected": ok and test_influx_connection()}
