from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from app import config

client = InfluxDBClient(
    url=config.INFLUX_URL,
    token=config.INFLUX_TOKEN,
    org=config.INFLUX_ORG
)

write_api = client.write_api(write_options=SYNCHRONOUS)
query_api = client.query_api()

def write_sample_point():
    try:
        point = Point("drilling_metrics").tag("rig", "rig-01").field("torque", 123.45)
        write_api.write(bucket=config.INFLUX_BUCKET, org=config.INFLUX_ORG, record=point)
        return True
    except Exception as e:
        print("InfluxDB write failed:", e)
        return False

def test_influx_connection():
    try:
        query = f'from(bucket:"{config.INFLUX_BUCKET}") |> range(start: -1h) |> limit(n:1)'
        result = query_api.query(org=config.INFLUX_ORG, query=query)
        return True
    except Exception as e:
        print("InfluxDB query failed:", e)
        return False
