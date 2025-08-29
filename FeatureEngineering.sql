-- 1) Staging (rådata med faste intervaller og partitionering)
CREATE TABLE IF NOT EXISTS staging_weather_raw (
  station_id  INT            NOT NULL,
  timestamp    TIMESTAMPTZ    NOT NULL,
  rain         DOUBLE PRECISION,
  temperature  DOUBLE PRECISION,
  pressure     DOUBLE PRECISION,
  humidity     DOUBLE PRECISION,
  cloud        DOUBLE PRECISION,
  wind_speed   DOUBLE PRECISION,
  sun          DOUBLE PRECISION,
  PRIMARY KEY (station_id, timestamp)
) PARTITION BY RANGE (timestamp);

-- Eksempel-partition pr. måned (udvid løbende)
CREATE TABLE IF NOT EXISTS staging_weather_2025_07
  PARTITION OF staging_weather_raw
  FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');

-- 2) Materialized view med alle features
CREATE MATERIALIZED VIEW IF NOT EXISTS weather_features AS
WITH base AS (
  SELECT
    station_id,
    timestamp,
    rain,
    temperature,
    pressure,
    humidity,
    cloud,
    wind_speed,
    sun,
    EXTRACT(HOUR  FROM timestamp) AS hour,
    EXTRACT(DOY   FROM timestamp) AS day_of_year
  FROM staging_weather_raw
),
-- 2a) Cykliske tidsegenskaber
time_feats AS (
  SELECT
    *,
    SIN(2*PI()*hour/24)::DOUBLE PRECISION  AS sin_hour,
    COS(2*PI()*hour/24)::DOUBLE PRECISION  AS cos_hour,
    SIN(2*PI()*day_of_year/365)::DOUBLE PRECISION AS sin_annual,
    COS(2*PI()*day_of_year/365)::DOUBLE PRECISION AS cos_annual
  FROM base
),
-- 2b) Rolling-aggregeringer over 1, 6 og 24 timer
rolling_feats AS (
  SELECT
    station_id,
    timestamp,
    sin_hour, cos_hour, sin_annual, cos_annual,
    AVG(temperature) OVER w1h   AS temp_avg_1h,
    STDDEV(temperature) OVER w1h AS temp_std_1h,
    AVG(pressure)    OVER w6h   AS pressure_avg_6h,
    STDDEV(pressure) OVER w6h   AS pressure_std_6h,
    AVG(humidity)    OVER w24h  AS humidity_avg_24h,
    MAX(cloud)       OVER w6h   AS cloud_max_6h,
    pressure - LAG(pressure) OVER (PARTITION BY station_id ORDER BY timestamp) AS pressure_delta,
    cloud    - LAG(cloud)    OVER (PARTITION BY station_id ORDER BY timestamp) AS cloud_delta
  FROM time_feats
  WINDOW
    w1h  AS (PARTITION BY station_id ORDER BY timestamp RANGE BETWEEN '1 hour'   PRECEDING AND CURRENT ROW),
    w6h  AS (PARTITION BY station_id ORDER BY timestamp RANGE BETWEEN '6 hours'  PRECEDING AND CURRENT ROW),
    w24h AS (PARTITION BY station_id ORDER BY timestamp RANGE BETWEEN '24 hours' PRECEDING AND CURRENT ROW)
),
-- 2c) Daglig regnprofil i fire blokke
rain_profile AS (
  SELECT
    station_id,
    timestamp::date          AS rain_day,
    SUM(CASE WHEN hour BETWEEN 0  AND  5 THEN rain ELSE 0 END)  AS rain_00_06,
    SUM(CASE WHEN hour BETWEEN 6  AND 11 THEN rain ELSE 0 END)  AS rain_06_12,
    SUM(CASE WHEN hour BETWEEN 12 AND 17 THEN rain ELSE 0 END)  AS rain_12_18,
    SUM(CASE WHEN hour BETWEEN 18 AND 23 THEN rain ELSE 0 END)  AS rain_18_00,
    SUM(rain)                                                   AS rain_total
  FROM base
  GROUP BY station_id, rain_day
)
-- Endeligt view: join rullende features med regnprofil
SELECT
  rf.station_id,
  rf.timestamp,
  rf.sin_hour,
  rf.cos_hour,
  rf.sin_annual,
  rf.cos_annual,
  rf.temp_avg_1h,
  rf.temp_std_1h,
  rf.pressure_avg_6h,
  rf.pressure_std_6h,
  rf.humidity_avg_24h,
  rf.cloud_max_6h,
  rf.pressure_delta,
  rf.cloud_delta,
  rp.rain_00_06,
  rp.rain_06_12,
  rp.rain_12_18,
  rp.rain_18_00,
  rp.rain_total
FROM rolling_feats rf
JOIN rain_profile rp
  ON rf.station_id = rp.station_id
 AND rf.timestamp::date = rp.rain_day
ORDER BY rf.timestamp, rf.station_id;
