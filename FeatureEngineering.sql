DROP MATERIALIZED VIEW IF EXISTS weather_features;

CREATE MATERIALIZED VIEW weather_features AS
SELECT *
FROM (
  SELECT
    d.*,

    -- Moving Averages (3h, 6h, 24h)
    AVG(rain) OVER (ORDER BY datetime ROWS 2 PRECEDING)  AS rain_ma3,
    AVG(rain) OVER (ORDER BY datetime ROWS 5 PRECEDING)  AS rain_ma6,
    AVG(rain) OVER (ORDER BY datetime ROWS 23 PRECEDING) AS rain_ma24,

    AVG(average_temperature) OVER (ORDER BY datetime ROWS 2 PRECEDING)  AS avg_temp_ma3,
    AVG(average_temperature) OVER (ORDER BY datetime ROWS 5 PRECEDING)  AS avg_temp_ma6,
    AVG(average_temperature) OVER (ORDER BY datetime ROWS 23 PRECEDING) AS avg_temp_ma24,

    AVG(maximum_temperature) OVER (ORDER BY datetime ROWS 5 PRECEDING)  AS max_temp_ma6,
    AVG(minimum_temperature) OVER (ORDER BY datetime ROWS 5 PRECEDING)  AS min_temp_ma6,

    AVG(average_windspeed) OVER (ORDER BY datetime ROWS 5 PRECEDING)  AS windspeed_ma6,
    AVG(average_windspeed) OVER (ORDER BY datetime ROWS 23 PRECEDING) AS windspeed_ma24,

    AVG(maximum_windspeed) OVER (ORDER BY datetime ROWS 5 PRECEDING)  AS gust_ma6,
    AVG(maximum_windspeed) OVER (ORDER BY datetime ROWS 23 PRECEDING) AS gust_ma24,

    AVG(sun) OVER (ORDER BY datetime ROWS 5 PRECEDING)  AS sun_ma6,
    AVG(sun) OVER (ORDER BY datetime ROWS 23 PRECEDING) AS sun_ma24,

    AVG(humidity) OVER (ORDER BY datetime ROWS 5 PRECEDING)  AS humidity_ma6,
    AVG(humidity) OVER (ORDER BY datetime ROWS 23 PRECEDING) AS humidity_ma24,

    AVG(pressure) OVER (ORDER BY datetime ROWS 5 PRECEDING)  AS pressure_ma6,
    AVG(pressure) OVER (ORDER BY datetime ROWS 23 PRECEDING) AS pressure_ma24,

    AVG(cloud) OVER (ORDER BY datetime ROWS 5 PRECEDING)  AS cloud_ma6,
    AVG(cloud) OVER (ORDER BY datetime ROWS 23 PRECEDING) AS cloud_ma24,

    -- Differences (1h, 24h, 7d)
    rain - LAG(rain) OVER (ORDER BY datetime) AS rain_diff,
    rain - LAG(rain, 24) OVER (ORDER BY datetime) AS rain_diff_24h,
    rain - LAG(rain, 24*7) OVER (ORDER BY datetime) AS rain_diff_7d,

    average_temperature - LAG(average_temperature) OVER (ORDER BY datetime) AS avg_temp_diff,
    average_temperature - LAG(average_temperature, 24) OVER (ORDER BY datetime) AS avg_temp_diff_24h,
    average_temperature - LAG(average_temperature, 24*7) OVER (ORDER BY datetime) AS avg_temp_diff_7d,

    pressure - LAG(pressure) OVER (ORDER BY datetime) AS pressure_diff,
    pressure - LAG(pressure, 24) OVER (ORDER BY datetime) AS pressure_diff_24h,
    pressure - LAG(pressure, 24*7) OVER (ORDER BY datetime) AS pressure_diff_7d,

    cloud - LAG(cloud) OVER (ORDER BY datetime) AS cloud_diff,

    -- Windows max/min (6h)
    MAX(rain) OVER (ORDER BY datetime ROWS 5 PRECEDING) AS rain_max6,
    MIN(rain) OVER (ORDER BY datetime ROWS 5 PRECEDING) AS rain_min6,
    MAX(average_temperature) OVER (ORDER BY datetime ROWS 5 PRECEDING) AS temp_max6,
    MIN(average_temperature) OVER (ORDER BY datetime ROWS 5 PRECEDING) AS temp_min6,

    -- Volatility (stddev over 6h and 24h)
    STDDEV(average_temperature) OVER (ORDER BY datetime ROWS 5 PRECEDING)  AS temp_volatility_6h,
    STDDEV(average_temperature) OVER (ORDER BY datetime ROWS 23 PRECEDING) AS temp_volatility_24h,

    STDDEV(average_windspeed) OVER (ORDER BY datetime ROWS 5 PRECEDING)  AS wind_volatility_6h,
    STDDEV(average_windspeed) OVER (ORDER BY datetime ROWS 23 PRECEDING) AS wind_volatility_24h,

    STDDEV(pressure) OVER (ORDER BY datetime ROWS 23 PRECEDING) AS pressure_volatility_24h,
    STDDEV(humidity) OVER (ORDER BY datetime ROWS 23 PRECEDING) AS humidity_volatility_24h,

    -- Cyclic features: time
    SIN(2 * PI() * EXTRACT(HOUR FROM datetime) / 24) AS hour_sin,
    COS(2 * PI() * EXTRACT(HOUR FROM datetime) / 24) AS hour_cos,
    SIN(2 * PI() * EXTRACT(DOY FROM datetime) / 365) AS doy_sin,
    COS(2 * PI() * EXTRACT(DOY FROM datetime) / 365) AS doy_cos,

    -- Cyclic features: wind_dir
    COS(RADIANS(wind_dir)) AS wind_dir_x,
    SIN(RADIANS(wind_dir)) AS wind_dir_y,

    -- Anomaly flag
    CASE WHEN ABS(average_temperature - LAG(average_temperature) OVER (ORDER BY datetime)) > 5 THEN 1 ELSE 0 END AS temp_spike_flag,
    CASE WHEN ABS(average_windspeed - LAG(average_windspeed) OVER (ORDER BY datetime)) > 5 THEN 1 ELSE 0 END AS wind_spike_flag

  FROM dataset d
  WHERE datetime IS NOT NULL
) AS features
WHERE 
  rain_ma24 IS NOT NULL
  AND avg_temp_ma24 IS NOT NULL
  AND avg_temp_diff_7d IS NOT NULL
  AND pressure_diff_7d IS NOT NULL
  AND temp_volatility_24h IS NOT NULL
  AND wind_volatility_24h IS NOT NULL
ORDER BY datetime;