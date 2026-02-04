PS D:\sad corps\crop yeild prediction> curl "https://api.open-meteo.com/v1/forecast?latitude=13.05&longitude=80.15&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean&timezone=Asia/Kolkata"
{"latitude":13.0,"longitude":80.125,"generationtime_ms":0.05829334259033203,"utc_offset_seconds":19800,"timezone":"Asia/Kolkata","timezone_abbreviation":"GMT+5:30","elevation":15.0,"daily_units":{"time":"iso8601","temperature_2m_mean":"°C","precipitation_sum":"mm","relative_humidity_2m_mean":"%"},"daily":{"time":["2026-02-03","2026-02-04","2026-02-05","2026-02-06","2026-02-07","2026-02-08","2026-02-09"],"temperature_2m_mean":[24.0,24.1,24.4,24.5,24.4,24.6,24.3],"precipitation_sum":[0.00,0.00,0.00,0.00,0.00,0.00,0.00],"relative_humidity_2m_mean":[72,72,77,76,77,76,79]}}
PS D:\sad corps\crop yeild prediction> curl "https://rest.isric.org/soilgrids/v2.0/properties/query?lat=13.05&lon=80.15&property=phh2o&depth=0-30cm"
{"type":"Feature","geometry":{"type":"Point","coordinates":[80.15,13.05]},"properties":{"layers":[]},"query_time_s":0.01587986946105957}
PS D:\sad corps\crop yeild prediction> curl "https://rest.isric.org/soilgrids/v2.0/properties/query?lat=13.05&lon=80.15&property=soc&depth=0-30cm"
10815620422363281}
PS D:\sad corps\crop yeild prediction> curl "https://rest.isric.org/soilgrids/v2.0/properties/query?lat=13.05&lon=80.15&property=sand&depth=0-30cm"
{"type":"Feature","geometry":{"type":"Point","coordinates":[80.15,13.05]},"properties":{"layers":[]},"query_time_s":0.009551525115966797}
PS D:\sad corps\crop yeild prediction> curl "https://rest.isric.org/soilgrids/v2.0/properties/query?lat=13.05&lon=80.1510815620422363281}
PS D:\sad corps\crop yeild prediction> curl "https://rest.isric.org/soilgrids/v2.0/properties/query?lat=13.05&lon=80.15&property=sand&depth=0-30cm"
{"type":"Feature","geometry":{"type":"Point","coordinates":[80.15,13.05]},"properties":{"layers":[]},"query_time_s":0.009551525115966797}
PS D:\sad corps\crop yeild prediction> curl "https://rest.isric.org/soilgrids/v2.0/properties/query?lat=13.05&lon=80.15&property=silt&depth=0-30cm"
{"type":"Feature","geometry":{"type":"Point","coordinates":[80.15,13.05]},"properties":{"layers":[]},"query_time_s":0.009637832641601562}
PS D:\sad corps\crop yeild prediction> curl "https://rest.isric.org/soilgrids/v2.0/properties/query?lat=13.05&lon=80.15&property=soc&depth=0-30cm"
{"type":"Feature","geometry":{"type":"Point","coordinates":[80.15,13.05]},"properties":{"layers":[]},"query_time_s":0.009069442749023438}
PS D:\sad corps\crop yeild prediction>