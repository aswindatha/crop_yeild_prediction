Dataset Summary:
Crops included: rice, sugarcane, cotton, pulses, millets, groundnut, coconut
Regions: coastal (40%), inland (50%), hills (10%)
Total samples: 500,000

First few rows:
    avg_temp    rainfall  crop_type  ...  farm_size_ha  tech_adoption_score  yield_kg
0  33.131229    6.149698    coconut  ...      1.993187             0.556998    5000.0
1  29.438862  159.373187     cotton  ...      5.296736             0.062003     500.0
2  27.210879    5.554851  groundnut  ...      2.187877             0.227709     500.0
3  25.386510   44.554925  groundnut  ...     10.000000             0.764255     500.0
4  32.615586   89.745438     pulses  ...      2.637442             0.259449     500.0

[5 rows x 20 columns]

Yield statistics by crop:
crop_type    mean      std      min        max

coconut     5000.39    48.63   5000.0   13946.02
cotton       842.44  1245.22    500.0   49149.82
groundnut    600.65   570.84    500.0   22403.40
millets      663.54   791.52    500.0   41750.68
pulses       550.61   335.23    500.0   15309.30
rice        2183.42  1267.20   2000.0   92002.64
sugarcane  30008.22   566.70  30000.0  121816.19

Regional distribution:
region
inland     50.1
coastal    39.9
hills      10.0
Name: proportion, dtype: float64