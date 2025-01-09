import pandas as pd

column_names = {
    '지점': "location_index",
    '일시': "date",
    '평균기온(°C)': "temp_avg",
    '최저기온(°C)': "temp_min",
    '최저기온 시각(hhmi)': "time_temp_min",
    '최고기온(°C)': "temp_max",
    '최고기온 시각(hhmi)': "time_temp_max",
    '일강수량(mm)': "precipitation",
    '최대 순간 풍속(m/s)': "wind_max",
    '최대 순간풍속 시각(hhmi)': "time_wind_max",
    '평균 풍속(m/s)': "wind_avg",
    '최대 순간 풍속 풍향(deg)': "direction_wind_max",
    '위도': "LAT",
    '경도': "LON",
    '노장해발고도(m)': "above_sea_level"
}

def read_csv(path: str, eng: bool = True, **kwargs):
    d = pd.read_csv(path, encoding="cp949", **kwargs)
    if eng:
        d = d.rename(columns=column_names)
    return d