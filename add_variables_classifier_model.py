import h5py
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.neighbors import BallTree
from ast import literal_eval
from datetime import datetime
import geopandas as gpd
import math
from shapely.geometry import Point, LineString, Polygon, box
import os
import tempfile

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.6f}'.format)

TZ = "Europe/Amsterdam"


def shrink_df_to_max_size(df: pd.DataFrame, max_size_mb=25, step=10_000):
    # Start with full df
    trimmed = df.copy()
    max_bytes = max_size_mb * 1024 * 1024

    # Temporary file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name

    while True:
        # Save current df slice to temp file
        trimmed.to_csv(tmp, index=False)
        size_bytes = os.path.getsize(tmp)

        if size_bytes <= max_bytes:
            return trimmed, size_bytes / (1024*1024)

        # If too large → drop from the top, keep only last rows
        rows = len(trimmed)
        if rows <= step:
            print("⚠ Warning: cannot shrink further without deleting all data.")
            return trimmed.iloc[0:0], 0.0

        trimmed = trimmed.iloc[-(rows - step):]  # keep last rows


def create_weather_location_df():
    ''' #uncomment code below for when we look at all locations and derive weather data for  centroids...
    data = """STN,LON_east,LAT_north,ALT_m,NAME
209,4.518,52.465,0.00,IJmond
210,4.430,52.171,-0.20,Valkenburg Zh
215,4.437,52.141,-1.10,Voorschoten
225,4.555,52.463,4.40,IJmuiden
235,4.781,52.928,1.20,De Kooy
240,4.790,52.318,-3.30,Schiphol
242,4.921,53.241,10.80,Vlieland
248,5.174,52.634,0.80,Wijdenes
249,4.979,52.644,-2.40,Berkhout
251,5.346,53.392,0.70,Hoorn Terschelling
257,4.603,52.506,8.50,Wijk aan Zee
258,5.401,52.649,7.30,Houtribdijk
260,5.180,52.100,1.90,De Bilt
265,5.274,52.130,13.90,Soesterberg
267,5.384,52.898,-1.30,Stavoren
269,5.520,52.458,-3.70,Lelystad
270,5.752,53.224,1.20,Leeuwarden
273,5.888,52.703,-3.30,Marknesse
275,5.873,52.056,48.20,Deelen
277,6.200,53.413,2.90,Lauwersoog
278,6.259,52.435,3.60,Heino
279,6.574,52.750,15.80,Hoogeveen
280,6.585,53.125,5.20,Eelde
283,6.657,52.069,29.10,Hupsel
285,6.399,53.575,0.00,Huibertgat
286,7.150,53.196,-0.20,Nieuw Beerta
290,6.891,52.274,34.80,Twenthe
308,3.379,51.381,0.00,Cadzand
310,3.596,51.442,8.00,Vlissingen
311,3.672,51.379,0.00,Hoofdplaat
312,3.622,51.768,0.00,Oosterschelde
313,3.242,51.505,0.00,Vlakte van De Raan
315,3.998,51.447,0.00,Hansweert
316,3.694,51.657,0.00,Schaar
319,3.861,51.226,1.70,Westdorpe
323,3.884,51.527,1.40,Wilhelminadorp
324,4.006,51.596,0.00,Stavenisse
330,4.122,51.992,11.90,Hoek van Holland
331,4.193,51.480,0.00,Tholen
340,4.342,51.449,19.20,Woensdrecht
343,4.313,51.893,3.50,Rotterdam Geulhaven
344,4.447,51.962,-4.30,Rotterdam
348,4.926,51.970,-0.70,Cabauw Mast
350,4.936,51.566,14.90,Gilze-Rijen
356,5.146,51.859,0.70,Herwijnen
370,5.377,51.451,22.60,Eindhoven
375,5.707,51.659,22.00,Volkel
377,5.763,51.198,30.00,Ell
380,5.762,50.906,114.30,Maastricht
391,6.197,51.498,19.50,Arcen
392,6.056189,51.486836,21.88,Horst
"""

    '''
    data = """STN,LON_east,LAT_north,ALT_m,NAME
    210,4.430,52.171,-0.20,Valkenburg Zh
    215,4.437,52.141,-1.10,Voorschoten
    240,4.790,52.318,-3.30,Schiphol
    """

    df = pd.read_csv(StringIO(data))
    weather_location_info = df[["STN", "LON_east", "LAT_north", "NAME"]].copy()
    # Optional: enforce dtypes
    weather_location_info["STN"] = weather_location_info["STN"].astype(int)
    weather_location_info["LON_east"] = weather_location_info["LON_east"].astype(float)
    weather_location_info["LAT_north"] = weather_location_info["LAT_north"].astype(float)
    return weather_location_info

def _to_list_generic(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return list(v)
    if pd.isna(v):
        return []
    s = str(v).strip()
    try:
        out = list(literal_eval(s))
        return out if isinstance(out, list) else [out]
    except Exception:
        try:
            return [int(x) for x in s.strip('[]').split(',') if str(x).strip() != '']
        except Exception:
            return []

def _ensure_ranked_station_cols(df, k_max=3):
    out = df.copy()
    if any(c in out.columns for c in ['nearest_STN', 'nearest_STN_2', 'nearest_STN_3']):
        for c in ['nearest_STN', 'nearest_STN_2', 'nearest_STN_3']:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors='coerce').astype('Int64')
        return out

    if 'nearest_stations_ids' not in out.columns:
        return out

    ids = out['nearest_stations_ids'].apply(_to_list_generic)
    for i in range(k_max):
        name = 'nearest_STN' if i == 0 else f'nearest_STN_{i+1}'
        out[name] = ids.apply(lambda xs, j=i: xs[j] if len(xs) > j else np.nan).astype('Int64')
    return out


def add_nearest_stations(df, df_weather_loc, k=10):
    # Ensure needed columns exist (handles both lat/lon and latitude/longitude)
    lat_col = 'lat' if 'lat' in df.columns else 'latitude'
    lon_col = 'lon' if 'lon' in df.columns else 'longitude'
    if lat_col not in df.columns or lon_col not in df.columns:
        raise KeyError("DataFrame must have 'lat'/'lon' (or 'latitude'/'longitude').")

    # Prepare station coordinates (radians)
    stations_deg = df_weather_loc[['LAT_north', 'LON_east']].to_numpy(dtype=float)
    stations_rad = np.radians(stations_deg)

    # Build BallTree with haversine metric
    tree = BallTree(stations_rad, metric='haversine')

    # Prepare query points (radians); handle possible NaNs
    pts = df[[lat_col, lon_col]].to_numpy(dtype=float)
    pts_rad = np.radians(pts)

    # Query k nearest (dist in radians); Earth radius (km)
    dist_rad, idx = tree.query(pts_rad, k=min(k, len(df_weather_loc)))
    dist_km = dist_rad * 6371.0088

    # Lookup station info
    stn_arr   = df_weather_loc['STN'].to_numpy()
    name_arr  = df_weather_loc['NAME'].to_numpy()
    lon_arr   = df_weather_loc['LON_east'].to_numpy()
    lat_arr   = df_weather_loc['LAT_north'].to_numpy()

    # Gather per-row lists
    nearest_stn_ids   = [stn_arr[i].tolist() for i in idx]
    nearest_stn_names = [name_arr[i].tolist() for i in idx]
    nearest_stn_lons  = [lon_arr[i].tolist() for i in idx]
    nearest_stn_lats  = [lat_arr[i].tolist() for i in idx]
    nearest_dist_km   = [row.tolist() for row in dist_km]

    # Attach to df (lists ordered from nearest → furthest)
    out = df.copy()
    out['nearest_stations_ids']   = nearest_stn_ids
    out['nearest_stations_names'] = nearest_stn_names
    out['nearest_stations_dist_km'] = nearest_dist_km
    return out


def merge_events_with_knmi(df, weather_df, k_max=10, rename_to_weather_prefix=True):
    df = _ensure_ranked_station_cols(df, k_max=3).copy()
    wx = weather_df.copy()
    wx.columns = wx.columns.str.strip()
    wx['STN'] = pd.to_numeric(wx['STN'], errors='coerce').astype('Int64')
    wx['YYYYMMDD'] = wx['YYYYMMDD'].astype(str).str.strip()
    wx['date'] = pd.to_datetime(wx['YYYYMMDD'], format='%Y%m%d', errors='coerce')
    wx['HH']   = pd.to_numeric(wx['HH'], errors='coerce')
    wx = wx.dropna(subset=['STN','date','HH'])

    extra_days = (wx['HH'] // 24).astype('Int64')
    hours      = (wx['HH'] % 24)
    wx['weather_time_wall'] = (
        wx['date']
        + pd.to_timedelta(hours, unit='h')
        + pd.to_timedelta(extra_days.fillna(0), unit='D')
    )
    wx = wx.sort_values(['STN','weather_time_wall'])

    def _localize_group(s):
        try:
            return s.dt.tz_localize('Europe/Amsterdam', ambiguous='infer', nonexistent='shift_forward')
        except Exception:
            return s.dt.tz_localize('Europe/Amsterdam', ambiguous=False, nonexistent='shift_forward')

    wx['weather_time_nl'] = wx.groupby('STN', group_keys=False)['weather_time_wall'].apply(_localize_group)
    wx = wx.drop_duplicates(['STN', 'weather_time_nl'])

    # --- events in NL-tijd, sleutel = afgeronde uur ---
    dt = pd.to_datetime(df['datetime'])
    if getattr(getattr(dt, 'dt', dt), 'tz', None) is None:
        df['datetime'] = dt.dt.tz_localize('Europe/Amsterdam')
    else:
        df['datetime'] = dt.dt.tz_convert('Europe/Amsterdam')

    dt_utc = df['datetime'].dt.tz_convert('UTC')
    df['key_nearest_hour'] = dt_utc.dt.round('h').dt.tz_convert('Europe/Amsterdam')

    # beschikbare variabelen
    V_LAST10  = [c for c in ['DD', 'FF'] if c in wx.columns]
    V_INSTANT = [c for c in ['T', 'P', 'U'] if c in wx.columns]

    w_last10  = wx[['STN','weather_time_nl'] + V_LAST10].rename(columns={'STN':'__stn','weather_time_nl':'__when'})
    w_instant = wx[['STN','weather_time_nl'] + V_INSTANT].rename(columns={'STN':'__stn','weather_time_nl':'__when'})

    out = df.copy()

    # Prioriteitslijst van stationkolommen (legacy eerst)
    branch_specs = []
    for c in ['nearest_STN', 'nearest_STN_2', 'nearest_STN_3']:
        if c in out.columns and out[c].notna().any():
            branch_specs.append(c)

    # Daarna extra posities uit de lijstkolom (boven 3), t/m k_max
    if 'nearest_stations_ids' in out.columns:
        ids = out['nearest_stations_ids'].apply(_to_list_generic)
        max_len = int(ids.map(len).max() if len(ids) else 0)
        max_len = min(max_len, k_max)
        start_pos = 0 if 'nearest_STN' not in out.columns else 3
        for pos in range(start_pos, max_len):
            tmpcol = f'__stn_k{pos+1}'
            out[tmpcol] = ids.apply(lambda xs, j=pos: xs[j] if len(xs) > j else np.nan).astype('Int64')
            branch_specs.append(tmpcol)

    if not branch_specs:
        # niets te mergen
        if rename_to_weather_prefix:
            return out  # niets toegevoegd
        else:
            return out

    # 1 branch merge helper (met midden-drops om overlap te voorkomen)
    def merge_for_branch(base_df, stn_col, tag):
        tmp = base_df
        if len(V_LAST10) > 0:
            tmp = tmp.merge(
                w_last10,
                left_on=[stn_col, 'key_nearest_hour'],
                right_on=['__stn', '__when'],
                how='left',
                suffixes=(None, None),
            )
            tmp = tmp.drop(columns=['__stn','__when'], errors='ignore')  # <-- voorkomt overlap

        if len(V_INSTANT) > 0:
            tmp = tmp.merge(
                w_instant,
                left_on=[stn_col, 'key_nearest_hour'],
                right_on=['__stn', '__when'],
                how='left',
                suffixes=(None, None),
            )
            tmp = tmp.drop(columns=['__stn','__when'], errors='ignore')  # <-- voorkomt overlap

        # branche-suffix zodat branches niet botsen
        ren = {}
        for c in set(V_LAST10 + V_INSTANT):
            if c in tmp.columns:
                ren[c] = f"{c}__{tag}"
        return tmp.rename(columns=ren)

    # voer merges uit voor alle branches
    tags = []
    cur = out
    for i, stn_col in enumerate(branch_specs, start=1):
        tag = f"s{i}"
        tags.append(tag)
        cur = merge_for_branch(cur, stn_col, tag)

    # combineer bronnen per variabele in prioriteitsvolgorde
    for var in set(V_LAST10 + V_INSTANT):
        candidates = [f"{var}__{t}" for t in tags if f"{var}__{t}" in cur.columns]
        if not candidates:
            continue
        if var not in cur.columns:
            cur[var] = pd.NA
        agg = cur[candidates[0]]
        for c in candidates[1:]:
            agg = agg.combine_first(cur[c])
        cur[var] = cur[var].combine_first(agg)

    # opruimen: branch-kolommen & tijdelijke stationkolommen
    drop_cols = []
    for var in set(V_LAST10 + V_INSTANT):
        drop_cols += [c for c in cur.columns if c.startswith(f"{var}__s")]
    drop_cols += [c for c in cur.columns if c.startswith('__stn_k')]
    cur = cur.drop(columns=list(set(drop_cols)), errors='ignore')

    # optioneel hernoemen naar weather_*
    if rename_to_weather_prefix:
        renf = {}
        if 'DD' in cur.columns: renf['DD'] = 'weather_DD'
        if 'FF' in cur.columns: renf['FF'] = 'weather_FF'
        if 'T'  in cur.columns: renf['T']  = 'weather_T'
        if 'P'  in cur.columns: renf['P']  = 'weather_P'
        if 'U'  in cur.columns: renf['U']  = 'weather_U'
        cur = cur.rename(columns=renf)
    return cur


def build_runway_lines(df_runways):
    # kolomnamen opschonen (spaties eraf)
    df_runways = df_runways.copy()
    df_runways.columns = df_runways.columns.str.strip()

    runway_lines = []

    for baan, group in df_runways.groupby("baan"):
        if len(group) < 2:
            # Mocht er ooit een baan maar één entry hebben, slaan we die over
            continue

        p1 = group.iloc[0]
        p2 = group.iloc[1]

        line = LineString([
            (p1["longitude"], p1["latitude"]),
            (p2["longitude"], p2["latitude"]),
        ])

        runway_lines.append({
            "baan": baan,
            "baannummer_1": p1["baannummer"],
            "baannummer_2": p2["baannummer"],
            "geometry": line
        })

    gdf_runway_lines = gpd.GeoDataFrame(runway_lines, geometry="geometry", crs="EPSG:4326")
    return gdf_runway_lines


def get_flight_id_column(df):
    if "flight_id" in df.columns:
        return "flight_id"
    elif "flight_nr" in df.columns:
        return "flight_nr"
    else:
        raise ValueError("No 'flight_id' or 'flight_nr' column found in df")


def extract_runway_reference_points(df):
    flight_id_col = get_flight_id_column(df)
    rows = []

    for flight_id, group in df.groupby(flight_id_col):
        # sorteren op tijd
        if "epoch" in group.columns:
            group_sorted = group.sort_values("epoch")
        elif "datetime" in group.columns:
            group_sorted = group.sort_values("datetime")
        else:
            group_sorted = group  # fallback

        # ---- determine landing_type from one-hot columns ----
        if "landing_type" in group_sorted.columns:
            # if you ever add a direct label column, use it
            landing_type = group_sorted["landing_type"].iloc[0]
        else:
            # infer from dummy columns like landing_type_CDA, landing_type_No landing, landing_type_SDA
            row0 = group_sorted.iloc[0]

            landing_type = "Unknown"
            # check each known dummy column in order of preference
            if "landing_type_No landing" in row0.index and bool(row0["landing_type_No landing"]):
                landing_type = "No landing"
            elif "landing_type_CDA" in row0.index and bool(row0["landing_type_CDA"]):
                landing_type = "CDA"
            elif "landing_type_SDA" in row0.index and bool(row0["landing_type_SDA"]):
                landing_type = "SDA"

        # Zorg dat altitude_plane er is
        if "altitude_plane" not in group_sorted.columns:
            raise ValueError("Column 'altitude_plane' not in df")

        alt = group_sorted["altitude_plane"].to_numpy()
        lat = group_sorted["lat"].to_numpy()
        lon = group_sorted["lon"].to_numpy()

        idx = None

        if landing_type == "No landing":
            # Take-off: laatste ground-point
            ground_mask = (alt == 0)
            if ground_mask.any():
                idx = np.where(ground_mask)[0][-1]
            else:
                # fallback: laatste punt
                idx = len(group_sorted) - 1
        else:
            # Landing (CDA/SDA/Unknown): laatste non-zero vóór eerste reeks nullen
            ground_mask = (alt == 0)
            if ground_mask.any():
                first_zero_idx = np.where(ground_mask)[0][0]
                if first_zero_idx > 0:
                    idx = first_zero_idx - 1
                else:
                    idx = 0
            else:
                # nooit op 0 → neem laatste punt
                idx = len(group_sorted) - 1

        rows.append({
            flight_id_col: flight_id,
            "landing_type": landing_type,
            "ref_lat": lat[idx],
            "ref_lon": lon[idx],
        })

    df_ref = pd.DataFrame(rows)
    return df_ref


def assign_runway_to_flights(df_ref, gdf_runway_lines):
    gdf_points = gpd.GeoDataFrame(
        df_ref.copy(),
        geometry=[Point(lon, lat) for lon, lat in zip(df_ref["ref_lon"], df_ref["ref_lat"])],
        crs="EPSG:4326"
    )
    gdf_points_28992 = gdf_points.to_crs("EPSG:28992")
    gdf_runways_28992 = gdf_runway_lines.to_crs("EPSG:28992")

    runway_names = []
    runway_dists = []

    for idx, pt in gdf_points_28992.geometry.items():
        # afstand naar alle runway lijnen
        distances = gdf_runways_28992.distance(pt)
        min_idx = distances.idxmin()
        runway_names.append(gdf_runways_28992.loc[min_idx, "baan"])
        runway_dists.append(distances.loc[min_idx])

    gdf_points_28992["runway_baan"] = runway_names
    gdf_points_28992["runway_dist_m"] = runway_dists

    # terug naar normaal DataFrame
    df_assigned = pd.DataFrame(gdf_points_28992.drop(columns="geometry"))
    return df_assigned



def heading_diff(h1, h2):
    diff = abs(h1 - h2) % 360
    return min(diff, 360 - diff)

def bearing_deg(lat1, lon1, lat2, lon2):
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon_rad = math.radians(lon2 - lon1)

    y = math.sin(dlon_rad) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)
    angle = math.degrees(math.atan2(y, x))  # kan negatief zijn
    return (angle + 360.0) % 360.0

def add_variables_from_traces_new(
    df: pd.DataFrame,
    gdf_runway_lines: gpd.GeoDataFrame,
    h5_path: str,
    start: int = 0
) -> pd.DataFrame:
    if start < 0:
        start = 0
    if start > 0:
        df = df.iloc[start:].reset_index(drop=True)

    # tijdelijk: alleen eerste 10 voor testen
    #df = df.head(10).reset_index(drop=True)
    #df = df.reset_index(drop=True)
    # zorg dat we een flight-id hebben (eventueel alleen flight_nr gebaseerd op index)
    df["flight_nr"] = df.index.astype(int)
    df["date_key"] = df["date_key"].astype(str)

    # outputkolommen voorbereiden
    df["time_diff_plane"] = np.nan
    df["altitude_plane"] = np.nan
    df["air_speed_plane"] = np.nan
    df["longitude"] = np.nan
    df["latitude"] = np.nan
    df["epoch"] = pd.NA
    df["abs_roll_deg"] = np.nan
    df["max_abs_vertical_acc_g"] = np.nan
    df["mean_vertical_acc_g"] = np.nan
    df["std_vertical_acc_g"] = np.nan
    df["trend_vertical_rate_ft_min"] = np.nan
    df["max_abs_roll_deg"] = np.nan
    df["std_roll"] = np.nan
    df["mean_roll"] = np.nan

    # Fallback column in traces.h5
    fallback_cols = [
        "epoch_time", "lat", "lon", "altitude", "ground_speed", "track", "vertical_rate",
        "geometric_altitude", "geometric_vertical_rate", "air_speed", "roll", "trace_id"
    ]

    def get_col_index_map(ds):
        if "columns" in ds.attrs:
            cols = [
                c.decode("utf-8") if isinstance(c, (bytes, bytearray)) else str(c)
                for c in ds.attrs["columns"]
            ]
        else:
            cols = fallback_cols
        return {c: i for i, c in enumerate(cols)}, cols

    n_total = len(df)
    n_no_group = 0
    n_no_link = 0
    n_filled = 0

    # hier slaan we per trace_id één referentiepunt op voor runway-bepaling
    df_ref_rows = []
    seen_trace_ids = set()

    print("Read in traces....")
    print(datetime.now())

    with h5py.File(h5_path, "r") as f:
        colmap_cache = {}

        for idx, row in df.iterrows():
            trace_id = row["trace_id"]
            if pd.isna(trace_id):
                continue
            try:
                trace_id = int(trace_id)
            except Exception:
                continue
            if trace_id <= 0:
                continue

            # t0 = sensor time in UTC seconds
            t0 = float(row["time"])
            duration = 50.0
            radius = max(0.0, duration / 2.0 - 0.5)  # 50/2 - 0.5
            t_min = t0 - radius
            t_max = t0 + radius

            date_key = str(row["date_key"])

            if date_key not in f:
                n_no_group += 1
                continue

            ds = f[date_key]

            # Build / reuse column index map
            if date_key in colmap_cache:
                ci, cols, alt_col = colmap_cache[date_key]
            else:
                ci, cols = get_col_index_map(ds)
                for need in ("trace_id", "epoch_time", "lat", "lon", "air_speed", "vertical_rate"):
                    if need not in ci:
                        raise KeyError(
                            f"HDF5 group {date_key} missing required column '{need}'. "
                            f"Found columns: {cols}"
                        )
                alt_col = "geometric_altitude" if "geometric_altitude" in ci else (
                    "altitude" if "altitude" in ci else None
                )
                if alt_col is None:
                    raise KeyError(f"HDF5 group {date_key} missing altitude column.")
                colmap_cache[date_key] = (ci, cols, alt_col)

            trace_col = np.asarray(ds[:, ci["trace_id"]])
            if np.issubdtype(trace_col.dtype, np.floating) or np.issubdtype(trace_col.dtype, np.integer):
                trace_vec = trace_col.astype(np.int64)
            else:
                trace_vec = pd.to_numeric(
                    pd.Series(trace_col).astype(str).str.strip(),
                    errors="coerce"
                ).fillna(-1).astype(np.int64).to_numpy()

            mask = (trace_vec == trace_id)
            if not np.any(mask):
                n_no_link += 1
                continue

            # volledige trace voor deze vlucht
            ep = np.asarray(ds[:, ci["epoch_time"]], dtype=float)[mask]
            la = np.asarray(ds[:, ci["lat"]], dtype=float)[mask]
            lo = np.asarray(ds[:, ci["lon"]], dtype=float)[mask]
            al = np.asarray(ds[:, ci[alt_col]], dtype=float)[mask]
            ae = np.asarray(ds[:, ci["air_speed"]], dtype=float)[mask]
            vr = np.asarray(ds[:, ci["vertical_rate"]], dtype=float)[mask]

            if "roll" in ci:
                rl = np.asarray(ds[:, ci["roll"]], dtype=float)[mask]
            else:
                rl = np.zeros(mask.sum(), dtype=float)
            rl = np.nan_to_num(rl, nan=0.0)

            if ep.size == 0:
                continue

            # --- 1) Scalars rondom t0 (voor jouw features) ---
            pos_closest = int(np.nanargmin(np.abs(ep - t0)))
            t_closest = float(ep[pos_closest])
            diff = float(abs(t_closest - t0))
            df.at[idx, "time_diff_plane"] = diff  # alleen ter info

            df.at[idx, "altitude_plane"] = al[pos_closest] if np.isfinite(al[pos_closest]) else np.nan
            df.at[idx, "air_speed_plane"] = ae[pos_closest] if np.isfinite(ae[pos_closest]) else np.nan
            df.at[idx, "longitude"] = lo[pos_closest] if np.isfinite(lo[pos_closest]) else np.nan
            df.at[idx, "latitude"] = la[pos_closest] if np.isfinite(la[pos_closest]) else np.nan
            df.at[idx, "epoch"] = int(t_closest) if np.isfinite(t_closest) else pd.NA
            df.at[idx, "abs_roll_deg"] = abs(rl[pos_closest]) if np.isfinite(rl[pos_closest]) else np.nan

            # --- 2) Window rond t0: [t0 - radius, t0 + radius] ---
            wmask = (ep >= t_min) & (ep <= t_max)
            n_win = int(np.count_nonzero(wmask))

            if n_win >= 2:
                order = np.argsort(ep[wmask])
                ep_w = ep[wmask][order]
                vr_w = vr[wmask][order]
                rl_w = rl[wmask][order]

                dt = np.diff(ep_w)
                dv = np.diff(vr_w)
                valid = np.isfinite(dt) & np.isfinite(dv) & (dt > 0)

                if np.any(valid):
                    a = dv[valid] / dt[valid]  # ft/min per second

                    df.at[idx, "max_abs_vertical_acc_g"] = float(np.max(np.abs(a)))
                    df.at[idx, "mean_vertical_acc_g"] = float(np.mean(a))
                    df.at[idx, "std_vertical_acc_g"] = float(np.std(a, ddof=0))
                    df.at[idx, "trend_vertical_rate_ft_min"] = float(vr_w[-1] - vr_w[0])
                else:
                    df.at[idx, "max_abs_vertical_acc_g"] = np.nan
                    df.at[idx, "mean_vertical_acc_g"] = np.nan
                    df.at[idx, "std_vertical_acc_g"] = np.nan
                    df.at[idx, "trend_vertical_rate_ft_min"] = np.nan

                if np.any(np.isfinite(rl_w)):
                    df.at[idx, "max_abs_roll_deg"] = float(np.nanmax(np.abs(rl_w)))
                    df.at[idx, "std_roll"] = float(np.nanstd(rl_w, ddof=0))
                    df.at[idx, "mean_roll"] = float(np.nanmean(rl_w))
                else:
                    df.at[idx, "max_abs_roll_deg"] = np.nan
                    df.at[idx, "std_roll"] = np.nan
                    df.at[idx, "mean_roll"] = np.nan
            else:
                for c in (
                    "max_abs_vertical_acc_g", "mean_vertical_acc_g",
                    "std_vertical_acc_g", "trend_vertical_rate_ft_min",
                    "max_abs_roll_deg", "std_roll", "mean_roll"
                ):
                    df.at[idx, c] = np.nan

            # --- 3) Referentiepunt per vlucht voor runway-bepaling ---
            # dit doen we maar één keer per trace_id
            if trace_id not in seen_trace_ids:
                seen_trace_ids.add(trace_id)

                # landing_type komt uit df-row (bijv. "CDA", "SDA", "No landing")
                landing_type = row.get("landing_type", "No landing")

                # zelfde logica als in extract_runway_reference_points, maar nu op de volledige trace (al, la, lo)
                ground_mask = (al == 0)
                if landing_type == "No landing":
                    # Take-off: laatste ground-point
                    if ground_mask.any():
                        ref_idx = np.where(ground_mask)[0][-1]
                    else:
                        ref_idx = len(al) - 1
                else:
                    # Landing: punt vlak vóór eerste lange serie nullen
                    if ground_mask.any():
                        first_zero_idx = np.where(ground_mask)[0][0]
                        if first_zero_idx > 0:
                            ref_idx = first_zero_idx - 1
                        else:
                            ref_idx = 0
                    else:
                        ref_idx = len(al) - 1

                df_ref_rows.append({
                    "trace_id": trace_id,
                    "landing_type": landing_type,
                    "ref_lat": float(la[ref_idx]),
                    "ref_lon": float(lo[ref_idx])
                })

            n_filled += 1

    # ---------------------------------------------
    # 4) Runways toekennen aan vluchten (op basis van df_ref_rows)
    # ---------------------------------------------
    if df_ref_rows:
        df_ref = pd.DataFrame(df_ref_rows)
        df_runway_assigned = assign_runway_to_flights(df_ref, gdf_runway_lines)

        # runway_baan (naam) aan df hangen via trace_id (die zit óók in df)
        df = df.merge(
            df_runway_assigned[["trace_id", "runway_baan"]],
            on="trace_id",
            how="left"
        )

        # runway-coördinaten uit de runway-lijnen pakken (centroid)
        runway_centroids = gdf_runway_lines.copy()
        runway_centroids["longitude_runway"] = runway_centroids.geometry.centroid.x
        runway_centroids["latitude_runway"] = runway_centroids.geometry.centroid.y
        runway_centroids = runway_centroids[["baan", "longitude_runway", "latitude_runway"]]

        # koppelen op baannaam
        df = df.merge(
            runway_centroids,
            left_on="runway_baan",
            right_on="baan",
            how="left"
        ).drop(columns=["baan"])

        df = df.rename(columns={"runway_baan": "name_runway"})
    else:
        print("[WARN] geen df_ref_rows opgebouwd, dus geen runways toegewezen.")

    # Diagnostics
    print(f"[INFO] rows total        : {n_total}")
    print(f"[INFO] no HDF5 group     : {n_no_group}")
    print(f"[INFO] trace not in group: {n_no_link}")
    print(f"[INFO] rows filled       : {n_filled}")

    return df


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlon/2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

def add_plane_sensor_distances(
    df,
    lat_sensor_col="latitude_sensor_loc",
    lon_sensor_col="longitude_sensor_loc",
    lat_plane_col="latitude",
    lon_plane_col="longitude",
    alt_plane_col="altitude_ft_plane",
    alt_unit="ft",
    sensor_elev_col=None,           # hoogte van sensor, evt. voor later, bijv. paar meter? nu doen we 0..
    min_range_m=10.0
):
    # Horizontal ground distance
    d_horiz_m = haversine_m(
        df[lat_sensor_col].to_numpy(),
        df[lon_sensor_col].to_numpy(),
        df[lat_plane_col].to_numpy(),
        df[lon_plane_col].to_numpy(),
    )

    # Vertical separation (assume sensor_elev = 0 if not provided)
    if sensor_elev_col is None:
        sensor_elev_m = 0.0
    else:
        sensor_elev_m = df[sensor_elev_col].fillna(0.0).to_numpy()

    alt_plane = df[alt_plane_col].to_numpy().astype(float)
    if alt_unit.lower() == "ft":
        alt_plane = alt_plane * 0.3048  # feet -> meters

    dz_m = alt_plane - sensor_elev_m
    # 3D slant range
    slant_m = np.sqrt(d_horiz_m**2 + dz_m**2)

    # Numerically stable variants for ML
    slant_clip = np.clip(slant_m, min_range_m, None)
    inv_r2 = 1.0 / (slant_clip ** 2)           # intensity ~ 1/r^2
    log_r = np.log1p(slant_clip)               # smooth monotone transform
    # elevation angle above horizon (deg)
    elev_rad = np.arctan2(dz_m, d_horiz_m + 1e-9)
    elev_deg = np.degrees(elev_rad)

    # Add columns
    out = df.copy()
    out["dist_horizontal_m"] = d_horiz_m
    out["dz_m"] = dz_m
    out["slant_range_m"] = slant_m
    out["elevation_deg"] = elev_deg
    out["inv_square"] = inv_r2
    out["log_slant_m"] = log_r
    out["twenty_log10_r"] = 20.0 * np.log10(slant_clip)
    return out


#h5_path = "/Users/elisabethfokker/PycharmProjects/AmsterdamMainportSimulatieTool/Schiphol_geluid-main/data/sensornet_all_coordinates_new/traces.h5"
#general_sample_path = "/Users/elisabethfokker/PycharmProjects/AmsterdamMainportSimulatieTool/Schiphol_geluid-main/data/extra_data_with_link_small.csv"
#weather_data_path = "/Users/elisabethfokker/PycharmProjects/AmsterdamMainportSimulatieTool/Schiphol_geluid-main/code_for_grid_map/data/weather_info_three_stations.txt"
#change
h5_path = 'path_to_h5/traces.h5'
general_sample_path = "data/extra_data_with_link_small.csv"
weather_data_path = "data/weather_info_three_stations.txt"

print('add weather data...')
df=pd.read_csv(general_sample_path)

#uncomment to make a smaller df with last x rows (needed for uploading)
'''
df=pd.read_csv('extra_data_with_link.csv')
df_small, final_size_mb = shrink_df_to_max_size(df, max_size_mb=25)
print(df_small.shape)
print(f"Final size: {final_size_mb:.2f} MB")
df_small.to_csv("extra_data_with_link_small.csv", index=False)
'''

df["datetime"] = (pd.to_datetime(df["time"], unit="s", utc=True)  .dt.tz_convert("Europe/Amsterdam"))

print('Clean altitude variable...')
# --- Schiphol bounding box (covers all runways) ---
lat_min_schiphol, lat_max_schiphol = 52.27, 52.34
lon_min_schiphol, lon_max_schiphol = 4.70, 4.83

inside_box = (
    (df['lat']  >= lat_min_schiphol) & (df['lat']  <= lat_max_schiphol) &
    (df['lon'] >= lon_min_schiphol) & (df['lon'] <= lon_max_schiphol)
)

# altitude < -14.8 ft inside Schiphol → clamp to -14.8 ===
rule1 = (df['geometric_altitude'] < -14.8) & inside_box
df.loc[rule1, 'geometric_altitude'] = 0

#  altitude is NaN inside Schiphol → fill with -14.8 ===
rule2 = df['geometric_altitude'].isna() & inside_box
df.loc[rule2, 'geometric_altitude'] = 0

# altitude < 0 ft outside Schiphol → set to NaN ===
rule3 = (df['geometric_altitude'] < 0) & (~inside_box)
df.loc[rule3, 'geometric_altitude'] = np.nan

print(f"Rule1 applied to {rule1.sum()} rows")
print(f"Rule2 applied to {rule2.sum()} rows")
print(f"Rule3 applied to {rule3.sum()} rows")

print('Add weather variables...')
df_weather_loc =create_weather_location_df()
df = add_nearest_stations(df, df_weather_loc, k=10)

weather_df = pd.read_csv(weather_data_path)
df = merge_events_with_knmi(df, weather_df)
df['winddirection'] = df['weather_DD']

# make sure rows are sorted in time order
df = df.sort_values('datetime')

weather_vars = ['weather_T', 'weather_FF', 'weather_DD', 'weather_P', 'weather_U']
df["winddirection"] = pd.to_numeric(df["winddirection"], errors="coerce")
for col in weather_vars:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill NaNs with the nearest non-NaN in time (actually nearest index = nearest time)
for col in weather_vars + ["winddirection"]:
    df[col] = df[col].interpolate(method="nearest", limit_direction="both")


print('Build runway lines...') #dit is voor de bearing vanaf vliegtuigen variabelen...
runway_text = """
    baan,baannummer,latitude,longitude,direction_landing
    Polderbaan,18R,52.362680,4.711942,182
    Polderbaan,36L,52.328337,4.708859,2
    Zwanenburgbaan,18C,52.331395,4.740047,182
    Zwanenburgbaan,36C,52.302756,4.737417,2
    Buitenveldertbaan,09,52.316620,4.745797,88
    Buitenveldertbaan,27,52.318406,4.797479,268
    Aalsmeerbaan,18L,52.321670,4.780185,181
    Aalsmeerbaan,36R,52.290561,4.777289,1
    Oostbaan,04,52.300376,4.783400,40
    Oostbaan,22,52.314053,4.803021,220
    Kaagbaan,06,52.288009,4.734251,58
    Kaagbaan,24,52.304457,4.777159,238
    """

df_runways = pd.read_csv(StringIO(runway_text))
gdf_runway_lines = build_runway_lines(df_runways)

df = add_variables_from_traces_new(
    df=df,
    h5_path=h5_path,
    gdf_runway_lines=gdf_runway_lines,
    start=0)

print('tussentijds df met extra data statistics opslaan....')
df.to_csv("output_filled_extra_data_statistics.csv", index=False)

print(df[['sensor_latitude', 'sensor_longitude', 'lat', 'lon', 'longitude', 'latitude']].head(100))

print('add plane-sensor distances...')
df = add_plane_sensor_distances(df,
    lat_sensor_col="sensor_latitude",
    lon_sensor_col="sensor_longitude",
    lat_plane_col="lat",
    lon_plane_col="lon",
    alt_plane_col="geometric_altitude",
    alt_unit="ft") #altitude zoals nu is in ft, dit zorgt ervoor dat code naar m vertaalt.

print('added plane sensor distances...')
print(df.head())

df_ref = extract_runway_reference_points(df)
print(df_ref.head())

df_ref = extract_runway_reference_points(df)
df_runway_assigned = assign_runway_to_flights(df_ref, gdf_runway_lines)

print(df_runway_assigned.head())

df = df.merge(df_runway_assigned[['flight_nr', 'ref_lat', 'ref_lon']], on='flight_nr')
print(df.head())

df["heading_diff_sensor_plane"] = [heading_diff(bearing_deg(x, y, x2, y2), bearing_deg(x, y, x3, y3)) for x, y, x2, y2, x3, y3 in zip(df["lat"], df["lon"], df["sensor_latitude"], df["sensor_longitude"], df["latitude_runway"], df["longitude_runway"])]
df["heading_diff_sensor_wind"] = [heading_diff(bearing_deg(x, y, x2, y2), x3) for x, y, x2, y2, x3 in zip(df["lat"], df["lon"], df["sensor_latitude"], df["sensor_longitude"], df["winddirection"])]
print(df.head())

df.to_csv("output_filled_extra_data_all.csv", index=False)
