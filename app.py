import streamlit as st
import pandas as pd
import numpy as np
import folium
import math
from scipy.stats import gaussian_kde
from shapely.geometry import Polygon
from shapely.ops import unary_union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
from folium.plugins import HeatMap

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Analyse GPS des animaux",
    page_icon="🐐",
    layout="wide"
)

st.title("🐐 Analyse des déplacements des animaux")
st.write(
    "Application d'analyse des données GPS des colliers des animaux."
)

EARTH_RADIUS_KM = 6371.0
EARTH_RADIUS_M = 6371000.0
COLONNES_REQUISES = ["Date/Time", "Latitude", "Longitude", "Altitude"]


# ============================================================
# PHASE 1 — FONCTIONS GPS
# ============================================================

def distance_gps(lat1, lon1, lat2, lon2):
    """Distance Haversine en kilomètres."""
    lat1_rad = math.radians(float(lat1))
    lat2_rad = math.radians(float(lat2))
    dlat = math.radians(float(lat2) - float(lat1))
    dlon = math.radians(float(lon2) - float(lon1))

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(dlon / 2) ** 2
    )

    a = min(1.0, max(0.0, a))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def nettoyer_gps(gps, vitesse_max=15):
    """
    Nettoyage GPS :
    - valeurs manquantes
    - coordonnées invalides
    - doublons
    - ordre chronologique
    - vitesses supérieures au seuil
    """
    gps = gps.copy()

    gps = gps.dropna(
        subset=["Date/Time", "Latitude", "Longitude"]
    )

    gps = gps[
        (gps["Latitude"] >= -90)
        & (gps["Latitude"] <= 90)
        & (gps["Longitude"] >= -180)
        & (gps["Longitude"] <= 180)
    ]

    gps = gps.sort_values("Date/Time")
    gps = gps.drop_duplicates(
        subset=["Date/Time", "Latitude", "Longitude"]
    )
    gps = gps.reset_index(drop=True)

    if len(gps) <= 1:
        return gps

    indices_valides = [0]

    for i in range(1, len(gps)):
        dernier = indices_valides[-1]

        lat1 = gps.loc[dernier, "Latitude"]
        lon1 = gps.loc[dernier, "Longitude"]
        lat2 = gps.loc[i, "Latitude"]
        lon2 = gps.loc[i, "Longitude"]

        distance = distance_gps(lat1, lon1, lat2, lon2)

        temps_h = (
            gps.loc[i, "Date/Time"]
            - gps.loc[dernier, "Date/Time"]
        ).total_seconds() / 3600

        if temps_h <= 0:
            continue

        vitesse = distance / temps_h

        if vitesse <= vitesse_max:
            indices_valides.append(i)

    gps = gps.iloc[indices_valides].reset_index(drop=True)
    return gps


def calculer_distances(gps):
    gps = gps.copy()
    distances = [0.0]

    for i in range(1, len(gps)):
        d = distance_gps(
            gps.loc[i - 1, "Latitude"],
            gps.loc[i - 1, "Longitude"],
            gps.loc[i, "Latitude"],
            gps.loc[i, "Longitude"]
        )
        distances.append(d)

    gps["Distance_km"] = distances
    return gps


def preparer_donnees(df):
    """Sélectionne et convertit les colonnes nécessaires."""
    manquantes = [
        c for c in COLONNES_REQUISES
        if c not in df.columns
    ]

    if manquantes:
        return None, manquantes

    gps = df[COLONNES_REQUISES].copy()

    for colonne in ["Latitude", "Longitude", "Altitude"]:
        gps[colonne] = pd.to_numeric(
            gps[colonne],
            errors="coerce"
        )

    gps["Date/Time"] = pd.to_datetime(
        gps["Date/Time"],
        errors="coerce",
        dayfirst=True
    )

    return gps, []


# ============================================================
# PHASE 6 — PROJECTION ET MCP
# ============================================================

def projection_locale_reference(
    latitudes,
    longitudes,
    latitude_reference,
    longitude_reference,
    latitude_centre
):
    """Projection locale en mètres avec une référence commune."""
    x = []
    y = []

    for lat, lon in zip(latitudes, longitudes):
        x_m = (
            EARTH_RADIUS_M
            * math.radians(float(lon) - float(longitude_reference))
            * math.cos(float(latitude_centre))
        )
        y_m = (
            EARTH_RADIUS_M
            * math.radians(float(lat) - float(latitude_reference))
        )
        x.append(x_m)
        y.append(y_m)

    return np.asarray(x), np.asarray(y)


def projection_locale(latitudes, longitudes):
    """Projection locale approximative en mètres."""
    latitudes = list(latitudes)
    longitudes = list(longitudes)

    latitude_centre = math.radians(
        float(np.mean(latitudes))
    )
    latitude_reference = float(latitudes[0])
    longitude_reference = float(longitudes[0])

    return projection_locale_reference(
        latitudes,
        longitudes,
        latitude_reference,
        longitude_reference,
        latitude_centre
    )


def enveloppe_convexe(points):
    """Algorithme monotone chain."""
    points = sorted(set(
        (float(p[0]), float(p[1]))
        for p in points
    ))

    if len(points) <= 2:
        return points

    def produit_vectoriel(o, a, b):
        return (
            (a[0] - o[0]) * (b[1] - o[1])
            - (a[1] - o[1]) * (b[0] - o[0])
        )

    bas = []
    for point in points:
        while (
            len(bas) >= 2
            and produit_vectoriel(
                bas[-2], bas[-1], point
            ) <= 0
        ):
            bas.pop()
        bas.append(point)

    haut = []
    for point in reversed(points):
        while (
            len(haut) >= 2
            and produit_vectoriel(
                haut[-2], haut[-1], point
            ) <= 0
        ):
            haut.pop()
        haut.append(point)

    return bas[:-1] + haut[:-1]


def aire_polygone(points):
    if len(points) < 3:
        return 0.0

    aire = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        aire += x1 * y2 - x2 * y1

    return abs(aire) / 2.0


def perimetre_polygone(points):
    if len(points) < 2:
        return 0.0

    perimetre = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        perimetre += math.hypot(x2 - x1, y2 - y1)

    return perimetre


def calculer_mcp(gps):
    if len(gps) < 3:
        return None

    x, y = projection_locale(
        gps["Latitude"],
        gps["Longitude"]
    )

    hull = enveloppe_convexe(zip(x, y))

    if len(hull) < 3:
        return None

    return {
        "hull": hull,
        "aire_ha": aire_polygone(hull) / 10000,
        "perimetre_km": perimetre_polygone(hull) / 1000,
        "x": x,
        "y": y
    }


def calculer_mcp_95_approx(gps):
    """
    Estimation simplifiée du MCP 95 % par exclusion
    des 5 % de positions les plus éloignées du centre.
    """
    if len(gps) < 3:
        return None

    x, y = projection_locale(
        gps["Latitude"],
        gps["Longitude"]
    )

    cx = np.mean(x)
    cy = np.mean(y)
    distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    seuil = np.quantile(distances, 0.95)

    masque = distances <= seuil
    points = list(zip(x[masque], y[masque]))

    if len(points) < 3:
        return None

    hull = enveloppe_convexe(points)

    if len(hull) < 3:
        return None

    return {
        "hull": hull,
        "aire_ha": aire_polygone(hull) / 10000,
        "perimetre_km": perimetre_polygone(hull) / 1000
    }


def xy_vers_latlon(points, gps):
    """Transforme des coordonnées locales en latitude/longitude."""
    if not points:
        return []

    lat_ref = float(gps["Latitude"].iloc[0])
    lon_ref = float(gps["Longitude"].iloc[0])
    lat_centre = math.radians(
        float(gps["Latitude"].mean())
    )

    resultat = []

    for px, py in points:
        lat = (
            lat_ref
            + math.degrees(py / EARTH_RADIUS_M)
        )
        lon = (
            lon_ref
            + math.degrees(
                px
                / (
                    EARTH_RADIUS_M
                    * max(0.000001, math.cos(lat_centre))
                )
            )
        )
        resultat.append([lat, lon])

    return resultat


# ============================================================
# KDE 50 % / 100 %
# ============================================================

def seuil_kde(densite, fraction, dx, dy):
    """Calcule le seuil contenant approximativement la fraction."""
    valeurs = np.asarray(densite).ravel()
    valeurs = valeurs[np.isfinite(valeurs)]

    if len(valeurs) == 0:
        return None

    valeurs = np.sort(valeurs)[::-1]
    masse = valeurs * dx * dy
    cumul = np.cumsum(masse)
    total = cumul[-1]

    if total <= 0:
        return None

    cible = fraction * total
    index = np.searchsorted(cumul, cible)
    index = min(index, len(valeurs) - 1)

    return float(valeurs[index])


def calculer_kde(gps, resolution=100, bandwidth=None):
    """
    Calcule une KDE sur une grille locale en mètres.
    """
    if len(gps) < 5:
        return None

    x, y = projection_locale(
        gps["Latitude"],
        gps["Longitude"]
    )

    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None

    try:
        kde = gaussian_kde(
            np.vstack([x, y]),
            bw_method=bandwidth
        )
    except Exception:
        return None

    marge_x = max(np.ptp(x) * 0.10, 50.0)
    marge_y = max(np.ptp(y) * 0.10, 50.0)

    xmin = x.min() - marge_x
    xmax = x.max() + marge_x
    ymin = y.min() - marge_y
    ymax = y.max() + marge_y

    resolution = int(max(50, min(200, resolution)))

    gx = np.linspace(xmin, xmax, resolution)
    gy = np.linspace(ymin, ymax, resolution)

    X, Y = np.meshgrid(gx, gy)

    positions = np.vstack([
        X.ravel(),
        Y.ravel()
    ])

    Z = kde(positions).reshape(X.shape)

    dx = float(gx[1] - gx[0])
    dy = float(gy[1] - gy[0])

    seuil_50 = seuil_kde(
        Z,
        0.50,
        dx,
        dy
    )

    # Une KDE théorique a des queues infinies.
    # Ici, le 100 % correspond à l'emprise de la grille.
    seuil_100 = float(np.nanmin(Z))

    return {
        "X": X,
        "Y": Y,
        "Z": Z,
        "seuil_50": seuil_50,
        "seuil_100": seuil_100
    }


def contours_kde(kde_result, seuil):
    """Transforme un niveau KDE en géométrie Shapely."""
    if kde_result is None or seuil is None:
        return None

    X = kde_result["X"]
    Y = kde_result["Y"]
    Z = kde_result["Z"]

    zmin = float(np.nanmin(Z))
    zmax = float(np.nanmax(Z))

    if not np.isfinite(zmin) or not np.isfinite(zmax):
        return None

    niveau = float(seuil)

    if niveau <= zmin:
        niveau = zmin + (zmax - zmin) * 1e-9

    if niveau >= zmax:
        niveau = zmax - (zmax - zmin) * 1e-9

    if niveau <= zmin or niveau >= zmax:
        return None

    try:
        fig, ax = plt.subplots()
        cs = ax.contour(X, Y, Z, levels=[niveau])
        segments = cs.allsegs[0]
        plt.close(fig)
    except Exception:
        plt.close("all")
        return None

    geometries = []

    for segment in segments:
        if len(segment) < 4:
            continue

        try:
            polygon = Polygon(segment)

            if polygon.is_empty:
                continue

            if not polygon.is_valid:
                polygon = polygon.buffer(0)

            if not polygon.is_empty:
                geometries.append(polygon)
        except Exception:
            continue

    if not geometries:
        return None

    try:
        union = unary_union(geometries)
        return union if not union.is_empty else None
    except Exception:
        return None


def geometrie_vers_geojson(geometry, gps):
    """Convertit une géométrie locale en GeoJSON."""
    if geometry is None:
        return None

    lat_ref = float(gps["Latitude"].iloc[0])
    lon_ref = float(gps["Longitude"].iloc[0])
    lat_centre = math.radians(
        float(gps["Latitude"].mean())
    )
    cos_lat = max(0.000001, math.cos(lat_centre))

    def transformer_coord(coord):
        x, y = coord
        lat = lat_ref + math.degrees(y / EARTH_RADIUS_M)
        lon = lon_ref + math.degrees(
            x / (EARTH_RADIUS_M * cos_lat)
        )
        return [lon, lat]

    def transformer_polygon(poly):
        return {
            "type": "Polygon",
            "coordinates": [[
                transformer_coord(c)
                for c in poly.exterior.coords
            ]]
        }

    if geometry.geom_type == "Polygon":
        return transformer_polygon(geometry)

    if geometry.geom_type == "MultiPolygon":
        return {
            "type": "MultiPolygon",
            "coordinates": [
                [[
                    transformer_coord(c)
                    for c in poly.exterior.coords
                ]]
                for poly in geometry.geoms
            ]
        }

    return None


# ============================================================
# RANGE OVERLAP — VI
# ============================================================

def calculer_vi_kde(gps1, gps2, resolution=100):
    """
    Volume of Intersection :
    VI = intégrale du minimum des deux densités KDE.
    0 = absence de chevauchement ; 1 = distributions identiques.
    """
    if len(gps1) < 5 or len(gps2) < 5:
        return None

    combine = pd.concat(
        [
            gps1[["Latitude", "Longitude"]],
            gps2[["Latitude", "Longitude"]]
        ],
        ignore_index=True
    )

    latitude_reference = float(
        combine["Latitude"].iloc[0]
    )
    longitude_reference = float(
        combine["Longitude"].iloc[0]
    )
    latitude_centre = math.radians(
        float(combine["Latitude"].mean())
    )

    x_all, y_all = projection_locale_reference(
        combine["Latitude"],
        combine["Longitude"],
        latitude_reference,
        longitude_reference,
        latitude_centre
    )

    x1, y1 = projection_locale_reference(
        gps1["Latitude"],
        gps1["Longitude"],
        latitude_reference,
        longitude_reference,
        latitude_centre
    )

    x2, y2 = projection_locale_reference(
        gps2["Latitude"],
        gps2["Longitude"],
        latitude_reference,
        longitude_reference,
        latitude_centre
    )

    if np.ptp(x_all) == 0 or np.ptp(y_all) == 0:
        return None

    try:
        kde1 = gaussian_kde(np.vstack([x1, y1]))
        kde2 = gaussian_kde(np.vstack([x2, y2]))
    except Exception:
        return None

    marge_x = max(np.ptp(x_all) * 0.15, 50.0)
    marge_y = max(np.ptp(y_all) * 0.15, 50.0)

    resolution = int(max(50, min(200, resolution)))

    gx = np.linspace(
        x_all.min() - marge_x,
        x_all.max() + marge_x,
        resolution
    )
    gy = np.linspace(
        y_all.min() - marge_y,
        y_all.max() + marge_y,
        resolution
    )

    X, Y = np.meshgrid(gx, gy)

    positions = np.vstack([
        X.ravel(),
        Y.ravel()
    ])

    Z1 = kde1(positions).reshape(X.shape)
    Z2 = kde2(positions).reshape(X.shape)

    dx = float(gx[1] - gx[0])
    dy = float(gy[1] - gy[0])

    vi = np.sum(
        np.minimum(Z1, Z2)
    ) * dx * dy

    return float(max(0.0, min(1.0, vi)))


# ============================================================
# SAISONS
# ============================================================

def saison_depuis_mois(mois):
    if mois in [12, 1, 2]:
        return "Hiver"
    if mois in [3, 4, 5]:
        return "Printemps"
    if mois in [6, 7, 8]:
        return "Été"
    return "Automne"


def ajouter_saison(gps):
    gps = gps.copy()
    gps["Saison"] = gps["Date/Time"].dt.month.map(
        saison_depuis_mois
    )
    return gps


# ============================================================
# CARTOGRAPHIE
# ============================================================

def ajouter_fond_carte(carte):
    folium.TileLayer(
        "OpenStreetMap",
        name="🗺️ Carte"
    ).add_to(carte)

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/World_Imagery/"
            "MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri",
        name="🛰️ Satellite"
    ).add_to(carte)


def ajouter_points_lignes(
    carte,
    gps,
    mode_affichage="Points + lignes"
):
    parcours = gps[
        ["Latitude", "Longitude"]
    ].values.tolist()

    if (
        mode_affichage in ["Lignes", "Points + lignes"]
        and len(parcours) > 1
    ):
        folium.PolyLine(
            parcours,
            weight=4,
            tooltip="Trajectoire GPS"
        ).add_to(carte)

    if mode_affichage in ["Points", "Points + lignes"]:
        for _, ligne in gps.iterrows():
            folium.CircleMarker(
                location=[
                    ligne["Latitude"],
                    ligne["Longitude"]
                ],
                radius=3,
                fill=True,
                fill_opacity=0.8,
                popup=str(ligne["Date/Time"])
            ).add_to(carte)


def ajouter_depart_arrivee(carte, gps):
    if len(gps) == 0:
        return

    folium.Marker(
        [
            gps["Latitude"].iloc[0],
            gps["Longitude"].iloc[0]
        ],
        tooltip="🟢 Départ",
        popup="Départ",
        icon=folium.DivIcon(
            html=(
                '<div style="'
                'background:#1b8a3a;color:white;'
                'border-radius:50%;width:28px;height:28px;'
                'text-align:center;line-height:28px;'
                'font-weight:bold;">D</div>'
            )
        )
    ).add_to(carte)

    folium.Marker(
        [
            gps["Latitude"].iloc[-1],
            gps["Longitude"].iloc[-1]
        ],
        tooltip="🔴 Arrivée",
        popup="Arrivée",
        icon=folium.DivIcon(
            html=(
                '<div style="'
                'background:#c62828;color:white;'
                'border-radius:50%;width:28px;height:28px;'
                'text-align:center;line-height:28px;'
                'font-weight:bold;">A</div>'
            )
        )
    ).add_to(carte)


# ============================================================
# FONCTIONS DE TRAITEMENT
# ============================================================

def charger_et_nettoyer_feuille(
    fichier,
    nom_animal,
    vitesse_max=15
):
    try:
        df = pd.read_excel(
            fichier,
            sheet_name=nom_animal
        )

        gps, manquantes = preparer_donnees(df)

        if gps is None:
            return None

        gps = nettoyer_gps(
            gps,
            vitesse_max=vitesse_max
        )

        if len(gps) == 0:
            return None

        return calculer_distances(gps)

    except Exception:
        return None


def appliquer_periode(gps, debut, fin):
    return gps[
        (gps["Date/Time"].dt.date >= debut)
        & (gps["Date/Time"].dt.date <= fin)
    ].copy()


# ============================================================
# APPLICATION
# ============================================================

fichier = st.file_uploader(
    "📂 Importer le fichier Excel GPS",
    type=["xlsx"]
)

if fichier is None:
    st.info(
        "👆 Commencez par importer votre fichier Excel GPS."
    )
    st.stop()


# ============================================================
# PHASE 2 — CHARGEMENT / SELECTION
# ============================================================

try:
    excel = pd.ExcelFile(fichier)
except Exception as erreur:
    st.error(
        f"❌ Impossible de lire le fichier Excel : {erreur}"
    )
    st.stop()

st.success("✅ Fichier Excel chargé avec succès !")

st.header("🐐 PHASE 2 — Sélection de l'animal")

animal = st.sidebar.selectbox(
    "Choisir l'animal",
    excel.sheet_names
)

try:
    df = pd.read_excel(
        fichier,
        sheet_name=animal
    )
except Exception as erreur:
    st.error(
        f"❌ Erreur de lecture de la feuille : {erreur}"
    )
    st.stop()

gps_brut, colonnes_manquantes = preparer_donnees(df)

if gps_brut is None:
    st.error(
        "❌ Colonnes manquantes : "
        + ", ".join(colonnes_manquantes)
    )
    st.stop()


# ============================================================
# PHASE 3 — PREPARATION
# ============================================================

st.header("⚙️ PHASE 3 — Préparation des données")

st.write(
    f"Feuille sélectionnée : **{animal}**"
)

st.write(
    f"Nombre de lignes initiales : **{len(gps_brut)}**"
)

st.write(
    "Colonnes utilisées : "
    + ", ".join(COLONNES_REQUISES)
)


# ============================================================
# PHASE 4 — NETTOYAGE
# ============================================================

st.header("🧹 PHASE 4 — Nettoyage GPS")

vitesse_max = st.sidebar.number_input(
    "Vitesse maximale pour le nettoyage (km/h)",
    min_value=1.0,
    max_value=100.0,
    value=15.0,
    step=1.0
)

nombre_avant = len(gps_brut)

gps = nettoyer_gps(
    gps_brut,
    vitesse_max=vitesse_max
)

nombre_apres = len(gps)
points_supprimes = nombre_avant - nombre_apres

st.info(
    f"🧹 Nettoyage GPS : "
    f"{nombre_avant} points au départ → "
    f"{nombre_apres} points conservés → "
    f"{points_supprimes} points supprimés."
)

if len(gps) == 0:
    st.error(
        "❌ Aucune donnée GPS valide après nettoyage."
    )
    st.stop()


# ============================================================
# FILTRE DATE
# ============================================================

st.sidebar.header("📅 Filtre temporel")

date_min = gps["Date/Time"].dt.date.min()
date_max = gps["Date/Time"].dt.date.max()

dates = st.sidebar.date_input(
    "Choisir la période",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max
)

if isinstance(dates, (tuple, list)) and len(dates) == 2:
    date_debut = dates[0]
    date_fin = dates[1]
else:
    date_debut = date_min
    date_fin = date_max

gps = appliquer_periode(
    gps,
    date_debut,
    date_fin
)

if len(gps) == 0:
    st.warning(
        "⚠️ Aucune donnée pour cette période."
    )
    st.stop()

gps = calculer_distances(gps)


# ============================================================
# PHASE 5 — ANALYSE DES DEPLACEMENTS
# ============================================================

st.header("📊 PHASE 5 — Analyse des déplacements")

distance_totale = gps["Distance_km"].sum()

if len(gps) > 1:
    duree_h = (
        gps["Date/Time"].iloc[-1]
        - gps["Date/Time"].iloc[0]
    ).total_seconds() / 3600
else:
    duree_h = 0.0

vitesse_moyenne = (
    distance_totale / duree_h
    if duree_h > 0
    else 0.0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("📍 Positions GPS", len(gps))
col2.metric(
    "📏 Distance totale",
    f"{distance_totale:.2f} km"
)
col3.metric(
    "⏱️ Durée",
    f"{duree_h:.2f} h"
)
col4.metric(
    "🚶 Vitesse moyenne",
    f"{vitesse_moyenne:.2f} km/h"
)


# ============================================================
# CARTE PRINCIPALE
# ============================================================

st.subheader("🗺️ Parcours GPS")

mode_affichage = st.radio(
    "Mode d'affichage",
    [
        "Points",
        "Lignes",
        "Points + lignes"
    ],
    horizontal=True
)

centre_latitude = float(gps["Latitude"].mean())
centre_longitude = float(gps["Longitude"].mean())

carte = folium.Map(
    location=[
        centre_latitude,
        centre_longitude
    ],
    zoom_start=15,
    control_scale=True
)

ajouter_fond_carte(carte)
ajouter_points_lignes(
    carte,
    gps,
    mode_affichage
)
ajouter_depart_arrivee(
    carte,
    gps
)
folium.LayerControl().add_to(carte)

st_folium(
    carte,
    width=1200,
    height=600
)


# ============================================================
# PHASE 6 — ANALYSE SPATIALE
# ============================================================

st.header("📍 PHASE 6 — Analyse spatiale avancée")

st.subheader("📍 Centre spatial")

col1, col2 = st.columns(2)

col1.metric(
    "Latitude moyenne",
    f"{centre_latitude:.6f}"
)
col2.metric(
    "Longitude moyenne",
    f"{centre_longitude:.6f}"
)

st.subheader("📐 Étendue géographique")

lat_min = gps["Latitude"].min()
lat_max = gps["Latitude"].max()
lon_min = gps["Longitude"].min()
lon_max = gps["Longitude"].max()

col1, col2 = st.columns(2)

col1.write(
    f"Latitude : **{lat_min:.6f} → {lat_max:.6f}**"
)
col2.write(
    f"Longitude : **{lon_min:.6f} → {lon_max:.6f}**"
)


# ------------------------------------------------------------
# MCP 100 / 95
# ------------------------------------------------------------

mcp = calculer_mcp(gps)
mcp_95 = calculer_mcp_95_approx(gps)

st.subheader("📐 MCP 100 % et MCP 95 %")

col1, col2, col3, col4 = st.columns(4)

if mcp is not None:
    col1.metric(
        "🔷 MCP 100 %",
        f"{mcp['aire_ha']:.2f} ha"
    )
    col2.metric(
        "Périmètre MCP 100 %",
        f"{mcp['perimetre_km']:.2f} km"
    )
else:
    col1.metric("🔷 MCP 100 %", "N/A")
    col2.metric("Périmètre MCP 100 %", "N/A")

if mcp_95 is not None:
    col3.metric(
        "🔶 MCP 95 %",
        f"{mcp_95['aire_ha']:.2f} ha"
    )
    col4.metric(
        "Périmètre MCP 95 %",
        f"{mcp_95['perimetre_km']:.2f} km"
    )
else:
    col3.metric("🔶 MCP 95 %", "N/A")
    col4.metric("Périmètre MCP 95 %", "N/A")

st.caption(
    "Le MCP 95 % présenté ici est une estimation simplifiée "
    "basée sur l'éloignement des positions par rapport au centre."
)


# ------------------------------------------------------------
# CARTE MCP
# ------------------------------------------------------------

st.subheader("🗺️ Carte des zones spatiales")

carte_mcp = folium.Map(
    location=[
        centre_latitude,
        centre_longitude
    ],
    zoom_start=15,
    control_scale=True
)

ajouter_fond_carte(carte_mcp)

ajouter_points_lignes(
    carte_mcp,
    gps,
    mode_affichage
)

ajouter_depart_arrivee(
    carte_mcp,
    gps
)

if mcp is not None:
    coordonnees_100 = xy_vers_latlon(
        mcp["hull"],
        gps
    )

    folium.Polygon(
        locations=coordonnees_100,
        tooltip="🔷 MCP 100 %",
        fill=True,
        fill_opacity=0.15,
        weight=2
    ).add_to(carte_mcp)

if mcp_95 is not None:
    coordonnees_95 = xy_vers_latlon(
        mcp_95["hull"],
        gps
    )

    folium.Polygon(
        locations=coordonnees_95,
        tooltip="🔶 MCP 95 %",
        fill=True,
        fill_opacity=0.20,
        weight=2
    ).add_to(carte_mcp)

folium.LayerControl().add_to(carte_mcp)

st_folium(
    carte_mcp,
    width=1200,
    height=600
)


# ------------------------------------------------------------
# HEATMAP
# ------------------------------------------------------------

st.subheader("🔥 HeatMap — zones fréquentées")

carte_heatmap = folium.Map(
    location=[
        centre_latitude,
        centre_longitude
    ],
    zoom_start=15,
    control_scale=True
)

ajouter_fond_carte(carte_heatmap)

points_heatmap = gps[
    ["Latitude", "Longitude"]
].values.tolist()

HeatMap(
    points_heatmap,
    radius=15,
    blur=20,
    min_opacity=0.4
).add_to(carte_heatmap)

folium.LayerControl().add_to(carte_heatmap)

st_folium(
    carte_heatmap,
    width=1200,
    height=600
)


# ------------------------------------------------------------
# KDE 50 / 100
# ------------------------------------------------------------

st.subheader("🟠 KDE 50 % et KDE 100 %")

col1, col2 = st.columns(2)

with col1:
    kde_resolution = st.slider(
        "Résolution de la KDE",
        min_value=50,
        max_value=150,
        value=100,
        step=10
    )

with col2:
    bandwidth_text = st.selectbox(
        "Lissage KDE",
        [
            "Automatique",
            "Scott",
            "Silverman"
        ]
    )

bandwidth = None

if bandwidth_text == "Scott":
    bandwidth = "scott"
elif bandwidth_text == "Silverman":
    bandwidth = "silverman"

if len(gps) < 5:
    st.warning(
        "⚠️ Au moins 5 positions sont nécessaires pour calculer la KDE."
    )
    kde_result = None
else:
    with st.spinner("Calcul de la KDE..."):
        kde_result = calculer_kde(
            gps,
            resolution=kde_resolution,
            bandwidth=bandwidth
        )

if kde_result is not None:
    col1, col2 = st.columns(2)

    geometry_50 = contours_kde(
        kde_result,
        kde_result["seuil_50"]
    )
    geometry_100 = contours_kde(
        kde_result,
        kde_result["seuil_100"]
    )

    geo_50 = (
        geometrie_vers_geojson(
            geometry_50,
            gps
        )
        if geometry_50 is not None
        else None
    )

    geo_100 = (
        geometrie_vers_geojson(
            geometry_100,
            gps
        )
        if geometry_100 is not None
        else None
    )

    if geometry_50 is not None:
        col1.metric(
            "🟠 Surface KDE 50 %",
            f"{geometry_50.area / 10000:.2f} ha"
        )

    if geometry_100 is not None:
        col2.metric(
            "🔵 Surface KDE 100 %",
            f"{geometry_100.area / 10000:.2f} ha"
        )

    carte_kde = folium.Map(
        location=[
            centre_latitude,
            centre_longitude
        ],
        zoom_start=15,
        control_scale=True
    )

    ajouter_fond_carte(carte_kde)

    ajouter_points_lignes(
        carte_kde,
        gps,
        mode_affichage
    )

    ajouter_depart_arrivee(
        carte_kde,
        gps
    )

    if geo_100 is not None:
        folium.GeoJson(
            geo_100,
            name="🔵 KDE 100 %",
            style_function=lambda feature: {
                "fillOpacity": 0.08,
                "weight": 2
            },
            tooltip="KDE 100 %"
        ).add_to(carte_kde)

    if geo_50 is not None:
        folium.GeoJson(
            geo_50,
            name="🟠 KDE 50 %",
            style_function=lambda feature: {
                "fillOpacity": 0.20,
                "weight": 3
            },
            tooltip="KDE 50 %"
        ).add_to(carte_kde)

    folium.LayerControl().add_to(carte_kde)

    st_folium(
        carte_kde,
        width=1200,
        height=600
    )

    st.caption(
        "La KDE 100 % est représentée ici par l'emprise "
        "de la grille de calcul. Les résultats KDE dépendent "
        "notamment du lissage et de la résolution."
    )
else:
    st.warning(
        "⚠️ KDE impossible à calculer avec ces données."
    )


# ============================================================
# RANGE OVERLAP / VI
# ============================================================

st.subheader(
    "🔄 Range Overlap — Volume of Intersection (VI)"
)

st.write(
    "Le VI compare deux distributions spatiales KDE. "
    "Une valeur proche de 0 indique peu de chevauchement, "
    "tandis qu'une valeur proche de 1 indique un fort "
    "chevauchement des distributions."
)

mode_comparaison = st.radio(
    "Type de comparaison spatiale",
    [
        "Deux animaux",
        "Deux jours",
        "Deux saisons"
    ],
    horizontal=True
)

gps_comparaison_1 = None
gps_comparaison_2 = None
nom_comparaison_1 = ""
nom_comparaison_2 = ""

if mode_comparaison == "Deux animaux":

    animaux_vi = st.multiselect(
        "Sélectionner deux animaux",
        excel.sheet_names,
        default=(
            excel.sheet_names[:2]
            if len(excel.sheet_names) >= 2
            else excel.sheet_names
        ),
        max_selections=2
    )

    if len(animaux_vi) == 2:
        gps_comparaison_1 = charger_et_nettoyer_feuille(
            fichier,
            animaux_vi[0],
            vitesse_max=vitesse_max
        )
        gps_comparaison_2 = charger_et_nettoyer_feuille(
            fichier,
            animaux_vi[1],
            vitesse_max=vitesse_max
        )

        if gps_comparaison_1 is not None:
            gps_comparaison_1 = appliquer_periode(
                gps_comparaison_1,
                date_debut,
                date_fin
            )

        if gps_comparaison_2 is not None:
            gps_comparaison_2 = appliquer_periode(
                gps_comparaison_2,
                date_debut,
                date_fin
            )

        nom_comparaison_1 = animaux_vi[0]
        nom_comparaison_2 = animaux_vi[1]

elif mode_comparaison == "Deux jours":

    jours_disponibles = sorted(
        gps["Date/Time"].dt.date.unique()
    )

    if len(jours_disponibles) >= 2:
        jours_vi = st.multiselect(
            "Sélectionner deux jours",
            jours_disponibles,
            default=jours_disponibles[:2],
            max_selections=2
        )

        if len(jours_vi) == 2:
            gps_comparaison_1 = gps[
                gps["Date/Time"].dt.date == jours_vi[0]
            ].copy()

            gps_comparaison_2 = gps[
                gps["Date/Time"].dt.date == jours_vi[1]
            ].copy()

            nom_comparaison_1 = str(jours_vi[0])
            nom_comparaison_2 = str(jours_vi[1])
    else:
        st.info(
            "Il faut au moins deux jours différents."
        )

else:
    gps_saisons = ajouter_saison(gps)

    saisons_disponibles = [
        s for s in [
            "Hiver",
            "Printemps",
            "Été",
            "Automne"
        ]
        if s in gps_saisons["Saison"].unique()
    ]

    if len(saisons_disponibles) >= 2:
        saisons_vi = st.multiselect(
            "Sélectionner deux saisons",
            saisons_disponibles,
            default=saisons_disponibles[:2],
            max_selections=2
        )

        if len(saisons_vi) == 2:
            gps_comparaison_1 = gps_saisons[
                gps_saisons["Saison"] == saisons_vi[0]
            ].copy()

            gps_comparaison_2 = gps_saisons[
                gps_saisons["Saison"] == saisons_vi[1]
            ].copy()

            nom_comparaison_1 = saisons_vi[0]
            nom_comparaison_2 = saisons_vi[1]
    else:
        st.info(
            "Il faut au moins deux saisons représentées."
        )

if (
    gps_comparaison_1 is not None
    and gps_comparaison_2 is not None
):
    if (
        len(gps_comparaison_1) >= 5
        and len(gps_comparaison_2) >= 5
    ):
        with st.spinner("Calcul du Range Overlap..."):
            vi = calculer_vi_kde(
                gps_comparaison_1,
                gps_comparaison_2,
                resolution=kde_resolution
            )

        if vi is not None:
            st.metric(
                f"VI — {nom_comparaison_1} / {nom_comparaison_2}",
                f"{vi:.3f}"
            )

            st.progress(
                vi,
                text=f"Chevauchement KDE estimé : {vi:.1%}"
            )
        else:
            st.warning(
                "⚠️ Impossible de calculer le VI."
            )
    else:
        st.warning(
            "⚠️ Chaque groupe doit contenir au moins "
            "5 positions pour calculer le VI."
        )


# ============================================================
# PHASE 7 — ANALYSE TEMPORELLE
# ============================================================

st.header("⏰ PHASE 7 — Analyse temporelle avancée")

gps["Jour"] = gps["Date/Time"].dt.date
gps["Heure"] = gps["Date/Time"].dt.hour

st.subheader("📏 Distance parcourue par jour")

distance_par_jour = gps.groupby("Jour")["Distance_km"].sum()
st.bar_chart(distance_par_jour)

st.subheader("📅 Nombre de positions par jour")

positions_par_jour = gps.groupby("Jour").size()
st.bar_chart(positions_par_jour)

st.subheader("🕐 Positions par heure")

positions_par_heure = gps.groupby("Heure").size()
st.bar_chart(positions_par_heure)

st.write(
    "Le nombre de positions par heure représente la "
    "répartition des observations GPS. Il ne constitue "
    "pas directement une mesure de l'activité biologique "
    "si la fréquence d'enregistrement varie."
)

st.subheader("📈 Distance cumulée")

distance_cumulative = gps["Distance_km"].cumsum()

graphique_distance = pd.DataFrame(
    {
        "Distance cumulée (km)": distance_cumulative.values
    },
    index=gps["Date/Time"]
)

st.line_chart(graphique_distance)

st.subheader("📊 Statistiques journalières")

moyenne_distance_jour = distance_par_jour.mean()

if len(distance_par_jour) > 0:
    jour_max_distance = distance_par_jour.idxmax()
    distance_max_jour = distance_par_jour.max()
else:
    jour_max_distance = "N/A"
    distance_max_jour = 0

col1, col2, col3 = st.columns(3)

col1.metric(
    "📏 Distance moyenne / jour",
    f"{moyenne_distance_jour:.2f} km"
)
col2.metric(
    "🏆 Jour avec plus grande distance",
    str(jour_max_distance)
)
col3.metric(
    "📈 Distance maximale",
    f"{distance_max_jour:.2f} km"
)

st.subheader("⛰️ Évolution de l'altitude")

altitude = gps[
    ["Date/Time", "Altitude"]
].dropna()

if len(altitude) > 0:
    st.line_chart(
        altitude.set_index("Date/Time")
    )

st.subheader("📏 Statistiques d'altitude")

if gps["Altitude"].notna().sum() > 0:
    altitude_moyenne = gps["Altitude"].mean()
    altitude_min = gps["Altitude"].min()
    altitude_max = gps["Altitude"].max()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Altitude moyenne",
        f"{altitude_moyenne:.2f} m"
    )
    col2.metric(
        "Altitude minimale",
        f"{altitude_min:.2f} m"
    )
    col3.metric(
        "Altitude maximale",
        f"{altitude_max:.2f} m"
    )


# ============================================================
# PHASE 8 — COMPARAISON ENTRE ANIMAUX
# ============================================================

st.header("🐐 PHASE 8 — Comparaison entre animaux")

resultats_animaux = []

for nom_animal in excel.sheet_names:

    donnees = charger_et_nettoyer_feuille(
        fichier,
        nom_animal,
        vitesse_max=vitesse_max
    )

    if donnees is None or len(donnees) == 0:
        continue

    # Même période que l'animal sélectionné.
    donnees = appliquer_periode(
        donnees,
        date_debut,
        date_fin
    )

    if len(donnees) == 0:
        continue

    donnees = calculer_distances(donnees)

    distance = donnees["Distance_km"].sum()

    if len(donnees) > 1:
        duree = (
            donnees["Date/Time"].iloc[-1]
            - donnees["Date/Time"].iloc[0]
        ).total_seconds() / 3600
    else:
        duree = 0.0

    vitesse = (
        distance / duree
        if duree > 0
        else 0.0
    )

    mcp_animal = calculer_mcp(donnees)
    mcp95_animal = calculer_mcp_95_approx(donnees)

    surface_mcp = (
        mcp_animal["aire_ha"]
        if mcp_animal is not None
        else 0.0
    )

    surface_mcp95 = (
        mcp95_animal["aire_ha"]
        if mcp95_animal is not None
        else 0.0
    )

    altitude_moyenne = donnees["Altitude"].mean()

    resultats_animaux.append(
        {
            "Animal": nom_animal,
            "Positions GPS": len(donnees),
            "Distance totale (km)": round(
                distance, 2
            ),
            "Durée (h)": round(
                duree, 2
            ),
            "Vitesse moyenne (km/h)": round(
                vitesse, 2
            ),
            "MCP 100 % (ha)": round(
                surface_mcp, 2
            ),
            "MCP 95 % (ha)": round(
                surface_mcp95, 2
            ),
            "Altitude moyenne (m)": round(
                altitude_moyenne, 2
            )
        }
    )

if resultats_animaux:

    tableau_comparaison = pd.DataFrame(
        resultats_animaux
    )

    st.dataframe(
        tableau_comparaison,
        width="stretch"
    )

    st.subheader("📏 Comparaison des distances")

    st.bar_chart(
        tableau_comparaison.set_index("Animal")[
            "Distance totale (km)"
        ]
    )

    st.subheader("🔷 Comparaison MCP 100 %")

    st.bar_chart(
        tableau_comparaison.set_index("Animal")[
            "MCP 100 % (ha)"
        ]
    )

    st.subheader("🔶 Comparaison MCP 95 %")

    st.bar_chart(
        tableau_comparaison.set_index("Animal")[
            "MCP 95 % (ha)"
        ]
    )

    st.subheader("🚶 Comparaison des vitesses")

    st.bar_chart(
        tableau_comparaison.set_index("Animal")[
            "Vitesse moyenne (km/h)"
        ]
    )

else:
    st.warning(
        "⚠️ Impossible de comparer les animaux."
    )


# ============================================================
# PHASE 9 — DONNEES ANALYSEES
# ============================================================

st.header("📋 PHASE 9 — Données GPS analysées")

st.dataframe(
    gps,
    width="stretch"
)


# ============================================================
# PHASE 10 — EXPORT
# ============================================================

st.header("📥 PHASE 10 — Export")

fichier_csv = gps.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="📥 Télécharger les données analysées",
    data=fichier_csv,
    file_name=f"{animal}_analyse_GPS.csv",
    mime="text/csv"
)

if resultats_animaux:

    comparaison_csv = tableau_comparaison.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Télécharger la comparaison des animaux",
        data=comparaison_csv,
        file_name="comparaison_animaux_GPS.csv",
        mime="text/csv"
    )


# ============================================================
# INFORMATIONS METHODOLOGIQUES
# ============================================================

with st.expander("ℹ️ Informations méthodologiques"):

    st.write(
        "• Le nettoyage utilise par défaut un seuil de "
        "15 km/h, modifiable dans la barre latérale."
    )

    st.write(
        "• Le MCP 100 % est une enveloppe convexe des "
        "positions et est sensible aux valeurs extrêmes."
    )

    st.write(
        "• Le MCP 95 % est une estimation simplifiée basée "
        "sur l'éloignement au centre."
    )

    st.write(
        "• La KDE dépend du nombre de positions, du lissage "
        "et de la résolution de calcul."
    )

    st.write(
        "• Le VI est calculé comme l'intégrale du minimum "
        "des deux densités KDE sur une grille commune."
    )

    st.write(
        "• La KDE 100 % affichée correspond à l'emprise "
        "de la grille de calcul ; une distribution KDE "
        "théorique possède des queues infinies."
    )
