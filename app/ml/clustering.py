"""Unsupervised energy-behaviour segmentation using K-Means.

Clusters are learned from measurable meter behaviour rather than arbitrary labels.
The API reports the selected k using silhouette score so the result is auditable.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

FEATURES = ["mean_daily_kwh", "std_daily_kwh", "peak_daily_kwh", "night_share", "weekend_ratio"]

def cluster_energy_behaviour(hourly: pd.DataFrame):
    if "meter_id" not in hourly.columns and "building" not in hourly.columns:
        return {"available": False, "message": "Clustering requires a meter_id or building column.", "clusters": [], "members": []}
    group_col = "building" if "building" in hourly.columns and hourly["building"].notna().any() else "meter_id"
    df = hourly.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5
    rows=[]
    for name,g in df.groupby(group_col):
        daily=g.groupby("date")["consumption_kwh"].sum()
        if len(daily)<7: continue
        night=float(g.loc[g.hour.isin([0,1,2,3,4,5,22,23]),"consumption_kwh"].sum())
        total=float(g["consumption_kwh"].sum()) or 1.0
        weekend=float(g.loc[g.is_weekend,"consumption_kwh"].sum())
        weekday=float(g.loc[~g.is_weekend,"consumption_kwh"].sum())
        rows.append({"name":str(name),"mean_daily_kwh":float(daily.mean()),"std_daily_kwh":float(daily.std(ddof=0)),"peak_daily_kwh":float(daily.max()),"night_share":night/total,"weekend_ratio":weekend/(weekday or 1.0)})
    if len(rows)<3:
        return {"available":False,"message":"Need at least 3 meters/buildings with 7+ days of data for behaviour clustering.","clusters":[],"members":[]}
    feat=pd.DataFrame(rows)
    X=StandardScaler().fit_transform(feat[FEATURES])
    max_k=min(5,len(feat)-1)
    scores=[]
    for k in range(2,max_k+1):
        labels=KMeans(n_clusters=k,n_init=20,random_state=42).fit_predict(X)
        if len(set(labels))>1:
            scores.append((k,float(silhouette_score(X,labels))))
    if not scores:
        return {"available":False,"message":"Meters/buildings are too similar to each other to form distinct behaviour clusters with this data.","clusters":[],"members":[]}
    best_k=max(scores,key=lambda x:x[1])[0]
    final=KMeans(n_clusters=best_k,n_init=20,random_state=42).fit(X)
    feat["cluster"]=final.labels_
    profiles=[]
    for c,g in feat.groupby("cluster"):
        mean=g[FEATURES].mean()
        if mean["night_share"]>.30: label="Night-heavy"
        elif mean["weekend_ratio"]>1.0: label="Weekend-heavy"
        elif mean["std_daily_kwh"]/(mean["mean_daily_kwh"] or 1)>.50: label="Irregular"
        elif mean["mean_daily_kwh"]>=feat["mean_daily_kwh"].median(): label="Higher-demand"
        else: label="Stable / lower-demand"
        profiles.append({"cluster":int(c),"label":label,"members":int(len(g)),"mean_daily_kwh":round(float(mean["mean_daily_kwh"]),3),"night_share_pct":round(float(mean["night_share"]*100),1),"weekend_ratio":round(float(mean["weekend_ratio"]),2)})
    members=[{"name":r["name"],"cluster":int(r["cluster"])} for _,r in feat.iterrows()]
    return {"available":True,"group_by":group_col,"algorithm":"K-Means + StandardScaler","selected_k":best_k,"silhouette_scores":[{"k":k,"score":round(score,3)} for k,score in scores],"clusters":sorted(profiles,key=lambda x:x["cluster"]),"members":members,"features":FEATURES}
