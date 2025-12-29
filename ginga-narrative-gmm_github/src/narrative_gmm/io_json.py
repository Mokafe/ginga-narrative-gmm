from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

def load_events_df(json_path: str | Path) -> pd.DataFrame:
    """
    Load narrative JSON (chapters -> time_series_data) into a flat DataFrame.
    Keeps text fields (event/evidence) so ML outputs can be mapped back to the story.

    Returns columns:
      chapter_title, scene_id, global_step, local_step, local_time, global_time,
      event, desc, evidence_1, evidence_2, m, iso, t, status, label
    """
    json_path = Path(json_path)
    chapters = json.loads(json_path.read_text(encoding="utf-8"))

    rows = []
    for ch in chapters:
        title = ch["chapter"]["title"]
        scene_id = ch["chapter"]["scene_id"]
        for p in ch["time_series_data"]:
            x = p["x"]
            status = p.get("status", "unlabeled")
            label = p.get("label", -1) if status == "labeled" else -1
            ev = p.get("evidence", []) or []
            rows.append({
                "chapter_title": title,
                "scene_id": scene_id,
                "global_step": int(p["global_step"]),
                "local_step": int(p["local_step"]),
                "local_time": float(p.get("local_time", np.nan)) if p.get("local_time") is not None else np.nan,
                "event": p.get("event", ""),
                "desc": p.get("desc", ""),
                "evidence_1": ev[0] if len(ev) > 0 else "",
                "evidence_2": ev[1] if len(ev) > 1 else "",
                "m": float(x[0]),
                "iso": float(x[1]),
                "t": float(x[2]) if len(x) > 2 else 0.0,
                "status": status,
                "label": int(label),
            })

    df = pd.DataFrame(rows).sort_values("global_step").reset_index(drop=True)
    gmin, gmax = df["global_step"].min(), df["global_step"].max()
    df["global_time"] = 0.0 if gmax == gmin else (df["global_step"] - gmin) / (gmax - gmin)
    return df
