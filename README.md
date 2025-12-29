# Narrative GMM Turning Points（物語×不確実性×GMM）

「銀河鉄道の夜」を **イベント＋原文引用(evidence)** 付きの時系列JSONにし、  
2次元特徴 **(m=Morality, iso=Isolation)** を半教師ありGMM（EM）で解析して、

- 状態クラスタ（アンカーラベルで意味を固定）
- 境界点Top20（揺れ：entropy高 / margin低）
- 章ごとの転調密度（turning density）
- クラスタ遷移回数（状態切替の頻度）
- 遷移した瞬間のイベント列（本文への還元）

を **図＋CSV** として出力します。

---

## Quickstart（Colab）

1. GitHubからこのrepoをColabに開く  
2. `notebooks/00_narrative_gmm_turning_points.ipynb` を上から実行

（Colabバッジは、アップ後にURLを書き換えてください）
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<USER>/<REPO>/blob/main/notebooks/00_narrative_gmm_turning_points.ipynb)

---

## Quickstart（Local）

```bash
pip install -r requirements.txt
python scripts/run_report.py --json data/sample/sample_g_1_reconstructed_keep_last9.json --out outputs --plots
```

---

## Outputs（`outputs/`）

- `events_all.csv` : JSONをDF化した全イベント（引用・要約つき）
- `preds_all.csv` : 各点のクラスタ/確率/entropy/margin を付与
- `cluster_name_map.csv` : クラスタ0/1を自動命名した対応表（Communal vs Self_Protection）
- `boundary_top20.csv` : 境界点Top20（本文へ還元する「揺れ」候補）
- `boundary_context.csv` : 境界点の前後±wイベント（文脈）
- `scene_turning_density.csv` : 章ごとの転調密度（entropy合計など）
- `scene_cluster_transitions.csv` : 章ごとのクラスタ遷移回数
- `cluster_transition_points.csv` : 物語全体で遷移が起きた点の一覧
- `gmm_params.npz` : 学習済みGMMパラメータ

---

## Data（JSONの前提）

`data/sample/sample_g_1_reconstructed_keep_last9.json` は、章ごとに `time_series_data[]` を持ちます。

- `x = [m, iso, t]`
- `event`（要約）, `evidence`（原文引用）
- `status` : `labeled` / `unlabeled`
- `label` : 0 = Communal_Happiness, 1 = Self_Protection（アンカーのみ）

---

## Reproducibility（再現性）

- `SEED` を固定すると、初期化が同じになり **教材として再現可能** になります
- `N_INIT`（初期化リスタート回数）を増やすと、seedを外しても比較的安定します

---

## Notes

- `outputs/` は生成物なのでGit管理しません（`.gitkeep` のみ）
- Bishop(PRML) 等の教科書PDFは本repoに同梱していません（各自で入手してください）
