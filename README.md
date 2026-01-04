# Narrative GMM Turning Points（物語×不確実性×GMM）

「銀河鉄道の夜」を **イベント＋原文引用（evidence）** 付きの時系列JSONにし、  
2次元特徴 **(m = Morality, iso = Isolation)** を半教師ありGMM（EM）で解析して、

- 状態クラスタ（アンカーラベルで意味を固定）
- 境界点 Top20（揺れ：entropy 高 / margin 低）
- 章ごとの転調密度（turning density）
- クラスタ遷移回数（状態切替の頻度）
- 遷移した瞬間のイベント列（本文への還元）

を **図＋CSV** として出力します。

---

## Paper（論文PDF）：物語イベントの状態遷移解析：半教師あり GMM を用いた転調点抽出

-　——宮沢賢治『銀河鉄道の夜』を例に

- Paper (PDF): `docs/paper/paper.pdf`
- Step log（実行手順・ログ）: `docs/ginga_step_log_script.md`

---

## Quickstart（Colab）

1. GitHubからこのrepoをColabに開く  
2. `notebooks/00_narrative_gmm_turning_points.ipynb` を上から実行

（「Open in Colab」バッジをクリックすると、ブラウザ上で直接コードを実行できます。環境構築は不要です）

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mokafe/ginga-narrative-gmm/blob/main/notebooks/00_narrative_gmm_turning_points.ipynb)

---

## Quickstart（Local）

```bash
pip install -r requirements.txt
python scripts/run_report.py --json data/sample/sample_g_1_reconstructed_keep_last9.json --out outputs --plots
