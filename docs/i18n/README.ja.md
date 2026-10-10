<div align="center">

<img src="../assets/trace-logo.svg" alt="TRACE: trajectory-routed causal memory" width="440">

## TRACE: Trajectory-Routed Causal Memory for Delayed-Evidence Visuomotor Imitation

**CoRL 2026 · 本会議採択**

**Zihao Li<sup>1,2</sup> · Ranpeng Qiu<sup>2</sup> · Yincong Chen<sup>2</sup> · Guoqiang Ren<sup>2</sup> · Weiming Zhi<sup>1</sup>**

<sup>1</sup> **The University of Sydney** &emsp; <sup>2</sup> **Zeno AI**

[![CI](https://github.com/Jeong-zju/corl-trace/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/Jeong-zju/corl-trace/actions/workflows/ci.yml)
[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b.svg)](https://jeong-zju.github.io/trace/assets/paper/trace-paper.pdf)
[![Project](https://img.shields.io/badge/Project-Webpage-4b7bec.svg)](https://jeong-zju.github.io/trace/)
[![Code](https://img.shields.io/badge/Code-TRACE-2b9348.svg)](https://github.com/Jeong-zju/corl-trace)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](../../LICENSE)

📄 [論文](https://jeong-zju.github.io/trace/assets/paper/trace-paper.pdf) |
💻 [コード](https://github.com/Jeong-zju/corl-trace) |
🌐 [プロジェクト](https://jeong-zju.github.io/trace/) |
📚 [ドキュメント](../installation.md)

[English](../../README.md) · [简体中文](README.zh-CN.md) · **日本語** · [한국어](README.ko.md)

---

<img src="../assets/teaser.png" alt="TRACE retains early evidence to resolve later ambiguous decisions" width="100%">

</div>

## ✨ 更新情報

- **[2026/10/08]** TRACE ACT／Diffusion の主手法コードと、英語・中国語・日本語・韓国語のドキュメントを公開しました。

---

## 🔍 研究の動機

**手がかりを覚え、必要なときに行動する。** TRACE は、後の判断に必要な手がかりが
先に見えなくなるタスクに対して、固定サイズの因果的メモリを視覚運動ポリシーに追加します。
経路シグネチャで実行済みのロボット状態軌跡を要約し、視覚・状態情報の書き込みと
読み出しを振り分けます。軽量なアダプターで、このメモリを ACT と Diffusion Policy に接続します。

## ⚙️ 主な特徴

| 構成要素 | 役割 |
| --- | --- |
| **軌跡によるルーティング** | 経路シグネチャと差分シグネチャが実行済みの状態軌跡を要約し、メモリの読み書きを振り分けます。 |
| **因果的なスロットメモリ** | 固定数の潜在スロットが視覚・状態の情報を保持し、初期の手がかりが見えなくなった後の判断に利用します。 |
| **2つのポリシーインターフェース** | 軽量アダプターが ACT と Diffusion Policy の既存の条件入力にメモリを接続します。 |
| **実行可能なソースコード** | データツール、参考設定、CPU 実行例、主要機能のテストを提供します。 |

---

## 📚 目次

- [公開内容](#release-contents)
- [インストールとクイックスタート](#installation)
- [学習](#training)
- [評価](#evaluation)
- [コード構成](#code-structure)
- [開発](#development)
- [引用](#citation)
- [ライセンスと謝辞](#license)

---

<a id="release-contents"></a>

## 📦 公開内容

- 回帰型（ACT）と拡散型のインターフェースを備えた **TRACE の完全な主手法**。
- 因果的なプレフィックスデータセット、シグネチャキャッシュ、学習、オフライン評価・シミュレーション評価。
- 合成観測と実際の経路シグネチャを使い、事前学習済み重みをダウンロードしない CPU 実行例。
- 参考設定、主要機能のテスト、4 言語のドキュメント。

このリリースは主手法のみを含みます。アブレーション実装、比較実験、非公開のロボット
配備設定は含まれません。データセット、学習済み重み、論文の各実験に対応する完全な
成果物は **同梱されていません**。参考設定は利用開始のためのもので、論文の実験結果を
厳密に再現できることを保証するものではありません。[公開範囲](../../docs/release-scope.md)を参照してください。

---

<a id="installation"></a>

## 🚀 インストールとクイックスタート

参照環境は Linux と Python 3.12 です。リポジトリをクローンし、
ルートディレクトリで以下のコマンドを実行してください。Signatory のビルド前に
システムの FFmpeg と C++ コンパイラーをインストールしてください。
GPU 学習にはドライバーと互換性のある PyTorch / CUDA 環境が必要です。

```bash
git clone https://github.com/Jeong-zju/corl-trace.git
cd corl-trace

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e depends/signatory --no-build-isolation
python examples/smoke_trace.py
```

この例では、ランダムに初期化した小さな CPU モデルを使用します。メモリを経由する
一連の計算が有限値の行動を出力することを確認しますが、タスク成功率は測定しません。
環境構築とビルドの詳細は[インストール手順](../../docs/installation.md)を参照してください。

---

<a id="training"></a>

## 🏋️ 学習

同期した RGB 観測、ロボット状態、行動を含むローカルの LeRobot v3 データセットを用意します。
`data/local/task` をデータセットのディレクトリに置き換えてください。前処理では別の
出力データセットを作成するため、入力データセットも引き続き利用できます。

```bash
python data/process_dataset.py data/local/task \
  --output-dir data/local/task_trace --signature-depth 3

python scripts/train_policy.py --policy streaming_act \
  --defaults-path configs/trace_act.yaml --dataset data/local/task_trace

python scripts/train_policy.py --policy prism_diffusion \
  --defaults-path configs/trace_diffusion.yaml --dataset data/local/task_trace
```

両方の設定は TRACE の完全な計算経路を使用し、乱数シードを固定しています。
実験トラッキングはデフォルトで無効です。データ分割、正規化、チェックポイントの詳細は
[データの準備](../../docs/data.md)と[学習](../../docs/training.md)を参照してください。

---

<a id="evaluation"></a>

## 🧪 評価

```bash
python scripts/eval_policy.py --policy streaming_act \
  --policy-path outputs/train/trace-act/RUN/checkpoints/STEP/pretrained_model \
  --dataset data/local/task_trace
```

チェックポイントのパスを実際の保存先に置き換えてください。オフラインの行動予測誤差は、
ロボットのタスク成功率ではありません。シミュレーションには対応する環境とアセットが
必要です。[評価](../../docs/evaluation.md)を参照してください。

---

<a id="code-structure"></a>

## 🗂️ コード構成

| パス | 用途 |
| --- | --- |
| `policy/lerobot_policy_streaming_act/src/` | TRACE メモリと ACT インターフェース |
| `policy/lerobot_policy_prism_diffusion/src/` | TRACE メモリと Diffusion Policy インターフェース |
| `configs/` | 公開する主手法の学習設定 |
| `scripts/` | 学習、評価、データセット用ユーティリティ |
| `data/` | LeRobot 形式への変換と経路シグネチャの前処理 |
| `examples/` | 実行可能な CPU 使用例 |
| `tests/` | 主手法、データ、リリースの検証 |
| `docs/` | インストール、手法、学習、評価、翻訳 |
| `depends/signatory/` | 同梱したシグネチャ実装と上流のライセンス |

`streaming_act` と `prism_diffusion` は、2 つの TRACE インターフェースで従来から
シリアライズに使われているポリシー名です。主手法のチェックポイントのパラメーター名を
維持するため、そのまま使用しています。対応関係と互換性の要件は
[アーキテクチャ](../../docs/architecture.md)を参照してください。

---

<a id="development"></a>

## 🛠️ 開発

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python scripts/check_release.py
python -m pytest
```

モデル関連の依存関係を入れずに小さな CPU ユーティリティテストを実行するには、
`requirements-test.txt` をインストールして `python -m pytest --lightweight` を実行します。
開発手順は [CONTRIBUTING.md](../../CONTRIBUTING.md) を参照してください。

---

<a id="citation"></a>

## 📄 引用

```bibtex
@inproceedings{li2026trace,
  title     = {TRACE: Trajectory-Routed Causal Memory for Delayed-Evidence Visuomotor Imitation},
  author    = {Li, Zihao and Qiu, Ranpeng and Chen, Yincong and Ren, Guoqiang and Zhi, Weiming},
  booktitle = {Conference on Robot Learning},
  year      = {2026},
  url       = {https://jeong-zju.github.io/trace/}
}
```

---

<a id="license"></a>

## 📜 ライセンスと謝辞

TRACE 独自の貢献部分は [Apache-2.0](../../LICENSE) ライセンスで公開されています。
本プロジェクトは LeRobot、ACT、Diffusion Policy、Signatory を基盤としています。
各プロジェクトの告知とライセンスは [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)
に記載しています。外部のデータと重みには、それぞれのライセンスが適用されます。

---

<div align="center">

**TRACE** · 手がかりを覚え、必要なときに行動する。

</div>
