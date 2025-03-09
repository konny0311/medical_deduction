# 医療費領収書画像処理ツール

医療費領収書の画像からテキスト情報を抽出し、医療費控除申請のためのデータを整理するPythonスクリプト。

## 機能

- 指定フォルダ内の医療費領収書画像を読み込む
- 画像から以下の情報を抽出：
  - 患者氏名
  - 医療機関名
  - 支払金額
- 同一医療機関・同一患者の領収書を自動的にグループ化
- 合計金額の計算
- CSV形式での結果出力

## ライブラリインストール

```bash
uv sync
```

## 環境変数設定

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## 使い方
単一リクエスト推論
```bash
uv run summarize_medical_expense.py <image_folder_path>
```

バッチ推論
```bash
uv run summarize_medical_expense_batch.py <image_folder_path>
```

## 出力ファイル
1. **メインCSVファイル** (デフォルト: `medical_receipts_data.csv`)
   - 医療機関別・患者別にグループ化されたデータ
   - 各グループの合計金額
   - 処理された領収書の数
   - 関連するファイル名のリスト

2. **詳細CSVファイル** (`medical_receipts_data_detail.csv`)
   - 各領収書の個別データ
   - デバッグや詳細確認用

## 使用モデル

ChatGPT: `gpt-4o-2024-11-20`
