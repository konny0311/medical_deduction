import os
import csv
import base64
from pathlib import Path
import requests
import json
from tqdm import tqdm
from collections import defaultdict

API_KEY = os.getenv("OPENAI_API_KEY")

PROMPT = """
この医療費領収書から以下の情報を抽出してください。
1.患者氏名（正確に抽出してください。周囲に「氏名」や「様」と記載されている場合が多いです。）
2.医療機関名（正式名称を抽出してください。1枚の領収書に薬局と病院の名前が印刷されている場合、薬局の名前を抽出してください。）
3.支払った医療費の金額（数字のみ）

### 出力フォーマット:
```json
{
  "患者氏名": "値1",
  "医療機関名": "値2",
  "支払った医療費の金額": "値3",
}

**[重要事項]**
- **このフォーマット以外の出力を禁止します**。
- JSONのフィールドは必ず指定された形式で出力してください。
"""

# マルチモーダルLLM APIを使って画像から情報を抽出する関数
def extract_info_from_image(image_path):
    """
    マルチモーダルLLM APIを使用して、医療費領収書の画像から必要な情報を抽出する
    
    Args:
        image_path (str): 画像ファイルのパス
        
    Returns:
        dict: 抽出された情報（患者名、医療機関名、支払金額）
    """
    # 画像をBase64エンコード
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    # OpenAI Vision API用のリクエスト設定（例としてGPT-4 Visionを使用）
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    # プロンプトの作成（日本語で明確な指示を与える）
    payload = {
        "model": "gpt-4o-2024-11-20",
        # "model": "gpt-4o-mini", # 4o-miniでもそこそこ正しく読み取れている
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 500
    }
    
    # APIリクエストの送信
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()  # エラーチェック
        
        # レスポンスから情報を抽出
        result = response.json()
        # print("RESULT:", result)
        content = result["choices"][0]["message"]["content"]
        
        # JSONレスポンスを解析
        try:
            # JSONの部分を探す
            import re
            json_match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
            if json_match:
                json_content = json_match.group(1)
            else:
                json_content = content
                
            data = json.loads(json_content)
            return {
                "patient_name": normalize_name(data.get("患者氏名", "不明")),
                "hospital_name": normalize_name(data.get("医療機関名", "不明")),
                "amount": data.get("支払った医療費の金額", "不明")
            }
        except json.JSONDecodeError:
            # JSON解析に失敗した場合は、テキスト解析を試みる
            print(f"JSON解析エラー: {content}")
            patient_name = "不明"
            hospital_name = "不明"
            amount = "不明"
            
            # テキスト内から情報を検索
            if "患者氏名:" in content:
                patient_name = content.split("患者氏名:")[1].split("\n")[0].strip()
            if "医療機関名:" in content:
                hospital_name = content.split("医療機関名:")[1].split("\n")[0].strip()
            if "支払った医療費の金額:" in content:
                amount = content.split("支払った医療費の金額:")[1].split("\n")[0].strip()
                
            return {
                "patient_name": normalize_name(patient_name),
                "hospital_name": normalize_name(hospital_name),
                "amount": amount
            }
            
    except requests.exceptions.RequestException as e:
        print(f"APIリクエストエラー: {e}")
        return {
            "patient_name": "エラー",
            "hospital_name": "エラー",
            "amount": "エラー"
        }

def normalize_name(name):
    """
    名前や医療機関名を正規化する関数（重複を避けるため）
    
    Args:
        name (str): 正規化する名前
        
    Returns:
        str: 正規化された名前
    """
    if not name:
        return "不明"
    
    # 基本的な正規化：全角スペース、半角スペース、記号などを削除
    normalized = name.replace("　", "").replace(" ", "").strip()
    
    # さん、様などの敬称を削除
    honorifics = ["さん", "様", "殿", "氏", "先生"]
    for honorific in honorifics:
        if normalized.endswith(honorific):
            normalized = normalized[:-len(honorific)]
    
    return normalized

def process_receipts_in_folder(folder_path, output_csv_path):
    """
    指定フォルダ内の全ての医療費領収書画像を処理し、医療機関と受診者名でまとめた結果をCSVファイルに出力する
    
    Args:
        folder_path (str): 医療費領収書画像が格納されているフォルダのパス
        output_csv_path (str): 出力先CSVファイルのパス
    """
    # 画像ファイルの拡張子リスト
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]
    
    # フォルダ内の画像ファイルを取得
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(folder_path).glob(f"*{ext}")))
    
    print(f"合計 {len(image_files)} 件の画像ファイルが見つかりました。")
    
    # 結果格納用リスト
    individual_results = []
    
    # 各画像を処理
    for image_path in tqdm(image_files, desc="画像処理中"):
        print(f"処理中: {image_path.name}")
        
        # 画像から情報を抽出
        info = extract_info_from_image(str(image_path))
        
        # 金額を数値に変換（可能な場合）
        try:
            # カンマや円記号、スペースを削除して数値化
            amount_str = str(info["amount"]).replace(",", "").replace("円", "").replace(" ", "").strip()
            amount_value = int(amount_str)
        except (ValueError, TypeError):
            # 数値に変換できない場合は元の値を使用
            amount_value = info["amount"]
        
        # ファイル名を含む結果を追加
        individual_results.append({
            "filename": image_path.name,
            "patient_name": info["patient_name"],
            "hospital_name": info["hospital_name"],
            "amount": amount_value
        })
    
    # 医療機関名と受診者名の組み合わせでデータをグループ化
    combined_data = defaultdict(list)
    
    for result in individual_results:
        # キーとして「医療機関名_受診者名」を使用
        key = f"{result["hospital_name"]}_{result["patient_name"]}"
        combined_data[key].append(result)
    
    # グループごとにまとめた結果を生成
    consolidated_results = []
    
    for key, receipts in combined_data.items():
        # キーから医療機関名と受診者名を取得
        hospital_name, patient_name = key.split("_", 1)
        
        # 金額の合計を計算
        total_amount = 0
        receipts_with_valid_amount = 0
        
        for receipt in receipts:
            if isinstance(receipt["amount"], (int, float)):
                total_amount += receipt["amount"]
                receipts_with_valid_amount += 1
        
        # 日付別領収書ファイル一覧を作成
        filenames = [r["filename"] for r in receipts]
        
        # 結果に追加
        consolidated_results.append({
            "hospital_name": hospital_name,
            "patient_name": patient_name,
            "medical_cure": "該当する",
            "medicine": "",
            "support": "",
            "others": "",
            "total_amount": total_amount,
            "receipt_count": len(receipts),
            "receipts_with_amount": receipts_with_valid_amount,
            "filenames": ", ".join(filenames)
        })
    
    # 結果をCSVファイルに保存（医療機関と受診者名でまとめたバージョン）
    with open(output_csv_path, "w", newline="", encoding="utf-8-sig") as csvfile:
        fieldnames = ["hospital_name", "patient_name", "medical_cure", "medicine", "support", "others", "total_amount", "receipt_count", "receipts_with_amount", "filenames"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in consolidated_results:
            writer.writerow(result)
    
    # 元の詳細データも保存（デバッグや詳細確認用）
    detail_csv_path = output_csv_path.replace(".csv", "_detail.csv")
    with open(detail_csv_path, "w", newline="", encoding="utf-8-sig") as csvfile:
        fieldnames = ["filename", "patient_name", "hospital_name", "amount"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in individual_results:
            writer.writerow(result)
    
    print(f"処理が完了しました。")
    print(f"集計結果: {output_csv_path}")
    print(f"詳細データ: {detail_csv_path}")

def main():
    """
    メイン関数：コマンドライン引数を処理し、処理を実行する
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="医療費領収書画像から情報を抽出してCSVに出力するプログラム")
    parser.add_argument("folder_path", help="医療費領収書画像が格納されているフォルダのパス")
    parser.add_argument("--output", "-o", default="medical_receipts_data.csv", 
                        help="出力CSVファイルのパス（デフォルト: medical_receipts_data.csv）")
    
    args = parser.parse_args()
    
    # 処理の実行
    process_receipts_in_folder(args.folder_path, args.output)

if __name__ == "__main__":
    main()