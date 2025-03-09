import os
import csv
import base64
from pathlib import Path
import requests
import json
import time
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures

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

# バッチ処理の設定
BATCH_SIZE = 10  # 一度に処理する画像の数
MAX_RETRIES = 3  # API呼び出しの最大リトライ回数
RETRY_DELAY = 2  # リトライ間の待機時間（秒）

def encode_image(image_path):
    """
    画像をBase64エンコードする
    
    Args:
        image_path (str): 画像ファイルのパス
        
    Returns:
        str: Base64エンコードされた画像データ
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def batch_extract_info_from_images(image_paths):
    """
    バッチ処理：複数の画像を一度にAPIに送信して情報を抽出する
    
    Args:
        image_paths (list): 画像ファイルパスのリスト
        
    Returns:
        list: 各画像から抽出された情報のリスト
    """
    # OpenAI API用のヘッダー
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    # バッチ処理用のリクエストを準備
    batch_requests = []
    
    # 各画像をBase64エンコードしてバッチリクエストを作成
    for image_path in image_paths:
        base64_image = encode_image(image_path)
        
        # 個別のリクエスト情報を作成
        request_data = {
            "model": "gpt-4o-2024-11-20",
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
        batch_requests.append(request_data)
    
    # 並列処理でAPIリクエストを送信
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(BATCH_SIZE, 5)) as executor:
        # 各リクエストを並列に実行
        future_to_path = {
            executor.submit(send_api_request, request, headers, i): image_paths[i] 
            for i, request in enumerate(batch_requests)
        }
        
        # 結果を取得
        for future in concurrent.futures.as_completed(future_to_path):
            image_path = future_to_path[future]
            try:
                response_data = future.result()
                
                # レスポンスから情報を抽出
                parsed_data = parse_api_response(response_data, image_path)
                results.append(parsed_data)
            except Exception as e:
                print(f"エラー発生 ({image_path}): {e}")
                # エラーの場合はデフォルト値を設定
                results.append({
                    "filename": Path(image_path).name,
                    "patient_name": "エラー",
                    "hospital_name": "エラー",
                    "amount": "エラー"
                })
    
    return results

def send_api_request(request_data, headers, request_index):
    """
    APIリクエストを送信し、リトライロジックを実装
    
    Args:
        request_data (dict): APIリクエストデータ
        headers (dict): APIリクエストヘッダー
        request_index (int): リクエストのインデックス（ログ用）
        
    Returns:
        dict: APIレスポンス
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=request_data,
                timeout=60  # タイムアウト設定
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            # リトライ可能なエラーの場合
            print(f"リクエスト {request_index} 失敗 (試行 {attempt+1}/{MAX_RETRIES}): {e}")
            
            # 最後の試行でなければリトライ
            if attempt < MAX_RETRIES - 1:
                # レート制限エラーの場合は長めに待機
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', RETRY_DELAY * 2))
                    print(f"レート制限に達しました。{retry_after}秒待機します...")
                    time.sleep(retry_after)
                else:
                    # 通常のリトライ
                    time.sleep(RETRY_DELAY)
            else:
                # 最大リトライ回数に達した場合は例外を発生
                raise Exception(f"APIリクエスト失敗 (最大リトライ回数到達): {e}")
    
    # ここには到達しないはずだが、念のため
    raise Exception("APIリクエスト失敗")

def parse_api_response(result, image_path):
    """
    APIレスポンスを解析して必要な情報を抽出する
    
    Args:
        result (dict): APIレスポンス
        image_path (str): 対応する画像ファイルのパス
        
    Returns:
        dict: 抽出された情報
    """
    try:
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
                "filename": Path(image_path).name,
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
                "filename": Path(image_path).name,
                "patient_name": normalize_name(patient_name),
                "hospital_name": normalize_name(hospital_name),
                "amount": amount
            }
            
    except Exception as e:
        print(f"レスポンス解析エラー: {e}")
        return {
            "filename": Path(image_path).name,
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
    指定フォルダ内の全ての医療費領収書画像をバッチ処理し、
    医療機関と受診者名でまとめた結果をCSVファイルに出力する
    
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
    
    # バッチ処理のためのチャンク分割
    for i in range(0, len(image_files), BATCH_SIZE):
        batch = image_files[i:i+BATCH_SIZE]
        batch_paths = [str(path) for path in batch]
        
        print(f"バッチ処理中: {i+1}〜{min(i+BATCH_SIZE, len(image_files))} / {len(image_files)}")
        
        # バッチ処理で情報を抽出
        batch_results = batch_extract_info_from_images(batch_paths)
        
        # 結果を追加
        individual_results.extend(batch_results)
    
    # 金額を数値に変換
    for result in individual_results:
        try:
            # カンマや円記号、スペースを削除して数値化
            amount_str = str(result["amount"]).replace(",", "").replace("円", "").replace(" ", "").strip()
            result["amount"] = int(amount_str)
        except (ValueError, TypeError):
            # 数値に変換できない場合は元の値を使用
            pass
    
    # 医療機関名と受診者名の組み合わせでデータをグループ化
    combined_data = defaultdict(list)
    
    for result in individual_results:
        # キーとして「医療機関名_受診者名」を使用
        key = f"{result['hospital_name']}_{result['patient_name']}"
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
    parser.add_argument("--batch-size", "-b", type=int, default=10,
                        help="一度に処理する画像の数（デフォルト: 10）")
    
    args = parser.parse_args()
    
    # バッチサイズを設定
    global BATCH_SIZE
    BATCH_SIZE = args.batch_size
    
    # 処理の実行
    process_receipts_in_folder(args.folder_path, args.output)

if __name__ == "__main__":
    main()