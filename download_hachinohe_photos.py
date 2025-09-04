#!/usr/bin/env python3
"""
八戸市の東日本大震災後の航空写真をダウンロードするスクリプト
国土地理院の災害情報サイトから航空写真を取得
"""

import os
import requests
from urllib.parse import urljoin
import time
from typing import List
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HachinohePhotoDownloader:
    def __init__(self):
        self.base_url = "http://saigai.gsi.go.jp/h23taiheiyo-ok/ortho/hachinohe/Ortho_JPEG/"
        self.save_dir = "assets/hatinohe"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # 八戸地区のエリアコード（WebページからHF**とIF**のパターンを生成）
        self.area_codes = self._generate_area_codes()
    
    def _generate_area_codes(self) -> List[str]:
        """エリアコードのリストを生成"""
        codes = []
        
        # HF700番台（HF701-HF799）
        for i in range(701, 800):
            codes.append(f"HF{i}")
            
        # IF000番台（IF001-IF099）  
        for i in range(1, 100):
            codes.append(f"IF{i:03d}")
            
        return codes
    
    def _create_directories(self):
        """必要なディレクトリを作成"""
        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(f"保存ディレクトリ: {self.save_dir}")
    
    def _download_image(self, area_code: str) -> bool:
        """
        指定されたエリアコードの画像をダウンロード
        
        Args:
            area_code: エリアコード（例: HF713, IF011）
            
        Returns:
            bool: ダウンロード成功時True
        """
        filename = f"20110907-10-{area_code}-a2.jpg"
        url = urljoin(self.base_url, filename)
        save_path = os.path.join(self.save_dir, filename)
        
        # 既にファイルが存在する場合はスキップ
        if os.path.exists(save_path):
            logger.info(f"スキップ（既存）: {filename}")
            return True
            
        try:
            logger.info(f"ダウンロード中: {filename}")
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"保存完了: {filename} ({len(response.content)} bytes)")
                return True
            elif response.status_code == 404:
                logger.debug(f"ファイル不存在: {filename}")
                return False
            else:
                logger.warning(f"ダウンロード失敗: {filename} (HTTP {response.status_code})")
                return False
                
        except Exception as e:
            logger.error(f"エラー: {filename} - {str(e)}")
            return False
    
    def download_all_photos(self):
        """すべての航空写真をダウンロード"""
        self._create_directories()
        
        success_count = 0
        total_count = len(self.area_codes)
        
        logger.info(f"ダウンロード開始: {total_count}個のエリアコードをチェック")
        
        for i, area_code in enumerate(self.area_codes, 1):
            logger.info(f"進捗: {i}/{total_count} - {area_code}")
            
            if self._download_image(area_code):
                success_count += 1
            
            # サーバー負荷軽減のため少し待機
            time.sleep(0.5)
        
        logger.info(f"ダウンロード完了: {success_count}個のファイルを取得")
    
    def download_sample_photos(self, sample_codes: List[str] = None):
        """サンプル画像をダウンロード（テスト用）"""
        if sample_codes is None:
            sample_codes = ["HF713", "HF714", "IF011", "IF012"]
        
        self._create_directories()
        
        logger.info(f"サンプルダウンロード開始: {len(sample_codes)}個")
        
        success_count = 0
        for code in sample_codes:
            if self._download_image(code):
                success_count += 1
        
        logger.info(f"サンプルダウンロード完了: {success_count}個のファイルを取得")

def main():
    """メイン関数"""
    downloader = HachinohePhotoDownloader()
    
    # まずサンプルをダウンロードしてテスト
    print("=== サンプル航空写真のダウンロード ===")
    downloader.download_sample_photos()
    
    # ユーザーに全件ダウンロードの確認
    print("\n=== 全航空写真のダウンロード ===")
    print("注意: 全ファイルのダウンロードには時間がかかる場合があります。")
    
    while True:
        choice = input("すべての航空写真をダウンロードしますか？ (y/n): ").lower().strip()
        if choice in ['y', 'yes', 'はい']:
            downloader.download_all_photos()
            break
        elif choice in ['n', 'no', 'いいえ']:
            print("サンプルのみダウンロードしました。")
            break
        else:
            print("y または n を入力してください。")

if __name__ == "__main__":
    main()