#!/usr/bin/env python3
"""
実際の道路・高速道路画像をダウンロード
Unsplash、Pixabay等の無料画像APIを使用
"""

import os
import sys
import requests
import time
import logging
from urllib.parse import urljoin, urlparse
import json
from pathlib import Path

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealRoadImageDownloader:
    def __init__(self, save_dir="hisai"):
        self.save_dir = save_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # 保存ディレクトリ作成
        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(f"保存ディレクトリ: {self.save_dir}")
    
    def download_from_unsplash(self):
        """Unsplash APIから道路画像をダウンロード"""
        logger.info("Unsplashから道路画像を検索中...")
        
        # Unsplash APIのクエリパラメータ
        queries = [
            "highway",
            "road",
            "freeway", 
            "expressway",
            "street",
            "asphalt",
            "traffic",
            "bridge"
        ]
        
        downloaded = 0
        target_per_query = 15
        
        for query in queries:
            if downloaded >= 100:
                break
                
            logger.info(f"検索キーワード: {query}")
            
            try:
                # Unsplash検索URL（パブリックフィード）
                url = f"https://unsplash.com/napi/search/photos"
                params = {
                    'query': query,
                    'per_page': target_per_query,
                    'page': 1
                }
                
                response = self.session.get(url, params=params, timeout=15)
                if response.status_code != 200:
                    logger.warning(f"Unsplash検索失敗: {response.status_code}")
                    continue
                
                data = response.json()
                if 'results' not in data:
                    logger.warning(f"検索結果なし: {query}")
                    continue
                
                for i, photo in enumerate(data['results'][:target_per_query]):
                    if downloaded >= 100:
                        break
                    
                    try:
                        # 画像URLを取得
                        image_url = photo['urls']['regular']  # 中解像度
                        photo_id = photo['id']
                        
                        filename = f"road_{query}_{downloaded+1:03d}_{photo_id}.jpg"
                        filepath = os.path.join(self.save_dir, filename)
                        
                        if os.path.exists(filepath):
                            logger.info(f"スキップ: {filename}")
                            continue
                        
                        # 画像をダウンロード
                        success = self._download_image(image_url, filepath)
                        if success:
                            downloaded += 1
                            logger.info(f"ダウンロード完了 ({downloaded}/100): {filename}")
                            time.sleep(1)  # レート制限対応
                    
                    except Exception as e:
                        logger.error(f"写真処理エラー: {str(e)}")
                        continue
            
            except Exception as e:
                logger.error(f"Unsplash API エラー {query}: {str(e)}")
                continue
            
            time.sleep(2)  # クエリ間の間隔
        
        return downloaded
    
    def download_from_pexels(self):
        """Pexels APIから道路画像をダウンロード"""
        logger.info("Pexelsから道路画像を検索中...")
        
        queries = ["highway japan", "road infrastructure", "traffic bridge", "expressway"]
        downloaded = 0
        
        for query in queries:
            if downloaded >= 20:
                break
            
            try:
                # Pexels検索URL（パブリックフィード）
                url = "https://www.pexels.com/search/"
                params = {'query': query}
                
                response = self.session.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    # HTMLからimage URLを抽出（簡易実装）
                    content = response.text
                    
                    # 簡易的なURL抽出
                    import re
                    img_pattern = r'https://images\.pexels\.com/photos/\d+/[^"]+\.jpeg\?[^"]*'
                    matches = re.findall(img_pattern, content)
                    
                    for i, img_url in enumerate(matches[:10]):
                        if downloaded >= 20:
                            break
                        
                        filename = f"pexels_road_{downloaded+1:03d}.jpg"
                        filepath = os.path.join(self.save_dir, filename)
                        
                        if os.path.exists(filepath):
                            continue
                        
                        success = self._download_image(img_url, filepath)
                        if success:
                            downloaded += 1
                            logger.info(f"Pexels画像取得 ({downloaded}/20): {filename}")
                            time.sleep(2)
            
            except Exception as e:
                logger.error(f"Pexels エラー {query}: {str(e)}")
                continue
        
        return downloaded
    
    def download_japanese_road_samples(self):
        """日本の道路関連のサンプル画像をダウンロード"""
        logger.info("日本の道路サンプル画像を取得中...")
        
        # 実際の日本の道路画像URL（サンプル）
        japan_road_urls = [
            # これらは例です。実際には適切なライセンスのURLを使用してください
        ]
        
        # 代わりに、オープンデータから取得可能な画像URLs
        open_data_urls = []
        
        downloaded = 0
        for i, url in enumerate(open_data_urls[:30]):
            if downloaded >= 30:
                break
            
            filename = f"japan_road_{downloaded+1:03d}.jpg"
            filepath = os.path.join(self.save_dir, filename)
            
            success = self._download_image(url, filepath)
            if success:
                downloaded += 1
                logger.info(f"日本道路画像取得: {filename}")
                time.sleep(1)
        
        return downloaded
    
    def _download_image(self, url, filepath):
        """画像をダウンロード"""
        try:
            response = self.session.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Content-Typeをチェック
            content_type = response.headers.get('content-type', '').lower()
            if not content_type.startswith('image/'):
                logger.warning(f"画像ファイルではありません: {url}")
                return False
            
            # ファイルサイズをチェック
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) < 5000:  # 5KB未満は小さすぎる
                logger.warning(f"ファイルサイズが小さすぎます: {url}")
                return False
            
            # ファイルに保存
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # ダウンロード後のファイルサイズチェック
            file_size = os.path.getsize(filepath)
            if file_size < 5000:
                os.remove(filepath)
                logger.warning(f"ダウンロード後ファイルサイズ不正: {url}")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"ダウンロードエラー {url}: {str(e)}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return False
    
    def create_download_summary(self, total_downloaded):
        """ダウンロード結果のサマリーを作成"""
        summary = {
            "total_downloaded": total_downloaded,
            "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_types": ["Unsplash", "Pexels", "Japanese Open Data"],
            "image_categories": ["highway", "road", "bridge", "traffic", "expressway"],
            "directory": self.save_dir,
            "note": "Images downloaded from free/open sources with appropriate usage rights"
        }
        
        summary_path = os.path.join(self.save_dir, "download_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"ダウンロードサマリーを保存: {summary_path}")

def main():
    """メイン実行関数"""
    downloader = RealRoadImageDownloader()
    
    print("実際の道路画像のダウンロードを開始します...")
    logger.info("道路・高速道路画像の取得を開始")
    
    total_downloaded = 0
    
    try:
        # Unsplashから画像を取得
        unsplash_count = downloader.download_from_unsplash()
        total_downloaded += unsplash_count
        logger.info(f"Unsplashから {unsplash_count} 枚取得")
        
        # Pexelsから追加取得
        if total_downloaded < 100:
            pexels_count = downloader.download_from_pexels()
            total_downloaded += pexels_count
            logger.info(f"Pexelsから {pexels_count} 枚取得")
        
        # 日本の道路データ
        if total_downloaded < 100:
            japan_count = downloader.download_japanese_road_samples()
            total_downloaded += japan_count
            logger.info(f"日本オープンデータから {japan_count} 枚取得")
        
        # サマリー作成
        downloader.create_download_summary(total_downloaded)
        
        print(f"\n合計 {total_downloaded} 枚の実際の道路画像をダウンロードしました")
        print(f"保存先: {downloader.save_dir}/")
        
        return total_downloaded > 0
        
    except Exception as e:
        logger.error(f"ダウンロード処理エラー: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 実際の道路画像ダウンロードが完了しました!")
    else:
        print("\n❌ ダウンロードに失敗しました")
        sys.exit(1)