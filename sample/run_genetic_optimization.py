#!/usr/bin/env python3
"""
八戸航空写真を使った遺伝的アルゴリズムによる道路検出最適化
RGB色空間の最適パラメータを進化的計算で発見
"""

import os
import sys
from genetic_road_detector import GeneticRoadDetector, RoadColorGene
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 八戸航空写真のパス
    hachinohe_images = [
        "/root/9_4_research/Grounded-Segment-Anything/assets/hatinohe/20110907-10-HF713-a2.jpg",
        "/root/9_4_research/Grounded-Segment-Anything/assets/hatinohe/20110907-10-HF714-a2.jpg", 
        "/root/9_4_research/Grounded-Segment-Anything/assets/hatinohe/20110907-10-HF723-a2.jpg",
        "/root/9_4_research/Grounded-Segment-Anything/assets/hatinohe/20110907-10-HF724-a2.jpg",
        "/root/9_4_research/Grounded-Segment-Anything/assets/hatinohe/20110907-10-IF011-a2.jpg"
    ]
    
    # 実際に存在する画像のみを選択
    existing_images = []
    for img_path in hachinohe_images:
        if os.path.exists(img_path):
            existing_images.append(img_path)
            logger.info(f"使用する画像: {img_path}")
        else:
            logger.warning(f"画像が見つかりません: {img_path}")
    
    if not existing_images:
        logger.error("使用できる画像が見つかりません")
        return False
    
    logger.info(f"合計 {len(existing_images)} 枚の八戸航空写真で道路検出を最適化します")
    
    try:
        # 遺伝的アルゴリズムによる最適化実行
        detector = GeneticRoadDetector(population_size=20, generations=15)
        
        logger.info("遺伝的アルゴリズムによる道路検出パラメータ最適化を開始...")
        best_gene = detector.evolve(existing_images)
        
        if best_gene:
            # 結果保存
            detector.save_best_gene("hachinohe_best_road_gene.json")
            
            # 各画像で検出結果を可視化
            logger.info("最適化されたパラメータでの検出結果を生成中...")
            for i, image_path in enumerate(existing_images):
                output_path = f"hachinohe_genetic_result_{i+1}.png"
                detector.visualize_detection_result(image_path, best_gene, output_path)
                logger.info(f"検出結果を保存: {output_path}")
            
            # 最良遺伝子の詳細を表示
            print("\n" + "="*60)
            print("遺伝的アルゴリズム最適化結果")
            print("="*60)
            print(f"最終適応度スコア: {detector.best_fitness:.4f}")
            print("\n【最適RGB色範囲】")
            print(f"  赤 (R): {best_gene.r_min} - {best_gene.r_max}")
            print(f"  緑 (G): {best_gene.g_min} - {best_gene.g_max}")
            print(f"  青 (B): {best_gene.b_min} - {best_gene.b_max}")
            
            print("\n【最適HSV色範囲】")
            print(f"  色相 (H): {best_gene.h_min} - {best_gene.h_max}")
            print(f"  彩度 (S): {best_gene.s_min} - {best_gene.s_max}")
            print(f"  明度 (V): {best_gene.v_min} - {best_gene.v_max}")
            
            print("\n【形態学的処理パラメータ】")
            print(f"  侵食カーネルサイズ: {best_gene.erosion_kernel}")
            print(f"  膨張カーネルサイズ: {best_gene.dilation_kernel}")
            
            print("\n【フィルタリングパラメータ】")
            print(f"  最小面積: {best_gene.min_area}")
            print(f"  アスペクト比閾値: {best_gene.aspect_ratio_threshold:.3f}")
            
            print("\n【保存ファイル】")
            print(f"  最良遺伝子: hachinohe_best_road_gene.json")
            print(f"  可視化結果: hachinohe_genetic_result_*.png")
            print("="*60)
            
            return True
        else:
            logger.error("最適化に失敗しました")
            return False
            
    except Exception as e:
        logger.error(f"最適化中にエラーが発生: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 遺伝的アルゴリズムによる道路検出最適化が完了しました！")
    else:
        print("\n❌ 最適化に失敗しました")
        sys.exit(1)
