#!/usr/bin/env python3
"""
遺伝的アルゴリズムによる道路検出最適化システム
RGB色空間パラメータを進化させて道路検出精度を向上
"""

import numpy as np
import cv2
import random
import json
import os
from typing import List, Tuple, Dict
from dataclasses import dataclass
from PIL import Image
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RoadColorGene:
    """道路色検出のための遺伝子"""
    # RGB範囲パラメータ
    r_min: int = 0
    r_max: int = 255
    g_min: int = 0
    g_max: int = 255  
    b_min: int = 0
    b_max: int = 255
    
    # HSV範囲パラメータ（補助的）
    h_min: int = 0
    h_max: int = 180
    s_min: int = 0
    s_max: int = 255
    v_min: int = 0
    v_max: int = 255
    
    # 形態学的処理パラメータ
    erosion_kernel: int = 3
    dilation_kernel: int = 5
    
    # フィルタリングパラメータ
    min_area: int = 100
    aspect_ratio_threshold: float = 0.3

    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            'rgb': {
                'r_range': [self.r_min, self.r_max],
                'g_range': [self.g_min, self.g_max], 
                'b_range': [self.b_min, self.b_max]
            },
            'hsv': {
                'h_range': [self.h_min, self.h_max],
                's_range': [self.s_min, self.s_max],
                'v_range': [self.v_min, self.v_max]
            },
            'morphology': {
                'erosion_kernel': self.erosion_kernel,
                'dilation_kernel': self.dilation_kernel
            },
            'filtering': {
                'min_area': self.min_area,
                'aspect_ratio_threshold': self.aspect_ratio_threshold
            }
        }

class GeneticRoadDetector:
    def __init__(self, population_size: int = 50, generations: int = 30):
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.fitness_scores = []
        self.best_gene = None
        self.best_fitness = 0.0
        
        # 道路の典型的なRGB値を基準として設定
        self.road_rgb_targets = [
            (120, 120, 120),  # アスファルト（灰色）
            (80, 80, 80),     # 濃いアスファルト
            (160, 160, 160),  # 明るいアスファルト  
            (100, 90, 80),    # 土の道路
            (140, 130, 120),  # コンクリート道路
        ]

    def create_initial_population(self):
        """初期個体群を生成"""
        logger.info("初期個体群を生成中...")
        self.population = []
        
        for _ in range(self.population_size):
            # 道路の典型色を中心とした範囲で初期化
            base_rgb = random.choice(self.road_rgb_targets)
            
            gene = RoadColorGene(
                # RGB範囲（基準色の±50の範囲）
                r_min=max(0, base_rgb[0] - random.randint(30, 70)),
                r_max=min(255, base_rgb[0] + random.randint(30, 70)),
                g_min=max(0, base_rgb[1] - random.randint(30, 70)),
                g_max=min(255, base_rgb[1] + random.randint(30, 70)),
                b_min=max(0, base_rgb[2] - random.randint(30, 70)),
                b_max=min(255, base_rgb[2] + random.randint(30, 70)),
                
                # HSV範囲
                h_min=random.randint(0, 30),
                h_max=random.randint(150, 180),
                s_min=random.randint(0, 50),
                s_max=random.randint(200, 255),
                v_min=random.randint(30, 100),
                v_max=random.randint(180, 255),
                
                # 形態学的処理
                erosion_kernel=random.randint(2, 5),
                dilation_kernel=random.randint(3, 8),
                
                # フィルタリング
                min_area=random.randint(50, 300),
                aspect_ratio_threshold=random.uniform(0.1, 0.8)
            )
            self.population.append(gene)

    def detect_roads_with_gene(self, image_path: str, gene: RoadColorGene) -> Tuple[np.ndarray, float]:
        """遺伝子に基づいて道路を検出"""
        try:
            # 画像読み込み
            image = cv2.imread(image_path)
            if image is None:
                return np.zeros((100, 100)), 0.0
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # RGB色空間でのマスク生成
            rgb_mask = cv2.inRange(image_rgb, 
                                 (gene.r_min, gene.g_min, gene.b_min),
                                 (gene.r_max, gene.g_max, gene.b_max))
            
            # HSV色空間でのマスク生成
            hsv_mask = cv2.inRange(image_hsv,
                                 (gene.h_min, gene.s_min, gene.v_min),
                                 (gene.h_max, gene.s_max, gene.v_max))
            
            # マスクを組み合わせ
            combined_mask = cv2.bitwise_and(rgb_mask, hsv_mask)
            
            # 形態学的処理
            erosion_kernel = np.ones((gene.erosion_kernel, gene.erosion_kernel), np.uint8)
            dilation_kernel = np.ones((gene.dilation_kernel, gene.dilation_kernel), np.uint8)
            
            processed_mask = cv2.erode(combined_mask, erosion_kernel, iterations=1)
            processed_mask = cv2.dilate(processed_mask, dilation_kernel, iterations=1)
            
            # 連結成分解析
            contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # フィルタリング
            filtered_mask = np.zeros_like(processed_mask)
            road_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > gene.min_area:
                    # アスペクト比チェック
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = min(w, h) / max(w, h)
                    
                    if aspect_ratio > gene.aspect_ratio_threshold:
                        cv2.fillPoly(filtered_mask, [contour], 255)
                        road_area += area
            
            return filtered_mask, road_area
            
        except Exception as e:
            logger.error(f"道路検出エラー: {str(e)}")
            return np.zeros((100, 100)), 0.0

    def calculate_fitness(self, gene: RoadColorGene, test_images: List[str]) -> float:
        """適応度を計算"""
        total_fitness = 0.0
        
        for image_path in test_images:
            mask, road_area = self.detect_roads_with_gene(image_path, gene)
            
            # 画像サイズに対する道路面積の割合
            image = cv2.imread(image_path)
            if image is not None:
                total_pixels = image.shape[0] * image.shape[1]
                road_ratio = road_area / total_pixels
                
                # 適応度計算（道路面積とマスクの連続性を考慮）
                connectivity_score = self.calculate_connectivity_score(mask)
                color_diversity_score = self.calculate_color_diversity_score(gene)
                
                fitness = (road_ratio * 0.4 + 
                          connectivity_score * 0.4 + 
                          color_diversity_score * 0.2)
                
                total_fitness += fitness
        
        return total_fitness / len(test_images) if test_images else 0.0

    def calculate_connectivity_score(self, mask: np.ndarray) -> float:
        """マスクの連続性スコアを計算"""
        if mask.sum() == 0:
            return 0.0
            
        # 連結成分の数と最大連結成分のサイズを評価
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        
        if num_labels <= 1:
            return 0.0
            
        # 最大連結成分のサイズ
        max_area = np.max(stats[1:, cv2.CC_STAT_AREA])
        total_area = mask.sum() / 255
        
        # 連続性スコア（大きな連結成分が多いほど高スコア）
        connectivity = (max_area / total_area) if total_area > 0 else 0.0
        return min(connectivity, 1.0)

    def calculate_color_diversity_score(self, gene: RoadColorGene) -> float:
        """色の多様性スコアを計算"""
        # RGB範囲の広さを評価（適度な範囲が望ましい）
        r_range = gene.r_max - gene.r_min
        g_range = gene.g_max - gene.g_min  
        b_range = gene.b_max - gene.b_min
        
        avg_range = (r_range + g_range + b_range) / 3
        
        # 適度な範囲（50-100）が最適
        if 50 <= avg_range <= 100:
            return 1.0
        elif avg_range < 50:
            return avg_range / 50
        else:
            return max(0.0, 1.0 - (avg_range - 100) / 100)

    def crossover(self, parent1: RoadColorGene, parent2: RoadColorGene) -> RoadColorGene:
        """交叉操作"""
        child = RoadColorGene()
        
        # RGB パラメータの交叉
        child.r_min = random.choice([parent1.r_min, parent2.r_min])
        child.r_max = random.choice([parent1.r_max, parent2.r_max])
        child.g_min = random.choice([parent1.g_min, parent2.g_min])
        child.g_max = random.choice([parent1.g_max, parent2.g_max])
        child.b_min = random.choice([parent1.b_min, parent2.b_min])
        child.b_max = random.choice([parent1.b_max, parent2.b_max])
        
        # HSV パラメータの交叉
        child.h_min = random.choice([parent1.h_min, parent2.h_min])
        child.h_max = random.choice([parent1.h_max, parent2.h_max])
        child.s_min = random.choice([parent1.s_min, parent2.s_min])
        child.s_max = random.choice([parent1.s_max, parent2.s_max])
        child.v_min = random.choice([parent1.v_min, parent2.v_min])
        child.v_max = random.choice([parent1.v_max, parent2.v_max])
        
        # その他パラメータの交叉
        child.erosion_kernel = random.choice([parent1.erosion_kernel, parent2.erosion_kernel])
        child.dilation_kernel = random.choice([parent1.dilation_kernel, parent2.dilation_kernel])
        child.min_area = random.choice([parent1.min_area, parent2.min_area])
        child.aspect_ratio_threshold = random.choice([parent1.aspect_ratio_threshold, parent2.aspect_ratio_threshold])
        
        return child

    def mutate(self, gene: RoadColorGene, mutation_rate: float = 0.1) -> RoadColorGene:
        """突然変異操作"""
        mutated = RoadColorGene(**gene.__dict__)
        
        if random.random() < mutation_rate:
            # RGB 突然変異
            if random.random() < 0.3:
                mutated.r_min = max(0, mutated.r_min + random.randint(-20, 20))
                mutated.r_max = min(255, mutated.r_max + random.randint(-20, 20))
            if random.random() < 0.3:
                mutated.g_min = max(0, mutated.g_min + random.randint(-20, 20))
                mutated.g_max = min(255, mutated.g_max + random.randint(-20, 20))
            if random.random() < 0.3:
                mutated.b_min = max(0, mutated.b_min + random.randint(-20, 20))
                mutated.b_max = min(255, mutated.b_max + random.randint(-20, 20))
            
            # 範囲の整合性チェック
            mutated.r_min = min(mutated.r_min, mutated.r_max)
            mutated.g_min = min(mutated.g_min, mutated.g_max)
            mutated.b_min = min(mutated.b_min, mutated.b_max)
        
        return mutated

    def evolve(self, test_images: List[str]):
        """遺伝的アルゴリズムを実行"""
        logger.info(f"遺伝的アルゴリズム開始: {self.generations}世代, {self.population_size}個体")
        
        self.create_initial_population()
        
        for generation in range(self.generations):
            logger.info(f"第{generation+1}世代を処理中...")
            
            # 適応度計算
            self.fitness_scores = []
            for i, gene in enumerate(self.population):
                fitness = self.calculate_fitness(gene, test_images)
                self.fitness_scores.append(fitness)
                
                # 最良個体を記録
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_gene = gene
            
            logger.info(f"第{generation+1}世代 最高適応度: {max(self.fitness_scores):.4f}")
            
            # 次世代の生成
            if generation < self.generations - 1:
                self.create_next_generation()
        
        logger.info(f"進化完了! 最終適応度: {self.best_fitness:.4f}")
        return self.best_gene

    def create_next_generation(self):
        """次世代を生成"""
        # エリート選択（上位20%を保持）
        elite_count = max(1, self.population_size // 5)
        elite_indices = np.argsort(self.fitness_scores)[-elite_count:]
        
        new_population = [self.population[i] for i in elite_indices]
        
        # 残りを交叉と突然変異で生成
        while len(new_population) < self.population_size:
            # トーナメント選択
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # 交叉
            child = self.crossover(parent1, parent2)
            
            # 突然変異
            child = self.mutate(child)
            
            new_population.append(child)
        
        self.population = new_population

    def tournament_selection(self, tournament_size: int = 3) -> RoadColorGene:
        """トーナメント選択"""
        tournament_indices = random.sample(range(len(self.population)), tournament_size)
        tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_index]

    def save_best_gene(self, filepath: str):
        """最良遺伝子を保存"""
        if self.best_gene:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'best_gene': self.best_gene.to_dict(),
                    'fitness': self.best_fitness,
                    'generation': self.generations
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"最良遺伝子を保存: {filepath}")

    def load_best_gene(self, filepath: str) -> RoadColorGene:
        """保存された遺伝子を読み込み"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                gene_data = data['best_gene']
                
                gene = RoadColorGene(
                    r_min=gene_data['rgb']['r_range'][0],
                    r_max=gene_data['rgb']['r_range'][1],
                    g_min=gene_data['rgb']['g_range'][0],
                    g_max=gene_data['rgb']['g_range'][1],
                    b_min=gene_data['rgb']['b_range'][0],
                    b_max=gene_data['rgb']['b_range'][1],
                    h_min=gene_data['hsv']['h_range'][0],
                    h_max=gene_data['hsv']['h_range'][1],
                    s_min=gene_data['hsv']['s_range'][0],
                    s_max=gene_data['hsv']['s_range'][1],
                    v_min=gene_data['hsv']['v_range'][0],
                    v_max=gene_data['hsv']['v_range'][1],
                    erosion_kernel=gene_data['morphology']['erosion_kernel'],
                    dilation_kernel=gene_data['morphology']['dilation_kernel'],
                    min_area=gene_data['filtering']['min_area'],
                    aspect_ratio_threshold=gene_data['filtering']['aspect_ratio_threshold']
                )
                
                self.best_gene = gene
                self.best_fitness = data['fitness']
                logger.info(f"遺伝子を読み込み: {filepath}, 適応度: {self.best_fitness:.4f}")
                return gene
                
        except Exception as e:
            logger.error(f"遺伝子読み込みエラー: {str(e)}")
            return None

    def visualize_detection_result(self, image_path: str, gene: RoadColorGene, output_path: str):
        """検出結果を可視化"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask, road_area = self.detect_roads_with_gene(image_path, gene)
        
        # マスクを重ねた画像を作成
        mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        result = cv2.addWeighted(image_rgb, 0.7, mask_colored, 0.3, 0)
        
        # 結果を保存
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image_rgb)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Road Detection Mask\nArea: {road_area:.0f} pixels')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(result)
        plt.title('Overlay Result')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"検出結果を保存: {output_path}")

def main():
    """メイン実行関数"""
    # テスト画像のパス
    test_images = [
        "assets/hatinohe/20110907-10-HF713-a2.jpg",
        "assets/hatinohe/20110907-10-HF714-a2.jpg",
        "assets/car/frame_0001.png"
    ]
    
    # 存在する画像のみを使用
    existing_images = [img for img in test_images if os.path.exists(img)]
    
    if not existing_images:
        logger.error("テスト画像が見つかりません")
        return
    
    logger.info(f"使用する画像: {existing_images}")
    
    # 遺伝的アルゴリズム実行
    detector = GeneticRoadDetector(population_size=30, generations=20)
    best_gene = detector.evolve(existing_images)
    
    if best_gene:
        # 結果保存
        detector.save_best_gene("best_road_gene.json")
        
        # 検出結果可視化
        for i, image_path in enumerate(existing_images):
            output_path = f"genetic_road_result_{i+1}.png"
            detector.visualize_detection_result(image_path, best_gene, output_path)
        
        # 最良遺伝子の詳細出力
        print("\n=== 最良遺伝子パラメータ ===")
        print(f"RGB範囲: R({best_gene.r_min}-{best_gene.r_max}), G({best_gene.g_min}-{best_gene.g_max}), B({best_gene.b_min}-{best_gene.b_max})")
        print(f"適応度: {detector.best_fitness:.4f}")
        print(f"保存ファイル: best_road_gene.json")

if __name__ == "__main__":
    main()
