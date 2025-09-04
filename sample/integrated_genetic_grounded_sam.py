#!/usr/bin/env python3
"""
遺伝的アルゴリズム最適化道路検出とGrounded-SAMの統合システム
"""

import sys
import os
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
import json
import logging
from typing import List, Tuple, Dict

# パス追加
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Segment Anything
from segment_anything import sam_model_registry, SamPredictor

# 遺伝的アルゴリズム
from genetic_road_detector import GeneticRoadDetector, RoadColorGene

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegratedRoadDetectionSystem:
    def __init__(self, 
                 grounding_config: str,
                 grounding_checkpoint: str,
                 sam_checkpoint: str,
                 device: str = "cpu"):
        
        self.device = device
        self.grounding_model = None
        self.sam_predictor = None
        self.genetic_detector = GeneticRoadDetector()
        self.best_gene = None
        
        # モデル読み込み
        self.load_models(grounding_config, grounding_checkpoint, sam_checkpoint)

    def load_models(self, grounding_config: str, grounding_checkpoint: str, sam_checkpoint: str):
        """GroundingDINOとSAMモデルを読み込み"""
        try:
            # GroundingDINO読み込み
            args = SLConfig.fromfile(grounding_config)
            args.device = self.device
            self.grounding_model = build_model(args)
            checkpoint = torch.load(grounding_checkpoint, map_location="cpu")
            self.grounding_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
            self.grounding_model.eval()
            logger.info("GroundingDINOモデル読み込み完了")

            # SAM読み込み
            sam_model = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
            self.sam_predictor = SamPredictor(sam_model.to(self.device))
            logger.info("SAMモデル読み込み完了")

        except Exception as e:
            logger.error(f"モデル読み込みエラー: {str(e)}")
            raise

    def optimize_road_detection(self, training_images: List[str], generations: int = 15):
        """遺伝的アルゴリズムで道路検出を最適化"""
        logger.info("遺伝的アルゴリズムによる道路検出最適化を開始")
        
        # 既存の最良遺伝子を読み込み試行
        if os.path.exists("best_road_gene.json"):
            logger.info("既存の最良遺伝子を読み込み中...")
            self.best_gene = self.genetic_detector.load_best_gene("best_road_gene.json")
        
        if not self.best_gene:
            # 新規最適化実行
            detector = GeneticRoadDetector(population_size=25, generations=generations)
            self.best_gene = detector.evolve(training_images)
            detector.save_best_gene("best_road_gene.json")
        
        return self.best_gene

    def preprocess_image_for_grounding(self, image_path: str) -> Tuple[Image.Image, torch.Tensor]:
        """GroundingDINO用画像前処理"""
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        image_pil = Image.open(image_path).convert("RGB")
        image_transformed, _ = transform(image_pil, None)
        return image_pil, image_transformed

    def generate_enhanced_road_prompt(self, gene: RoadColorGene) -> str:
        """遺伝子パラメータに基づいて最適化されたプロンプトを生成"""
        # RGB値から道路タイプを推定
        avg_r = (gene.r_min + gene.r_max) / 2
        avg_g = (gene.g_min + gene.g_max) / 2
        avg_b = (gene.b_min + gene.b_max) / 2
        
        road_descriptors = []
        
        # 色味に基づく記述追加
        if avg_r < 100 and avg_g < 100 and avg_b < 100:
            road_descriptors.append("dark asphalt road")
        elif 100 <= avg_r <= 150 and 100 <= avg_g <= 150 and 100 <= avg_b <= 150:
            road_descriptors.append("gray concrete road")
        else:
            road_descriptors.append("light colored road")
        
        # 基本的な道路キーワード
        road_descriptors.extend([
            "paved road", "street", "highway", "pathway", 
            "roadway", "asphalt", "pavement"
        ])
        
        return ". ".join(road_descriptors[:4]) + "."

    def detect_roads_with_grounding_dino(self, image_path: str, gene: RoadColorGene, 
                                       box_threshold: float = 0.3, text_threshold: float = 0.25):
        """GroundingDINOで道路検出"""
        try:
            image_pil, image_transformed = self.preprocess_image_for_grounding(image_path)
            
            # 最適化されたプロンプト生成
            text_prompt = self.generate_enhanced_road_prompt(gene)
            logger.info(f"使用プロンプト: {text_prompt}")
            
            # GroundingDINOで検出
            image_tensor = image_transformed.to(self.device)
            self.grounding_model = self.grounding_model.to(self.device)
            
            with torch.no_grad():
                outputs = self.grounding_model(image_tensor[None], captions=[text_prompt])
            
            prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]
            prediction_boxes = outputs["pred_boxes"].cpu()[0]
            
            mask = prediction_logits.max(dim=1)[0] > box_threshold
            logits = prediction_logits[mask]
            boxes = prediction_boxes[mask]
            
            if len(boxes) == 0:
                logger.warning("GroundingDINOで道路が検出されませんでした")
                return [], [], []
            
            # フレーズ取得
            tokenizer = self.grounding_model.tokenizer
            tokenized = tokenizer(text_prompt)
            phrases = [
                get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
                for logit in logits
            ]
            
            return boxes, logits.max(dim=1)[0], phrases

        except Exception as e:
            logger.error(f"GroundingDINO検出エラー: {str(e)}")
            return [], [], []

    def refine_detection_with_genetic_mask(self, image_path: str, boxes: torch.Tensor, gene: RoadColorGene):
        """遺伝的アルゴリズムのマスクでGroundingDINOの結果を洗練"""
        if len(boxes) == 0:
            return []
        
        # 遺伝的アルゴリズムで道路マスク生成
        genetic_mask, _ = self.genetic_detector.detect_roads_with_gene(image_path, gene)
        
        image = cv2.imread(image_path)
        if image is None:
            return []
        
        h, w = image.shape[:2]
        refined_boxes = []
        
        for box in boxes:
            # ボックス座標を画像座標に変換
            box_scaled = box * torch.tensor([w, h, w, h])
            x1, y1, x2, y2 = box_scaled
            
            # ボックス内の遺伝的マスクの密度をチェック
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                box_mask = genetic_mask[y1:y2, x1:x2]
                mask_ratio = np.sum(box_mask > 0) / (box_mask.shape[0] * box_mask.shape[1])
                
                # マスクの密度が閾値以上の場合のみ採用
                if mask_ratio > 0.3:  # 30%以上道路ピクセル
                    refined_boxes.append(box)
        
        return torch.stack(refined_boxes) if refined_boxes else torch.empty((0, 4))

    def generate_sam_masks(self, image_path: str, boxes: torch.Tensor):
        """SAMでマスク生成"""
        if len(boxes) == 0:
            return None
        
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image_rgb)
        
        h, w = image.shape[:2]
        
        # ボックスをSAM用に変換
        boxes_xyxy = boxes * torch.tensor([w, h, w, h])
        boxes_xyxy[:, 2:] = boxes_xyxy[:, :2] + boxes_xyxy[:, 2:]  # cxcywh -> xyxy
        
        input_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes_xyxy, image_rgb.shape[:2]).to(self.device)
        
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=input_boxes,
            multimask_output=False,
        )
        
        return masks

    def integrated_road_detection(self, image_path: str, output_dir: str = "outputs_integrated"):
        """統合道路検出システム実行"""
        logger.info(f"統合道路検出を開始: {image_path}")
        
        # 出力ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.best_gene:
            logger.error("遺伝子最適化が実行されていません")
            return
        
        # Step 1: GroundingDINOで道路検出
        boxes, logits, phrases = self.detect_roads_with_grounding_dino(image_path, self.best_gene)
        
        if len(boxes) == 0:
            logger.warning("道路が検出されませんでした")
            return
        
        # Step 2: 遺伝的アルゴリズムのマスクで結果を洗練
        refined_boxes = self.refine_detection_with_genetic_mask(image_path, boxes, self.best_gene)
        
        if len(refined_boxes) == 0:
            logger.warning("洗練後に道路が残りませんでした")
            refined_boxes = boxes  # 元の結果を使用
        
        # Step 3: SAMでマスク生成
        masks = self.generate_sam_masks(image_path, refined_boxes)
        
        # Step 4: 結果可視化・保存
        self.visualize_integrated_results(image_path, refined_boxes, masks, phrases, logits, output_dir)
        
        logger.info("統合道路検出完了")

    def visualize_integrated_results(self, image_path: str, boxes: torch.Tensor, masks, 
                                   phrases: List[str], logits: torch.Tensor, output_dir: str):
        """統合結果を可視化"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # 遺伝的アルゴリズムのマスク
        genetic_mask, _ = self.genetic_detector.detect_roads_with_gene(image_path, self.best_gene)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 元画像
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 遺伝的アルゴリズムの結果
        axes[0, 1].imshow(genetic_mask, cmap='gray')
        axes[0, 1].set_title('Genetic Algorithm Mask')
        axes[0, 1].axis('off')
        
        # GroundingDINOのボックス
        image_with_boxes = image_rgb.copy()
        if len(boxes) > 0:
            boxes_scaled = boxes * torch.tensor([w, h, w, h])
            for i, box in enumerate(boxes_scaled):
                x1, y1, x2, y2 = box
                cv2.rectangle(image_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                if i < len(phrases):
                    cv2.putText(image_with_boxes, f"{phrases[i]}", (int(x1), int(y1-10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        axes[0, 2].imshow(image_with_boxes)
        axes[0, 2].set_title('GroundingDINO Detection')
        axes[0, 2].axis('off')
        
        # SAMマスク
        if masks is not None and len(masks) > 0:
            combined_mask = torch.zeros((h, w), dtype=torch.bool)
            for mask in masks:
                combined_mask |= mask.cpu()[0]
            
            axes[1, 0].imshow(combined_mask, cmap='gray')
            axes[1, 0].set_title('SAM Segmentation Mask')
            axes[1, 0].axis('off')
            
            # 最終統合結果
            final_result = image_rgb.copy()
            mask_colored = np.zeros((h, w, 3), dtype=np.uint8)
            mask_colored[combined_mask] = [255, 0, 0]  # 赤色
            final_result = cv2.addWeighted(final_result, 0.7, mask_colored, 0.3, 0)
            
            axes[1, 1].imshow(final_result)
            axes[1, 1].set_title('Final Integrated Result')
            axes[1, 1].axis('off')
        else:
            axes[1, 0].text(0.5, 0.5, 'No SAM masks', ha='center', va='center')
            axes[1, 0].axis('off')
            axes[1, 1].text(0.5, 0.5, 'No final result', ha='center', va='center') 
            axes[1, 1].axis('off')
        
        # 統計情報
        stats_text = f"""
        Detection Statistics:
        - GroundingDINO boxes: {len(boxes)}
        - Best fitness: {self.genetic_detector.best_fitness:.4f}
        - RGB range: R({self.best_gene.r_min}-{self.best_gene.r_max})
                    G({self.best_gene.g_min}-{self.best_gene.g_max})
                    B({self.best_gene.b_min}-{self.best_gene.b_max})
        """
        
        axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, va='center')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # 保存
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"integrated_result_{base_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"統合結果を保存: {output_path}")

def main():
    """メイン実行関数"""
    # 設定
    config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounding_checkpoint = "groundingdino_swint_ogc.pth"
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    device = "cpu"
    
    # テスト画像
    test_images = [
        "assets/hatinohe/20110907-10-HF713-a2.jpg",
        "assets/hatinohe/20110907-10-HF714-a2.jpg",
        "assets/car/frame_0001.png"
    ]
    
    existing_images = [img for img in test_images if os.path.exists(img)]
    
    if not existing_images:
        logger.error("テスト画像が見つかりません")
        return
    
    try:
        # 統合システム初期化
        system = IntegratedRoadDetectionSystem(
            config_path, grounding_checkpoint, sam_checkpoint, device
        )
        
        # 遺伝的アルゴリズム最適化
        logger.info("道路検出パラメータを最適化中...")
        system.optimize_road_detection(existing_images, generations=10)
        
        # 各画像で統合検出実行
        for image_path in existing_images:
            system.integrated_road_detection(image_path)
        
        logger.info("全ての処理が完了しました")
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {str(e)}")
        raise

if __name__ == "__main__":
    main()
