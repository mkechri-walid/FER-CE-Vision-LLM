# FER-CE : Vision-LLM pour la Reconnaissance Faciale des Émotions Composées

## Description
Projet de reconnaissance des émotions composées (RAF-CE) utilisant un pipeline Vision-LLM combinant ConvNeXt, Grad-CAM et LLaVA pour la classification et l'explication des émotions faciales.

## Objectif
- Classifier 11 émotions composées sur le dataset RAF-CE
- Générer des explications textuelles via LLaVA
- Valider le CEG (Causal Emotion Grounding) via Grad-CAM

## Architecture du Pipeline

1. **Image Visage** — Upload photo de visage
2. **ConvNeXt** — Classification émotion composée
3. **Grad-CAM** — Heatmap zones faciales CEG
4. **LLaVA** — Explication textuelle AUs

## Résultats

| Modèle | Accuracy | F1 Macro |
|--------|----------|----------|
| Baseline ConvNeXt | 56.13% | 44.25% |
| LLaVA fine-tuné | 30.59% | 15.12% |

### Scores textuels LLaVA

| Métrique | Score |
|----------|-------|
| BLEU | 0.67 |
| ROUGE-1 | 0.84 |
| ROUGE-2 | 0.74 |
| ROUGE-L | 0.81 |

## Structure du Projet

- notebooks/
  - 01_baseline_convnext.ipynb
  - 02_finetune_llava.ipynb
  - 03_evaluation_ceg.ipynb
- results/
  - confusion_matrix_convnext.png
  - gradcam images

## Installation

    pip install torch torchvision transformers
    pip install peft bitsandbytes accelerate
    pip install scikit-learn matplotlib seaborn
    pip install nltk rouge-score gradio

## Dataset
- **RAF-CE** : 3954 images réelles
- 11 émotions composées
- Split : 3162 train / 792 test
- Source : http://whdeng.cn/RAF/model4.html

## Modèles utilisés
- **ConvNeXt-Tiny** : Baseline classification
- **LLaVA-1.5-7B** : Vision-LLM avec LoRA
- **Grad-CAM** : Visualisation CEG

## Papers de référence
1. Du et al. (2014) - Compound Facial Expressions
2. Li et al. (2017) - RAF-DB Dataset
3. Selvaraju et al. (2017) - Grad-CAM
4. Li et al. (2023) - BLIP-2
5. Liu et al. (2023) - LLaVA
6. Liu et al. (2022) - ConvNeXt

## Auteur
Mkechri Walid — Projet FER-CE 2026
