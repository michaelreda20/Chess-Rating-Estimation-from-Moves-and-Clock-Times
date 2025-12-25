# Chess Rating Estimation from Moves and Clock Times

This project implements a Deep Learning system to infer a chess player's skill level (Elo) directly from their gameplay behavior. [cite_start]Unlike traditional systems like Elo or Glicko-2 that rely on match outcomes [cite: 5, 309][cite_start], this model analyzes the quality of move sequences and time management on a move-by-move basis[cite: 7].


## 🚀 Features
* [cite_start]**Hybrid Architecture:** Combines a CNN for spatial board features and a Bidirectional LSTM for temporal move sequences[cite: 30, 318, 319].
* [cite_start]**Attention Mechanisms:** Utilizes Multi-Head Self-Attention and Move Importance Weighting to identify critical game-defining moves[cite: 58, 59, 700].
* [cite_start]**Clock Integration:** Incorporates standardized remaining clock time to significantly improve prediction accuracy[cite: 25, 321].
* [cite_start]**GUI Tool:** A dedicated PyQt5 application for analyzing individual game files and visualizing model predictions[cite: 802].

## 📊 Performance
[cite_start]The project successfully improved upon the baseline "RatingNet" model, achieving a **22.1 Elo reduction** in Mean Absolute Error (MAE)[cite: 90, 838].

| Model | Test MAE (Elo) | Improvement |
| :--- | :--- | :--- |
| Baseline | 308.6 | - |
| **Final (Attention + All Features)** | **286.5** | **-22.1** |

[cite_start]*Detailed findings indicate that longer time controls (Classical) are significantly easier to predict than fast-paced games (UltraBullet)[cite: 71, 821].*

## 📁 Repository Contents
* `/src`: Contains the improved training pipeline and the PyQt5 GUI analysis tool.
* `/docs`: Includes the initial AML Proposal, the technical poster, and the final milestone presentation.
* `/website`: Source code for the project's web-based showcase.
* `/demo`: Video presentation of the final milestone.

## 🛠️ Installation

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/your-username/chess-rating-prediction.git](https://github.com/your-username/chess-rating-prediction.git)
   cd chess-rating-prediction
