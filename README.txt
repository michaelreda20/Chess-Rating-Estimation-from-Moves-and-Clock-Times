CHESS RATING PREDICTION SYSTEM
===============================

DEPENDENCIES
------------
- Python 3.7+
- torch (PyTorch)
- numpy
- scikit-learn
- PyQt5
- matplotlib
- tensorboard

INSTALLATION
------------
1. Create virtual environment:
   python3 -m venv rating_env
   source rating_env/bin/activate  # On Windows: rating_env\Scripts\activate

2. Install PyTorch:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

3. Install other dependencies:
   pip install numpy scikit-learn PyQt5 matplotlib tensorboard

USAGE
-----
Run game analysis GUI:
   python3 game_analysis_qt.py

Run rating network training:
   python3 chess_rating_net_improved.py
