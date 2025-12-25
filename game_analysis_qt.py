import sys
import os
import threading
import torch
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from chess_rating_net_improved import ImprovedChessEloPredictor, AttentionChessEloPredictor, time_to_seconds

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QLabel, QLineEdit, QPushButton, QTextEdit, QProgressBar,
                              QFileDialog, QSpinBox, QComboBox, QGroupBox, QGridLayout,
                              QMessageBox, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QColor, QPalette

class AnalysisWorker(QThread):
    """Worker thread for running analysis without blocking GUI"""
    progress_update = pyqtSignal(int, str)  # progress, status
    log_update = pyqtSignal(str)  # log message
    finished_signal = pyqtSignal(dict)  # results
    error_signal = pyqtSignal(str)  # error message

    def __init__(self, config):
        super().__init__()
        self.config = config

    def log(self, message):
        self.log_update.emit(message)

    def run(self):
        try:
            self.log("="*60)
            self.log("Starting Game Analysis")
            self.log("="*60)

            self.progress_update.emit(10, "Loading model...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.log(f"Using device: {device}")

            # Load checkpoint
            checkpoint = torch.load(self.config['model_path'], map_location=device, weights_only=False)

            if 'params' in checkpoint:
                saved_params = checkpoint['params']
                self.log(f"Found saved parameters (val_loss: {checkpoint.get('val_loss', 'N/A')})")
            else:
                saved_params = {
                    'conv_filters': 32, 'lstm_layers': 5, 'bidirectional': True,
                    'dropout_rate': 0.4, 'lstm_h': 128, 'fc1_h': 64,
                    'use_residual': True, 'use_attention': self.config['model_type'] == 'attention',
                    'num_attention_heads': 8, 'use_move_importance': True
                }

            self.progress_update.emit(20, "Initializing model...")

            # Initialize model
            if saved_params.get('use_attention', False) or self.config['model_type'] == 'attention':
                model = AttentionChessEloPredictor(
                    **{k: saved_params[k] for k in ['conv_filters', 'lstm_layers', 'dropout_rate',
                       'lstm_h', 'fc1_h', 'bidirectional', 'use_residual', 'num_attention_heads',
                       'use_move_importance']}
                ).to(device)
                use_attention = True
                self.log("Using AttentionChessEloPredictor")
            else:
                model = ImprovedChessEloPredictor(
                    **{k: saved_params[k] for k in ['conv_filters', 'lstm_layers', 'dropout_rate',
                       'lstm_h', 'fc1_h', 'bidirectional', 'use_residual']}
                ).to(device)
                use_attention = False
                self.log("Using ImprovedChessEloPredictor")

            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            self.log("Model loaded successfully")

            self.progress_update.emit(30, "Loading game files...")

            # Load data
            all_files = [os.path.join(self.config['data_dir'], f)
                        for f in os.listdir(self.config['data_dir']) if f.endswith('.pkl')]
            self.log(f"Found {len(all_files)} game files")

            train_val_files, test_files = train_test_split(all_files, test_size=0.1, random_state=42)
            self.log(f"Test set: {len(test_files)} games")

            sample_size = min(self.config['sample_size'], len(test_files))
            sampled_files = np.random.choice(test_files, sample_size, replace=False)
            self.log(f"Analyzing {sample_size} games...")

            self.progress_update.emit(40, f"Processing games...")

            # Normalization params
            ratings_mean, ratings_std = 1514, 366
            clocks_mean, clocks_std = 273, 380

            # Analyze games
            lowest_mae = float('inf')
            highest_mae = float('-inf')
            lowest_mae_lines, highest_mae_lines = None, None
            lowest_mae_file, highest_mae_file = None, None

            for idx, file_path in enumerate(sampled_files):
                progress = 40 + int((idx / sample_size) * 50)
                self.progress_update.emit(progress, f'Processing game {idx + 1}/{sample_size}...')

                with open(file_path, 'rb') as f:
                    game_info = pickle.load(f)

                clocks = [time_to_seconds(c) for c in game_info.get('Clocks', [])]
                clocks = [(c - clocks_mean) / clocks_std for c in clocks]
                clocks = torch.tensor(clocks, dtype=torch.float).to(device)

                positions = torch.stack(game_info['Positions']).to(device)

                with torch.no_grad():
                    model_output = model(positions.unsqueeze(0), clocks.unsqueeze(0),
                                       torch.tensor([len(positions)]))

                    if use_attention:
                        all_outputs, last_output, attention_weights = model_output
                    else:
                        all_outputs, last_output = model_output

                    predictions = all_outputs.squeeze().cpu().numpy()

                predictions = predictions * ratings_std + ratings_mean
                white_elo = float(game_info['WhiteElo'])
                black_elo = float(game_info['BlackElo'])

                mae_white = np.abs(predictions[:, 0] - white_elo)
                mae_black = np.abs(predictions[:, 1] - black_elo)
                mae = np.mean(mae_white + mae_black)

                # Prepare lines
                lines = []
                moves = game_info.get('Moves', [])
                clocks_denorm = [c * clocks_std + clocks_mean for c in clocks.cpu().numpy()]

                for i in range(min(len(moves), 100)):
                    white_pred = predictions[i, 0]
                    black_pred = predictions[i, 1]
                    line = (f"Move: {moves[i]}, ClockTime: {clocks_denorm[i]:.1f}s, "
                           f"PredictedWhiteRating: {white_pred:.1f}, PredictedBlackRating: {black_pred:.1f}, "
                           f"ActualWhiteElo: {white_elo}, ActualBlackElo: {black_elo}")
                    lines.append(line)

                if mae < lowest_mae:
                    lowest_mae = mae
                    lowest_mae_lines = lines
                    lowest_mae_file = file_path

                if mae > highest_mae:
                    highest_mae = mae
                    highest_mae_lines = lines
                    highest_mae_file = file_path

            # Save results
            self.progress_update.emit(95, "Saving results...")
            output_dir = self.config['output_dir']
            os.makedirs(output_dir, exist_ok=True)

            if lowest_mae_lines:
                lowest_output = self.save_game_txt(lowest_mae_file, lowest_mae_lines,
                                                   output_dir, 'lowest_mae', lowest_mae)
                self.log(f"Best prediction saved: {os.path.basename(lowest_output)}")

            if highest_mae_lines:
                highest_output = self.save_game_txt(highest_mae_file, highest_mae_lines,
                                                    output_dir, 'highest_mae', highest_mae)
                self.log(f"Worst prediction saved: {os.path.basename(highest_output)}")

            self.log("="*60)
            self.log(f"Analysis complete!")
            self.log(f"Lowest MAE: {lowest_mae:.2f} Elo")
            self.log(f"Highest MAE: {highest_mae:.2f} Elo")
            self.log("="*60)

            # Emit results
            results = {
                'lowest_mae': lowest_mae,
                'highest_mae': highest_mae,
                'lowest_file': lowest_output,
                'highest_file': highest_output,
                'output_dir': output_dir
            }

            self.progress_update.emit(100, "Complete!")
            self.finished_signal.emit(results)

        except Exception as e:
            self.error_signal.emit(str(e))

    def save_game_txt(self, file_path, lines, output_dir, suffix, mae):
        """Save game to text file"""
        output_file_name = os.path.splitext(os.path.basename(file_path))[0] + f'_{suffix}.txt'
        output_file_path = os.path.join(output_dir, output_file_name)

        with open(output_file_path, 'w') as output_file:
            for line in lines:
                output_file.write(line + '\n')
            output_file.write(f'\nMAE: {mae:.2f} Elo points')

        return output_file_path


class GameAnalysisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Chess Game Analysis Tool")
        self.setGeometry(100, 100, 1000, 700)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("Chess Game Analysis Tool")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("QLabel { background-color: #2c3e50; color: white; padding: 15px; border-radius: 5px; }")
        main_layout.addWidget(title)

        # Configuration Group
        config_group = QGroupBox("Configuration")
        config_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12pt; }")
        config_layout = QGridLayout()
        config_layout.setSpacing(10)

        # Model Path
        config_layout.addWidget(QLabel("Model Path:"), 0, 0)
        self.model_path_edit = QLineEdit("models/improved_model_residual/best_model.pth")
        config_layout.addWidget(self.model_path_edit, 0, 1)
        model_browse_btn = QPushButton("Browse")
        model_browse_btn.clicked.connect(self.browse_model)
        config_layout.addWidget(model_browse_btn, 0, 2)

        # Data Directory
        config_layout.addWidget(QLabel("Data Directory:"), 1, 0)
        self.data_dir_edit = QLineEdit("data/processed_games_1gb")
        config_layout.addWidget(self.data_dir_edit, 1, 1)
        data_browse_btn = QPushButton("Browse")
        data_browse_btn.clicked.connect(self.browse_data)
        config_layout.addWidget(data_browse_btn, 1, 2)

        # Output Directory
        config_layout.addWidget(QLabel("Output Directory:"), 2, 0)
        self.output_dir_edit = QLineEdit("games_with_lowest_highest_mae_improved")
        config_layout.addWidget(self.output_dir_edit, 2, 1)
        output_browse_btn = QPushButton("Browse")
        output_browse_btn.clicked.connect(self.browse_output)
        config_layout.addWidget(output_browse_btn, 2, 2)

        # Sample Size and Model Type
        config_layout.addWidget(QLabel("Sample Size:"), 3, 0)
        self.sample_size_spin = QSpinBox()
        self.sample_size_spin.setRange(10, 100000)
        self.sample_size_spin.setValue(500)
        config_layout.addWidget(self.sample_size_spin, 3, 1)

        config_layout.addWidget(QLabel("Model Type:"), 4, 0)
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["attention", "improved"])
        config_layout.addWidget(self.model_type_combo, 4, 1)

        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)

        # Analyze Button
        self.analyze_btn = QPushButton("Start Analysis")
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-size: 14pt;
                font-weight: bold;
                padding: 15px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.analyze_btn.clicked.connect(self.start_analysis)
        main_layout.addWidget(self.analyze_btn)

        # Progress Section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                height: 30px;
            }
            QProgressBar::chunk {
                background-color: #3498db;
            }
        """)
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready to analyze games")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #7f8c8d; font-size: 10pt;")
        progress_layout.addWidget(self.status_label)

        progress_group.setLayout(progress_layout)
        main_layout.addWidget(progress_group)

        # Results Section
        results_layout = QHBoxLayout()

        # Log
        log_group = QGroupBox("Analysis Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #2c3e50;
                color: #ecf0f1;
                font-family: monospace;
                font-size: 9pt;
            }
        """)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        results_layout.addWidget(log_group)

        # Results Cards
        cards_layout = QVBoxLayout()

        # Best Result Card
        self.best_card = self.create_result_card("Best Prediction (Lowest MAE)", "#27ae60")
        cards_layout.addWidget(self.best_card)

        # Worst Result Card
        self.worst_card = self.create_result_card("Worst Prediction (Highest MAE)", "#e74c3c")
        cards_layout.addWidget(self.worst_card)

        cards_widget = QWidget()
        cards_widget.setLayout(cards_layout)
        results_layout.addWidget(cards_widget)

        main_layout.addLayout(results_layout, 1)

    def create_result_card(self, title, color):
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet(f"""
            QFrame {{
                border: 3px solid {color};
                border-radius: 8px;
                background-color: white;
            }}
        """)

        layout = QVBoxLayout()

        # Header
        header = QLabel(title)
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet(f"""
            QLabel {{
                background-color: {color};
                color: white;
                font-weight: bold;
                font-size: 11pt;
                padding: 10px;
                border-radius: 5px;
            }}
        """)
        layout.addWidget(header)

        # MAE Label
        mae_label = QLabel("MAE: --")
        mae_label.setAlignment(Qt.AlignCenter)
        mae_label.setStyleSheet("font-size: 16pt; font-weight: bold; color: #2c3e50;")
        setattr(frame, 'mae_label', mae_label)
        layout.addWidget(mae_label)

        # File Label
        file_label = QLabel("File: --")
        file_label.setAlignment(Qt.AlignCenter)
        file_label.setStyleSheet("font-size: 9pt; color: #7f8c8d;")
        file_label.setWordWrap(True)
        setattr(frame, 'file_label', file_label)
        layout.addWidget(file_label)

        # Buttons
        btn_layout = QHBoxLayout()
        view_btn = QPushButton("View Details")
        view_btn.setEnabled(False)
        view_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                padding: 5px 15px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {color};
                opacity: 0.8;
            }}
            QPushButton:disabled {{
                background-color: #95a5a6;
            }}
        """)
        setattr(frame, 'view_btn', view_btn)
        btn_layout.addWidget(view_btn)

        open_btn = QPushButton("Open Folder")
        open_btn.setEnabled(False)
        open_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                padding: 5px 15px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)
        setattr(frame, 'open_btn', open_btn)
        btn_layout.addWidget(open_btn)

        layout.addLayout(btn_layout)
        frame.setLayout(layout)

        return frame

    def browse_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "Model Files (*.pth)")
        if file_name:
            self.model_path_edit.setText(file_name)

    def browse_data(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if dir_name:
            self.data_dir_edit.setText(dir_name)

    def browse_output(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_name:
            self.output_dir_edit.setText(dir_name)

    def start_analysis(self):
        # Validate inputs
        if not os.path.exists(self.model_path_edit.text()):
            QMessageBox.critical(self, "Error", "Model file does not exist!")
            return

        if not os.path.exists(self.data_dir_edit.text()):
            QMessageBox.critical(self, "Error", "Data directory does not exist!")
            return

        # Disable button
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("Analyzing...")

        # Clear log
        self.log_text.clear()

        # Reset progress
        self.progress_bar.setValue(0)

        # Prepare config
        config = {
            'model_path': self.model_path_edit.text(),
            'data_dir': self.data_dir_edit.text(),
            'output_dir': self.output_dir_edit.text(),
            'sample_size': self.sample_size_spin.value(),
            'model_type': self.model_type_combo.currentText()
        }

        # Start worker thread
        self.worker = AnalysisWorker(config)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.log_update.connect(self.append_log)
        self.worker.finished_signal.connect(self.on_analysis_complete)
        self.worker.error_signal.connect(self.on_analysis_error)
        self.worker.start()

    def update_progress(self, value, status):
        self.progress_bar.setValue(value)
        self.status_label.setText(status)

    def append_log(self, message):
        self.log_text.append(message)

    def on_analysis_complete(self, results):
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("Start Analysis")

        # Update best card
        self.best_card.mae_label.setText(f"MAE: {results['lowest_mae']:.2f} Elo")
        self.best_card.file_label.setText(f"File: {os.path.basename(results['lowest_file'])}")
        self.best_card.view_btn.setEnabled(True)
        self.best_card.view_btn.clicked.connect(lambda: self.view_file(results['lowest_file']))
        self.best_card.open_btn.setEnabled(True)
        self.best_card.open_btn.clicked.connect(lambda: self.open_folder(results['output_dir']))

        # Update worst card
        self.worst_card.mae_label.setText(f"MAE: {results['highest_mae']:.2f} Elo")
        self.worst_card.file_label.setText(f"File: {os.path.basename(results['highest_file'])}")
        self.worst_card.view_btn.setEnabled(True)
        self.worst_card.view_btn.clicked.connect(lambda: self.view_file(results['highest_file']))
        self.worst_card.open_btn.setEnabled(True)
        self.worst_card.open_btn.clicked.connect(lambda: self.open_folder(results['output_dir']))

        QMessageBox.information(self, "Complete",
                               f"Analysis complete!\n\n"
                               f"Best MAE: {results['lowest_mae']:.2f} Elo\n"
                               f"Worst MAE: {results['highest_mae']:.2f} Elo")

    def on_analysis_error(self, error):
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("Start Analysis")
        QMessageBox.critical(self, "Error", f"Analysis failed:\n\n{error}")

    def view_file(self, file_path):
        import subprocess
        if sys.platform == 'win32':
            os.startfile(file_path)
        elif sys.platform == 'darwin':
            subprocess.call(['open', file_path])
        else:
            subprocess.call(['xdg-open', file_path])

    def open_folder(self, folder_path):
        import subprocess
        if sys.platform == 'win32':
            os.startfile(folder_path)
        elif sys.platform == 'darwin':
            subprocess.call(['open', folder_path])
        else:
            subprocess.call(['xdg-open', folder_path])


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    window = GameAnalysisGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
