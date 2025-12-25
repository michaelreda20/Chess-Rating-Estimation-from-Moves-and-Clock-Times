import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import os
import pickle
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
import time
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict
from copy import deepcopy


def time_to_seconds(time_str):

    parts = time_str.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])



class ChessGamesDataset(Dataset):
    
    def __init__(self, filenames, max_moves=100, ratings_mean=1514, ratings_std=366,
                 clocks_mean=273, clocks_std=380, augment=False):
        self.filenames = filenames
        self.max_moves = max_moves
        self.ratings_mean = ratings_mean
        self.ratings_std = ratings_std
        self.clocks_mean = clocks_mean
        self.clocks_std = clocks_std
        self.augment = augment

    def __len__(self):
        return len(self.filenames)

    def augment_board(self, positions, targets=None):
       
        swap_colors = False

        if not self.augment:
            return positions, targets, swap_colors


        if np.random.rand() > 0.5:
            positions = torch.flip(positions, dims=[3])


        if np.random.rand() > 0.5:
            swap_colors = True


            white_pieces = positions[:, 0:6, :, :]
            black_pieces = positions[:, 6:12, :, :]

            white_pieces_flipped = torch.flip(white_pieces, dims=[2])
            black_pieces_flipped = torch.flip(black_pieces, dims=[2])

            positions = torch.cat([black_pieces_flipped, white_pieces_flipped], dim=1)

            if targets is not None:

                targets = torch.tensor([targets[1].item(), targets[0].item()], dtype=torch.float)

        return positions, targets, swap_colors

    def __getitem__(self, idx):
        
        with open(self.filenames[idx], 'rb') as f:
            game_info = pickle.load(f)

        clocks = [time_to_seconds(c) for c in game_info.get('Clocks', [])]
        clocks = [(c - self.clocks_mean) / self.clocks_std for c in clocks]
        clocks = torch.tensor(clocks, dtype=torch.float)[:self.max_moves]

        white = False
        if "white" in game_info:
            white = game_info["white"]

        last_rating = None
        if "rating_after_last_game" in game_info:
            last_rating = game_info["rating_after_last_game"]
            last_rating = (last_rating - self.ratings_mean) / self.ratings_std
            last_rating = torch.tensor(last_rating, dtype=torch.float)

        positions = torch.stack(game_info['Positions'])[:self.max_moves]

        white_elo, black_elo = float(game_info['WhiteElo']), float(game_info['BlackElo'])
        targets = torch.tensor([white_elo, black_elo], dtype=torch.float)

        positions, targets, swap_colors = self.augment_board(positions, targets)

        targets = (targets - self.ratings_mean) / self.ratings_std

        length = len(positions)

        initial_time, increment = map(int, game_info['Time'].split('+'))
        estimated_duration = initial_time + 40 * increment
        time_control = self.categorize_time_control(estimated_duration)

        result = None
        if "Result" in game_info:
            result = game_info["Result"]

        return {'positions': positions, 'clocks': clocks, 'targets': targets, 'length': length,
                'time_control': time_control, 'white': white, 'last_rating': last_rating, 'result': result}

    def categorize_time_control(self, estimated_duration):
        
        if estimated_duration < 29:
            return 'ultrabullet'
        elif estimated_duration < 179:
            return 'bullet'
        elif estimated_duration < 479:
            return 'blitz'
        elif estimated_duration < 1499:
            return 'rapid'
        else:
            return 'classical'


def collate_fn(batch):
    
    positions = pad_sequence([item['positions'] for item in batch], batch_first=True)
    clocks = pad_sequence([item['clocks'] for item in batch], batch_first=True)

    targets = torch.stack([item['targets'] for item in batch])
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.int)

    time_controls = [item['time_control'] for item in batch]
    white = torch.tensor([item['white'] for item in batch])

    last_rating = None
    if batch[0]['last_rating'] is not None:
        last_rating = torch.stack([item['last_rating'] for item in batch])

    if batch[0]['result']:
        results = [item['result'] for item in batch]
        return {'positions': positions, 'clocks': clocks, 'targets': targets, 'lengths': lengths,
                'time_controls': time_controls, 'white': white, 'last_rating': last_rating, 'results': results}

    return {'positions': positions, 'clocks': clocks, 'targets': targets, 'lengths': lengths,
            'time_controls': time_controls, 'white': white, 'last_rating': last_rating}

class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        
        residual = x

        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)

        out = F.leaky_relu(out)
        return out


class ImprovedChessEloPredictor(nn.Module):
    
    def __init__(self, conv_filters=32, lstm_layers=4, dropout_rate=0.5, lstm_h=128, 
                 fc1_h=64, bidirectional=True, use_residual=True):
        super(ImprovedChessEloPredictor, self).__init__()
        
        self.use_residual = use_residual
        
        if use_residual:
            self.conv1 = nn.Conv2d(12, conv_filters, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(conv_filters)
            
            self.res1 = ResidualBlock(conv_filters, conv_filters * 2, stride=2)
            self.res2 = ResidualBlock(conv_filters * 2, conv_filters * 4, stride=2)
            self.res3 = ResidualBlock(conv_filters * 4, conv_filters * 8, stride=2)
            
            self.conv_final = nn.Conv2d(conv_filters * 8, conv_filters * 8, kernel_size=3, padding=1)
            self.bn_final = nn.BatchNorm2d(conv_filters * 8)
            
        else:
            self.conv1 = nn.Conv2d(12, conv_filters, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(conv_filters)
            self.conv2 = nn.Conv2d(conv_filters, conv_filters * 2, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(conv_filters * 2)
            self.conv3 = nn.Conv2d(conv_filters * 2, conv_filters * 4, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(conv_filters * 4)
            self.conv4 = nn.Conv2d(conv_filters * 4, conv_filters * 8, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(conv_filters * 8)
            self.conv5 = nn.Conv2d(conv_filters * 8, conv_filters * 8, kernel_size=3, padding=1)
            self.bn5 = nn.BatchNorm2d(conv_filters * 8)
            self.conv6 = nn.Conv2d(conv_filters * 8, conv_filters * 8, kernel_size=3, padding=1)
            self.bn6 = nn.BatchNorm2d(conv_filters * 8)
        
        self.pool = nn.AvgPool2d(2, 2)
        self.dropout_cnn = nn.Dropout(dropout_rate * 0.5)
        self.dropout_lstm = nn.Dropout(dropout_rate) 
        
        self.lstm = nn.LSTM(
            input_size=conv_filters * 8 + 1, 
            hidden_size=lstm_h, 
            num_layers=lstm_layers, 
            batch_first=True, 
            bidirectional=bidirectional,
            dropout=dropout_rate if lstm_layers > 1 else 0 
        )
        
        lstm_output_size = lstm_h * 2 if bidirectional else lstm_h
        self.fc1 = nn.Linear(lstm_output_size, fc1_h)
        self.fc2 = nn.Linear(fc1_h, 2)
        
        self.layer_norm = nn.LayerNorm(lstm_output_size)

    def forward(self, positions, clocks, lengths):
        batch_size = positions.size(0)
        sequence_length = positions.size(1)
        positions = positions.view(-1, 12, 8, 8)
        
        if self.use_residual:
            x = F.leaky_relu(self.bn1(self.conv1(positions)))
            x = self.res1(x)
            x = self.res2(x)
            x = self.res3(x)
            x = F.leaky_relu(self.bn_final(self.conv_final(x)))
        else:
            x = F.leaky_relu(self.bn1(self.conv1(positions)))
            x = self.pool(x)
            x = F.leaky_relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = F.leaky_relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            x = F.leaky_relu(self.bn4(self.conv4(x)))
            x = F.leaky_relu(self.bn5(self.conv5(x)))
            x = F.leaky_relu(self.bn6(self.conv6(x)))
        
        x = self.dropout_cnn(x)
        x = x.view(batch_size, sequence_length, -1)
        
        clocks = clocks.unsqueeze(2)
        lstm_input = torch.cat((x, clocks), dim=2)
        
        packed_input = pack_padded_sequence(lstm_input, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        lstm_output = self.layer_norm(lstm_output)
        
        y = F.leaky_relu(self.fc1(lstm_output))
        y = self.dropout_lstm(y)
        y = self.fc2(y)

        idx = torch.arange(batch_size)
        last_time_step_output = y[idx, lengths - 1, :]
        return y, last_time_step_output



class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization for residual connection
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        
        batch_size, seq_len, _ = x.shape

        attn_mask = None
        if lengths is not None:

            attn_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
            for i, length in enumerate(lengths):
                if length < seq_len:
                    attn_mask[i, length:] = True


            attn_mask = attn_mask.unsqueeze(1).expand(-1, seq_len, -1)
            attn_mask = attn_mask.repeat(self.num_heads, 1, 1)


        attn_output, attn_weights = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False 
        )


        attn_output = self.dropout(attn_output)
        output = self.layer_norm(x + attn_output)

        return output, attn_weights


class MoveImportanceAttention(nn.Module):
    
    def __init__(self, hidden_size, dropout=0.1):
        super(MoveImportanceAttention, self).__init__()


        self.importance_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        
        batch_size, seq_len, hidden_size = x.shape


        importance_scores = self.importance_net(x)


        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
        for i, length in enumerate(lengths):
            if length < seq_len:
                mask[i, length:] = True


        importance_scores = importance_scores.squeeze(-1)
        importance_scores = importance_scores.masked_fill(mask, -1e4)

        importance_weights = torch.softmax(importance_scores, dim=1)

        weighted_output = torch.sum(
            importance_weights.unsqueeze(-1) * x,
            dim=1
        )

        weighted_output = self.dropout(weighted_output)

        return weighted_output, importance_weights


class AttentionChessEloPredictor(nn.Module):
   
    def __init__(self, conv_filters=32, lstm_layers=4, dropout_rate=0.5, lstm_h=128,
                 fc1_h=64, bidirectional=True, use_residual=True,
                 num_attention_heads=8, use_move_importance=True):
        super(AttentionChessEloPredictor, self).__init__()

        self.use_residual = use_residual
        self.use_move_importance = use_move_importance

        if use_residual:
            self.conv1 = nn.Conv2d(12, conv_filters, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(conv_filters)

            self.res1 = ResidualBlock(conv_filters, conv_filters * 2, stride=2)
            self.res2 = ResidualBlock(conv_filters * 2, conv_filters * 4, stride=2)
            self.res3 = ResidualBlock(conv_filters * 4, conv_filters * 8, stride=2)

            self.conv_final = nn.Conv2d(conv_filters * 8, conv_filters * 8, kernel_size=3, padding=1)
            self.bn_final = nn.BatchNorm2d(conv_filters * 8)

        else:
            self.conv1 = nn.Conv2d(12, conv_filters, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(conv_filters)
            self.conv2 = nn.Conv2d(conv_filters, conv_filters * 2, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(conv_filters * 2)
            self.conv3 = nn.Conv2d(conv_filters * 2, conv_filters * 4, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(conv_filters * 4)
            self.conv4 = nn.Conv2d(conv_filters * 4, conv_filters * 8, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(conv_filters * 8)
            self.conv5 = nn.Conv2d(conv_filters * 8, conv_filters * 8, kernel_size=3, padding=1)
            self.bn5 = nn.BatchNorm2d(conv_filters * 8)
            self.conv6 = nn.Conv2d(conv_filters * 8, conv_filters * 8, kernel_size=3, padding=1)
            self.bn6 = nn.BatchNorm2d(conv_filters * 8)

        self.pool = nn.AvgPool2d(2, 2)
        self.dropout_cnn = nn.Dropout(dropout_rate * 0.5)
        self.dropout_lstm = nn.Dropout(dropout_rate)

        self.lstm = nn.LSTM(
            input_size=conv_filters * 8 + 1, 
            hidden_size=lstm_h,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if lstm_layers > 1 else 0
        )

        lstm_output_size = lstm_h * 2 if bidirectional else lstm_h

        self.self_attention = MultiHeadSelfAttention(
            hidden_size=lstm_output_size,
            num_heads=num_attention_heads,
            dropout=dropout_rate * 0.5
        )

        if use_move_importance:
            self.move_importance = MoveImportanceAttention(
                hidden_size=lstm_output_size,
                dropout=dropout_rate * 0.5
            )

        self.layer_norm = nn.LayerNorm(lstm_output_size)

        self.fc1 = nn.Linear(lstm_output_size, fc1_h)
        self.fc2 = nn.Linear(fc1_h, 2)

        if use_move_importance:
            self.fc1_importance = nn.Linear(lstm_output_size, fc1_h)
            self.fc2_importance = nn.Linear(fc1_h, 2)

    def forward(self, positions, clocks, lengths):
        
        batch_size = positions.size(0)
        sequence_length = positions.size(1)

        positions = positions.view(-1, 12, 8, 8)

        if self.use_residual:
            x = F.leaky_relu(self.bn1(self.conv1(positions)))
            x = self.res1(x)
            x = self.res2(x)
            x = self.res3(x)
            x = F.leaky_relu(self.bn_final(self.conv_final(x)))
        else:
            x = F.leaky_relu(self.bn1(self.conv1(positions)))
            x = self.pool(x)
            x = F.leaky_relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = F.leaky_relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            x = F.leaky_relu(self.bn4(self.conv4(x)))
            x = F.leaky_relu(self.bn5(self.conv5(x)))
            x = F.leaky_relu(self.bn6(self.conv6(x)))

        x = self.dropout_cnn(x)
        x = x.view(batch_size, sequence_length, -1)

        clocks = clocks.unsqueeze(2)
        lstm_input = torch.cat((x, clocks), dim=2)

        packed_input = pack_padded_sequence(lstm_input, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        lstm_output = self.layer_norm(lstm_output)

        attended_output, self_attn_weights = self.self_attention(lstm_output, lengths)

        y = F.leaky_relu(self.fc1(attended_output))
        y = self.dropout_lstm(y)
        y = self.fc2(y)

        idx = torch.arange(batch_size, device=positions.device)
        last_time_step_output = y[idx, lengths - 1, :]

  
        attention_weights_dict = {'self_attention': self_attn_weights}

        if self.use_move_importance:
     
            weighted_features, importance_weights = self.move_importance(attended_output, lengths)
            attention_weights_dict['move_importance'] = importance_weights

        
            importance_pred = F.leaky_relu(self.fc1_importance(weighted_features))
            importance_pred = self.dropout_lstm(importance_pred)
            importance_pred = self.fc2_importance(importance_pred)

        
            final_output = 0.5 * last_time_step_output + 0.5 * importance_pred
        else:
            final_output = last_time_step_output

        return y, final_output, attention_weights_dict




class ModelEMA:
    
    def __init__(self, model, decay=0.999, device=None):
        self.model = model
        self.decay = decay
        self.device = device if device is not None else torch.device('cpu')

        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)

    def update(self, model=None):
        
        if model is not None:
            self.model = model

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow, f"Parameter {name} not in shadow dict"
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self):
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup, f"Parameter {name} not in backup dict"
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {
            'decay': self.decay,
            'shadow': self.shadow
        }

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']


class ModelEnsemble:
    
    def __init__(self, models, device):
        self.models = models
        self.device = device

        for model in self.models:
            model.eval()
            model.to(device)

        print(f"Ensemble created with {len(models)} models")

    def predict(self, positions, clocks, lengths):
        
        with torch.no_grad():
            all_predictions = []
            last_predictions = []

            for model in self.models:
                all_out, last_out = model(positions, clocks, lengths)
                all_predictions.append(all_out)
                last_predictions.append(last_out)


            avg_all_outputs = torch.stack(all_predictions).mean(dim=0)
            avg_last_outputs = torch.stack(last_predictions).mean(dim=0)

            return avg_all_outputs, avg_last_outputs

    @staticmethod
    def load_ensemble(model_paths, model_class, model_params, device):
        
        models = []
        for path in model_paths:
            model = model_class(**model_params).to(device)

            if os.path.exists(path):
                checkpoint = torch.load(path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from {path}")
            else:
                print(f"Warning: Model path {path} does not exist, skipping")
                continue

            models.append(model)

        if len(models) == 0:
            raise ValueError("No models were successfully loaded for ensemble")

        return ModelEnsemble(models, device)




def train_one_epoch(model, train_loader, device, criterion, optimizer, ratings_mean=1514,
                    ratings_std=366, use_mixed_precision=True, ema=None, use_attention=False,
                    gradient_clip_value=None):

    model.train()
    total_train_loss = 0
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and device.type == 'cuda' else None

    for batch in train_loader:
        positions = batch['positions'].to(device)
        clocks = batch['clocks'].to(device)
        targets = batch['targets'].to(device)
        lengths = batch['lengths']

        optimizer.zero_grad()

        if use_mixed_precision and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                model_output = model(positions, clocks, lengths)
                if use_attention:
                    all_outputs, outputs, attention_weights = model_output
                else:
                    all_outputs, outputs = model_output
                loss = criterion(outputs * ratings_std + ratings_mean, targets * ratings_std + ratings_mean)

            scaler.scale(loss).backward()
            if gradient_clip_value is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
            scaler.step(optimizer)
            scaler.update()
        else:
            model_output = model(positions, clocks, lengths)
            if use_attention:
                all_outputs, outputs, attention_weights = model_output
            else:
                all_outputs, outputs = model_output
            loss = criterion(outputs * ratings_std + ratings_mean, targets * ratings_std + ratings_mean)
            loss.backward()
            if gradient_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        total_train_loss += loss.item()

    return total_train_loss / len(train_loader)


def validate(model, val_loader, device, criterion, ratings_mean=1514, ratings_std=366, use_attention=False):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            positions = batch['positions'].to(device)
            clocks = batch['clocks'].to(device)
            targets = batch['targets'].to(device)
            lengths = batch['lengths']
            model_output = model(positions, clocks, lengths)
            if use_attention:
                all_outputs, outputs, attention_weights = model_output
            else:
                all_outputs, outputs = model_output
            loss = criterion(outputs * ratings_std + ratings_mean, targets * ratings_std + ratings_mean)
            total_val_loss += loss.item()
    return total_val_loss / len(val_loader)


def mae_per_item(outputs, targets, ratings_mean, ratings_std):
    outputs_rescaled = outputs * ratings_std + ratings_mean
    targets_rescaled = targets * ratings_std + ratings_mean
    mae = torch.abs(outputs_rescaled - targets_rescaled)
    return mae.mean(dim=1)


def mse_per_item(outputs, targets, ratings_mean, ratings_std):
    outputs_rescaled = outputs * ratings_std + ratings_mean
    targets_rescaled = targets * ratings_std + ratings_mean
    mse = (outputs_rescaled - targets_rescaled) ** 2
    return mse.mean(dim=1)


def test(model, test_loader, device, criterion, ratings_mean=1514, ratings_std=366, use_attention=False):
    model.eval()
    total_test_loss = 0
    loss_by_time_control = {'ultrabullet': [], 'bullet': [], 'blitz': [], 'rapid': [], 'classical': []}
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            positions = batch['positions'].to(device)
            clocks = batch['clocks'].to(device)
            targets = batch['targets'].to(device)
            lengths = batch['lengths']
            time_controls = batch['time_controls']

            model_output = model(positions, clocks, lengths)
            
            if use_attention:
                all_outputs, outputs, attention_weights = model_output
            else:
                all_outputs, outputs = model_output
            loss = criterion(outputs * ratings_std + ratings_mean, targets * ratings_std + ratings_mean)
            total_test_loss += loss.item()
            
           
            outputs_rescaled = outputs * ratings_std + ratings_mean
            targets_rescaled = targets * ratings_std + ratings_mean
            
        
            all_predictions.extend(outputs_rescaled.cpu().numpy())
            all_targets.extend(targets_rescaled.cpu().numpy())
            
        
            mae = mae_per_item(outputs, targets, ratings_mean, ratings_std)
            
            for idx, time_control in enumerate(time_controls):
                loss_by_time_control[time_control].append(mae[idx].item())

    avg_loss_by_time_control = {}
    for key in loss_by_time_control:
        if len(loss_by_time_control[key]) > 0:
            avg_loss_by_time_control[key] = np.mean(loss_by_time_control[key])
        else:
            avg_loss_by_time_control[key] = 0
    
    return (total_test_loss / len(test_loader), avg_loss_by_time_control, 
            np.array(all_predictions), np.array(all_targets))


def plot_training_curves(train_losses, val_losses, save_path='training_curves.png'):

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MAE)', fontsize=12)
    plt.title('Training and Validation Loss Over Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_time_control_comparison(baseline_results, improved_results, save_path='time_control_comparison.png'):

    time_controls = list(baseline_results.keys())
    baseline_values = [baseline_results[tc] for tc in time_controls]
    improved_values = [improved_results[tc] for tc in time_controls]
    
    x = np.arange(len(time_controls))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, baseline_values, width, label='Baseline', color='#3498db', alpha=0.8)
    plt.bar(x + width/2, improved_values, width, label='Improved Model', color='#e74c3c', alpha=0.8)
    
    plt.xlabel('Time Control', fontsize=12)
    plt.ylabel('Mean Absolute Error (Elo)', fontsize=12)
    plt.title('Model Performance Across Time Controls', fontsize=14, fontweight='bold')
    plt.xticks(x, time_controls, rotation=15)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Time control comparison saved to {save_path}")


def plot_prediction_scatter(predictions, targets, save_path='prediction_scatter.png'):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    

    ax1.scatter(targets[:, 0], predictions[:, 0], alpha=0.3, s=10, color='#3498db')
    ax1.plot([targets[:, 0].min(), targets[:, 0].max()], 
             [targets[:, 0].min(), targets[:, 0].max()], 
             'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Rating (White)', fontsize=11)
    ax1.set_ylabel('Predicted Rating (White)', fontsize=11)
    ax1.set_title('White Player Rating Predictions', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    

    ax2.scatter(targets[:, 1], predictions[:, 1], alpha=0.3, s=10, color='#e74c3c')
    ax2.plot([targets[:, 1].min(), targets[:, 1].max()], 
             [targets[:, 1].min(), targets[:, 1].max()], 
             'r--', lw=2, label='Perfect Prediction')
    ax2.set_xlabel('Actual Rating (Black)', fontsize=11)
    ax2.set_ylabel('Predicted Rating (Black)', fontsize=11)
    ax2.set_title('Black Player Rating Predictions', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Prediction scatter plot saved to {save_path}")


def plot_error_distribution(predictions, targets, save_path='error_distribution.png'):
   
    errors = predictions - targets
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
 
    ax1.hist(errors[:, 0], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax1.set_xlabel('Prediction Error (Elo)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title(f'White Player Error Distribution\nMean: {errors[:, 0].mean():.2f}, Std: {errors[:, 0].std():.2f}', 
                  fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    

    ax2.hist(errors[:, 1], bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax2.set_xlabel('Prediction Error (Elo)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title(f'Black Player Error Distribution\nMean: {errors[:, 1].mean():.2f}, Std: {errors[:, 1].std():.2f}', 
                  fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Error distribution saved to {save_path}")


def save_results_summary(results, save_path='results_summary.json'):

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results summary saved to {save_path}")


def plot_learning_rate_schedule(learning_rates, save_path='learning_rate_schedule.png'):
  
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(learning_rates) + 1)

    plt.plot(epochs, learning_rates, 'g-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log') 
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Learning rate schedule saved to {save_path}")


def plot_epoch_durations(epoch_durations, save_path='epoch_durations.png'):
    
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(epoch_durations) + 1)

    plt.bar(epochs, epoch_durations, color='#9b59b6', alpha=0.7, edgecolor='black')
    plt.axhline(y=np.mean(epoch_durations), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(epoch_durations):.2f} min')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Duration (minutes)', fontsize=12)
    plt.title('Training Time per Epoch', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Epoch durations plot saved to {save_path}")


def plot_detailed_training_metrics(train_losses, val_losses, learning_rates,
                                   epoch_durations, save_path='detailed_metrics.png'):
 
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    epochs = range(1, len(train_losses) + 1)


    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss (MAE)', fontsize=11)
    ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)


    ax2.plot(epochs, learning_rates, 'g-', linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Learning Rate', fontsize=11)
    ax2.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)


    ax3.bar(epochs, epoch_durations, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax3.axhline(y=np.mean(epoch_durations), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(epoch_durations):.2f} min')
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Duration (minutes)', fontsize=11)
    ax3.set_title('Training Time per Epoch', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')


    loss_gap = [val - train for train, val in zip(train_losses, val_losses)]
    ax4.plot(epochs, loss_gap, 'm-', linewidth=2, marker='o', markersize=4)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.fill_between(epochs, loss_gap, 0, where=[g >= 0 for g in loss_gap],
                     color='red', alpha=0.3, label='Overfitting')
    ax4.fill_between(epochs, loss_gap, 0, where=[g < 0 for g in loss_gap],
                     color='green', alpha=0.3, label='Underfitting')
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Val Loss - Train Loss', fontsize=11)
    ax4.set_title('Generalization Gap (Overfitting Indicator)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Detailed metrics plot saved to {save_path}")


def save_epoch_metrics(train_losses, val_losses, learning_rates, epoch_durations,
                      best_epoch, save_path='epoch_metrics.json'):
  
    metrics = {
        'epochs': list(range(1, len(train_losses) + 1)),
        'train_losses': [float(loss) for loss in train_losses],
        'val_losses': [float(loss) for loss in val_losses],
        'learning_rates': [float(lr) for lr in learning_rates],
        'epoch_durations_minutes': [float(dur) for dur in epoch_durations],
        'best_epoch': int(best_epoch + 1),
        'best_val_loss': float(min(val_losses)),
        'final_train_loss': float(train_losses[-1]) if train_losses else None,
        'final_val_loss': float(val_losses[-1]) if val_losses else None,
        'total_training_time_minutes': float(sum(epoch_durations)),
        'avg_epoch_duration_minutes': float(np.mean(epoch_durations)) if epoch_durations else None
    }

    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Epoch metrics saved to {save_path}")


def main():

    data_dir = "data/processed_games_10gb"
    experiment_name = "improved_model_residual"
    train_model = True
    use_mixed_precision = True
    

    model_type = "attention"

    criterion = nn.L1Loss()

    #
    params = {
        'train_batch_size': 8,
        'val_batch_size': 128,
        'num_workers': 4,
        'learning_rate': 0.0001,
        'weight_decay': 1e-4,  
        'epochs': 10,
        'optimizer': 'Adam',
        'patience': 7,
        'lr_factor': 0.5,
        'early_stopping_patience': 15, 
        'gradient_clip_value': 1.0,  
        "conv_filters": 32,
        "lstm_layers": 5, 
        "bidirectional": True,
        "dropout_rate": 0.4, 
        "lstm_h": 128, 
        "fc1_h": 64, 
        "use_residual": True,
        "data_augmentation": True,
        "use_cosine_scheduler": True,
        "model_type": model_type,
        
        "use_attention": model_type == "attention",
        "num_attention_heads": 8,  
        "use_move_importance": True,  
    }
    
    
    model_dir = os.path.join('models', experiment_name)
    os.makedirs(model_dir, exist_ok=True)
    log_dir = os.path.join('runs', experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    results_dir = os.path.join('results', experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    
    
    print("Loading data...")
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pkl')]
    print(f"Total files: {len(all_files)}")
    
    
    train_val_files, test_files = train_test_split(all_files, test_size=0.1, random_state=42)
    train_files, val_files = train_test_split(train_val_files, test_size=0.2, random_state=42)
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    
    train_dataset = ChessGamesDataset(train_files, augment=params['data_augmentation'])
    val_dataset = ChessGamesDataset(val_files, augment=False)
    test_dataset = ChessGamesDataset(test_files, augment=False)
    
   
    train_loader = DataLoader(train_dataset, batch_size=params["train_batch_size"], 
                             shuffle=True, collate_fn=collate_fn, num_workers=params['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=params['val_batch_size'], 
                           shuffle=False, collate_fn=collate_fn, num_workers=params['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=params['val_batch_size'], 
                            shuffle=False, collate_fn=collate_fn, num_workers=params['num_workers'])
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if model_type == "attention":
        model = AttentionChessEloPredictor(
            params["conv_filters"], params["lstm_layers"], params["dropout_rate"],
            params["lstm_h"], params["fc1_h"], params["bidirectional"],
            params["use_residual"], params["num_attention_heads"],
            params["use_move_importance"]
        ).to(device)
        print("Using Attention Model with Multi-Head Self-Attention and Move Importance")
    elif model_type == "improved":
        model = ImprovedChessEloPredictor(
            params["conv_filters"], params["lstm_layers"], params["dropout_rate"],
            params["lstm_h"], params["fc1_h"], params["bidirectional"],
            params["use_residual"]
        ).to(device)
        print("Using Improved Model with Residual Connections")
    else:
     
        from chess_rating_net import ChessEloPredictor
        model = ChessEloPredictor(
            params["conv_filters"], params["lstm_layers"], params["dropout_rate"],
            params["lstm_h"], params["fc1_h"], params["bidirectional"]
        ).to(device)
        print("Using Baseline Model")
    

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], 
                                weight_decay=params['weight_decay'])
    

    if params['use_cosine_scheduler']:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        print("Using Cosine Annealing Warm Restarts scheduler")
    else:
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=params['patience'], 
                                     factor=params['lr_factor'])
        print("Using ReduceLROnPlateau scheduler")
    
    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []
    learning_rates = []
    epoch_durations = []
    epochs_without_improvement = 0 


    epoch_log_path = os.path.join(results_dir, 'training_log.csv')
    with open(epoch_log_path, 'w') as f:
        f.write('epoch,train_loss,val_loss,learning_rate,duration_min,best_val_loss,is_best\n')

    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    start = time.time()

    if train_model:
        for epoch in range(params['epochs']):
            epoch_start = time.time()

            train_loss = train_one_epoch(model, train_loader, device, criterion, optimizer,
                                        use_mixed_precision=use_mixed_precision,
                                        use_attention=params['use_attention'],
                                        gradient_clip_value=params['gradient_clip_value'])
            val_loss = validate(model, val_loader, device, criterion,
                              use_attention=params['use_attention'])

            train_losses.append(train_loss)
            val_losses.append(val_loss)

  
            if params['use_cosine_scheduler']:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
            else:
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']

            learning_rates.append(current_lr)

            epoch_duration = (time.time() - epoch_start) / 60
            epoch_durations.append(epoch_duration)

            print(f'Epoch {epoch + 1}/{params["epochs"]} | '
                  f'Train Loss: {train_loss:.4f} | '
                  f'Val Loss: {val_loss:.4f} | '
                  f'LR: {current_lr:.6f} | '
                  f'Time: {epoch_duration:.2f} min')

   
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
            writer.add_scalar('Timing/Epoch_Duration', epoch_duration, epoch)


            is_best = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                is_best = True
                epochs_without_improvement = 0
                best_path = os.path.join(model_dir, f'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'params': params
                }, best_path)
                print(f'✓ Saved best model (Val Loss: {val_loss:.4f})')
            else:
                epochs_without_improvement += 1
                print(f'  No improvement for {epochs_without_improvement} epoch(s)')


            with open(epoch_log_path, 'a') as f:
                f.write(f'{epoch+1},{train_loss:.6f},{val_loss:.6f},{current_lr:.8f},'
                       f'{epoch_duration:.4f},{best_val_loss:.6f},{int(is_best)}\n')


            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(model_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'params': params
                }, checkpoint_path)

   
            if epochs_without_improvement >= params['early_stopping_patience']:
                print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
                print(f"  No improvement for {epochs_without_improvement} consecutive epochs")
                print(f"  Best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")
                break
        
        training_duration = (time.time() - start) / 60
        print(f"\n{'='*50}")
        print(f"Training Complete!")
        print(f"Total Duration: {training_duration:.2f} minutes")
        print(f"Best Val Loss: {best_val_loss:.4f} at Epoch {best_epoch + 1}")
        print(f"{'='*50}\n")
        
        writer.close()


        print("\nGenerating training visualizations...")


        plot_training_curves(train_losses, val_losses,
                           os.path.join(results_dir, 'training_curves.png'))


        plot_learning_rate_schedule(learning_rates,
                                    os.path.join(results_dir, 'learning_rate_schedule.png'))

        plot_epoch_durations(epoch_durations,
                            os.path.join(results_dir, 'epoch_durations.png'))

        plot_detailed_training_metrics(train_losses, val_losses, learning_rates, epoch_durations,
                                       os.path.join(results_dir, 'detailed_metrics.png'))

        save_epoch_metrics(train_losses, val_losses, learning_rates, epoch_durations,
                          best_epoch, os.path.join(results_dir, 'epoch_metrics.json'))

    best_path = os.path.join(model_dir, 'best_model.pth')
    if os.path.exists(best_path):
        print(f"Loading best model from {best_path}")
        checkpoint = torch.load(best_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    

    print("\n" + "="*50)
    print("Testing Model")
    print("="*50)
    test_loss, loss_by_time_control, predictions, targets = test(model, test_loader, device, criterion,
                                                                  use_attention=params['use_attention'])
    
    print(f"\nOverall Test Loss (MAE): {test_loss:.4f}")
    print("\nLoss by Time Control:")
    for tc, loss in loss_by_time_control.items():
        print(f"  {tc:15s}: {loss:.4f}")
    

    print("\nGenerating visualizations...")
    plot_prediction_scatter(predictions, targets, 
                           os.path.join(results_dir, 'prediction_scatter.png'))
    plot_error_distribution(predictions, targets, 
                          os.path.join(results_dir, 'error_distribution.png'))
    

    results_summary = {
        'model_type': model_type,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'test_loss': float(test_loss),
        'loss_by_time_control': {k: float(v) for k, v in loss_by_time_control.items()},
        'best_val_loss': float(best_val_loss),
        'best_epoch': int(best_epoch + 1),
        'training_duration_minutes': float(training_duration) if train_model else 0,
        'hyperparameters': params
    }
    
    save_results_summary(results_summary, 
                        os.path.join(results_dir, 'results_summary.json'))
    
    print(f"\n{'='*50}")
    print("All results saved to:", results_dir)
    print(f"{'='*50}")
    
    return results_summary


if __name__ == "__main__":
    main()