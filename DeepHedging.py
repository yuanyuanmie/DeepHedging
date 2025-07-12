# 在文件最顶部添加
from collections import defaultdict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time
import os
from scipy.stats import norm
import scipy.optimize as optimize
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Set random seed to ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Check if GPU is available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(f"Using device: {device}")

# Heston model parameters
class HestonParams:
    def __init__(self):
        self.S0 = 100.0       # Initial stock price
        self.V0 = 0.04        # Initial volatility
        self.kappa = 2.0      # Mean reversion speed of volatility
        self.theta = 0.04     # Long-term mean of volatility
        self.sigma = 0.3      # Volatility of volatility
        self.rho = -0.7       # Correlation between price and volatility
        self.r = 0.01         # Risk-free rate
        self.T = 1.0          # Option maturity (years)
        self.K = 100.0        # Option strike price
        self.trading_days = 253  # Trading days per year
        self.Kv = 0.04        # Variance Swap' strike price
        self.c = 0.003        # Transaction cost rate 
# # Simulate Heston model
# def simulate_heston(params, n_paths, n_steps):
#     dt = params.T / n_steps
#     sqrt_dt = np.sqrt(dt)
    
#     # Initialize paths
#     S = np.zeros((n_paths, n_steps + 1))
#     V = np.zeros((n_paths, n_steps + 1))
#     integrated_var = np.zeros((n_paths, n_steps + 1))
    
#     # Set initial values
#     S[:, 0] = params.S0
#     V[:, 0] = params.V0
    
#     # Generate correlated random numbers using PyTorch
#     Z1 = torch.randn(n_paths, n_steps).numpy()
#     Z2 = params.rho * Z1 + np.sqrt(1 - params.rho**2) * torch.randn(n_paths, n_steps).numpy()
    
#     # Simulate paths
#     for t in range(n_steps):
#         # Ensure volatility is positive
#         V[:, t] = np.maximum(V[:, t], 0)
        
#         # Update stock price
#         S[:, t+1] = S[:, t] * np.exp((params.r - 0.5 * V[:, t]) * dt + np.sqrt(V[:, t]) * sqrt_dt * Z1[:, t])
        
#         # Update volatility
#         V[:, t+1] = V[:, t] + params.kappa * (params.theta - V[:, t]) * dt + params.sigma * np.sqrt(V[:, t]) * sqrt_dt * Z2[:, t]
        
#         # Calculate integrated variance
#         integrated_var[:, t+1] = integrated_var[:, t] + V[:, t] * dt
    
#     return S, V, integrated_var
def simulate_heston(params, n_paths, n_steps):
    dt = params.T / n_steps
    sqrt_dt = torch.sqrt(torch.tensor(dt, device=device))
    
    S = torch.zeros((n_paths, n_steps + 1), device=device)
    V = torch.zeros((n_paths, n_steps + 1), device=device)
    integrated_var = torch.zeros((n_paths, n_steps + 1), device=device)
    
    S[:, 0] = params.S0
    V[:, 0] = params.V0
    
    Z1 = torch.randn(n_paths, n_steps, device=device)
    Z2 = params.rho * Z1 + torch.sqrt(1 - torch.tensor(params.rho**2, device=device)) * torch.randn(n_paths, n_steps, device=device)
    
    for t in range(n_steps):
        V[:, t] = torch.clamp(V[:, t], min=0)
        S[:, t+1] = S[:, t] * torch.exp((params.r - 0.5 * V[:, t]) * dt + torch.sqrt(V[:, t]) * sqrt_dt * Z1[:, t])
        V[:, t+1] = V[:, t] + params.kappa * (params.theta - V[:, t]) * dt + params.sigma * torch.sqrt(V[:, t]) * sqrt_dt * Z2[:, t]
        integrated_var[:, t+1] = integrated_var[:, t] + V[:, t] * dt
    
    return S, V, integrated_var



# Calculate European put option payoff
def put_option_payoff(S, K):
    if isinstance(S, torch.Tensor):
        return torch.maximum(torch.tensor(K, device=S.device) - S, torch.tensor(0.0, device=S.device))
    else:
        return np.maximum(K - S, 0)

# # Calculate variance swap value based on formula (31)
# def calculate_variance_swap(integrated_var_t, V_t, t, T, params):
#     """
#     Calculate variance swap value based on S_t^2 = ∫_0^t V_s ds + L(t, V_t)
#     where L(t,v) = (v-h)/α * (1-e^(-α(T-t))) + h(T-t)
    
#     Parameters:
#     integrated_var_t - Realized variance from 0 to t
#     V_t - Current instantaneous variance
#     t - Current time
#     T - Maturity
#     params - Model parameters
    
#     Returns:
#     variance_swap_value - Variance swap value


#     20250410 Modified by shenys
#     方差互换的支付应该改 variance_swap_value - K_v 
    
#     """
#     # Calculate L(t, V_t) term
#     h = params.theta      # Long-term variance mean
#     alpha = params.kappa  # Mean reversion rate
    
#     remaining_T = T - t
#     L_t_v = (V_t - h) / alpha * (1 - np.exp(-alpha * remaining_T)) + h * remaining_T
    
#     # Variance swap value = realized variance + expected future variance
#     variance_swap_value = integrated_var_t + L_t_v

#     # shenys modify:
#     # variance_swap_value 价值计算 需要
#     variance_swap_value - params.Kv
#     return variance_swap_value

def calculate_variance_swap(integrated_var_t, V_t, t, T, params):
    h = torch.tensor(params.theta, device=integrated_var_t.device)
    alpha = torch.tensor(params.kappa, device=integrated_var_t.device)
    remaining_T = torch.tensor(T - t, device=integrated_var_t.device)
    Kv = torch.tensor(params.Kv, device=integrated_var_t.device)
    
    exp_term = torch.exp(-alpha * remaining_T)
    L_t_v = (V_t - h) / alpha * (1 - exp_term) + h * remaining_T
    variance_swap_value = integrated_var_t + L_t_v - Kv
    return variance_swap_value


# Calculate sensitivity of variance swap to instantaneous variance
def variance_swap_sensitivity(V_t, t, T, params):
    """
    Calculate ∂L(t,V_t)/∂V_t for hedging purposes
    
    Parameters:
    V_t - Current instantaneous variance
    t - Current time
    T - Maturity
    params - Model parameters
    
    Returns:
    sensitivity - Partial derivative of L with respect to V
    """
    alpha = params.kappa
    remaining_T = T - t
    
    # From the formula: ∂L(t,v)/∂v = 1/α * (1-e^(-α(T-t)))
    sensitivity = (1 - np.exp(-alpha * remaining_T)) / alpha
    
    return sensitivity

class DeepHedgingTransformerLSTM(nn.Module):
    def __init__(self, n_features, hidden_dim=64, nhead=4, num_layers=2, seq_len=10):
        super().__init__()
        self.seq_len = seq_len  # 输入序列长度（时间窗口）
        self.position_memory = {}  # 改为普通字典存储头寸
        
        # 输入嵌入层（适配时序数据）
        self.embedding = nn.Linear(n_features, hidden_dim)
        
        # LSTM模块
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        # Transformer编码器
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=0.1)
        encoder_layers = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim*4,
            dropout=0.1
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        # 输出解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 2)
        )
        
        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
                
    def forward(self, x):
        """
        x shape: (batch_size, seq_len, n_features)
        """
        # 嵌入层
        x = self.embedding(x)  # (B, S, H)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)  # (B, S, H)
        
        # 位置编码
        x = self.pos_encoder(lstm_out)
        
        # Transformer处理
        x = x.permute(1, 0, 2)  # (S, B, H)
        transformer_out = self.transformer_encoder(x)
        last_step = transformer_out[-1]  # 取最后时间步 (B, H)
        
        # 解码输出
        return self.decoder(last_step)
        
    def get_batch_positions(self, indices):
        """批量获取当前头寸状态"""
        return torch.stack([self.position_memory[idx] for idx in indices])

    def update_batch_positions(self, indices, new_positions):
        """批量更新头寸状态"""
        for idx, pos in zip(indices, new_positions):
            self.position_memory[idx] = pos        

    def update_position_memory(self, batch_indices, outputs, timesteps):
        """更新头寸状态 (简化参数传递)"""
        # outputs形状: (batch_size, 2)
        deltas = outputs[:, 0]  # 股票头寸变化
        var_swaps = outputs[:, 1]  # 方差互换头寸变化
        
        # 假设batch_indices是元组 (path_ids, timesteps)
        path_ids, step_ids = batch_indices  

        
        
        for pid, t, d, v in zip(path_ids, step_ids, deltas, var_swaps):
            key = f"path{pid}_step{t}"
            self.position_memory[key] = {
                'delta': float(d.item()),
                'var_swap': float(v.item())
            }
    def get_previous_positions(self, path_ids: list, prev_timesteps: list) -> torch.Tensor:
        """批量获取前一头寸"""
        positions = []
        for pid, t in zip(path_ids, prev_timesteps):
            key = f"path{pid}_step{t}"
            if key in self.position_memory:
                pos = self.position_memory[key]
            else:  # 处理初始状态
                pos = {'delta': 0.0, 'var_swap': 0.0}
            positions.append([pos['delta'], pos['var_swap']])
        return torch.tensor(positions, device=self.device)            


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

class PositionMemory:
    def __init__(self):
        self.memory = defaultdict(lambda: torch.zeros(2))  # 存储(delta, var_swap)
    
    def __getitem__(self, key):
        return self.memory[key]
    
    def __setitem__(self, key, value):
        self.memory[key] = value    


# Calculate VaR according to traditional definition (positive value)
def compute_var(portfolio_values, alpha=0.05):
    """
    Calculate Value at Risk (VaR) as a positive value
    
    VaR_alpha(X) = inf{m | P(X < -m) <= alpha}
    
    Parameters:
    portfolio_values - Portfolio P&L values (positive = profit, negative = loss)
    alpha - Confidence level (e.g., 0.05 for 95% confidence)
    
    Returns:
    VaR as a positive value
    """
    # Sort portfolio values in ascending order
    sorted_values, _ = torch.sort(portfolio_values)
    
    # Determine threshold index
    n = sorted_values.size(0)
    alpha_index = int(torch.ceil(torch.tensor(alpha * n))) - 1
    alpha_index = torch.max(torch.tensor(0), torch.tensor(alpha_index))
    
    # Get the value at the threshold index 
    var_value = sorted_values[alpha_index]
    
    # VaR is the negative of this value (to make losses positive)
    # but if var_value is positive (profit), VaR should be 0
    return torch.max(torch.tensor(0.0, device=portfolio_values.device), -var_value)

# Calculate CVaR (Expected Shortfall) - revised to be positive for losses
def compute_cvar(portfolio_values, alpha=0.05):
    """
    Calculate Conditional Value at Risk (CVaR) as a positive value
    
    CVaR_alpha(X) = 1/alpha * E[X | X <= -VaR_alpha(X)]
    
    Parameters:
    portfolio_values - Portfolio P&L values (positive = profit, negative = loss)
    alpha - Confidence level (e.g., 0.05 for 95% confidence)
    
    Returns:
    CVaR as a positive value
    """
    # Sort portfolio values in ascending order
    sorted_values, _ = torch.sort(portfolio_values)
    
    # Determine threshold index
    n = sorted_values.size(0)
    alpha_index = int(torch.ceil(torch.tensor(alpha * n)))
    alpha_index = torch.max(torch.tensor(1), torch.tensor(alpha_index))
    
    # Extract values below threshold
    tail_values = sorted_values[:alpha_index]
    
    # Calculate CVaR as the negative of the mean (to make losses positive)
    # If mean is positive (average profit in worst cases), CVaR should be 0
    return torch.max(torch.tensor(0.0, device=portfolio_values.device), -torch.mean(tail_values))

# Calculate trading costs
def calculate_trading_cost(trades, cost_factor=0.003):
    # Calculate cost of each trade, proportional to the trade volume
    return cost_factor * torch.sum(torch.abs(trades))


"""
Modified by shenys

"""
def train_deep_hedging_model(model, params, n_paths, n_steps, epochs=100, batch_size=64, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.7, verbose=True)
    
    # 新增：定义序列长度（需与模型初始化参数一致）
    seq_len = model.seq_len  # 从模型属性获取序列长度

    
    # 数据预处理函数：将路径数据转换为序列样本
    def create_sequences(S, V, integrated_var):
        # 从数据维度获取实际步数
        n_paths, total_steps = S.shape
        seq_len = model.seq_len  # 从模型获取序列长度
        
        # 计算有效时间步范围（避免越界）
        num_samples = n_paths * (total_steps - seq_len + 1)
        sequences = torch.zeros((num_samples, seq_len, 4), device=device)
        targets = torch.zeros((num_samples, 2), device=device)
        time_indices = []  # 存储每个样本的时间步信息
        sample_idx = 0
        for path in range(n_paths):
            # 正确的时间步范围应为 total_steps - seq_len
            for t in range(total_steps - seq_len + 1):
                # 提取序列特征 [价格, 波动率, 已实现方差, 标准化时间]
                sequences[sample_idx] = torch.stack([
                    S[path, t:t+seq_len],
                    V[path, t:t+seq_len],
                    integrated_var[path, t:t+seq_len],
                    torch.linspace(0, 1, seq_len, device=device)  # 时间标准化
                ], dim=-1)
            
                # 目标值：当前时刻的预期头寸（使用下一时刻的持仓）
                targets[sample_idx] = torch.tensor([
                    model.position_memory.get((path, t+seq_len-1), 0.0),  # 股票头寸
                    model.position_memory.get((path, t+seq_len-1), 0.0)   # 方差互换头寸
                ], device=device)
                sample_idx += 1
                # 记录时间步 (窗口最后一个时间点)
                time_indices.append((path, t + seq_len - 1))                  
        return sequences, targets, time_indices

    # 生成验证集（序列格式）
    val_size = int(0.1 * n_paths)
    S_val, V_val, integrated_var_val = simulate_heston(params, val_size, n_steps)
    # val_sequences, val_targets = create_sequences(S_val, V_val, integrated_var_val)
    val_sequences, val_targets, val_time = create_sequences(S_val, V_val, integrated_var_val)  # 解包三个变量    
    val_dataset =  TensorDataset(val_sequences, val_targets, val_time)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 训练循环
    start_total_time = time.time()
    best_loss = float('inf')
    loss_history = []
    val_loss_history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # 生成训练数据并转换为序列
        S, V, integrated_var = simulate_heston(params, n_paths, n_steps)
        train_sequences, train_targets, train_time = create_sequences(S, V, integrated_var)  # 接收三个变量
        # 数据加载部分调整为包含时间步
        # train_dataset = TensorDataset(train_sequences, train_targets, train_time)
        train_loader = DataLoader(
            TensorDataset(train_sequences, train_targets, train_time),
            batch_size=batch_size, 
            shuffle=True
        )
        
        # 批次训练
        for batch_seq, batch_target, batch_time in train_loader:
            optimizer.zero_grad()
            
            # 前向传播（输入整个序列）
            outputs = model(batch_seq)  # shape: (batch_size, 2)
            # 拆分输出
            deltas = outputs[:, 0]
            var_swaps = outputs[:, 1]
            # 计算交易成本（基于头寸变化）
            if epoch == 0:  # 初始头寸为0
                prev_positions = torch.zeros_like(outputs)
            else:
                prev_positions = model.position_memory.get_batch(batch_seq.indices)
                
            delta_positions = outputs - prev_positions
            transaction_cost = 0.003 * torch.sum(torch.abs(delta_positions), dim=1)
            
            # 计算组合价值
            portfolio_values = compute_portfolio_value(
                batch_seq, outputs, params, transaction_cost
            )
            
            # CVaR损失计算
            loss = compute_cvar(portfolio_values)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 更新头寸记忆
            model.update_position_memory(
                batch_indices=batch_time,  # (path_ids, timesteps)
                outputs=outputs,
                timesteps=batch_time[:, 1]  # 时间步信息
            )
            epoch_loss += loss.item()

        # 验证步骤
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for al_seq, val_tgt, val_t in val_loader:
                val_output = model(val_seq)
                val_cost = 0.003 * torch.sum(torch.abs(val_output - val_target))
                val_portfolio = compute_portfolio_value(val_seq, val_output, params, val_cost)
                val_loss += compute_cvar(val_portfolio).item()
        
        # 记录损失历史
        avg_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        loss_history.append(avg_loss)
        val_loss_history.append(avg_val_loss)
        
        # 保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        # 打印训练信息
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    # 加载最佳模型
    model.load_state_dict(torch.load("best_model.pth"))
    return loss_history, val_loss_history

# 新增辅助函数
def compute_portfolio_value(sequences, positions, params, transaction_cost):
    """
    计算组合价值（序列版本）
    sequences: (batch_size, seq_len, 4) [S, V, integrated_var, time]
    positions: (batch_size, 2) [delta, var_swap]
    """
    batch_size, seq_len, _ = sequences.shape
    portfolio = torch.zeros(batch_size, device=sequences.device)
    
    # 获取最终时刻的价格信息
    final_prices = sequences[:, -1, 0]  # 最后时间步的价格
    K = params.K
    
    # 计算期权支付
    option_payoff = torch.relu(K - final_prices)
    
    # 计算投资组合价值
    portfolio = positions[:, 0] * final_prices + positions[:, 1] * params.Kv - option_payoff - transaction_cost
    return portfolio











    
# Evaluate model
def evaluate_model(model, params, n_paths=1000, n_steps=30):
    model.eval()  # Set to evaluation mode
    print("\nStarting Deep Hedging Model evaluation...")
    
    # Generate new paths for evaluation
    print(f"Generating {n_paths} paths with {n_steps} steps for evaluation...")
    S, V, integrated_var = simulate_heston(params, n_paths, n_steps)
    
    # Calculate option payoffs
    option_payoff_np = put_option_payoff(S[:, -1], params.K)
    
    # Initialize
    portfolio_values = np.zeros(n_paths)
    stock_positions = np.zeros((n_paths, n_steps + 1))
    var_swap_positions = np.zeros((n_paths, n_steps + 1))
    trading_costs = np.zeros(n_paths)
    
    # Set up progress monitoring
    print(f"Running Deep Hedging Model on paths...")
    progress_interval = max(1, n_steps // 5)  # Update every 20% of time steps
    start_time = time.time()
    
    with torch.no_grad():  # Don't calculate gradients during evaluation
        for t in range(n_steps):
            if t % progress_interval == 0 or t == n_steps - 1:
                elapsed = time.time() - start_time
                progress = (t + 1) / n_steps * 100
                print(f"  Progress: {progress:.1f}% - Step {t+1}/{n_steps}, Time: {elapsed:.1f}s")
            
            curr_t = t * params.T / n_steps
            remaining_T = params.T - curr_t


            # 向量化计算 variance_swap_t 和 variance_swap_t_plus_dt
            h = torch.tensor(params.theta, device=device)
            alpha = torch.tensor(params.kappa, device=device)
            remaining_T_tensor = torch.tensor(remaining_T, device=device)
            Kv_tensor = torch.tensor(params.Kv, device=device)

            exp_term = torch.exp(-alpha * remaining_T_tensor)
            L_t_v = (V[:, t] - h) / alpha * (1 - exp_term) + h * remaining_T_tensor
            var_swap_t = integrated_var[:, t] + L_t_v - Kv_tensor

            remaining_T_plus_dt = remaining_T - (params.T / n_steps)
            exp_term_plus = torch.exp(-alpha * remaining_T_plus_dt)
            L_t_v_plus = (V[:, t+1] - h) / alpha * (1 - exp_term_plus) + h * remaining_T_plus_dt
            var_swap_t_plus_dt = integrated_var[:, t+1] + L_t_v_plus - Kv_tensor
            
            
            
            # Feature vectors
            features = torch.cat([
                torch.tensor(S[:, t], dtype=torch.float32).view(n_paths, 1),
                torch.tensor(V[:, t], dtype=torch.float32).view(n_paths, 1),
                torch.tensor(integrated_var[:, t], dtype=torch.float32).view(n_paths, 1),
                torch.ones(n_paths, 1) * t / n_steps
            ], dim=1).to(device)
            
            # Process in batches to avoid memory issues
            batch_size = 200
            all_trades = []
            
            for i in range(0, n_paths, batch_size):
                end_idx = min(i + batch_size, n_paths)
                batch_features = features[i:end_idx]
                batch_trades = model(batch_features)
                all_trades.append(batch_trades)
            
            # Combine results from all batches
            trades = torch.cat(all_trades, dim=0).cpu().numpy()
            stock_trades = trades[:, 0]
            var_swap_trades = trades[:, 1]
            
            # Calculate trading costs
            cost = params.c * np.sum(np.abs(trades), axis=1)
            trading_costs += cost
            
            # Update positions
            stock_positions[:, t+1] = stock_positions[:, t] + stock_trades
            var_swap_positions[:, t+1] = var_swap_positions[:, t] + var_swap_trades
            
            # Calculate variance swap values
            var_swap_t = np.array([
                calculate_variance_swap(
                    integrated_var[i, t], 
                    V[i, t], 
                    curr_t, 
                    params.T, 
                    params
                ) for i in range(n_paths)
            ])
            
            var_swap_t_plus_dt = np.array([
                calculate_variance_swap(
                    integrated_var[i, t+1], 
                    V[i, t+1], 
                    curr_t + params.T/n_steps, 
                    params.T, 
                    params
                ) for i in range(n_paths)
            ])
            
            # Update portfolio value
            stock_pnl = stock_positions[:, t+1] * (S[:, t+1] - S[:, t])
            var_swap_pnl = var_swap_positions[:, t+1] * (var_swap_t_plus_dt - var_swap_t)
            
            portfolio_values += stock_pnl + var_swap_pnl - cost
    
    # Final portfolio value (minus option payoff)
    final_portfolio = portfolio_values - option_payoff_np
    
    # Calculate statistics
    mean_pnl = np.mean(final_portfolio)
    std_pnl = np.std(final_portfolio)
    
    # Calculate VaR and CVaR - correctly as positive values for losses
    alpha = 0.05
    sorted_pnl = np.sort(final_portfolio)
    var_index = int(np.ceil(alpha * n_paths)) - 1
    var_index = max(0, var_index)
    var = max(0, -sorted_pnl[var_index])
    cvar = max(0, -np.mean(sorted_pnl[:var_index+1]))
    
    total_time = time.time() - start_time
    print(f"Deep Hedging evaluation completed in {total_time:.2f} seconds")
    print(f"Deep Hedging Evaluation Results:")
    print(f"Mean P&L: {mean_pnl:.4f}")
    print(f"P&L Standard Deviation: {std_pnl:.4f}")
    print(f"VaR(95%): {var:.4f}")
    print(f"CVaR(95%): {cvar:.4f}")
    print(f"Average Trading Cost: {np.mean(trading_costs):.4f}")
    
    return final_portfolio, stock_positions, var_swap_positions

# Heston option pricing and hedging functions

# Heston model characteristic function
def heston_characteristic_function(phi, S, V, T, params):
    a = params.kappa * params.theta
    b = params.kappa
    
    d = np.sqrt((params.rho * params.sigma * phi * 1j - b)**2 + (params.sigma**2) * (phi * 1j + phi**2))
    g = (b - params.rho * params.sigma * phi * 1j - d) / (b - params.rho * params.sigma * phi * 1j + d)
    
    exp1 = np.exp(params.r * T * phi * 1j)
    exp2 = np.exp(a * params.T * (b - params.rho * params.sigma * phi * 1j - d) / (params.sigma**2))
    exp3 = np.exp((V * (b - params.rho * params.sigma * phi * 1j - d)) / (params.sigma**2 * (1 - g * np.exp(-d * T))))
    
    return exp1 * exp2 * exp3

# Calculate option price under Heston model using numerical methods
def heston_option_price_fft(S, V, K, T, params, option_type='put'):
    import numpy as np
    from scipy.integrate import quad
    from scipy import interpolate
    
    # Characteristic function integral
    def integrand_call(phi, S, V, K, T, params):
        numerator = np.exp(-phi * np.log(K) * 1j) * heston_characteristic_function(phi - 1j, S, V, T, params)
        denominator = phi * 1j
        return np.real(numerator / denominator)
    
    # Calculate integral
    result, _ = quad(integrand_call, 0, 100, args=(S, V, K, T, params), limit=100)
    call_price = S / 2 + result / np.pi
    
    # Use put-call parity to get put option price
    if option_type.lower() == 'call':
        return call_price
    else:  # Put option
        return call_price - S + K * np.exp(-params.r * T)

# Calculate option Delta under Heston model
def heston_option_delta(S, V, K, T, params, option_type='put'):
    h = 0.01  # Small price change for numerical differentiation
    
    price_up = heston_option_price_fft(S + h, V, K, T, params, option_type)
    price_down = heston_option_price_fft(S - h, V, K, T, params, option_type)
    
    delta = (price_up - price_down) / (2 * h)
    return delta

# Calculate option Vega (sensitivity to variance) under Heston model
def heston_option_vega(S, V, K, T, params, option_type='put'):
    h = 0.0001  # Small variance change for numerical differentiation
    
    price_up = heston_option_price_fft(S, V + h, K, T, params, option_type)
    price_down = heston_option_price_fft(S, V - h, K, T, params, option_type)
    
    vega = (price_up - price_down) / (2 * h)
    return vega

# Evaluate Delta-Vega hedging strategy based on Formulas (32) and (33)
def evaluate_delta_vega_hedging(params, n_paths=1000, n_steps=30, use_vega=True):
    """
    Evaluate Delta-Vega hedging strategy under Heston model using formula (33)
    δ¹ := ∂u(t,S¹,Vt)/∂S¹ and δ² := ∂u(t,S¹,Vt)/∂L(t,Vt)
    
    Parameters:
    params: Heston model parameters
    n_paths: Number of simulation paths
    n_steps: Number of hedging steps
    use_vega: Whether to use Vega hedging (if False, only Delta hedging)
    
    Returns:
    final_portfolio: Final portfolio values
    delta_positions: Delta hedging positions
    vega_positions: Vega hedging positions


    Modify by shenys
    
    """
    strategy_name = "Delta-Vega Hedging" if use_vega else "Delta-Only Hedging"
    print(f"\nStarting {strategy_name} evaluation...")
    
    # Generate price paths
    print("Generating price paths...")
    S, V, integrated_var = simulate_heston(params, n_paths, n_steps)
    dt = params.T / n_steps
    
    # Initialize portfolio and positions
    portfolio_values = np.zeros(n_paths)
    delta_positions = np.zeros((n_paths, n_steps + 1))
    vega_positions = np.zeros((n_paths, n_steps + 1))
    trading_costs = np.zeros(n_paths)
    
    # Calculate option payoffs
    option_payoff_np = put_option_payoff(S[:, -1], params.K)
    
    # Set up progress monitoring
    print(f"Starting hedging calculations for {n_paths} paths over {n_steps} steps...")
    total_calculations = n_steps * n_paths
    progress_interval = max(1, n_paths // 10)
    last_progress_time = time.time()
    start_time = time.time()
    
    # Hedge at each time step
    for t in range(n_steps):
        curr_t = t * params.T / n_steps
        remaining_T = params.T - curr_t
        
        print(f"Time step {t+1}/{n_steps} - Remaining T: {remaining_T:.4f}")
        paths_done = 0
        
        # Process each path
        for i in range(n_paths):
            # Calculate Delta (Formula 33: δ¹ := ∂u(t,S¹,Vt)/∂S¹)
            delta = heston_option_delta(S[i, t], V[i, t], params.K, remaining_T, params, 'put')
            delta_trade = -delta - delta_positions[i, t]  # New position minus old position
            delta_positions[i, t+1] = -delta  # Negative delta for hedge
            
            # If using Vega hedging
            if use_vega:
                # Calculate option's sensitivity to variance (Formula 33)
                option_dv_sensitivity = heston_option_vega(S[i, t], V[i, t], params.K, remaining_T, params, 'put')
                
                # Calculate sensitivity of variance swap to instantaneous variance (∂L(t,Vt)/∂Vt)
                vs_sensitivity = variance_swap_sensitivity(V[i, t], curr_t, params.T, params)
                
                # Calculate vega hedge ratio according to formula (33): δ² := ∂u(t,S¹,Vt)/∂L(t,Vt)
                # Here we convert from ∂u/∂V to ∂u/∂L using the chain rule: (∂u/∂V) = (∂u/∂L) * (∂L/∂V)
                # So δ² = (∂u/∂V) / (∂L/∂V)
                vega_hedge_ratio = -option_dv_sensitivity / vs_sensitivity if vs_sensitivity != 0 else 0
                
                vega_trade = vega_hedge_ratio - vega_positions[i, t]
                vega_positions[i, t+1] = vega_hedge_ratio
            else:
                vega_trade = 0
                vega_positions[i, t+1] = 0
            
            # Calculate trading costs
            cost = 0.003 * (np.abs(delta_trade) * S[i, t] + np.abs(vega_trade))
            trading_costs[i] += cost
            
            # Calculate variance swap values for current and next time step
            var_swap_t = calculate_variance_swap(
                integrated_var[i, t], V[i, t], curr_t, params.T, params
            )
            
            var_swap_t_plus_dt = calculate_variance_swap(
                integrated_var[i, t+1], V[i, t+1], curr_t + dt, params.T, params
            )
            
            # Calculate PnL
            delta_pnl = delta_positions[i, t] * (S[i, t+1] - S[i, t])
            vega_pnl = vega_positions[i, t] * (var_swap_t_plus_dt - var_swap_t)
            
            # Update portfolio value
            portfolio_values[i] += delta_pnl + vega_pnl - cost
            
            paths_done += 1
            
            # Update progress
            if paths_done % progress_interval == 0 or paths_done == n_paths:
                current_time = time.time()
                elapsed = current_time - start_time
                progress_pct = (t * n_paths + paths_done) / total_calculations * 100
                
                # Only update if at least 1 second has passed since last update
                if current_time - last_progress_time >= 1.0:
                    last_progress_time = current_time
                    
                    # Estimate time remaining
                    if progress_pct > 0:
                        total_estimated_time = elapsed / (progress_pct / 100)
                        remaining_time = total_estimated_time - elapsed
                        
                        print(f"  Progress: {progress_pct:.1f}% - Paths: {paths_done}/{n_paths} in step {t+1}/{n_steps}")
                        print(f"  Elapsed time: {elapsed:.1f}s, Est. remaining: {remaining_time:.1f}s")
    
    # Final portfolio value (minus option payoff)
    final_portfolio = portfolio_values - option_payoff_np
    
    # Calculate statistics
    mean_pnl = np.mean(final_portfolio)
    std_pnl = np.std(final_portfolio)
    
    # Calculate VaR and CVaR
    alpha = 0.05
    sorted_pnl = np.sort(final_portfolio)
    var_index = int(np.ceil(alpha * n_paths)) - 1
    var_index = max(0, var_index)
    var = max(0, -sorted_pnl[var_index])
    cvar = max(0, -np.mean(sorted_pnl[:var_index+1]))
    
    total_time = time.time() - start_time
    print(f"\n{strategy_name} calculation complete in {total_time:.2f} seconds")
    print(f"{strategy_name} Evaluation Results:")
    print(f"Mean P&L: {mean_pnl:.4f}")
    print(f"P&L Standard Deviation: {std_pnl:.4f}")
    print(f"VaR(95%): {var:.4f}")
    print(f"CVaR(95%): {cvar:.4f}")
    print(f"Average Trading Cost: {np.mean(trading_costs):.4f}")
    
    return final_portfolio, delta_positions, vega_positions

# Evaluate BS Delta hedging strategy as a benchmark
def evaluate_bs_delta_hedging(params, n_paths=1000, n_steps=30):
    """
    Evaluate Black-Scholes Delta hedging strategy (assuming Heston model for the real world)
    """
    print("\nStarting Black-Scholes Delta Hedging evaluation...")
    
    # Generate price paths
    print("Generating price paths...")
    S, V, integrated_var = simulate_heston(params, n_paths, n_steps)
    dt = params.T / n_steps
    
    # Initialize portfolio and positions
    portfolio_values = np.zeros(n_paths)
    delta_positions = np.zeros((n_paths, n_steps + 1))
    trading_costs = np.zeros(n_paths)
    
    # Calculate option payoffs
    option_payoff_np = put_option_payoff(S[:, -1], params.K)
    
    # Set up progress monitoring
    print(f"Starting hedging calculations for {n_paths} paths over {n_steps} steps...")
    total_calculations = n_steps * n_paths
    progress_interval = max(1, n_paths // 10)
    last_progress_time = time.time()
    start_time = time.time()
    
    # Hedge at each time step
    for t in range(n_steps):
        remaining_T = params.T - t * dt
        
        print(f"Time step {t+1}/{n_steps} - Remaining T: {remaining_T:.4f}")
        paths_done = 0
        
        for i in range(n_paths):
            # Calculate Delta using Black-Scholes formula (with current implied volatility)
            current_vol = np.sqrt(V[i, t])
            d1 = (np.log(S[i, t] / params.K) + (params.r + 0.5 * current_vol**2) * remaining_T) / (current_vol * np.sqrt(remaining_T))
            bs_delta = norm.cdf(-d1)  # Delta for put option
            
            # Calculate Delta trade
            delta_trade = -bs_delta - delta_positions[i, t]
            delta_positions[i, t+1] = -bs_delta
            
            # Calculate trading costs
            cost = params.c * np.abs(delta_trade) * S[i, t]
            trading_costs[i] += cost
            
            # Calculate PnL
            delta_pnl = delta_positions[i, t] * (S[i, t+1] - S[i, t])
            
            # Update portfolio value
            portfolio_values[i] += delta_pnl - cost
            
            paths_done += 1
            
            # Update progress
            if paths_done % progress_interval == 0 or paths_done == n_paths:
                current_time = time.time()
                elapsed = current_time - start_time
                progress_pct = (t * n_paths + paths_done) / total_calculations * 100
                
                # Only update if at least 1 second has passed since last update
                if current_time - last_progress_time >= 1.0:
                    last_progress_time = current_time
                    
                    # Estimate time remaining
                    if progress_pct > 0:
                        total_estimated_time = elapsed / (progress_pct / 100)
                        remaining_time = total_estimated_time - elapsed
                        
                        print(f"  Progress: {progress_pct:.1f}% - Paths: {paths_done}/{n_paths} in step {t+1}/{n_steps}")
                        print(f"  Elapsed time: {elapsed:.1f}s, Est. remaining: {remaining_time:.1f}s")
    
    # Final portfolio value (minus option payoff)
    final_portfolio = portfolio_values - option_payoff_np
    
    # Calculate statistics
    mean_pnl = np.mean(final_portfolio)
    std_pnl = np.std(final_portfolio)
    
    # Calculate VaR and CVaR
    alpha = 0.05
    sorted_pnl = np.sort(final_portfolio)
    var_index = int(np.ceil(alpha * n_paths)) - 1
    var_index = max(0, var_index)
    var = max(0, -sorted_pnl[var_index])
    cvar = max(0, -np.mean(sorted_pnl[:var_index+1]))
    
    total_time = time.time() - start_time
    print(f"\nBlack-Scholes Delta Hedging calculation complete in {total_time:.2f} seconds")
    print(f"Black-Scholes Delta Hedging Evaluation Results:")
    print(f"Mean P&L: {mean_pnl:.4f}")
    print(f"P&L Standard Deviation: {std_pnl:.4f}")
    print(f"VaR(95%): {var:.4f}")
    print(f"CVaR(95%): {cvar:.4f}")
    print(f"Average Trading Cost: {np.mean(trading_costs):.4f}")
    
    return final_portfolio, delta_positions

# Compare different hedging strategies
def compare_hedging_strategies(params, deep_model, n_paths=500, n_steps=30):
    """
    Compare performance of different hedging strategies
    """
    print("\nStarting comparison of different hedging strategies...")
    
    # Generate the same set of paths for fair comparison
    S, V, integrated_var = simulate_heston(params, n_paths, n_steps)
    option_payoff_np = put_option_payoff(S[:, -1], params.K)
    
    # Deep hedging strategy
    print("\nEvaluating Deep Hedging strategy:")
    deep_pnl, deep_stock, deep_var = evaluate_model(deep_model, params, n_paths, n_steps)
    
    # Heston Delta-Vega hedging
    print("\nEvaluating Heston Delta-Vega hedging strategy:")
    dv_pnl, dv_delta, dv_vega = evaluate_delta_vega_hedging(params, n_paths, n_steps, use_vega=True)
    
    # Heston Delta-only hedging
    print("\nEvaluating Heston Delta-only hedging strategy:")
    d_pnl, d_delta, _ = evaluate_delta_vega_hedging(params, n_paths, n_steps, use_vega=False)
    
    # Black-Scholes Delta hedging
    print("\nEvaluating Black-Scholes Delta hedging strategy:")
    bs_pnl, bs_delta = evaluate_bs_delta_hedging(params, n_paths, n_steps)
    
    # Plot comparison results
    plt.figure(figsize=(15, 12))
    
    # Plot P&L distribution comparison
    plt.subplot(2, 2, 1)
    plt.hist(deep_pnl, bins=30, alpha=0.5, label='Deep Hedging')
    plt.hist(dv_pnl, bins=30, alpha=0.5, label='Heston Delta-Vega')
    plt.hist(d_pnl, bins=30, alpha=0.5, label='Heston Delta')
    plt.hist(bs_pnl, bins=30, alpha=0.5, label='Black-Scholes Delta')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('P&L Distribution Comparison Across Strategies')
    plt.xlabel('P&L')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Box plot comparison
    plt.subplot(2, 2, 2)
    plt.boxplot([deep_pnl, dv_pnl, d_pnl, bs_pnl], 
                labels=['Deep Hedging', 'Heston Delta-Vega', 'Heston Delta', 'BS Delta'])
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('P&L Box Plot Comparison Across Strategies')
    plt.ylabel('P&L')
    
    # Select a sample path, compare different strategy positions
    sample_idx = np.random.randint(0, n_paths)
    
    plt.subplot(2, 2, 3)
    plt.plot(S[sample_idx])
    plt.title(f'Sample Price Path #{sample_idx}')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    
    plt.subplot(2, 2, 4)
    plt.plot(deep_stock[sample_idx], label='Deep Hedging-Stock')
    plt.plot(deep_var[sample_idx], label='Deep Hedging-Variance Swap')
    plt.plot(dv_delta[sample_idx], label='Heston Delta')
    plt.plot(bs_delta[sample_idx], label='BS Delta')
    plt.title(f'Hedging Positions Comparison for Path #{sample_idx}')
    plt.xlabel('Time Step')
    plt.ylabel('Position Size')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('hedging_strategies_comparison.png')
    plt.show()
    
    # Numerical comparison
    print("\nNumerical Performance Comparison Across Strategies:")
    strategies = ['Deep Hedging', 'Heston Delta-Vega', 'Heston Delta', 'BS Delta']
    pnls = [deep_pnl, dv_pnl, d_pnl, bs_pnl]
    
    # Calculate various risk metrics
    results = []
    for i, strategy in enumerate(strategies):
        mean_pnl = np.mean(pnls[i])
        std_pnl = np.std(pnls[i])
        
        # VaR and CVaR - correctly as positive values for losses
        alpha = 0.05
        sorted_pnl = np.sort(pnls[i])
        var_index = int(np.ceil(alpha * len(pnls[i]))) - 1
        var_index = max(0, var_index)
        var = max(0, -sorted_pnl[var_index])
        cvar = max(0, -np.mean(sorted_pnl[:var_index+1]))
        
        # Probability of profit
        profit_prob = np.mean(pnls[i] > 0)
        
        results.append([mean_pnl, std_pnl, var, cvar, profit_prob])
    
    # Output comparison table
    header = ["Strategy", "Mean P&L", "Std Dev", "VaR(95%)", "CVaR(95%)", "Profit Prob"]
    row_format = "{:>15}" * (len(header))
    print(row_format.format(*header))
    
    for i, strategy in enumerate(strategies):
        row = [strategy] + [f"{val:.4f}" for val in results[i]]
        print(row_format.format(*row))
    
    # Return PnLs from each strategy for further analysis
    return deep_pnl, dv_pnl, d_pnl, bs_pnl, S, V, integrated_var

# Create advanced comparison plots
def create_advanced_comparison_plots(deep_pnl, dv_pnl, d_pnl, bs_pnl, 
                                     S, V, integrated_var, sample_idx=0):
    """
    Create more advanced comparison plots, including risk-return analysis and backtest analysis
    """
    print("\nCreating advanced comparison plots...")
    
    # 1. Risk-return analysis (mean-std scatterplot)
    strategies = ['Deep Hedging', 'Heston Delta-Vega', 'Heston Delta', 'BS Delta']
    pnls = [deep_pnl, dv_pnl, d_pnl, bs_pnl]
    
    means = [np.mean(p) for p in pnls]
    stds = [np.std(p) for p in pnls]
    sharpe = [m/s if s != 0 else 0 for m, s in zip(means, stds)]
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.scatter(stds, means, s=100)
    for i, strategy in enumerate(strategies):
        plt.annotate(strategy, (stds[i], means[i]), 
                    fontsize=10, xytext=(5, 5), textcoords='offset points')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Risk-Return Analysis')
    plt.xlabel('Risk (Standard Deviation)')
    plt.ylabel('Return (Mean P&L)')
    
    # 2. Sharpe ratio comparison across strategies
    plt.subplot(2, 2, 2)
    plt.bar(strategies, sharpe)
    plt.title('Sharpe Ratio Comparison')
    plt.ylabel('Sharpe Ratio')
    plt.xticks(rotation=45)
    
    # 3. Gain-to-Pain ratio across strategies
    gain_pain = []
    for p in pnls:
        gains = np.sum(p[p>0])
        pains = np.abs(np.sum(p[p<0]))
        ratio = gains / pains if pains != 0 else float('inf')
        gain_pain.append(ratio)
    
    plt.subplot(2, 2, 3)
    plt.bar(strategies, gain_pain)
    plt.title('Gain-to-Pain Ratio')
    plt.ylabel('Gain-to-Pain Ratio')
    plt.xticks(rotation=45)
    
    # 4. Hedging effectiveness analysis: P&L vs volatility
    plt.subplot(2, 2, 4)
    
    # Calculate overall realized volatility
    realized_vol = np.std(np.diff(np.log(S)), axis=1) * np.sqrt(252)
    
    # Create scatterplot to examine P&L vs realized volatility
    plt.scatter(realized_vol, deep_pnl, alpha=0.5, label='Deep Hedging')
    plt.scatter(realized_vol, dv_pnl, alpha=0.5, label='Heston Delta-Vega')
    plt.axhline(y=0, color='r', linestyle='--')
    
    # Add trend lines
    for pnl, label in zip([deep_pnl, dv_pnl], ['Deep Hedging', 'Heston Delta-Vega']):
        z = np.polyfit(realized_vol, pnl, 1)
        p = np.poly1d(z)
        plt.plot(np.sort(realized_vol), p(np.sort(realized_vol)), 
                '--', label=f'{label} Trend')
    
    plt.title('P&L vs Realized Volatility')
    plt.xlabel('Realized Volatility')
    plt.ylabel('P&L')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('advanced_comparison.png')
    plt.show()
    
    # Create P&L evolution over time plot
    plt.figure(figsize=(15, 8))
    
    # Select some sample paths
    sample_paths = np.random.choice(len(deep_pnl), size=5, replace=False)
    
    # For each sample path, recalculate P&L evolution over time
    n_steps = S.shape[1] - 1
    
    for sample_idx in sample_paths:
        # Recalculate deep hedging P&L evolution
        deep_pnl_evolution = np.zeros(n_steps + 1)
        
        # This is just a simplified example, ideally would rerun deep hedging and traditional methods
        # to get P&L evolution at each time step
        
        plt.plot(deep_pnl_evolution, label=f'Sample {sample_idx} - Deep Hedging')
        
    plt.title('P&L Evolution on Sample Paths')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative P&L')
    plt.legend()
    plt.grid(True)
    plt.savefig('pnl_evolution.png')
    plt.show()

# Main function
def main():
    print("Starting hedging strategy comparison under Heston model...")
    
    # Set parameters
    params = HestonParams()
    n_paths = 1000    # Number of training paths
    n_steps = 30      # Number of hedging steps (30 trading days)
    n_features = 4    # Number of features: price, volatility, integrated variance, time
    
    # Build deep hedging model
    # model = DeepHedgingTransformerLSTM(n_features).to(device)
    # print(model)

    model = DeepHedgingTransformerLSTM(
    n_features=4,
    hidden_dim=32,
    seq_len=3,
    nhead=1,
    num_layers=1
    )
    print(model)
    model.to('cpu')
    
    # Train model with increased epochs (100)
    loss_history, val_loss_history = train_deep_hedging_model(
        model, 
        params, 
        n_paths=n_paths, 
        n_steps=n_steps, 
        epochs=100,  # Increased to 100 epochs
        batch_size=16,
        learning_rate=0.001
    )
    
    # Plot loss history
    plt.figure(figsize=(12, 6))
    plt.plot(loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.title('Loss History Over 100 Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (CVaR)')
    plt.grid(True)
    plt.legend()
    plt.savefig('training_loss.png')
    plt.show()
    
    # Compare different hedging strategies
    deep_pnl, dv_pnl, d_pnl, bs_pnl, S, V, integrated_var = compare_hedging_strategies(
        params, model, n_paths=500, n_steps=n_steps
    )
    
    # Generate advanced comparison plots
    create_advanced_comparison_plots(
        deep_pnl, dv_pnl, d_pnl, bs_pnl, 
        S, V, integrated_var
    )
    
    # Save model
    torch.save(model.state_dict(), 'deep_hedging_model_100epochs.pth')
    print("Model saved as 'deep_hedging_model_100epochs.pth'")

if __name__ == "__main__":
    main()
