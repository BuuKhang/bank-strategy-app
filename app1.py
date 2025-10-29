# app.py
# =========================================================
# Webapp Streamlit: Tối ưu chiến lược giao dịch cổ phiếu ngân hàng (VN)
# 4 bước: Lọc -> Tối ưu tỷ trọng (SA/GA) -> Grid WMA Top% -> Ensemble + luật VN + Backtest
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import random
import logging

from scipy.optimize import dual_annealing
from deap import base, creator, tools, algorithms
from vnstock import Quote

from ta.trend import wma_indicator
from joblib import Parallel, delayed

from backtesting import Backtest, Strategy

# =========================================================
# PHẦN 1. HÀM TIỆN ÍCH / LOGIC CHO 4 BƯỚC
# =========================================================

# ---------- Bước 1: Lọc cổ phiếu ----------
def fetch_price_history(ticker, start_date, end_date):
    """
    Lấy dữ liệu giá đóng cửa ngày cho 1 mã.
    Trả về DataFrame index=Date, cột 'Close'.
    Nếu gọi API thất bại (ví dụ bị chặn trên Cloud) -> trả về None thay vì làm app crash.
    """
    try:
        q = Quote(source='vci', symbol=ticker)
        df = q.history(start=start_date, end=end_date, interval='1D')
    except Exception as e:
        # log nhẹ để debug, nhưng đừng kill app
        print(f"[WARN] fetch_price_history({ticker}) lỗi khi gọi API: {e}")
        return None

    if df is None or df.empty:
        print(f"[WARN] fetch_price_history({ticker}): API trả về rỗng.")
        return None

    df = df[['time', 'close']].rename(columns={'time': 'Date', 'close': 'Close'})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    return df



def screen_bank_stocks(
    bank_list,
    start_date='2015-01-01',
    end_date='2023-12-31',
    max_missing_ratio=0.4
):
    """
    Quét chất lượng dữ liệu từng mã bank.
    Nếu API bị chặn => df = None => mã đó sẽ bị đánh dấu NO DATA
    App vẫn tiếp tục chạy.
    """
    results = []
    price_panel = {}

    total_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1

    for ticker in bank_list:
        df = fetch_price_history(ticker, start_date, end_date)

        if df is None or df.empty:
            results.append({
                "Ticker": ticker,
                "Status": "NO DATA",
                "MissingRatio": 1.0,
                "Ndays": 0
            })
            continue

        available_days = df.shape[0]
        missing_ratio = 1 - (available_days / total_days)

        status = "OK" if missing_ratio <= max_missing_ratio else "DROP"

        results.append({
            "Ticker": ticker,
            "Status": status,
            "MissingRatio": round(missing_ratio, 4),
            "Ndays": available_days
        })

        if status == "OK":
            price_panel[ticker] = df['Close']

    summary_df = pd.DataFrame(results)

    if len(price_panel) > 0:
        merged_prices = pd.DataFrame(price_panel)
    else:
        merged_prices = pd.DataFrame()

    return summary_df, merged_prices



# ---------- Bước 2: Tối ưu tỷ trọng ----------
def get_return_cov_matrix(price_df):
    """
    price_df: mỗi cột là 1 mã cổ phiếu, mỗi dòng là giá đóng cửa theo ngày.
    Trả về lợi suất kỳ vọng năm hóa và ma trận hiệp phương sai năm hóa.
    """
    returns = price_df.pct_change().dropna()
    mean_returns = returns.mean() * 252          # annualized expected return
    cov_matrix   = returns.cov() * 252           # annualized covariance
    return mean_returns, cov_matrix


def portfolio_performance(weights, mean_returns, cov_matrix, rf=0.035):
    """
    Tính Return, Volatility, Sharpe Ratio của một vector trọng số.
    """
    port_ret = np.dot(weights, mean_returns)
    port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    sharpe = (port_ret - rf) / port_vol if port_vol > 0 else -999
    return port_ret, port_vol, sharpe


def normalize_weights(w_raw: np.ndarray):
    """
    Chuẩn hóa vector trọng số về:
    - không âm
    - tổng = 1
    """
    w = np.clip(w_raw, 0, None)  # ép âm -> 0
    s = w.sum()
    if s == 0:
        w[:] = 1.0 / len(w)
    else:
        w = w / s
    return w


def sharpe_ratio(weights, mean_returns, cov_matrix, rf=0.035):
    """
    Tính Sharpe Ratio của một vector trọng số (đã chuẩn hóa).
    """
    w = normalize_weights(np.array(weights, dtype=float))
    port_ret = np.dot(w, mean_returns)
    port_vol = np.sqrt(w.T @ cov_matrix @ w)
    if port_vol <= 0:
        return -999
    return (port_ret - rf) / port_vol


def negative_sharpe_ratio(weights, mean_returns, cov_matrix, rf=0.035):
    """
    Hàm mục tiêu cho bộ tối ưu: MINIMIZE(-Sharpe) để 'tối đa Sharpe'.
    """
    return -sharpe_ratio(weights, mean_returns, cov_matrix, rf=rf)


def optimize_weights_sa(mean_returns, cov_matrix, rf=0.035):
    """
    Tối ưu trọng số bằng Simulated Annealing (dual_annealing).
    ràng buộc: mỗi weight trong [0,1], sau đó chuẩn hóa lại tổng=1.
    """
    num_assets = len(mean_returns)
    bounds = [(0, 1) for _ in range(num_assets)]

    def objective(w):
        return negative_sharpe_ratio(w, mean_returns.values, cov_matrix.values, rf=rf)

    result_sa = dual_annealing(objective, bounds=bounds)
    optimal_weights_sa = normalize_weights(result_sa.x)

    ret, vol, sh = portfolio_performance(
        optimal_weights_sa,
        mean_returns.values,
        cov_matrix.values,
        rf=rf
    )

    return {
        "weights": optimal_weights_sa,
        "return": ret,
        "vol": vol,
        "sharpe": sh,
        "method": "SA"
    }


def optimize_weights_ga(mean_returns, cov_matrix, rf=0.035,
                        population_size=100, ngen=50,
                        cxpb=0.7, mutpb=0.2):
    """
    Tối ưu trọng số bằng Genetic Algorithm (DEAP) – rerun-safe cho Streamlit.
    """
    num_assets = len(mean_returns)

    # RERUN-SAFE: chỉ tạo class 1 lần
    try:
        creator.FitnessMin
    except AttributeError:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    try:
        creator.Individual
    except AttributeError:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=num_assets)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        return (
            negative_sharpe_ratio(
                individual,
                mean_returns.values,
                cov_matrix.values,
                rf=rf
            ),
        )

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0.5, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_size)

    algorithms.eaSimple(
        population,
        toolbox,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=ngen,
        verbose=False
    )

    best_ind = tools.selBest(population, k=1)[0]
    optimal_weights_ga = normalize_weights(np.array(best_ind, dtype=float))

    ret, vol, sh = portfolio_performance(
        optimal_weights_ga,
        mean_returns.values,
        cov_matrix.values,
        rf=rf
    )

    return {
        "weights": optimal_weights_ga,
        "return": ret,
        "vol": vol,
        "sharpe": sh,
        "method": "GA"
    }


def pretty_weights_report(tickers, weights_arr):
    return pd.DataFrame({
        "Ticker": tickers,
        "Weight(%)": [round(w*100, 2) for w in weights_arr]
    })


# ---------- Bước 3: Tối ưu tham số WMA ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_stock_data_for_backtesting(ticker, start_date, end_date):
    """
    Lấy dữ liệu OHLCV cho backtesting lib.
    Trả về DataFrame index=Datetime, cột ['Open','High','Low','Close','Volume'].
    """
    try:
        q = Quote(source='vci', symbol=ticker)
        data = q.history(start=start_date, end=end_date, interval='1D')

        if data is None or data.empty:
            logging.warning(f"⚠️ Không có dữ liệu cho mã {ticker}")
            return None

        data = data.copy()
        data.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'time': 'Time'
        }, inplace=True)

        data['Time'] = pd.to_datetime(data['Time'])
        data.set_index('Time', inplace=True)

        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index()
        data.fillna(method='ffill', inplace=True)

        return data

    except Exception as e:
        logging.error(f"❌ Lỗi khi lấy dữ liệu mã {ticker}: {e}")
        return None


class MAStrategyWMA(Strategy):
    """
    Chiến lược MA crossover dùng WMA.
    Vào lệnh khi WMA ngắn cắt lên WMA dài; thoát khi cắt xuống.
    """
    short_period = 10
    long_period = 50

    def init(self):
        self.ma_short = self.I(
            lambda x: wma_indicator(pd.Series(x), window=int(self.short_period)),
            self.data.Close
        )
        self.ma_long = self.I(
            lambda x: wma_indicator(pd.Series(x), window=int(self.long_period)),
            self.data.Close
        )

    def next(self):
        if self.ma_short[-1] > self.ma_long[-1] and self.ma_short[-2] <= self.ma_long[-2]:
            self.buy()
        elif self.ma_short[-1] < self.ma_long[-1] and self.ma_short[-2] >= self.ma_long[-2]:
            self.position.close()


def run_backtest_wma(short_period, long_period, data_ohlc):
    """
    Chạy backtest WMA(short,long) trên dữ liệu OHLC.
    Trả về Sharpe Ratio. Nếu short>=long thì loại.
    """
    if short_period >= long_period:
        return -1

    bt = Backtest(
        data_ohlc,
        MAStrategyWMA,
        cash=100_000,
        commission=0.002  # 0.2%
    )

    stats = bt.run(short_period=int(short_period), long_period=int(long_period))
    sharpe = stats.get('Sharpe Ratio', -1)
    return sharpe


def optimize_wma_grid_for_ticker(ticker, train_start, train_end, top_percent=0.1):
    """
    Grid WMA cho 1 mã:
    - short in [5..45 step 5], long in [51..191 step 10]
    - trả về top 10% theo Sharpe
    """
    data_ohlc = get_stock_data_for_backtesting(
        ticker,
        start_date=train_start,
        end_date=train_end
    )
    if data_ohlc is None or data_ohlc.empty:
        return None

    results = []
    for short_p in range(5, 50, 5):
        for long_p in range(51, 200, 10):
            sharpe = run_backtest_wma(short_p, long_p, data_ohlc)
            results.append({
                "Ticker": ticker,
                "MA": "WMA",
                "Short": short_p,
                "Long": long_p,
                "Sharpe": sharpe
            })

    df = pd.DataFrame(results)
    df = df[df['Sharpe'] > -1]
    if df.empty:
        return None

    df_sorted = df.sort_values(by='Sharpe', ascending=False).reset_index(drop=True)
    top_n = max(1, int(len(df_sorted) * top_percent))
    df_top = df_sorted.head(top_n)
    return df_top


# ---------- Bước 4: Backtest danh mục cuối ----------
# ==== THAM SỐ GIAO DỊCH VN DÙNG CHO DANH MỤC CUỐI ====
K = 3                         # số ứng viên/mã để ensemble
VOTE_THRESHOLD = 0.5          # >50% phiếu "mua" thì vào lệnh
TCOST_BPS = 20                # 20 bps = 0.20%
RF_ANNUAL = 0.035             # lãi suất phi rủi ro năm
PERIODS = 252                 # số phiên/năm

# RÀNG BUỘC VN
MIN_HOLD_DAYS = 2             # T+2: giữ >=2 phiên trước khi bán
STOP_LOSS_PCT = 0.07          # SL 7%
COOLDOWN_DAYS = 5             # sau SL, cooldown X phiên
T2_CASH_DAYS = 2              # tiền bán về sau 2 phiên mới dùng lại


def fetch_prices_multi(tickers, start_date, end_date):
    """
    Lấy giá đóng cửa cho nhiều mã, trả về DataFrame: index=ngày, cột=ticker.
    """
    price = pd.DataFrame()
    for t in tickers:
        try:
            q = Quote(source='vci', symbol=t)
            df = q.history(start=start_date, end=end_date, interval='1D')
            if df is not None and not df.empty:
                df = df[["time", "close"]].copy()
                df["time"] = pd.to_datetime(df["time"])
                df.set_index("time", inplace=True)
                price = pd.concat(
                    [price, df.rename(columns={"close": t})],
                    axis=1
                )
            else:
                logging.warning(f"⚠️ Không có dữ liệu cho mã {t}")
        except Exception as e:
            logging.error(f"❌ Lỗi khi lấy giá {t}: {e}")
    return price.sort_index()


def gen_signal_wma_for_price(close_series: pd.Series, short_w: int, long_w: int):
    """
    Tạo tín hiệu giao dịch WMA crossover.
    1 = nắm giữ; 0 = không nắm giữ.
    Dịch 1 ngày (shift(1)) để tránh nhìn tương lai.
    """
    short_line = wma_indicator(close_series, window=int(short_w))
    long_line  = wma_indicator(close_series, window=int(long_w))
    raw_sig = (short_line > long_line).astype(int)
    return raw_sig.shift(1).fillna(0).astype(int)


def apply_vn_rules_one(sig_raw: pd.Series,
                       px: pd.Series,
                       min_hold_days: int,
                       stop_loss_pct: float,
                       cooldown_days: int):
    """
    Điều chỉnh tín hiệu theo luật VN:
    - Giữ >= min_hold_days phiên
    - Stop-loss nếu drawdown <= -stop_loss_pct
    - Sau SL, cooldown X phiên không được mua lại
    """
    sig = sig_raw.copy().astype(int)
    sig_adj = sig.copy()

    in_pos = False
    hold_cnt = 0
    cool_cnt = 0
    entry_price = np.nan

    for i, _ in enumerate(sig.index):
        if cool_cnt > 0:
            sig_adj.iloc[i] = 0
            cool_cnt -= 1
        else:
            sig_adj.iloc[i] = 1 if sig.iloc[i] == 1 else 0

        if not in_pos and sig_adj.iloc[i] == 1:
            in_pos = True
            hold_cnt = 0
            entry_price = px.iloc[i]

        elif in_pos and sig_adj.iloc[i] == 1:
            hold_cnt += 1
            if entry_price and entry_price > 0:
                dd = px.iloc[i] / entry_price - 1.0
                if dd <= -abs(stop_loss_pct):
                    sig_adj.iloc[i] = 0
                    in_pos = False
                    hold_cnt = 0
                    cool_cnt = cooldown_days
                    entry_price = np.nan

        elif in_pos and sig_adj.iloc[i] == 0:
            if hold_cnt < (min_hold_days - 1):
                sig_adj.iloc[i] = 1
                hold_cnt += 1
            else:
                in_pos = False
                hold_cnt = 0
                entry_price = np.nan

        if not in_pos and sig_adj.iloc[i] == 0:
            hold_cnt = 0

    return sig_adj.astype(int)


def build_portfolio_nav(price_df: pd.DataFrame,
                        wma_params_by_ticker: dict,
                        weights_by_ticker: dict,
                        tcost_bps: float,
                        min_hold_days: int,
                        stop_loss_pct: float,
                        cooldown_days: int,
                        t2_cash_days: int):
    """
    (Dùng khi mỗi mã chỉ có 1 bộ WMA). Ở app này ta dùng ensemble nhiều bộ nên dùng hàm dưới.
    Hàm này vẫn giữ lại nếu bạn cần tham khảo.
    """
    tickers = [
        tk for tk in wma_params_by_ticker.keys()
        if tk in price_df.columns and weights_by_ticker.get(tk, 0) > 0
    ]
    if len(tickers) == 0:
        raise RuntimeError("Không có mã hợp lệ (không có tham số WMA hoặc weight > 0).")

    # 1. Tín hiệu WMA cơ bản
    raw_signals = {}
    for tk in tickers:
        close_series = price_df[tk].dropna()
        params = wma_params_by_ticker[tk]
        sig_raw = gen_signal_wma_for_price(
            close_series,
            short_w=params["short"],
            long_w=params["long"]
        )
        raw_signals[tk] = sig_raw

    # 2. Áp luật VN
    adj_signals = {}
    for tk in tickers:
        px = price_df[tk].reindex(raw_signals[tk].index).fillna(method='ffill')
        adj_signals[tk] = apply_vn_rules_one(
            sig_raw=raw_signals[tk],
            px=px,
            min_hold_days=min_hold_days,
            stop_loss_pct=stop_loss_pct,
            cooldown_days=cooldown_days
        )

    # 3. Chuẩn bị kết hợp danh mục + T+2 tiền
    idx_union = sorted(set().union(*[s.index for s in adj_signals.values()]))
    idx_union = pd.to_datetime(idx_union)

    weights_mat  = pd.DataFrame(0.0, index=idx_union, columns=tickers)
    signals_mat  = pd.DataFrame(0,   index=idx_union, columns=tickers)
    returns_mat  = pd.DataFrame(0.0, index=idx_union, columns=tickers)
    costs_mat    = pd.DataFrame(0.0, index=idx_union, columns=tickers)

    for tk in tickers:
        w_fixed = weights_by_ticker.get(tk, 0.0)
        px = price_df[tk].reindex(idx_union).fillna(method='ffill')
        sig = adj_signals[tk].reindex(idx_union).fillna(0).astype(int)

        ret = px.pct_change().fillna(0.0)
        turn = sig.diff().abs().fillna(0.0)
        cost = turn * (tcost_bps/10000.0)

        weights_mat[tk] = float(w_fixed)
        signals_mat[tk] = sig
        returns_mat[tk] = ret
        costs_mat[tk]   = cost

    sell_events = (signals_mat.diff() == -1).astype(int)

    blocked_weight = pd.Series(0.0, index=idx_union)
    kernel = np.ones(int(t2_cash_days), dtype=float)
    for tk in tickers:
        w_i = float(weights_mat[tk].iloc[0]) if len(weights_mat[tk]) else 0.0
        seq = sell_events[tk].fillna(0).astype(float).to_numpy() * w_i
        conv = np.convolve(seq, kernel, mode='full')[:len(seq)]
        blocked_weight = blocked_weight.add(pd.Series(conv, index=idx_union), fill_value=0.0)
    blocked_weight = blocked_weight.clip(0.0, 1.0)

    active_weight = (weights_mat * signals_mat).sum(axis=1)
    capacity = (1.0 - blocked_weight).clip(0.0, 1.0)
    scale = pd.Series(1.0, index=idx_union)
    mask = active_weight > 1e-12
    scale.loc[mask] = np.minimum(1.0, (capacity.loc[mask] / active_weight.loc[mask]).astype(float))

    eff_weights = (weights_mat * signals_mat).mul(scale, axis=0)

    per_stock_pnl = eff_weights * (signals_mat * returns_mat - costs_mat)
    port_ret = per_stock_pnl.sum(axis=1).rename("port_ret")
    nav_series = (1 + port_ret).cumprod()
    nav_df = nav_series.to_frame(name="NAV")

    return nav_df, port_ret, {}


def sharpe_annual(daily_ret: pd.Series, rf_annual=RF_ANNUAL, periods=PERIODS):
    if len(daily_ret) == 0 or daily_ret.std() == 0:
        return np.nan
    er = daily_ret.mean()*periods - rf_annual
    vol = daily_ret.std()*np.sqrt(periods)
    return er/vol if vol != 0 else np.nan


def cagr_from_nav(nav: pd.Series, periods=PERIODS):
    if len(nav) < 2 or nav.iloc[0] <= 0:
        return np.nan
    total_return = nav.iloc[-1] / nav.iloc[0]
    years = len(nav) / periods
    return total_return**(1/years) - 1 if years > 0 else np.nan


def max_drawdown(nav: pd.Series):
    if len(nav) == 0:
        return np.nan
    roll_max = nav.cummax()
    dd = nav/roll_max - 1.0
    return dd.min()


def slice_series(s: pd.Series, start: str, end: str):
    if s.empty:
        return s
    return s.loc[(s.index >= pd.to_datetime(start)) & (s.index <= pd.to_datetime(end))]


def build_portfolio_from_ensemble(
    wma_results_by_ticker: dict,
    weights_by_ticker: dict,
    train_start_str: str,
    train_end_str: str,
    test_start_str: str,
    test_end_str: str,
    vote_threshold: float = VOTE_THRESHOLD,
    top_k: int = K,
    tcost_bps: float = TCOST_BPS,
    min_hold_days: int = MIN_HOLD_DAYS,
    stop_loss_pct: float = STOP_LOSS_PCT,
    cooldown_days: int = COOLDOWN_DAYS,
    t2_cash_days: int = T2_CASH_DAYS
):
    """
    Ensemble top-K bộ WMA tốt nhất của từng mã (từ Tab 3), áp luật VN, ghép danh mục theo trọng số (Tab 2),
    xử lý T+2 tiền, rồi trả về NAV + metrics Train/Test.
    """

    # 1) Tickers hợp lệ
    usable_tickers = []
    for tk, w in weights_by_ticker.items():
        if w > 0 and (tk in wma_results_by_ticker) and (wma_results_by_ticker[tk] is not None) and (not wma_results_by_ticker[tk].empty):
            usable_tickers.append(tk)
    usable_tickers = sorted(list(set(usable_tickers)))
    if not usable_tickers:
        raise RuntimeError("Không có mã hợp lệ để ensemble (cần weight>0 và có kết quả WMA ở Bước 3).")

    # 2) Tải giá Train→Test
    price_df = fetch_prices_multi(usable_tickers, start_date=train_start_str, end_date=test_end_str)
    if price_df is None or price_df.empty:
        raise RuntimeError("Không tải được dữ liệu giá cho danh mục.")

    # 3) Ensemble vote trên từng mã bằng top-K Sharpe
    base_signals = {}
    picked_rows = []
    for tk in usable_tickers:
        df_top = wma_results_by_ticker[tk].copy()

        if "Sharpe Ratio" in df_top.columns:
            df_top = df_top.sort_values("Sharpe Ratio", ascending=False)
            sharp_col = "Sharpe Ratio"
        else:
            df_top = df_top.sort_values("Sharpe", ascending=False)
            sharp_col = "Sharpe"

        df_top = df_top.head(int(max(1, top_k)))

        px = price_df[tk].dropna()
        if px.empty:
            continue

        sig_list = []
        for _, r in df_top.iterrows():
            sp = int(r["Short"]); lp = int(r["Long"])
            s = gen_signal_wma_for_price(px, sp, lp)
            sig_list.append(s.reindex(px.index).fillna(0).astype(int))
            picked_rows.append([tk, "WMA", sp, lp, float(r[sharp_col])])

        if sig_list:
            mat = pd.concat(sig_list, axis=1)
            ens = (mat.mean(axis=1) > float(vote_threshold)).astype(int)
            base_signals[tk] = ens.reindex(px.index).fillna(0).astype(int)

    if not base_signals:
        raise RuntimeError("Không tạo được tín hiệu ensemble cho bất kỳ mã nào.")

    # 4) Áp luật VN
    adj_signals = {}
    for tk, sig in base_signals.items():
        px = price_df[tk].reindex(sig.index).fillna(method='ffill')
        adj = apply_vn_rules_one(
            sig_raw=sig,
            px=px,
            min_hold_days=min_hold_days,
            stop_loss_pct=stop_loss_pct,
            cooldown_days=cooldown_days
        )
        adj_signals[tk] = adj

    # 5) T+2 tiền + chi phí
    idx_union = sorted(set().union(*[sig.index for sig in adj_signals.values()]))
    idx_union = pd.to_datetime(idx_union)

    weights_mat  = pd.DataFrame(0.0, index=idx_union, columns=usable_tickers)
    signals_mat  = pd.DataFrame(0,   index=idx_union, columns=usable_tickers)
    returns_mat  = pd.DataFrame(0.0, index=idx_union, columns=usable_tickers)
    costs_mat    = pd.DataFrame(0.0, index=idx_union, columns=usable_tickers)

    for tk in usable_tickers:
        w_fixed = float(weights_by_ticker.get(tk, 0.0))
        px = price_df[tk].reindex(idx_union).fillna(method='ffill')
        sig = adj_signals[tk].reindex(idx_union).fillna(0).astype(int)

        ret = px.pct_change().fillna(0.0)
        turn = sig.diff().abs().fillna(0.0)
        cost = turn * (tcost_bps/10000.0)

        weights_mat[tk] = w_fixed
        signals_mat[tk] = sig
        returns_mat[tk] = ret
        costs_mat[tk]   = cost

    sell_events = (signals_mat.diff() == -1).astype(int)

    blocked_weight = pd.Series(0.0, index=idx_union)
    kernel = np.ones(int(t2_cash_days), dtype=float)
    for tk in usable_tickers:
        w_i = float(weights_mat[tk].iloc[0]) if len(weights_mat[tk]) else 0.0
        seq = sell_events[tk].fillna(0).astype(float).to_numpy() * w_i
        conv = np.convolve(seq, kernel, mode='full')[:len(seq)]
        blocked_weight = blocked_weight.add(pd.Series(conv, index=idx_union), fill_value=0.0)
    blocked_weight = blocked_weight.clip(0.0, 1.0)

    active_weight = (weights_mat * signals_mat).sum(axis=1)
    capacity = (1.0 - blocked_weight).clip(0.0, 1.0)
    scale = pd.Series(1.0, index=idx_union)
    mask = active_weight > 1e-12
    scale.loc[mask] = np.minimum(1.0, (capacity.loc[mask] / active_weight.loc[mask]).astype(float))

    eff_weights = (weights_mat * signals_mat).mul(scale, axis=0)

    per_stock_pnl = eff_weights * (signals_mat * returns_mat - costs_mat)
    port_ret = per_stock_pnl.sum(axis=1).rename("port_ret")
    nav_df = (1 + port_ret).cumprod().to_frame(name="NAV")

    # 6) Metrics
    metrics_all = {
        "Sharpe": sharpe_annual(port_ret, rf_annual=RF_ANNUAL, periods=PERIODS),
        "CAGR":   cagr_from_nav(nav_df["NAV"], periods=PERIODS),
        "MaxDD":  max_drawdown(nav_df["NAV"])
    }
    pr_train = slice_series(port_ret, train_start_str, train_end_str)
    pr_test  = slice_series(port_ret, test_start_str,  test_end_str)

    def _metrics(seg):
        if seg.empty:
            return {"Sharpe": np.nan, "CAGR": np.nan, "MaxDD": np.nan}
        nv = (1 + seg).cumprod()
        return {
            "Sharpe": sharpe_annual(seg, rf_annual=RF_ANNUAL, periods=PERIODS),
            "CAGR":   cagr_from_nav(nv, periods=PERIODS),
            "MaxDD":  max_drawdown(nv)
        }

    metrics_train = _metrics(pr_train)
    metrics_test  = _metrics(pr_test)

    picked_rows_debug = pd.DataFrame(
        picked_rows, columns=["Ticker","MA","Short","Long","Sharpe"]
    )

    return {
        "nav_df": nav_df,
        "port_ret": port_ret,
        "eff_weights": eff_weights,
        "metrics_all": metrics_all,
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
        "picked_rows_debug": picked_rows_debug
    }


# =========================================================
# PHẦN 2. UI STREAMLIT (4 TAB)
# =========================================================

st.set_page_config(page_title="Bank Strategy Workflow", layout="wide")
st.title("📈 Quy trình tối ưu chiến lược giao dịch ngân hàng Việt Nam")

# Khởi tạo session_state
if "selected_stocks" not in st.session_state:
    st.session_state.selected_stocks = []

if "final_weights" not in st.session_state:
    st.session_state.final_weights = {}  # {ticker: weight_float}

if "wma_results" not in st.session_state:
    st.session_state["wma_results"] = {}

# Sidebar cấu hình chung
st.sidebar.header("Cấu hình chung")
train_start = st.sidebar.date_input("Train start", value=date(2020,1,1))
train_end   = st.sidebar.date_input("Train end",   value=date(2023,12,31))
test_start  = st.sidebar.date_input("Test start",  value=date(2024,1,1))
test_end    = st.sidebar.date_input("Test end",    value=date.today())

# Lưu chuỗi ngày để truyền hàm
train_start_str = pd.to_datetime(train_start).strftime("%Y-%m-%d")
train_end_str   = pd.to_datetime(train_end).strftime("%Y-%m-%d")
test_start_str  = pd.to_datetime(test_start).strftime("%Y-%m-%d")
test_end_str    = pd.to_datetime(test_end).strftime("%Y-%m-%d")

rf_annual = st.sidebar.number_input("Lãi suất phi rủi ro (năm)", min_value=0.0, max_value=0.2, value=0.035)
tcost_bps = st.sidebar.number_input("Phí giao dịch (bps)", min_value=0, max_value=200, value=20)

bank_universe_default = [
    'VCB','BID','CTG','TCB','MBB','VPB',
    'STB','TPB','VIB','HDB','LPB','SHB'
]

st.sidebar.caption("Mỗi bước: bạn chọn → bấm lưu → bước sau dùng dữ liệu đã lưu.")

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Lọc cổ phiếu",
    "2️⃣ Tỷ trọng danh mục",
    "3️⃣ Tối ưu tham số WMA",
    "4️⃣ Ensemble + Backtest"
])

# ============= TAB 1: LỌC CỔ PHIẾU =============
with tab1:
    st.header("Bước 1. Lọc cổ phiếu ngân hàng đầu vào")

    # Hiển thị đã lưu
    st.markdown("**Các mã đã lưu hiện tại:**")
    st.write(
        st.session_state.selected_stocks
        if len(st.session_state.selected_stocks) > 0
        else "Chưa có"
    )

    universe = st.multiselect(
        "Chọn universe ngân hàng để kiểm tra dữ liệu",
        options=bank_universe_default,
        default=bank_universe_default
    )

    # nút chạy lọc
    run_screen = st.button("📊 Chạy lọc dữ liệu (Screen)")

    # tạo biến session_state nếu chưa có
    if "last_screen_summary" not in st.session_state:
        st.session_state.last_screen_summary = None
    if "last_good_stocks" not in st.session_state:
        st.session_state.last_good_stocks = []

    if run_screen:
        # Giai đoạn train để đánh giá độ đầy đủ dữ liệu
        train_start_str = train_start.strftime("%Y-%m-%d")
        train_end_str   = train_end.strftime("%Y-%m-%d")

        try:
            summary_df, merged_prices = screen_bank_stocks(
                universe,
                start_date=train_start_str,
                end_date=train_end_str,
                max_missing_ratio=0.4
            )

            st.session_state.last_screen_summary = summary_df

            if summary_df is not None and not summary_df.empty:
                ok_list = summary_df[summary_df["Status"] == "OK"]["Ticker"].tolist()
            else:
                ok_list = []

            st.session_state.last_good_stocks = ok_list

            if len(ok_list) == 0:
                st.warning(
                    "Không mã nào lấy được dữ liệu (có thể server chứng khoán chặn IP Streamlit Cloud). "
                    "Bạn vẫn tiếp tục xem luồng chiến lược được nhưng dữ liệu thực tế có thể cần chạy tại máy local."
                )
            else:
                st.success(f"Các mã đạt yêu cầu dữ liệu: {ok_list}")

        except Exception as e:
            # QUAN TRỌNG: nếu vnstock fail (RetryError) thì ta không crash app
            st.error(
                "Không gọi được dữ liệu từ nguồn chứng khoán (có thể bị giới hạn IP Streamlit Cloud). "
                "Hãy thử chạy lại trên máy local. "
                f"Chi tiết lỗi: {e}"
            )

    # Hiển thị lại summary nếu đã có từ lần trước (kể cả sau rerun trang)
    if st.session_state.last_screen_summary is not None:
        st.subheader("Chất lượng dữ liệu các mã (lần quét gần nhất):")
        st.dataframe(st.session_state.last_screen_summary)

        # Cho người dùng chọn mã để lưu cho bước sau
        chosen = st.multiselect(
            "Chọn các mã bạn muốn GIỮ lại cho bước sau",
            options=st.session_state.last_good_stocks,
            default=st.session_state.last_good_stocks,
            key="stock_picker_step1"
        )

        if st.button("💾 Lưu danh sách mã đã chọn"):
            st.session_state.selected_stocks = chosen
            st.success(f"ĐÃ LƯU: {chosen}")
    else:
        st.info("Bấm nút '📊 Chạy lọc dữ liệu (Screen)' để kiểm tra chất lượng dữ liệu và chọn mã.")


# ============= TAB 2: TỐI ƯU TỶ TRỌNG =============
with tab2:
    st.header("Bước 2. Đề xuất tỷ trọng & chỉnh tay (dựa trên SA / GA)")

    st.markdown("**Danh sách mã đã lưu từ bước 1:**")
    st.write(st.session_state.selected_stocks if len(st.session_state.selected_stocks) > 0 else "Chưa có")

    if len(st.session_state.selected_stocks) == 0:
        st.warning("Chưa có mã nào được lưu ở Bước 1. Qua tab 1 để lưu trước.")
    else:
        # tải dữ liệu train cho từng mã đã chọn
        prices_dict = {}
        for tk in st.session_state.selected_stocks:
            df_price = fetch_price_history(tk, train_start_str, train_end_str)
            if df_price is not None and not df_price.empty:
                prices_dict[tk] = df_price['Close']

        if len(prices_dict) == 0:
            st.error("Không lấy được dữ liệu train cho các mã đã chọn.")
        else:
            merged_prices = pd.DataFrame(prices_dict).dropna()

            mean_returns, cov_matrix = get_return_cov_matrix(merged_prices)

            st.subheader("🔥 Tối ưu tỷ trọng bằng Simulated Annealing (SA)")
            best_sa = optimize_weights_sa(mean_returns, cov_matrix, rf=rf_annual)
            tickers = mean_returns.index.tolist()
            st.write("Kết quả SA:")
            st.dataframe(pretty_weights_report(tickers, best_sa["weights"]))
            st.write(
                f"Sharpe={best_sa['sharpe']:.3f} | Return năm ~ {best_sa['return']*100:.2f}% | "
                f"Vol năm ~ {best_sa['vol']*100:.2f}%"
            )

            st.subheader("🧬 (Tuỳ chọn) Tối ưu bằng Genetic Algorithm (GA)")
            run_ga = st.checkbox("Chạy GA để so sánh với SA (mất thêm thời gian)")
            if run_ga:
                best_ga = optimize_weights_ga(
                    mean_returns,
                    cov_matrix,
                    rf=rf_annual,
                    population_size=100,
                    ngen=50,
                    cxpb=0.7,
                    mutpb=0.2
                )
                st.write("Kết quả GA:")
                st.dataframe(pretty_weights_report(tickers, best_ga["weights"]))
                st.write(
                    f"Sharpe={best_ga['sharpe']:.3f} | Return năm ~ {best_ga['return']*100:.2f}% | "
                    f"Vol năm ~ {best_ga['vol']*100:.2f}%"
                )
            else:
                best_ga = None

            st.markdown("### ✍ Chỉnh tay tỷ trọng cuối cùng bạn muốn dùng")
            manual_weights = {}
            total_preview = 0.0

            # đề xuất mặc định để user chỉnh: dùng SA
            default_weights_source = best_sa["weights"]

            for i, tk in enumerate(tickers):
                default_val_pct = float(default_weights_source[i]*100)
                val_pct = st.number_input(
                    f"Tỷ trọng {tk} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=round(default_val_pct,2),
                    step=0.5,
                    key=f"weight_input_{tk}"
                )
                manual_weights[tk] = val_pct/100.0
                total_preview += val_pct

            st.write(f"Tổng % bạn nhập: {round(total_preview,2)}% (mục tiêu ~100%)")

            if st.button("💾 Lưu trọng số cuối cùng để dùng cho bước 4"):
                st.session_state.final_weights = manual_weights
                st.success(f"ĐÃ LƯU trọng số cuối: {manual_weights}")


# ============= TAB 3: TỐI ƯU THAM SỐ WMA =============
with tab3:
    st.header("Bước 3. Tối ưu WMA cho từng mã (grid search) và lưu TOP tham số tốt nhất")

    if len(st.session_state.selected_stocks) == 0:
        st.warning("Chưa có mã từ Bước 1.")
    elif len(st.session_state.final_weights) == 0:
        st.warning("Chưa có trọng số từ Bước 2.")
    else:
        usable_tickers = [
            tk for tk in st.session_state.selected_stocks
            if st.session_state.final_weights.get(tk, 0) > 0
        ]

        st.write("Các mã sẽ tối ưu WMA (trọng số > 0):", usable_tickers)
        st.write(f"Giai đoạn TRAIN: {train_start_str} → {train_end_str}")

        run_opt = st.button("🚀 Chạy grid search WMA và lấy top 10% Sharpe cho từng mã")

        if run_opt:
            try:
                results_list = Parallel(n_jobs=-1)(
                    delayed(optimize_wma_grid_for_ticker)(
                        tk, train_start_str, train_end_str, top_percent=0.1
                    ) for tk in usable_tickers
                )
            except Exception as e:
                # Fallback chạy tuần tự nếu joblib lỗi môi trường
                st.warning(f"Joblib Parallel gặp lỗi ({e}), chuyển sang chạy tuần tự.")
                results_list = [
                    optimize_wma_grid_for_ticker(tk, train_start_str, train_end_str, top_percent=0.1)
                    for tk in usable_tickers
                ]

            for tk, df_top in zip(usable_tickers, results_list):
                if df_top is not None:
                    df_temp = df_top.copy()
                    if "Sharpe Ratio" not in df_temp.columns and "Sharpe" in df_temp.columns:
                        df_temp.rename(columns={"Sharpe": "Sharpe Ratio"}, inplace=True)
                    df_temp["Ticker"] = tk
                    df_temp["MA"] = "WMA"
                    st.session_state["wma_results"][tk] = df_temp

        st.subheader("Top tham số WMA theo Sharpe (đã lưu)")
        if len(st.session_state["wma_results"]) == 0:
            st.info("Chưa có kết quả tối ưu. Hãy nhấn nút chạy ở trên.")
        else:
            for tk, df_top in st.session_state["wma_results"].items():
                st.markdown(f"### {tk}")
                st.dataframe(df_top)
                st.caption("Các bộ Short/Long này sẽ được ENSEMBLE ở Bước 4. Bạn KHÔNG cần chọn tay 1 bộ.")


# ============= TAB 4: ENSEMBLE + BACKTEST =============
with tab4:
    st.header("Bước 4. Ensemble top-K WMA, áp luật thị trường VN, ghép danh mục và backtest")

    st.subheader("📌 Input từ các bước trước")
    st.write("Danh sách mã đã chọn:", st.session_state.selected_stocks)
    st.write("Trọng số cuối (Bước 2):", st.session_state.final_weights)
    st.write("Kết quả tối ưu WMA (Bước 3):",
             f"{len(st.session_state['wma_results'])} mã" if "wma_results" in st.session_state else "Chưa có")

    ready = True
    if len(st.session_state.selected_stocks) == 0:
        st.warning("Thiếu cổ phiếu từ Bước 1.")
        ready = False
    if len(st.session_state.final_weights) == 0:
        st.warning("Thiếu trọng số từ Bước 2.")
        ready = False
    if "wma_results" not in st.session_state or len(st.session_state["wma_results"]) == 0:
        st.warning("Thiếu kết quả tối ưu WMA từ Bước 3.")
        ready = False

    st.markdown("#### Giai đoạn backtest")
    st.write(f"Train: {train_start_str} → {train_end_str}")
    st.write(f"Test:  {test_start_str} → {test_end_str}")
    st.caption("Chiến lược được huấn luyện (chọn top tham số) trên TRAIN nhưng áp dụng & đánh giá Train+Test với luật T+2, stop-loss,...")

    run_final = st.button("🚀 Chạy ensemble + backtest danh mục cuối")

    if run_final and ready:
        try:
            results = build_portfolio_from_ensemble(
                wma_results_by_ticker=st.session_state["wma_results"],
                weights_by_ticker=st.session_state.final_weights,
                train_start_str=train_start_str,
                train_end_str=train_end_str,
                test_start_str=test_start_str,
                test_end_str=test_end_str,
                vote_threshold=VOTE_THRESHOLD,
                top_k=K,
                tcost_bps=tcost_bps,
                min_hold_days=MIN_HOLD_DAYS,
                stop_loss_pct=STOP_LOSS_PCT,
                cooldown_days=COOLDOWN_DAYS,
                t2_cash_days=T2_CASH_DAYS
            )

            nav_df = results["nav_df"]
            metrics_all = results["metrics_all"]
            metrics_train = results["metrics_train"]
            metrics_test = results["metrics_test"]
            eff_weights = results["eff_weights"]
            picked_debug = results["picked_rows_debug"]

            st.subheader("📉 NAV Danh Mục (Train+Test)")
            st.line_chart(nav_df["NAV"])

            st.subheader("📊 Hiệu suất danh mục")
            colA, colB, colC = st.columns(3)
            colA.metric("ALL Sharpe", f"{metrics_all['Sharpe']:.2f}" if pd.notna(metrics_all['Sharpe']) else "NaN")
            colB.metric("ALL CAGR",   f"{metrics_all['CAGR']*100:.2f}%" if pd.notna(metrics_all['CAGR']) else "NaN")
            colC.metric("ALL MaxDD",  f"{metrics_all['MaxDD']*100:.2f}%" if pd.notna(metrics_all['MaxDD']) else "NaN")

            st.markdown("**Train period (in-sample):**")
            st.write(metrics_train)

            st.markdown("**Test period (out-of-sample):**")
            st.write(metrics_test)

            st.subheader("⚖ Trọng số hiệu lực sau ràng buộc T+2 (eff_weights)")
            st.dataframe(eff_weights.tail())

            st.subheader("🔍 Các cặp WMA đã được ensemble (top-K mỗi mã)")
            st.dataframe(picked_debug)

            csv_data = nav_df.to_csv().encode('utf-8')
            st.download_button(
                "💾 Tải NAV danh mục (CSV)",
                csv_data,
                file_name="portfolio_nav_wma_ensemble.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi backtest danh mục: {e}")
    elif run_final and not ready:
        st.error("Thiếu dữ liệu đầu vào. Hãy đảm bảo bạn đã chạy và lưu xong Bước 1, 2, 3.")
