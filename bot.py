from telegram.ext import Updater, CommandHandler
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import io
import numpy as np
from datetime import datetime

# ==== Helper Functions ====
def safe_last(series):
    """Ambil nilai terakhir dari Series atau DataFrame dengan aman"""
    if series is None or len(series) == 0:
        return None

    val = series.iloc[-1]

    # kalau ternyata masih Series (multi kolom), ambil Close/Volume dulu
    if isinstance(val, pd.Series):
        val = val.iloc[0]

    try:
        val = float(val)
    except Exception:
        return None

    return val if pd.notna(val) else None

def get_current_price(symbol):
    """Mendapatkan harga real-time"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='1d', interval='1m')
        if len(data) > 0:
            return data['Close'].iloc[-1]
        else:
            # Fallback ke data harian
            data = ticker.history(period='1d')
            return data['Close'].iloc[-1] if len(data) > 0 else None
    except Exception:
        return None

def calculate_rsi(data, period=14):
    """Menghitung RSI"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def find_support_resistance(data, current_price, lookback_days=60):
    """Mencari level support dan resistance yang realistis"""
    recent_data = data.tail(lookback_days)

    # Support: cari level di BAWAH harga saat ini
    price_below = recent_data[recent_data['Low'] < current_price]['Low']
    if len(price_below) > 0:
        support = float(price_below.max())  # Convert ke float
    else:
        support = float(recent_data['Low'].min())

    # Resistance: cari level di ATAS harga saat ini
    price_above = recent_data[recent_data['High'] > current_price]['High']
    if len(price_above) > 0:
        resistance = float(price_above.min())  # Convert ke float
    else:
        resistance = float(recent_data['High'].max())

    return support, resistance

def find_improved_support_resistance(data, current_price, ma20, ma50, ma200, lookback_days=60):
    """Mencari level support dan resistance yang lebih akurat"""
    recent_data = data.tail(lookback_days)
    
    # Method 1: Recent highs and lows
    recent_high = float(recent_data['High'].max())
    recent_low = float(recent_data['Low'].min())
    
    # Method 2: Psychological levels (round numbers)
    base_level = round(current_price / 100) * 100
    psychological_levels = [base_level + i * 100 for i in range(-5, 6)]
    
    # Method 3: Moving averages as dynamic levels
    ma_levels = []
    if ma20: ma_levels.append(float(ma20))
    if ma50: ma_levels.append(float(ma50))
    if ma200: ma_levels.append(float(ma200))
    
    # Combine all levels
    all_levels = psychological_levels + ma_levels + [recent_high, recent_low]
    
    # Filter levels
    support_candidates = [level for level in all_levels if level < current_price * 0.99]
    resistance_candidates = [level for level in all_levels if level > current_price * 1.01]
    
    # Take strongest levels
    support = max(support_candidates) if support_candidates else recent_low
    resistance = min(resistance_candidates) if resistance_candidates else recent_high
    
    return float(support), float(resistance)

# ==== START ====
def start(update, context):
    update.message.reply_text(
        "Halo! Saya bot trading saham Indonesia.\n\n"
        "Saya dapat membantu analisis teknikal saham-saham BEI.\n\n"
        "Ketik /menu untuk melihat fitur yang tersedia."
    )

# ==== MENU ====
def menu(update, context):
    text = (
        "📌 Menu Bot Trading Saham:\n\n"
        "/ma <kode> → Cek Moving Average (MA10, MA20, MA50, MA100, MA200)\n"
        "/alert <kode> → Deteksi sinyal trading & volume\n"
        "/chart <kode> → Chart lengkap (Candle + Volume + RSI + MACD)\n"
        "/analysis <kode> → Analisis teknikal mendalam\n"
        "/faq → Penjelasan istilah trading\n\n"
        "Contoh: /ma BBCA"
    )
    update.message.reply_text(text)

# ==== MA ====
def ma(update, context):
    if len(context.args) != 1:
        update.message.reply_text("❌ Format salah. Contoh: /ma BBCA")
        return

    kode = context.args[0].upper()
    symbol = kode + ".JK"

    try:
        # Get data
        data = yf.download(symbol, period="1y", interval="1d")
        if len(data) == 0:
            update.message.reply_text(f"❌ Tidak menemukan data untuk {kode}")
            return

        # Calculate MAs
        data['MA10'] = data['Close'].rolling(10).mean()
        data['MA20'] = data['Close'].rolling(20).mean()
        data['MA50'] = data['Close'].rolling(50).mean()
        data['MA100'] = data['Close'].rolling(100).mean()
        data['MA200'] = data['Close'].rolling(200).mean()

        # Get current price (real-time)
        current_price = get_current_price(symbol)
        if current_price is None:
            current_price = safe_last(data['Close'])

        # Get MA values
        ma10 = safe_last(data['MA10'])
        ma20 = safe_last(data['MA20'])
        ma50 = safe_last(data['MA50'])
        ma100 = safe_last(data['MA100'])
        ma200 = safe_last(data['MA200'])

        # Format output
        harga_text = f"Rp {current_price:,.2f}" if current_price is not None else "Belum tersedia"

        ma_values = [
            ("MA10", ma10),
            ("MA20", ma20),
            ("MA50", ma50),
            ("MA100", ma100),
            ("MA200", ma200)
        ]

        ma_text = ""
        for name, value in ma_values:
            if value is not None:
                status = "🟢 DI ATAS" if current_price > value else "🔴 DI BAWAH"
                ma_text += f"{name}: Rp {value:,.2f} ({status})\n"
            else:
                ma_text += f"{name}: Belum tersedia\n"

        # Determine trend
        if all(v is not None for v in [ma10, ma20, ma50, ma100, ma200]):
            if ma10 > ma20 > ma50 > ma100 > ma200:
                trend = "📈 BULLISH KUAT"
            elif ma10 > ma20 > ma50:
                trend = "📈 BULLISH"
            elif ma200 > ma100 > ma50 > ma20 > ma10:
                trend = "📉 BEARISH KUAT"
            elif ma50 > ma20 > ma10:
                trend = "📉 BEARISH"
            else:
                trend = "➡️ SIDEWAYS"
        else:
            trend = "ℹ️ Data belum cukup"

        pesan = (
            f"📊 MOVING AVERAGE {kode}\n"
            f"💰 Harga Saat Ini: {harga_text}\n\n"
            f"{ma_text}\n"
            f"📈 Trend: {trend}"
        )
        update.message.reply_text(pesan)

    except Exception as e:
        update.message.reply_text(f"❌ Error: {str(e)}")

# ==== ALERT ====
def alert(update, context):
    if len(context.args) != 1:
        update.message.reply_text("❌ Format salah. Contoh: /alert BBCA")
        return

    kode = context.args[0].upper()
    symbol = kode + ".JK"

    try:
        data = yf.download(symbol, period="3mo", interval="1d")
        if len(data) == 0:
            update.message.reply_text(f"❌ Tidak menemukan data untuk {kode}")
            return

        # Calculate indicators
        data['MA20'] = data['Close'].rolling(20).mean()
        data['Vol20'] = data['Volume'].rolling(20).mean()
        data['RSI'] = calculate_rsi(data)

        # Get current values
        current_price = get_current_price(symbol) or safe_last(data['Close'])
        volume = safe_last(data['Volume'])
        ma20 = safe_last(data['MA20'])
        vol20 = safe_last(data['Vol20'])
        rsi = safe_last(data['RSI'])

        # Format values
        harga_text = f"Rp {current_price:,.2f}" if current_price is not None else "Belum tersedia"
        vol20_text = f"{vol20:,.0f}" if vol20 is not None else "Belum tersedia"
        rsi_text = f"{rsi:.2f}" if rsi is not None else "Belum tersedia"

        pesan = (
            f"🚨 ALERT {kode}\n"
            f"💰 Harga: {harga_text}\n"
            f"📊 Volume: {volume:,.0f}\n"
            f"📈 Rata2 Vol20: {vol20_text}\n"
            f"📊 RSI: {rsi_text}\n\n"
        )

        # Generate signals
        signals = []

        # Volume analysis
        if vol20 is not None and volume is not None:
            if volume > 2 * vol20:
                signals.append("🚨 VOLUME TINGGI: Kemungkinan ada aksi bandar")
            elif volume > 1.5 * vol20:
                signals.append("📈 Volume di atas rata-rata")

        # Price vs MA analysis
        if ma20 is not None and current_price is not None:
            if current_price > ma20:
                signals.append("🟢 Harga di atas MA20 (Bullish)")
            else:
                signals.append("🔴 Harga di bawah MA20 (Bearish)")

        # RSI analysis
        if rsi is not None:
            if rsi > 70:
                signals.append("⚠️ RSI Overbought")
            elif rsi < 30:
                signals.append("⚠️ RSI Oversold")

        # Determine overall signal
        if len(signals) == 0:
            signals.append("ℹ️ Tidak ada sinyal kuat")

        # Add signals to message
        for signal in signals:
            pesan += f"• {signal}\n"

        # Overall recommendation
        bullish_count = sum(1 for s in signals if "Bullish" in s or "di atas" in s)
        bearish_count = sum(1 for s in signals if "Bearish" in s or "di bawah" in s)

        if bullish_count >= 2:
            pesan += "\n🎯 Sinyal: BULLISH KUAT"
        elif bullish_count == 1:
            pesan += "\n🎯 Sinyal: BULLISH LEMAH"
        elif bearish_count >= 2:
            pesan += "\n🎯 Sinyal: BEARISH KUAT"
        elif bearish_count == 1:
            pesan += "\n🎯 Sinyal: BEARISH LEMAH"
        else:
            pesan += "\n🎯 Sinyal: SIDEWAYS / NETRAL"

        update.message.reply_text(pesan)

    except Exception as e:
        update.message.reply_text(f"❌ Error: {str(e)}")

# === CHART ===
def chart(update, context):
    try:
        if len(context.args) != 1:
            update.message.reply_text("❌ Format salah. Contoh: /chart BBCA")
            return

        kode = context.args[0].upper()
        symbol = kode + ".JK"

        # Download data
        data = yf.download(symbol, period="6mo", interval="1d", auto_adjust=True)

        if len(data) == 0:
            update.message.reply_text(f"❌ Tidak menemukan data untuk {kode}")
            return

        # Clean data
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        data = data.dropna()

        if data.empty or len(data) < 50:
            update.message.reply_text(f"⚠️ Data untuk {kode} terlalu sedikit.")
            return

        # Calculate indicators
        data["MA9"] = data["Close"].rolling(9).mean()
        data["MA20"] = data["Close"].rolling(20).mean()
        data["MA50"] = data["Close"].rolling(50).mean()
        data["RSI"] = calculate_rsi(data)

        # Calculate MACD
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal

        # Get current price untuk menentukan support/resistance
        current_price = get_current_price(symbol) or safe_last(data['Close'])

        # ===== IMPROVED SUPPORT/RESISTANCE UNTUK CHART =====
        def find_chart_support_resistance(data, current_price, lookback_days=60):
            """Mencari level support dan resistance yang optimal untuk chart"""
            recent_data = data.tail(lookback_days)

            # Method 1: Recent significant highs and lows
            recent_high = float(recent_data['High'].max())
            recent_low = float(recent_data['Low'].min())

            # Method 2: Psychological levels (round numbers)
            base_level = round(current_price / 100) * 100
            psychological_levels = [base_level + i * 100 for i in range(-5, 6)]

            # Method 3: Moving averages as dynamic levels
            ma9_val = safe_last(data["MA9"])
            ma20_val = safe_last(data["MA20"])
            ma50_val = safe_last(data["MA50"])

            ma_levels = []
            if ma9_val: ma_levels.append(float(ma9_val))
            if ma20_val: ma_levels.append(float(ma20_val))
            if ma50_val: ma_levels.append(float(ma50_val))

            # Combine all levels
            all_levels = psychological_levels + ma_levels + [recent_high, recent_low]

            # Filter dan pilih yang paling signifikan
            support_candidates = [level for level in all_levels if level < current_price * 0.99]
            resistance_candidates = [level for level in all_levels if level > current_price * 1.01]

            # Ambil 2 support terkuat (tertinggi) dan 2 resistance terkuat (terendah)
            support_levels = sorted(support_candidates, reverse=True)[:2] if support_candidates else [recent_low]
            resistance_levels = sorted(resistance_candidates)[:2] if resistance_candidates else [recent_high]

            # Pastikan ada jarak minimal 2% antara level
            min_gap = current_price * 0.02
            filtered_support = []
            filtered_resistance = []

            for level in support_levels:
                if not filtered_support or (filtered_support[-1] - level) >= min_gap:
                    filtered_support.append(level)

            for level in resistance_levels:
                if not filtered_resistance or (level - filtered_resistance[-1]) >= min_gap:
                    filtered_resistance.append(level)

            return filtered_support, filtered_resistance

        # Gunakan improved support resistance untuk chart
        support_levels, resistance_levels = find_chart_support_resistance(data, current_price, 60)

        # Create subplots
        apds = [
            mpf.make_addplot(data["MA9"], color='orange', width=0.7, label='MA9'),
            mpf.make_addplot(data["MA20"], color='blue', width=0.7, label='MA20'),
            mpf.make_addplot(data["MA50"], color='red', width=0.7, label='MA50'),
            mpf.make_addplot(data["RSI"], panel=1, color='purple', width=0.7, label='RSI'),
            mpf.make_addplot([70] * len(data), panel=1, color='red', linestyle='--', width=0.5),
            mpf.make_addplot([30] * len(data), panel=1, color='green', linestyle='--', width=0.5),
            mpf.make_addplot(macd, panel=2, color='blue', width=0.7, label='MACD'),
            mpf.make_addplot(signal, panel=2, color='red', width=0.7, label='SIGNAL'),
            mpf.make_addplot(histogram, type='bar', panel=2, color='gray', alpha=0.3, width=0.7)
        ]

        # Create the plot
        fig, axlist = mpf.plot(
            data,
            type="candle",
            style="yahoo",
            addplot=apds,
            volume=True,
            returnfig=True,
            figsize=(12, 10),
            panel_ratios=(3,1,1),
            tight_layout=True
        )

        ax_main = axlist[0]
        ax_rsi = axlist[2]
        ax_macd = axlist[3]

        # Plot support levels dengan warna dan annotation yang berbeda
        colors_support = ['green', 'lime']
        for i, s in enumerate(support_levels):
            if s < current_price:  # Pastikan support di bawah harga
                color = colors_support[i] if i < len(colors_support) else 'green'
                ax_main.axhline(s, color=color, linestyle='--', linewidth=2, alpha=0.8)
                ax_main.annotate(
                    f"S{i+1}: {s:.0f}", xy=(0.02, s),
                    xycoords=("axes fraction", "data"),
                    xytext=(0, 5), textcoords="offset points",
                    ha="left", va="bottom", color=color, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.8)
                )

        # Plot resistance levels dengan warna dan annotation yang berbeda
        colors_resistance = ['red', 'orange']
        for i, r in enumerate(resistance_levels):
            if r > current_price:  # Pastikan resistance di atas harga
                color = colors_resistance[i] if i < len(colors_resistance) else 'red'
                ax_main.axhline(r, color=color, linestyle='--', linewidth=2, alpha=0.8)
                ax_main.annotate(
                    f"R{i+1}: {r:.0f}", xy=(0.02, r),
                    xycoords=("axes fraction", "data"),
                    xytext=(0, 5), textcoords="offset points",
                    ha="left", va="bottom", color=color, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.8)
                )

        ax_main.set_title(f'Chart {kode} - {datetime.now().strftime("%Y-%m-%d")}', fontsize=14, fontweight='bold')
        ax_main.legend()

        ax_rsi.set_ylabel('RSI')
        ax_rsi.legend()

        ax_macd.set_ylabel('MACD')
        ax_macd.legend()

        # Save and send
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)

        price_text = f"Rp {current_price:,.2f}" if current_price else "N/A"

        update.message.reply_photo(
            photo=buf,
            caption=f"📊 Chart {kode}\n💰 Harga: {price_text}\n📈 Periode: 6 Bulan\n\nCandle + MA + RSI + MACD\nS: Support | R: Resistance"
        )
        plt.close(fig)

    except Exception as e:
        update.message.reply_text(f"❌ Error membuat chart: {str(e)}")

# ==== REVISED ENTRY POINT CALCULATION ====
def calculate_entry_points(current_price, support, resistance, trend_type):
    """Menghitung entry point yang logis untuk saham Indonesia (hanya bisa profit dari kenaikan)"""
    # Validasi input
    valid_current = current_price if current_price and not pd.isna(current_price) else 1000
    valid_support = support if support and not pd.isna(support) else valid_current * 0.95
    valid_resistance = resistance if resistance and not pd.isna(resistance) else valid_current * 1.05

    entry_points = {}

    if trend_type == "bullish":
        # BULLISH: Buy pada pullback ke support atau breakout resistance
        entry_points['conservative'] = valid_support * 1.02  # Sedikit di atas support
        entry_points['aggressive'] = min(valid_current * 0.995, valid_resistance * 0.98)
        entry_points['breakout'] = valid_resistance * 1.01   # Setelah breakout resistance
        entry_points['recommended'] = entry_points['conservative']

    elif trend_type == "bearish":
        # BEARISH: TUNGGU REVERSAL - jangan entry di trend turun
        # Untuk saham Indonesia, hindari entry saat bearish, tunggu konfirmasi reversal
        entry_points['wait_reversal'] = valid_support * 0.98  # Tunggu bounce dari support
        entry_points['wait_breakout'] = valid_resistance * 1.02  # Tunggu breakout resistance
        entry_points['avoid_entry'] = "HINDARI - Trend turun"
        entry_points['recommended'] = "TUNGGU KONFIRMASI REVERSAL"

    else:  # sideways
        # SIDEWAYS: Buy di support, sell di resistance
        entry_points['buy_near_support'] = valid_support * 1.02
        entry_points['sell_near_resistance'] = valid_resistance * 0.98
        entry_points['recommended'] = entry_points['buy_near_support']

    # Validasi final: pastikan tidak ada NaN dan nilai reasonable
    for key, value in entry_points.items():
        if isinstance(value, (int, float)) and (pd.isna(value) or value <= 0):
            entry_points[key] = valid_current

    return entry_points

# ==== REVISED TARGET CALCULATION ====
def calculate_targets_stoploss(current_price, support, resistance, trend_type, entry_points):
    """Menghitung target dan stop loss untuk saham Indonesia (hanya profit dari kenaikan)"""
    valid_current = current_price if current_price and not pd.isna(current_price) else 1000
    valid_support = support if support and not pd.isna(support) else valid_current * 0.95
    valid_resistance = resistance if resistance and not pd.isna(resistance) else valid_current * 1.05

    price_range = valid_resistance - valid_support

    if trend_type == "bullish":
        # BULLISH: TP1 di resistance, TP2 di atas resistance
        tp1 = valid_resistance
        tp2 = valid_resistance + price_range * 0.3  # 30% dari range di atas resistance
        
        # Pastikan TP2 > TP1 untuk bullish
        if tp2 <= tp1:
            tp2 = tp1 * 1.05

        # Stop loss di bawah support
        stop_loss = valid_support * 0.98

    elif trend_type == "bearish":
        # BEARISH: HINDARI TRADING - berikan target konservatif jika terpaksa
        # Untuk saham Indonesia, target harus selalu di atas entry (tidak bisa short)
        tp1 = valid_current * 1.02  # Target kecil 2%
        tp2 = valid_current * 1.05  # Target 5% (konservatif)
        
        # Stop loss ketat
        stop_loss = valid_current * 0.97

    else:  # sideways
        # SIDEWAYS: TP buy di resistance
        tp1 = valid_resistance
        tp2 = valid_resistance * 1.03  # Sedikit di atas resistance
        
        # Stop loss di bawah support
        stop_loss = valid_support * 0.98

    # Validasi akhir: pastikan target selalu lebih tinggi dari current price untuk profit
    if tp1 < valid_current:
        tp1 = valid_current * 1.03
    if tp2 < tp1:
        tp2 = tp1 * 1.02

    # Pastikan tidak ada NaN
    if pd.isna(tp1): tp1 = valid_current * 1.03
    if pd.isna(tp2): tp2 = valid_current * 1.06
    if pd.isna(stop_loss): stop_loss = valid_current * 0.97

    return float(tp1), float(tp2), float(stop_loss)

# ==== ANALYSIS ====
def analysis(update, context):
    if len(context.args) != 1:
        update.message.reply_text("❌ Format salah. Contoh: /analysis BBCA")
        return

    kode = context.args[0].upper()
    symbol = kode + ".JK"

    try:
        # Get data dengan periode lebih panjang untuk MA200
        data = yf.download(symbol, period="1y", interval="1d", auto_adjust=True)
        if len(data) == 0:
            update.message.reply_text(f"❌ Tidak menemukan data untuk {kode}")
            return

        # Calculate indicators
        data['MA9'] = data['Close'].rolling(9).mean()
        data['MA20'] = data['Close'].rolling(20).mean()
        data['MA50'] = data['Close'].rolling(50).mean()
        data['MA200'] = data['Close'].rolling(200).mean()
        data['RSI'] = calculate_rsi(data)
        data['Vol20'] = data['Volume'].rolling(20).mean()

        # Calculate MACD
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line

        # Get current values dengan safe extraction
        current_price = get_current_price(symbol) or safe_last(data['Close'])

        # Extract values properly dari Series
        def safe_extract(value):
            if value is None:
                return None
            if isinstance(value, pd.Series):
                return float(value.iloc[0]) if len(value) > 0 else None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        ma9 = safe_extract(safe_last(data['MA9']))
        ma20 = safe_extract(safe_last(data['MA20']))
        ma50 = safe_extract(safe_last(data['MA50']))
        ma200 = safe_extract(safe_last(data['MA200']))
        rsi = safe_extract(safe_last(data['RSI']))
        current_macd = safe_extract(safe_last(macd_line))
        current_signal = safe_extract(safe_last(signal_line))
        current_histogram = safe_extract(safe_last(histogram))
        current_volume = safe_extract(safe_last(data['Volume']))
        avg_volume = safe_extract(safe_last(data['Vol20']))

        # Gunakan improved support resistance
        support, resistance = find_improved_support_resistance(
            data, current_price, ma20, ma50, ma200, 60
        )

        # ===== TREND DETERMINATION =====
        trend_type = "sideways"
        trend_strength = "lemah"
        trend_explanation = ""

        if ma20 is not None and ma50 is not None and ma200 is not None:
            # Strong Bullish
            if ma9 > ma20 > ma50 > ma200:
                trend_type = "bullish"
                trend_strength = "kuat"
                trend_explanation = f"Semua Moving Average sejajar naik, trend BULLISH kuat"
            # Bullish
            elif ma20 > ma50 and ma50 > ma200:
                trend_type = "bullish"
                trend_strength = "sedang"
                trend_explanation = f"MA20 > MA50 > MA200, trend BULLISH"
            # Weak Bullish
            elif ma20 > ma50:
                trend_type = "bullish"
                trend_strength = "lemah"
                trend_explanation = f"MA20 > MA50, trend BULLISH lemah"
            # Strong Bearish
            elif ma200 > ma50 > ma20 > ma9:
                trend_type = "bearish"
                trend_strength = "kuat"
                trend_explanation = f"Semua Moving Average sejajar turun, trend BEARISH kuat"
            # Bearish
            elif ma50 > ma20:
                trend_type = "bearish"
                trend_strength = "sedang"
                trend_explanation = f"MA50 > MA20, trend BEARISH"
            # Sideways
            else:
                trend_type = "sideways"
                trend_explanation = f"MA saling berpotongan, trend SIDEWAYS"

        # ===== RSI ANALYSIS =====
        rsi_analysis = ""
        rsi_signal = "netral"

        if rsi is not None:
            if rsi > 70:
                rsi_signal = "overbought"
                rsi_analysis = f"RSI {rsi:.1f} berada di zona overbought (>70), menunjukkan potensi koreksi atau konsolidasi"
            elif rsi > 60:
                rsi_signal = "bullish kuat"
                rsi_analysis = f"RSI {rsi:.1f} di zona bullish kuat (60-70), momentum naik masih berpotensi lanjut"
            elif rsi > 50:
                rsi_signal = "bullish lemah"
                rsi_analysis = f"RSI {rsi:.1f} di zona bullish lemah (50-60), momentum positif tapi perlu konfirmasi"
            elif rsi > 40:
                rsi_signal = "netral"
                rsi_analysis = f"RSI {rsi:.1f} di zona netral (40-50), tidak ada momentum kuat"
            elif rsi > 30:
                rsi_signal = "bearish lemah"
                rsi_analysis = f"RSI {rsi:.1f} di zona bearish lemah (30-40), momentum negatif tapi perlu konfirmasi"
            else:
                rsi_signal = "oversold"
                rsi_analysis = f"RSI {rsi:.1f} berada di zona oversold (<30), menunjukkan potensi bounce atau reversal"

        # ===== MACD ANALYSIS =====
        macd_analysis = ""
        macd_signal = "netral"

        if current_macd is not None and current_signal is not None and current_histogram is not None:
            if current_macd > current_signal and current_histogram > 0:
                macd_signal = "bullish"
                macd_analysis = "MACD di atas signal line dan histogram positif, mengindikasikan momentum bullish"
            elif current_macd < current_signal and current_histogram < 0:
                macd_signal = "bearish"
                macd_analysis = "MACD di bawah signal line dan histogram negatif, mengindikasikan momentum bearish"
            else:
                macd_signal = "netral"
                macd_analysis = "MACD dan signal line berdekatan, momentum tidak jelas"

        # ===== VOLUME ANALYSIS =====
        volume_analysis = ""
        volume_signal = "normal"

        if current_volume is not None and avg_volume is not None:
            volume_ratio = current_volume / avg_volume
            if volume_ratio > 2.5:
                volume_signal = "sangat tinggi"
                volume_analysis = f"Volume {volume_ratio:.1f}x rata-rata, kemungkinan ada aksi institusi atau berita"
            elif volume_ratio > 1.5:
                volume_signal = "tinggi"
                volume_analysis = f"Volume {volume_ratio:.1f}x rata-rata, minat trading meningkat"
            elif volume_ratio > 0.8:
                volume_signal = "normal"
                volume_analysis = "Volume dalam range normal"
            else:
                volume_signal = "rendah"
                volume_analysis = f"Volume hanya {volume_ratio:.1f}x rata-rata, minat trading rendah"

        # ===== SUPPORT/RESISTANCE ANALYSIS =====
        support_resistance_analysis = ""

        # Hitung jarak ke support dan resistance
        distance_to_support = ((current_price - support) / current_price) * 100
        distance_to_resistance = ((resistance - current_price) / current_price) * 100

        if distance_to_support < 3:
            support_resistance_analysis = f"Harga sangat dekat dengan support ({distance_to_support:.1f}%), perlu waspada breakdown"
        elif distance_to_resistance < 3:
            support_resistance_analysis = f"Harga sangat dekat dengan resistance ({distance_to_resistance:.1f}%), peluang breakout"
        else:
            support_resistance_analysis = f"Harga berada di tengah range, {distance_to_support:.1f}% ke support dan {distance_to_resistance:.1f}% ke resistance"

        # ===== CALCULATE ENTRY POINTS =====
        entry_points = calculate_entry_points(current_price, support, resistance, trend_type)

        # ===== CALCULATE TARGETS & STOP LOSS =====
        tp1, tp2, stop_loss = calculate_targets_stoploss(current_price, support, resistance, trend_type, entry_points)

        # ===== GENERATE RECOMMENDATIONS =====
        recommendations = []
        explanation_parts = []

        # Trend-based recommendations - REVISED untuk saham Indonesia
        if trend_type == "bullish" and trend_strength == "kuat":
            recommendations.extend([
                "✅ BUY: Trend bullish kuat, preferensi beli pada pullback",
                "🎯 Entry: Di area support atau breakout resistance",
                "📈 Target: Multiple resistance levels",
                "🛑 Stop Loss: Di bawah support terdekat"
            ])
            explanation_parts.append("Kondisi ideal untuk trading dengan bias beli (buy on dip)")

        elif trend_type == "bullish":
            recommendations.extend([
                "🟡 BUY dengan konfirmasi: Trend bullish tapi perlu konfirmasi volume",
                "🎯 Entry: Tunggu konfirmasi breakout atau bounce dari support",
                "📈 Target: Resistance berikutnya",
                "🛑 Stop Loss: Ketat di bawah support"
            ])
            explanation_parts.append("Bias beli tapi perlu konfirmasi tambahan")

        elif trend_type == "bearish" and trend_strength == "kuat":
            # REVISED: Untuk bearish kuat, hindari entry dan tunggu reversal
            recommendations.extend([
                "🔴 HINDARI BUY: Trend bearish kuat, tunggu konfirmasi reversal",
                "💡 Strategy: Wait & See - tunggu bounce dari support atau breakout resistance",
                "⚠️ Warning: Jangan mencoba catch falling knife",
                "🔄 Alternatif: Cari saham lain dengan trend bullish"
            ])
            explanation_parts.append("Kondisi risk tinggi, hindari posisi beli")

        elif trend_type == "bearish":
            # REVISED: Untuk bearish, sangat hati-hati
            recommendations.extend([
                "🟡 HATI-HATI: Trend bearish, hanya untuk trader experienced",
                "🎯 Entry: HANYA jika ada konfirmasi reversal kuat",
                "📈 Target: Konservatif (2-5%)",
                "🛑 Stop Loss: Sangat ketat (3% dari entry)"
            ])
            explanation_parts.append("Bias hindari, hanya trade dengan konfirmasi kuat")

        else:  # sideways
            recommendations.extend([
                "🟡 RANGE TRADING: Harga bergerak sideways",
                "🎯 Entry Buy: Di dekat support",
                "🎯 Entry Sell: Di dekat resistance (profit taking)",
                "📊 Target: Sisi berlawanan dari range",
                "🛑 Stop Loss: Di luar range"
            ])
            explanation_parts.append("Trading range-bound, buy low sell high")

        # RSI-based adjustments
        if rsi_signal == "overbought":
            recommendations.append("⚠️ Hati-hati: RSI overbought, pertimbangkan take profit")
            explanation_parts.append("Kondisi overbought berpotensi koreksi")
        elif rsi_signal == "oversold" and trend_type == "bearish":
            recommendations.append("💡 Opportunity: RSI oversold, pantau reversal pattern")
            explanation_parts.append("Kondisi oversold berpotensi bounce, tapi tunggu konfirmasi")
        elif rsi_signal == "oversold":
            recommendations.append("💡 Opportunity: RSI oversold, peluang buy dengan konfirmasi")
            explanation_parts.append("Kondisi oversold berpotensi bounce")

        # Volume-based adjustments
        if volume_signal in ["tinggi", "sangat tinggi"]:
            recommendations.append("📊 Volume tinggi mengkonfirmasi pergerakan harga")
        elif volume_signal == "rendah":
            recommendations.append("⚠️ Volume rendah, kurang konfirmasi - hati-hati false signal")

        # ===== FINAL ANALYSIS TEXT =====
        # Format entry points text berdasarkan trend type
        if trend_type == "bullish":
            entry_text = f"""🎪 ENTRY POINTS:
• Conservative: Rp {entry_points['conservative']:,.0f}
• Aggressive: Rp {entry_points['aggressive']:,.0f}
• Breakout: Rp {entry_points['breakout']:,.0f}"""
        elif trend_type == "bearish":
            # REVISED: Untuk bearish, tampilkan warning
            entry_text = f"""🎪 ENTRY POINTS (HATI-HATI!):
• Tunggu Reversal: Rp {entry_points['wait_reversal']:,.0f}
• Breakout Bullish: Rp {entry_points['wait_breakout']:,.0f}
• {entry_points['avoid_entry']}"""
        else:
            entry_text = f"""🎪 ENTRY POINTS:
• Buy Near Support: Rp {entry_points['buy_near_support']:,.0f}
• Sell Near Resistance: Rp {entry_points['sell_near_resistance']:,.0f}"""

        # Hitung risk/reward ratio
        if 'recommended' in entry_points and isinstance(entry_points['recommended'], (int, float)):
            entry_price = entry_points['recommended']
            reward = (tp1 - entry_price) / entry_price * 100
            risk = (entry_price - stop_loss) / entry_price * 100
            if risk > 0:
                rr_ratio = reward / risk
                rr_text = f"{rr_ratio:.1f}:1"
            else:
                rr_text = "N/A"
        else:
            rr_text = "N/A"

        analysis_text = f"""
📊 Analisa {kode} Harian

🎯 TREND: {trend_type.upper()} ({trend_strength.upper()})
💰 Harga Saat Ini: Rp {current_price:,.2f}
📊 Support Terdekat: Rp {support:,.0f}
📈 Resistance Terdekat: Rp {resistance:,.0f}

{entry_text}

🎯 TARGET & RISK:
• TP1: Rp {tp1:,.0f} (+{((tp1-current_price)/current_price*100):.1f}%)
• TP2: Rp {tp2:,.0f} (+{((tp2-current_price)/current_price*100):.1f}%)
• Stop Loss: Rp {stop_loss:,.0f} (-{((current_price-stop_loss)/current_price*100):.1f}%)
• Risk/Reward: {rr_text}

📈 INDIKATOR TEKNIKAL:
• RSI: {rsi:.1f} ({rsi_signal})
• MACD: {macd_signal}
• Volume: {volume_signal}

🤖 PENJELASAN DETIL:
{trend_explanation}

{rsi_analysis}

{macd_analysis}

{volume_analysis}

{support_resistance_analysis}

💡 REKOMENDASI TRADING:
"""

        # Add recommendations
        for rec in recommendations:
            analysis_text += f"• {rec}\n"

        # Add final notes dengan penekanan pada safety
        analysis_text += f"""
📝 STRATEGI EXECUTION:
- {'Hold dan monitor level kunci' if trend_type == 'bullish' else 'Tunggu di pinggir dan pantau reversal' if trend_type == 'bearish' else 'Trade dalam range'}
- Money management: Jangan risk lebih dari 2% equity per trade
- {'Bias BUY dengan konfirmasi' if trend_type == 'bullish' else 'BIAS HINDARI - tunggu reversal' if trend_type == 'bearish' else 'Bias BUY di support, SELL di resistance'}

⚠️ DISCLAIMER KHUSUS UNTUK BEARISH:
• Saham Indonesia TIDAK BISA SHORT
• Profit HANYA dari kenaikan harga
• Hindari catching falling knife
• Tunggu konfirmasi reversal yang kuat

📚 Analisis ini hanya untuk edukasi, bukan rekomendasi beli/jual.
Pastikan konfirmasi dengan analisis fundamental dan kondisi market.
"""

        update.message.reply_text(analysis_text)

    except Exception as e:
        update.message.reply_text(f"❌ Error dalam analisis: {str(e)}")

# ==== FAQ ====
def faq(update, context):
    faq_text = """
📚 FAQ - Istilah Trading

• MA (Moving Average) = Rata-rata harga dalam periode tertentu
• MA20 = Rata-rata harga 20 hari terakhir
• Support = Level harga dimana biasanya terjadi pembelian
• Resistance = Level harga dimana biasanya terjadi penjualan
• RSI (Relative Strength Index) = Indikator momentum (0-100)
  - >70 = Overbought (jenuh beli)
  - <30 = Oversold (jenuh jual)
• MACD = Indikator trend dan momentum
• Volume = Jumlah saham yang diperdagangkan
• Trend = Arah pergerakan harga
• Bullish = Kondisi market naik
• Bearish = Kondisi market turun
• Sideways = Harga bergerak dalam range tertentu

⚠️ PENTING UNTUK SAHAM INDONESIA:
- Hanya bisa profit dari kenaikan harga (tidak bisa short)
- Hindari trading saat trend bearish kuat
- Tunggu konfirmasi reversal untuk entry
- Risk management adalah kunci utama

📖 Tips:
- Gunakan multiple indikator untuk konfirmasi
- Selalu gunakan stop loss
- Jangan emotional trading
- Risk management yang baik kunci sukses
"""

    update.message.reply_text(faq_text)

# ==== MAIN ====
def main():
    TOKEN = "8252806375:AAFS9jvvyeAHVDpu2dfRtTag73tLBmB1xCY"
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("menu", menu))
    dp.add_handler(CommandHandler("ma", ma))
    dp.add_handler(CommandHandler("alert", alert))
    dp.add_handler(CommandHandler("chart", chart))
    dp.add_handler(CommandHandler("analysis", analysis))
    dp.add_handler(CommandHandler("faq", faq))

    print("Bot sedang berjalan...")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()