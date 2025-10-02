from telegram.ext import Updater, CommandHandler
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import io
import numpy as np
from datetime import datetime, timedelta
import os
import logging
import requests
from bs4 import BeautifulSoup
import time

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

# ==== DATA SOURCE ALTERNATIVES ====
def get_indonesia_stock_data(symbol, period="6mo"):
    """
    Mendapatkan data saham Indonesia dari berbagai sumber
    """
    # Coba yfinance dulu
    try:
        yf_symbol = symbol + ".JK"
        data = yf.download(yf_symbol, period=period, interval="1d", progress=False)
        if len(data) > 0:
            logger.info(f"Data found via yfinance for {symbol}")
            return data
    except Exception as e:
        logger.warning(f"yfinance failed for {symbol}: {e}")
    
    # Fallback: Gunakan data dummy/simulasi untuk testing
    logger.info(f"Using simulated data for {symbol}")
    return create_sample_data(symbol)

def create_sample_data(symbol):
    """
    Membuat data sample untuk testing ketika data real tidak tersedia
    """
    dates = pd.date_range(end=datetime.now(), periods=180, freq='D')
    
    # Harga dasar berdasarkan simbol
    base_price = {
        'BBCA': 9500, 'BBRI': 4500, 'BMRI': 8500, 'BBNI': 9500,
        'TLKM': 3500, 'ASII': 5200, 'UNVR': 3800, 'ICBP': 9500,
        'GOTO': 80, 'BRIS': 1500, 'ANTM': 1500, 'ADRO': 2000,
        'BBHI': 1200, 'BUKA': 150, 'EMTK': 500, 'ARTO': 3000
    }
    
    base = base_price.get(symbol, 1000)
    
    np.random.seed(hash(symbol) % 10000)
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = base * (1 + np.cumsum(returns))
    
    data = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 50000000, len(dates))
    }, index=dates)
    
    return data

def get_current_price_fallback(symbol):
    """
    Fallback untuk mendapatkan harga current
    """
    try:
        # Coba dari yfinance
        ticker = yf.Ticker(symbol + ".JK")
        data = ticker.history(period='1d', interval='1m', progress=False)
        if len(data) > 0:
            return data['Close'].iloc[-1]
    except:
        pass
    
    # Fallback ke harga simulasi
    base_price = {
        'BBCA': 9500, 'BBRI': 4500, 'BMRI': 8500, 'BBNI': 9500,
        'TLKM': 3500, 'ASII': 5200, 'UNVR': 3800, 'ICBP': 9500,
        'GOTO': 80, 'BRIS': 1500, 'ANTM': 1500, 'ADRO': 2000
    }
    return base_price.get(symbol, 1000)

# ==== UPDATED HELPER FUNCTIONS ====
def safe_last(series):
    """Ambil nilai terakhir dari Series atau DataFrame dengan aman"""
    if series is None or len(series) == 0:
        return None

    val = series.iloc[-1]

    if isinstance(val, pd.Series):
        val = val.iloc[0]

    try:
        val = float(val)
    except Exception:
        return None

    return val if pd.notna(val) else None

def get_current_price(symbol):
    """Mendapatkan harga real-time dengan fallback"""
    price = get_current_price_fallback(symbol)
    if price is None:
        # Final fallback
        return 1000
    return price

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
    
    price_below = recent_data[recent_data['Low'] < current_price]['Low']
    if len(price_below) > 0:
        support = float(price_below.max())
    else:
        support = float(recent_data['Low'].min())
    
    price_above = recent_data[recent_data['High'] > current_price]['High']
    if len(price_above) > 0:
        resistance = float(price_above.min())
    else:
        resistance = float(recent_data['High'].max())
    
    return support, resistance

# ==== IMPROVED ENTRY POINT CALCULATION FOR STOCKS ====
def calculate_entry_points(current_price, support, resistance, trend_type):
    """Menghitung entry point untuk saham Indonesia (LONG ONLY)"""
    valid_current = current_price if current_price and not pd.isna(current_price) else 1000
    valid_support = support if support and not pd.isna(support) else valid_current * 0.95
    valid_resistance = resistance if resistance and not pd.isna(resistance) else valid_current * 1.05
    
    entry_points = {}
    
    if trend_type == "bullish":
        # BULLISH: Entry di pullback ke support atau breakout
        entry_points['conservative'] = valid_support * 1.01  # Sedikit di atas support
        entry_points['aggressive'] = valid_current * 0.995   # Harga saat ini
        entry_points['breakout'] = valid_resistance * 1.005  # Setelah breakout resistance
        entry_points['recommended'] = entry_points['conservative']  # Paling aman
        
    elif trend_type == "bearish":
        # BEARISH: HINDARI atau entry sangat konservatif
        entry_points['avoid'] = "HINDARI - Trend Bearish"
        entry_points['conservative'] = valid_support * 1.02  # Hanya jika bounce dari support
        entry_points['wait'] = "TUNGGU reversal pattern"
        entry_points['recommended'] = entry_points['conservative']
        
    else:  # sideways
        # SIDEWAYS: Buy di support area
        entry_points['buy_support'] = valid_support * 1.01
        entry_points['current_price'] = valid_current
        entry_points['breakout'] = valid_resistance * 1.005
        entry_points['recommended'] = entry_points['buy_support']
    
    # Validasi: pastikan entry points reasonable
    for key, value in entry_points.items():
        if isinstance(value, (int, float)) and (pd.isna(value) or value <= 0):
            entry_points[key] = valid_current
    
    return entry_points

# ==== IMPROVED TARGET CALCULATION FOR STOCKS (LONG ONLY) ====
def calculate_targets_stoploss(current_price, support, resistance, trend_type, entry_points):
    """Menghitung target dan stop loss untuk saham Indonesia (LONG ONLY)"""
    valid_current = current_price if current_price and not pd.isna(current_price) else 1000
    valid_support = support if support and not pd.isna(support) else valid_current * 0.95
    valid_resistance = resistance if resistance and not pd.isna(resistance) else valid_current * 1.05
    
    # Untuk saham Indonesia, kita hanya consider LONG positions
    # TP1 dan TP2 harus selalu > entry price
    # Stop loss harus < entry price
    
    if trend_type == "bullish":
        # BULLISH: Multiple target di atas
        tp1 = valid_resistance
        tp2 = valid_resistance * 1.05  # Target lebih tinggi
        stop_loss = valid_support * 0.98  # Stop di bawah support
        
        # Pastikan TP1 > current_price dan TP2 > TP1
        if tp1 <= current_price:
            tp1 = current_price * 1.03
        if tp2 <= tp1:
            tp2 = tp1 * 1.03
            
    elif trend_type == "bearish":
        # BEARISH: Hindari entry atau target konservatif
        # Untuk bearish, kita tunggu di support atau hindari
        tp1 = valid_current * 1.02  # Target sangat konservatif
        tp2 = valid_current * 1.05  # Target kecil
        stop_loss = valid_current * 0.95  # Stop ketat
        
    else:  # sideways
        # SIDEWAYS: Buy di support, target di resistance
        tp1 = valid_resistance * 0.98  # Target di bawah resistance
        tp2 = valid_resistance * 1.02  # Target breakout
        stop_loss = valid_support * 0.98  # Stop di bawah support
        
        # Pastikan target > current price untuk sideways
        if tp1 <= current_price:
            tp1 = current_price * 1.02
        if tp2 <= tp1:
            tp2 = tp1 * 1.02
    
    # Validasi akhir: pastikan selalu TP2 > TP1 > Entry > Stop Loss
    recommended_entry = entry_points.get('recommended', valid_current)
    
    if isinstance(recommended_entry, (int, float)):
        if tp1 <= recommended_entry:
            tp1 = recommended_entry * 1.03
        if tp2 <= tp1:
            tp2 = tp1 * 1.03
        if stop_loss >= recommended_entry:
            stop_loss = recommended_entry * 0.97
    
    return float(tp1), float(tp2), float(stop_loss)

# ==== IMPROVED SUPPORT/RESISTANCE CALCULATION =====
def find_improved_support_resistance(data, current_price, ma20, ma50, ma200, lookback_days=60):
    """Mencari level support dan resistance yang lebih akurat"""
    recent_data = data.tail(lookback_days)
    
    # Method 1: Recent significant highs and lows
    recent_high = float(recent_data['High'].max())
    recent_low = float(recent_data['Low'].min())
    
    # Method 2: Psychological levels (round numbers)
    base_level = round(current_price / 100) * 100
    psychological_levels = [base_level + i * 100 for i in range(-5, 6)]
    
    # Method 3: Moving averages as dynamic levels
    ma_levels = []
    if ma20 is not None: 
        ma20_val = float(ma20) if not isinstance(ma20, pd.Series) else float(ma20.iloc[0])
        ma_levels.append(ma20_val)
    if ma50 is not None: 
        ma50_val = float(ma50) if not isinstance(ma50, pd.Series) else float(ma50.iloc[0])
        ma_levels.append(ma50_val)
    if ma200 is not None: 
        ma200_val = float(ma200) if not isinstance(ma200, pd.Series) else float(ma200.iloc[0])
        ma_levels.append(ma200_val)
    
    # Combine all levels
    all_levels = psychological_levels + ma_levels + [recent_high, recent_low]
    
    # Filter support (below current price) and resistance (above current price)
    support_candidates = [level for level in all_levels if level < current_price * 0.99]
    resistance_candidates = [level for level in all_levels if level > current_price * 1.01]
    
    # Ambil support tertinggi dan resistance terendah
    support = max(support_candidates) if support_candidates else recent_low
    resistance = min(resistance_candidates) if resistance_candidates else recent_high
    
    # Pastikan ada jarak minimal 2% dari current price
    min_distance = current_price * 0.02
    if current_price - support < min_distance:
        support = max(support, current_price - min_distance)
    if resistance - current_price < min_distance:
        resistance = min(resistance, current_price + min_distance)
    
    # Pastikan support < resistance
    if support >= resistance:
        if current_price > recent_high * 0.8:  # Jika harga cukup tinggi
            resistance = current_price * 1.05
            support = current_price * 0.95
        else:
            support = current_price * 0.95
            resistance = current_price * 1.05
    
    return float(support), float(resistance)

# ==== COMMAND FUNCTIONS ====
def start(update, context):
    update.message.reply_text(
        "Halo! Saya bot trading saham Indonesia.\n\n"
        "Saya dapat membantu analisis teknikal saham-saham BEI.\n\n"
        "Ketik /menu untuk melihat fitur yang tersedia."
    )

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

def ma(update, context):
    if len(context.args) != 1:
        update.message.reply_text("❌ Format salah. Contoh: /ma BBCA")
        return

    kode = context.args[0].upper()
    
    try:
        # Gunakan data source alternatif
        data = get_indonesia_stock_data(kode, period="1y")
        
        if len(data) == 0:
            update.message.reply_text(f"❌ Tidak menemukan data untuk {kode}")
            return

        # Calculate MAs
        data['MA10'] = data['Close'].rolling(10).mean()
        data['MA20'] = data['Close'].rolling(20).mean()
        data['MA50'] = data['Close'].rolling(50).mean()
        data['MA100'] = data['Close'].rolling(100).mean()
        data['MA200'] = data['Close'].rolling(200).mean()

        # Get current price
        current_price = get_current_price(kode)
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
            f"📈 Trend: {trend}\n\n"
            f"💡 Catatan: Menggunakan data simulasi untuk demo"
        )
        update.message.reply_text(pesan)

    except Exception as e:
        logger.error(f"Error in /ma: {e}")
        update.message.reply_text(f"❌ Error: {str(e)}")

def alert(update, context):
    if len(context.args) != 1:
        update.message.reply_text("❌ Format salah. Contoh: /alert BBCA")
        return

    kode = context.args[0].upper()
    
    try:
        data = get_indonesia_stock_data(kode, period="3mo")
        if len(data) == 0:
            update.message.reply_text(f"❌ Tidak menemukan data untuk {kode}")
            return

        # Calculate indicators
        data['MA20'] = data['Close'].rolling(20).mean()
        data['Vol20'] = data['Volume'].rolling(20).mean()
        data['RSI'] = calculate_rsi(data)

        # Get current values
        current_price = get_current_price(kode) or safe_last(data['Close'])
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

        pesan += "\n\n💡 Catatan: Menggunakan data simulasi untuk demo"

        update.message.reply_text(pesan)

    except Exception as e:
        logger.error(f"Error in /alert: {e}")
        update.message.reply_text(f"❌ Error: {str(e)}")

def chart(update, context):
    try:
        if len(context.args) != 1:
            update.message.reply_text("❌ Format salah. Contoh: /chart BBCA")
            return

        kode = context.args[0].upper()
        
        # Download data dengan source alternatif
        data = get_indonesia_stock_data(kode, period="6mo")
        
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
        current_price = get_current_price(kode) or safe_last(data['Close'])
        
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
            caption=f"📊 Chart {kode}\n💰 Harga: {price_text}\n📈 Periode: 6 Bulan\n\nCandle + MA + RSI + MACD\n💡 Data Simulasi"
        )
        plt.close(fig)

    except Exception as e:
        logger.error(f"Error in /chart: {e}")
        update.message.reply_text(f"❌ Error membuat chart: {str(e)}")

def analysis(update, context):
    if len(context.args) != 1:
        update.message.reply_text("❌ Format salah. Contoh: /analysis BBCA")
        return

    kode = context.args[0].upper()
    
    try:
        # Get data dengan source alternatif
        data = get_indonesia_stock_data(kode, period="1y")
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
        current_price = get_current_price(kode) or safe_last(data['Close'])
        
        # FIXED: Extract values properly dari Series
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

        # Gunakan improved support resistance dengan parameter yang benar
        support, resistance = find_improved_support_resistance(data, current_price, ma20, ma50, ma200, 60)

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
            elif ma20 < ma50 and ma50 < ma200:
                trend_type = "bearish"
                trend_strength = "kuat"
                trend_explanation = f"Semua Moving Average sejajar turun, trend BEARISH kuat"
            # Bearish  
            elif ma20 < ma50:
                trend_type = "bearish"
                trend_strength = "sedang"
                trend_explanation = f"MA20 < MA50, trend BEARISH"
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

        # Trend-based recommendations untuk saham Indonesia
        if trend_type == "bullish" and trend_strength == "kuat":
            recommendations.extend([
                "✅ BUY: Trend bullish kuat, preferensi beli pada pullback",
                "🎯 Entry: Di area support atau breakout resistance", 
                "📈 Target: Multiple resistance levels",
                "🛑 Stop Loss: Di bawah support terdekat",
                "💰 Strategy: Buy on dip"
            ])
            explanation_parts.append("Kondisi ideal untuk accumulation")
            
        elif trend_type == "bullish":
            recommendations.extend([
                "🟡 BUY dengan konfirmasi: Trend bullish tapi perlu konfirmasi volume",
                "🎯 Entry: Tunggu konfirmasi breakout atau bounce dari support",
                "📈 Target: Resistance berikutnya", 
                "🛑 Stop Loss: Ketat di bawah support",
                "💰 Strategy: Wait for confirmation"
            ])
            explanation_parts.append("Bias beli tapi perlu konfirmasi tambahan")
            
        elif trend_type == "bearish" and trend_strength == "kuat":
            recommendations.extend([
                "🔴 HINDARI: Trend bearish kuat, jangan buka posisi baru",
                "💡 Action: Jika sudah hold, consider cut loss atau averaging",
                "🎯 Wait For: Reversal pattern atau bounce dari support kuat",
                "📉 Risk: High probability of further downside",
                "💰 Strategy: Stay cash, wait for better entry"
            ])
            explanation_parts.append("Kondisi risk tinggi, preferensi hindari")
            
        elif trend_type == "bearish":
            recommendations.extend([
                "🔴 HINDARI / TUNGGU: Trend bearish, risk tinggi",
                "💡 Khusus: Hanya untuk trader experienced dengan risk management ketat", 
                "🎯 Entry: Hanya jika ada bounce dari support kuat dengan volume",
                "📈 Target: Sangat konservatif (2-3%)",
                "🛑 Stop Loss: Sangat ketat (3-5%)",
                "💰 Strategy: Avoid or very small position"
            ])
            explanation_parts.append("Bias hindari, hanya untuk experienced trader")
            
        else:  # sideways
            recommendations.extend([
                "🟡 RANGE TRADING: Harga bergerak sideways",
                "🎯 Entry Buy: Di dekat support dengan konfirmasi bounce",
                "🎯 Entry Sell: Take profit di resistance",
                "📊 Target: Sisi resistance dari range",
                "🛑 Stop Loss: Di bawah support range",
                "💰 Strategy: Buy low, sell high dalam range"
            ])
            explanation_parts.append("Trading range-bound, jangan FOMO breakout")

        # RSI-based adjustments
        if rsi_signal == "overbought":
            recommendations.append("⚠️ Hati-hati: RSI overbought, pertimbangkan take profit")
            explanation_parts.append("Kondisi overbought berpotensi koreksi")
        elif rsi_signal == "oversold":
            recommendations.append("💡 Opportunity: RSI oversold, pantau reversal pattern")
            explanation_parts.append("Kondisi oversold berpotensi bounce")

        # Volume-based adjustments  
        if volume_signal in ["tinggi", "sangat tinggi"]:
            recommendations.append("📊 Volume tinggi mengkonfirmasi pergerakan harga")
        elif volume_signal == "rendah":
            recommendations.append("⚠️ Volume rendah, kurang konfirmasi - hati-hati false signal")

        # ===== FINAL ANALYSIS TEXT =====
        # Format entry points text berdasarkan trend type
        if trend_type == "bullish":
            entry_text = f"""🎪 ENTRY POINTS (BIAS BUY):
• Conservative (BUY): Rp {entry_points['conservative']:,.0f} - Pullback ke support
• Aggressive (BUY): Rp {entry_points['aggressive']:,.0f} - Current price  
• Breakout (BUY): Rp {entry_points['breakout']:,.0f} - Above resistance"""
        elif trend_type == "bearish":
            if 'avoid' in entry_points:
                entry_text = f"""🎪 ENTRY POINTS (BIAS AVOID):
• {entry_points['avoid']}
• Conservative (RISKY): Rp {entry_points['conservative']:,.0f} - Hanya jika bounce kuat
• {entry_points['wait']}"""
            else:
                entry_text = f"""🎪 ENTRY POINTS (BIAS AVOID):
• HINDARI posisi baru
• Tunggu reversal confirmation"""
        else:
            entry_text = f"""🎪 ENTRY POINTS (RANGE TRADING):
• Buy Support: Rp {entry_points['buy_support']:,.0f} - Area support
• Current Price: Rp {entry_points['current_price']:,.0f}
• Breakout Buy: Rp {entry_points['breakout']:,.0f} - Above resistance"""

        # Hitung risk/reward ratio
        if 'recommended' in entry_points and isinstance(entry_points['recommended'], (int, float)):
            entry_price = entry_points['recommended']
            risk = entry_price - stop_loss
            reward = tp1 - entry_price
            
            if risk > 0:
                rr_ratio = reward / risk
                rr_text = f"{rr_ratio:.1f}:1"
                if rr_ratio >= 2:
                    rr_rating = "🟢 EXCELLENT"
                elif rr_ratio >= 1.5:
                    rr_rating = "🟡 GOOD" 
                elif rr_ratio >= 1:
                    rr_rating = "🟠 FAIR"
                else:
                    rr_rating = "🔴 POOR"
            else:
                rr_text = "N/A"
                rr_rating = "⚠️ Check"
        else:
            rr_text = "N/A" 
            rr_rating = "⚠️ Check"

        analysis_text = f"""
📊 Analisa {kode} Harian

🎯 TREND: {trend_type.upper()} ({trend_strength.upper()})
💰 Harga Saat Ini: Rp {current_price:,.2f}
📊 Support Terdekat: Rp {support:,.0f}
📈 Resistance Terdekat: Rp {resistance:,.0f}

{entry_text}

🎯 TARGET & RISK:
• TP1: Rp {tp1:,.0f} (+{((tp1/current_price)-1)*100:.1f}%)
• TP2: Rp {tp2:,.0f} (+{((tp2/current_price)-1)*100:.1f}%) 
• Stop Loss: Rp {stop_loss:,.0f} (-{((1-stop_loss/current_price))*100:.1f}%)
• Risk/Reward: {rr_text} {rr_rating}

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

        # Add final notes
        analysis_text += f"""
📝 STRATEGI EXECUTION:
- Money management: Jangan risk lebih dari 2% equity per trade
- Position sizing: Sesuaikan dengan risk tolerance
- Trading plan: Patuhi rencana yang sudah dibuat

⚠️ DISCLAIMER: 
Analisis ini menggunakan DATA SIMULASI untuk demo bot.
Bukan rekomendasi beli/jual. Pastikan analisis fundamental.
"""

        update.message.reply_text(analysis_text)

    except Exception as e:
        logger.error(f"Error in /analysis: {e}")
        update.message.reply_text(f"❌ Error dalam analisis: {str(e)}")

def faq(update, context):
    faq_text = """
📚 FAQ - Istilah Trading Saham Indonesia

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

📖 Tips Trading Saham:
- Hanya LONG positions (buy low, sell high)
- Gunakan stop loss untuk proteksi
- Risk management yang baik kunci sukses
- Jangan emotional trading
- Diversifikasi portfolio
"""

    update.message.reply_text(faq_text)

# ==== MAIN ====
def main():
    TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', "7766255048:AAF2DO86mIOZaEJnnYSLqXMOjGY5SxDVYA8")
    
    updater = Updater(TOKEN)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("menu", menu))
    dp.add_handler(CommandHandler("ma", ma))
    dp.add_handler(CommandHandler("alert", alert))
    dp.add_handler(CommandHandler("chart", chart))
    dp.add_handler(CommandHandler("analysis", analysis))
    dp.add_handler(CommandHandler("faq", faq))

    def error_handler(update, context):
        logger.error(f"Update {update} caused error {context.error}")

    dp.add_error_handler(error_handler)

    try:
        updater.bot.delete_webhook()
        logger.info("Bot sedang berjalan dengan data simulasi...")
        updater.start_polling(timeout=30, clean=True)
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
    
    updater.idle()

if __name__ == "__main__":
    main()