from telegram.ext import Updater, CommandHandler
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import io
import numpy as np
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

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
        # Coba ambil data real-time terlebih dahulu
        data = ticker.history(period='1d', interval='1m')
        if len(data) > 0:
            return data['Close'].iloc[-1]
        else:
            # Fallback ke data harian
            data = ticker.history(period='1d')
            return data['Close'].iloc[-1] if len(data) > 0 else None
    except Exception as e:
        logger.error(f"Error getting current price for {symbol}: {e}")
        return None

def get_stock_data(symbol, period="1y"):
    """Mendapatkan data saham dengan handling error"""
    try:
        data = yf.download(symbol, period=period, interval="1d", progress=False)
        if len(data) == 0:
            logger.warning(f"No data found for {symbol}")
            return None
        return data
    except Exception as e:
        logger.error(f"Error downloading data for {symbol}: {e}")
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
        # Get data dengan periode yang cukup untuk semua MA
        data = get_stock_data(symbol, period="2y")  # 2 tahun untuk MA200
        if data is None or len(data) == 0:
            update.message.reply_text(f"❌ Tidak menemukan data untuk {kode}")
            return

        # Calculate MAs dengan handling untuk data yang kurang
        data['MA10'] = data['Close'].rolling(10).mean()
        data['MA20'] = data['Close'].rolling(20).mean()
        data['MA50'] = data['Close'].rolling(50).mean()
        data['MA100'] = data['Close'].rolling(100).mean()
        data['MA200'] = data['Close'].rolling(200).mean()

        # Get current price (real-time)
        current_price = get_current_price(symbol)
        if current_price is None:
            current_price = safe_last(data['Close'])

        # Get MA values dengan handling untuk MA yang belum tersedia
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
        available_ma_count = 0
        for name, value in ma_values:
            if value is not None:
                status = "🟢 DI ATAS" if current_price > value else "🔴 DI BAWAH"
                ma_text += f"{name}: Rp {value:,.2f} ({status})\n"
                available_ma_count += 1
            else:
                ma_text += f"{name}: Data belum cukup\n"

        # Determine trend berdasarkan MA yang tersedia
        if available_ma_count >= 3:  # Minimal 3 MA untuk menentukan trend
            if ma10 is not None and ma20 is not None and ma50 is not None:
                if ma10 > ma20 > ma50:
                    if ma100 is not None and ma200 is not None and ma50 > ma100 > ma200:
                        trend = "📈 BULLISH KUAT"
                    else:
                        trend = "📈 BULLISH"
                elif ma10 < ma20 < ma50:
                    if ma100 is not None and ma200 is not None and ma50 < ma100 < ma200:
                        trend = "📉 BEARISH KUAT"
                    else:
                        trend = "📉 BEARISH"
                else:
                    trend = "➡️ SIDEWAYS"
            else:
                trend = "➡️ SIDEWAYS"
        else:
            trend = "ℹ️ Data belum cukup untuk analisis trend"

        # Tambahkan info data terakhir
        last_date = data.index[-1].strftime("%d-%m-%Y") if len(data) > 0 else "N/A"
        
        pesan = (
            f"📊 MOVING AVERAGE {kode}\n"
            f"💰 Harga Saat Ini: {harga_text}\n"
            f"📅 Data Terakhir: {last_date}\n\n"
            f"{ma_text}\n"
            f"📈 Trend: {trend}"
        )
        update.message.reply_text(pesan)

    except Exception as e:
        logger.error(f"Error in /ma: {e}")
        update.message.reply_text(f"❌ Error: {str(e)}")

# ==== ALERT ====
def alert(update, context):
    if len(context.args) != 1:
        update.message.reply_text("❌ Format salah. Contoh: /alert BBCA")
        return

    kode = context.args[0].upper()
    symbol = kode + ".JK"

    try:
        data = get_stock_data(symbol, period="3mo")
        if data is None or len(data) == 0:
            update.message.reply_text(f"❌ Tidak menemukan data untuk {kode}")
            return

        # Calculate indicators
        data['MA20'] = data['Close'].rolling(20).mean()
        data['Vol20'] = data['Volume'].rolling(20).mean()
        data['RSI'] = calculate_rsi(data)

        # Get current values
        current_price = get_current_price(symbol)
        if current_price is None:
            current_price = safe_last(data['Close'])
            
        volume = safe_last(data['Volume'])
        ma20 = safe_last(data['MA20'])
        vol20 = safe_last(data['Vol20'])
        rsi = safe_last(data['RSI'])

        # Format values
        harga_text = f"Rp {current_price:,.2f}" if current_price is not None else "Belum tersedia"
        vol20_text = f"{vol20:,.0f}" if vol20 is not None else "Belum tersedia"
        rsi_text = f"{rsi:.2f}" if rsi is not None else "Belum tersedia"

        # Tambahkan info waktu data
        last_date = data.index[-1].strftime("%d-%m-%Y %H:%M") if len(data) > 0 else "N/A"
        
        pesan = (
            f"🚨 ALERT {kode}\n"
            f"💰 Harga: {harga_text}\n"
            f"📊 Volume: {volume:,.0f}\n"
            f"📈 Rata2 Vol20: {vol20_text}\n"
            f"📊 RSI: {rsi_text}\n"
            f"🕒 Update: {last_date}\n\n"
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
        logger.error(f"Error in /alert: {e}")
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
        data = get_stock_data(symbol, period="6mo")
        if data is None or len(data) == 0:
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
        current_price = get_current_price(symbol)
        if current_price is None:
            current_price = safe_last(data['Close'])

        # Create subplots - FIXED: remove label parameter yang bermasalah
        apds = [
            mpf.make_addplot(data["MA9"], color='orange', width=0.7),
            mpf.make_addplot(data["MA20"], color='blue', width=0.7),
            mpf.make_addplot(data["MA50"], color='red', width=0.7),
            mpf.make_addplot(data["RSI"], panel=1, color='purple', width=0.7),
            mpf.make_addplot([70] * len(data), panel=1, color='red', linestyle='--', width=0.5),
            mpf.make_addplot([30] * len(data), panel=1, color='green', linestyle='--', width=0.5),
            mpf.make_addplot(macd, panel=2, color='blue', width=0.7),
            mpf.make_addplot(signal, panel=2, color='red', width=0.7),
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

        # Add manual legends
        ax_main.plot([], [], color='orange', label='MA9')
        ax_main.plot([], [], color='blue', label='MA20')
        ax_main.plot([], [], color='red', label='MA50')
        ax_main.legend()
        
        ax_rsi.plot([], [], color='purple', label='RSI')
        ax_rsi.legend()
        
        ax_macd.plot([], [], color='blue', label='MACD')
        ax_macd.plot([], [], color='red', label='SIGNAL')
        ax_macd.legend()

        ax_main.set_title(f'Chart {kode} - {datetime.now().strftime("%Y-%m-%d")}', fontsize=14, fontweight='bold')
        ax_rsi.set_ylabel('RSI')
        ax_macd.set_ylabel('MACD')

        # Save and send
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        
        price_text = f"Rp {current_price:,.2f}" if current_price else "N/A"
        last_date = data.index[-1].strftime("%d-%m-%Y") if len(data) > 0 else "N/A"
        
        update.message.reply_photo(
            photo=buf, 
            caption=f"📊 Chart {kode}\n💰 Harga: {price_text}\n📅 Data: {last_date}\n📈 Periode: 6 Bulan\n\nCandle + MA + RSI + MACD"
        )
        plt.close(fig)

    except Exception as e:
        logger.error(f"Error in /chart: {e}")
        update.message.reply_text(f"❌ Error membuat chart: {str(e)}")

# ==== IMPROVED ANALYSIS FUNCTIONS FOR STOCKS (LONG ONLY) ====
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
    
    # Validasi akhir
    recommended_entry = entry_points.get('recommended', valid_current)
    
    if isinstance(recommended_entry, (int, float)):
        if tp1 <= recommended_entry:
            tp1 = recommended_entry * 1.03
        if tp2 <= tp1:
            tp2 = tp1 * 1.03
        if stop_loss >= recommended_entry:
            stop_loss = recommended_entry * 0.97
    
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
        data = get_stock_data(symbol, period="2y")
        if data is None or len(data) == 0:
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

        # Get current values
        current_price = get_current_price(symbol)
        if current_price is None:
            current_price = safe_last(data['Close'])

        # Extract values
        ma9 = safe_last(data['MA9'])
        ma20 = safe_last(data['MA20'])
        ma50 = safe_last(data['MA50'])
        ma200 = safe_last(data['MA200'])
        rsi = safe_last(data['RSI'])
        current_macd = safe_last(macd_line)
        current_signal = safe_last(signal_line)
        current_histogram = safe_last(histogram)
        current_volume = safe_last(data['Volume'])
        avg_volume = safe_last(data['Vol20'])

        # Cari support resistance
        support, resistance = find_support_resistance(data, current_price, 60)

        # ===== TREND DETERMINATION =====
        trend_type = "sideways"
        trend_strength = "lemah"
        trend_explanation = ""

        # Tentukan trend berdasarkan MA yang tersedia
        available_ma = [ma for ma in [ma9, ma20, ma50, ma200] if ma is not None]
        
        if len(available_ma) >= 3:
            if ma9 is not None and ma20 is not None and ma50 is not None:
                if ma9 > ma20 > ma50:
                    if ma200 is not None and ma50 > ma200:
                        trend_type = "bullish"
                        trend_strength = "kuat"
                        trend_explanation = "Semua MA sejajar naik, trend BULLISH kuat"
                    else:
                        trend_type = "bullish"
                        trend_strength = "sedang"
                        trend_explanation = "MA jangka pendek naik, trend BULLISH"
                elif ma9 < ma20 < ma50:
                    if ma200 is not None and ma50 < ma200:
                        trend_type = "bearish"
                        trend_strength = "kuat"
                        trend_explanation = "Semua MA sejajar turun, trend BEARISH kuat"
                    else:
                        trend_type = "bearish"
                        trend_strength = "sedang"
                        trend_explanation = "MA jangka pendek turun, trend BEARISH"
                else:
                    trend_type = "sideways"
                    trend_explanation = "MA saling berpotongan, trend SIDEWAYS"
        else:
            trend_type = "sideways"
            trend_explanation = "Data MA belum cukup untuk analisis trend"

        # ===== INDIKATOR ANALYSIS =====
        rsi_analysis = ""
        rsi_signal = "netral"
        
        if rsi is not None:
            if rsi > 70:
                rsi_signal = "overbought"
                rsi_analysis = f"RSI {rsi:.1f} > 70 (Overbought), potensi koreksi"
            elif rsi < 30:
                rsi_signal = "oversold"
                rsi_analysis = f"RSI {rsi:.1f} < 30 (Oversold), potensi rebound"
            elif rsi > 50:
                rsi_signal = "bullish"
                rsi_analysis = f"RSI {rsi:.1f} > 50 (Bullish zone)"
            else:
                rsi_signal = "bearish"
                rsi_analysis = f"RSI {rsi:.1f} < 50 (Bearish zone)"

        macd_analysis = ""
        macd_signal = "netral"
        
        if current_macd is not None and current_signal is not None:
            if current_macd > current_signal:
                macd_signal = "bullish"
                macd_analysis = "MACD > Signal (Bullish momentum)"
            else:
                macd_signal = "bearish"
                macd_analysis = "MACD < Signal (Bearish momentum)"

        volume_analysis = ""
        volume_signal = "normal"
        
        if current_volume is not None and avg_volume is not None:
            volume_ratio = current_volume / avg_volume
            if volume_ratio > 2:
                volume_signal = "sangat tinggi"
                volume_analysis = f"Volume {volume_ratio:.1f}x rata-rata"
            elif volume_ratio > 1.2:
                volume_signal = "tinggi"
                volume_analysis = f"Volume {volume_ratio:.1f}x rata-rata"
            elif volume_ratio < 0.8:
                volume_signal = "rendah"
                volume_analysis = f"Volume {volume_ratio:.1f}x rata-rata"

        # ===== CALCULATE TRADING LEVELS =====
        entry_points = calculate_entry_points(current_price, support, resistance, trend_type)
        tp1, tp2, stop_loss = calculate_targets_stoploss(current_price, support, resistance, trend_type, entry_points)

        # ===== GENERATE RECOMMENDATIONS =====
        recommendations = []

        if trend_type == "bullish":
            recommendations.extend([
                "✅ BIAS BELI: Trend bullish",
                "🎯 Entry: Pullback ke support atau breakout resistance",
                "📈 Target: Multiple resistance levels",
                "🛑 Stop Loss: Di bawah support"
            ])
        elif trend_type == "bearish":
            recommendations.extend([
                "🔴 HINDARI: Trend bearish, risk tinggi",
                "💡 Action: Tunggu reversal confirmation",
                "🎯 Entry: Hanya jika bounce kuat dari support",
                "📈 Target: Konservatif (2-3%)"
            ])
        else:
            recommendations.extend([
                "🟡 RANGE TRADING: Harga sideways",
                "🎯 Entry: Buy di support, sell di resistance",
                "📊 Target: Sisi berlawanan range",
                "🛑 Stop Loss: Di luar range"
            ])

        # Additional signals
        if rsi_signal == "overbought":
            recommendations.append("⚠️ Hati-hati: RSI overbought")
        elif rsi_signal == "oversold":
            recommendations.append("💡 Opportunity: RSI oversold")

        if volume_signal in ["tinggi", "sangat tinggi"]:
            recommendations.append("📊 Volume tinggi konfirmasi pergerakan")

        # ===== FINAL ANALYSIS TEXT =====
        last_date = data.index[-1].strftime("%d-%m-%Y %H:%M") if len(data) > 0 else "N/A"
        
        analysis_text = f"""
📊 ANALISIS TEKNIKAL {kode}
🕒 Update: {last_date}

🎯 TREND: {trend_type.upper()} ({trend_strength.upper()})
💰 Harga: Rp {current_price:,.2f}
📊 Support: Rp {support:,.0f}
📈 Resistance: Rp {resistance:,.0f}

📈 INDIKATOR:
• RSI: {rsi:.1f} ({rsi_signal})
• MACD: {macd_signal}
• Volume: {volume_signal}

🎪 REKOMENDASI ENTRY:"""

        # Format entry points
        if trend_type == "bullish":
            analysis_text += f"""
• Conservative: Rp {entry_points['conservative']:,.0f}
• Aggressive: Rp {entry_points['aggressive']:,.0f}
• Breakout: Rp {entry_points['breakout']:,.0f}"""
        elif trend_type == "bearish":
            analysis_text += f"""
• {entry_points['avoid']}
• Conservative (RISKY): Rp {entry_points['conservative']:,.0f}"""
        else:
            analysis_text += f"""
• Buy Support: Rp {entry_points['buy_support']:,.0f}
• Breakout: Rp {entry_points['breakout']:,.0f}"""

        analysis_text += f"""

🎯 TARGET & RISK:
• TP1: Rp {tp1:,.0f} (+{((tp1/current_price)-1)*100:.1f}%)
• TP2: Rp {tp2:,.0f} (+{((tp2/current_price)-1)*100:.1f}%)
• Stop Loss: Rp {stop_loss:,.0f} (-{((1-stop_loss/current_price))*100:.1f}%)

💡 REKOMENDASI:
"""

        for rec in recommendations:
            analysis_text += f"• {rec}\n"

        analysis_text += f"""

📝 CATATAN:
{trend_explanation}
{rsi_analysis}
{macd_analysis}
{volume_analysis}

⚠️ DISCLAIMER: 
Analisis teknikal untuk edukasi, bukan rekomendasi beli/jual.
Pastikan analisis fundamental dan kondisi market.
"""

        update.message.reply_text(analysis_text)

    except Exception as e:
        logger.error(f"Error in /analysis: {e}")
        update.message.reply_text(f"❌ Error dalam analisis: {str(e)}")

# ==== FAQ ====
def faq(update, context):
    faq_text = """
📚 FAQ - Istilah Trading Saham

• MA (Moving Average) = Rata-rata harga periode tertentu
• Support = Level harga dimana biasanya terjadi pembelian
• Resistance = Level harga dimana biasanya terjadi penjualan  
• RSI = Indikator momentum (30-70)
• MACD = Indikator trend dan momentum
• Volume = Jumlah saham yang diperdagangkan

📖 Tips Trading:
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
        logger.info("Bot sedang berjalan...")
        updater.start_polling(timeout=30, clean=True)
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
    
    updater.idle()

if __name__ == "__main__":
    main()