import os
import time
import schedule
from openai import OpenAI
import ccxt
import pandas as pd
import re
from dotenv import load_dotenv
import json
import requests
from datetime import datetime, timedelta

import builtins

# 保存原始print函数
original_print = builtins.print

def ts_print(*args, **kwargs):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    original_print(f"[{timestamp}]", *args, **kwargs)

# 替换内置print函数
builtins.print = ts_print

load_dotenv()

# 初始化AI客户端
ai_clients = {
    'deepseek': OpenAI(
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com"
    ),
    'qwen': OpenAI(
        api_key=os.getenv('QWEN_API_KEY'),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 默认为北京地域的模型(免费额度) 如新加坡改成https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    )
}

# 初始化OKX交易所
exchange = ccxt.okx({
    'options': {
        'defaultType': 'swap',  # OKX使用swap表示永续合约
    },
    'apiKey': os.getenv('OKX_API_KEY'),
    'secret': os.getenv('OKX_SECRET'),
    'password': os.getenv('OKX_PASSWORD'),  # OKX需要交易密码
})

# 交易参数配置 - 结合两个版本的优点
TRADE_CONFIG = {
    'ai_provider': 'qwen',  # AI提供商选择：'deepseek' 或 'qwen'
    'symbol': 'BTC/USDT:USDT',  # OKX的合约符号格式
    'leverage': 10,  # 杠杆倍数,只影响保证金不影响下单价值
    'sleepTime': 3,         # 轮询休息时间间隔，默认3m
    'baseTimeFrame': 15,    # 默认为15分钟信号线为基准，其他选择将同时扩展数据
    'settingTimeframe': 1,  # 使用15分钟K线，还可选 5m,3m,1m
    'test_mode': False,     # 测试模式
    'data_points': 96*3,    # 24*3小时数据（96根15分钟K线）
    'kline_num': 20,        # K线数量
    'analysis_periods': {
        'short_term': 20,   # 短期均线
        'medium_term': 50,  # 中期均线
        'long_term': 96     # 长期趋势
    },
    # 新增智能仓位参数
    'position_management': {
        'enable_intelligent_position': True,  # 🆕 新增：是否启用智能仓位管理
        'base_usdt_amount': 100,  # USDT投入下单基数
        'high_confidence_multiplier': 1.5,
        'medium_confidence_multiplier': 1.0,
        'low_confidence_multiplier': 0.5,
        'max_position_ratio': 10,  # 单次最大仓位比例
        'trend_strength_multiplier': 1.2
    }
}

#设置交易所参数 - 强制全仓模式
def setup_exchange():
    """设置交易所参数 - 强制全仓模式"""
    try:

        # 首先获取合约规格信息
        print("🔍 获取BTC合约规格...")
        markets = exchange.load_markets()
        btc_market = markets[TRADE_CONFIG['symbol']]

        # 获取合约乘数
        contract_size = float(btc_market['contractSize'])
        print(f"✅ 合约规格: 1张 = {contract_size} BTC")

        # 设定时间帧（K线间隔）
        TRADE_CONFIG['timeframe'] = str(TRADE_CONFIG['settingTimeframe']) + 'm'
        # 计算因子 baseTimeFrame / settingTimeframe
        TRADE_CONFIG['factor'] = TRADE_CONFIG['baseTimeFrame'] / TRADE_CONFIG['settingTimeframe']
        # 重设置获取K线数量
        TRADE_CONFIG['data_points'] = int(TRADE_CONFIG['data_points'] * TRADE_CONFIG['factor'])
        # 重设置k线数量
        TRADE_CONFIG['kline_num'] = int(TRADE_CONFIG['kline_num'] * TRADE_CONFIG['factor'])

        # 存储合约规格到全局配置
        TRADE_CONFIG['contract_size'] = contract_size
        TRADE_CONFIG['min_amount'] = btc_market['limits']['amount']['min']

        print(f"📏 最小交易量: {TRADE_CONFIG['min_amount']} 张")

        # 先检查现有持仓
        print("🔍 检查现有持仓模式...")
        positions = exchange.fetch_positions([TRADE_CONFIG['symbol']])

        has_isolated_position = False
        isolated_position_info = None

        for pos in positions:
            if pos['symbol'] == TRADE_CONFIG['symbol']:
                contracts = float(pos.get('contracts', 0))
                mode = pos.get('mgnMode')

                if contracts > 0 and mode == 'isolated':
                    has_isolated_position = True
                    isolated_position_info = {
                        'side': pos.get('side'),
                        'size': contracts,
                        'entry_price': pos.get('entryPrice'),
                        'mode': mode
                    }
                    break

        # 2. 如果有逐仓持仓，提示并退出
        if has_isolated_position:
            print("❌ 检测到逐仓持仓，程序无法继续运行！")
            print(f"📊 逐仓持仓详情:")
            print(f"   - 方向: {isolated_position_info['side']}")
            print(f"   - 数量: {isolated_position_info['size']}")
            print(f"   - 入场价: {isolated_position_info['entry_price']}")
            print(f"   - 模式: {isolated_position_info['mode']}")
            print("\n🚨 解决方案:")
            print("1. 手动平掉所有逐仓持仓")
            print("2. 或者将逐仓持仓转为全仓模式")
            print("3. 然后重新启动程序")
            return False

        # 3. 设置单向持仓模式
        print("🔄 设置单向持仓模式...")
        try:
            exchange.set_position_mode(False, TRADE_CONFIG['symbol'])  # False表示单向持仓
            print("✅ 已设置单向持仓模式")
        except Exception as e:
            print(f"⚠️ 设置单向持仓模式失败 (可能已设置): {e}")

        # 4. 设置全仓模式和杠杆
        print("⚙️ 设置全仓模式和杠杆...")
        exchange.set_leverage(
            TRADE_CONFIG['leverage'],
            TRADE_CONFIG['symbol'],
            {'mgnMode': 'cross'}  # 强制全仓模式
        )
        print(f"✅ 已设置全仓模式，杠杆倍数: {TRADE_CONFIG['leverage']}x")

        # 5. 验证设置
        print("🔍 验证账户设置...")
        balance = exchange.fetch_balance()
        usdt_balance = balance['USDT']['free']
        print(f"💰 当前USDT余额: {usdt_balance:.2f}")

        # 获取当前持仓状态
        current_pos = get_current_position()
        if current_pos:
            print(f"📦 当前持仓: {current_pos['side']}仓 {current_pos['size']}张")
        else:
            print("📦 当前无持仓")

        print("🎯 程序配置完成：全仓模式 + 单向持仓")
        return True

    except Exception as e:
        print(f"❌ 交易所设置失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# 全局变量存储历史数据
price_history = []
signal_history = []
position = None

#计算智能仓位大小 - 修复版
def calculate_intelligent_position(signal_data, price_data, current_position):
    """计算智能仓位大小 - 修复版"""
    config = TRADE_CONFIG['position_management']

    # 🆕 新增：如果禁用智能仓位，使用固定仓位
    if not config.get('enable_intelligent_position', True):
        fixed_contracts = 0.1  # 固定仓位大小，可以根据需要调整
        print(f"🔧 智能仓位已禁用，使用固定仓位: {fixed_contracts} 张")
        return fixed_contracts

    try:
        # 获取账户余额
        balance = exchange.fetch_balance()
        usdt_balance = balance['USDT']['free']

        # TODO 投入基数修改为余额
        # 基础USDT投入
        # base_usdt = config['base_usdt_amount']
        base_usdt = usdt_balance
        print(f"💰 可用USDT余额: {usdt_balance:.2f}, 下单基数{base_usdt}")

        # 根据信心程度调整 - 修复这里
        confidence_multiplier = {
            'HIGH': config['high_confidence_multiplier'],
            'MEDIUM': config['medium_confidence_multiplier'],
            'LOW': config['low_confidence_multiplier']
        }.get(signal_data['confidence'], 1.0)  # 添加默认值

        # 根据趋势强度调整
        trend = price_data['trend_analysis'].get('overall', '震荡整理')
        if trend in ['强势上涨', '强势下跌']:
            trend_multiplier = config['trend_strength_multiplier']
        else:
            trend_multiplier = 1.0

        # 根据RSI状态调整（超买超卖区域减仓）
        rsi = price_data['technical_data'].get('rsi_14', 50)
        if rsi > 75 or rsi < 25:
            rsi_multiplier = 0.7
        else:
            rsi_multiplier = 1.0

        # 计算建议投入USDT金额
        suggested_usdt = base_usdt * confidence_multiplier * trend_multiplier * rsi_multiplier

        # 风险管理：不超过总资金的指定比例 - 删除重复定义
        max_usdt = usdt_balance * config['max_position_ratio']
        final_usdt = min(suggested_usdt, max_usdt)

        # 正确的合约张数计算！
        # 公式：合约张数 = (投入USDT) / (当前价格 * 合约乘数)
        contract_size = (final_usdt) / (price_data['price'] * TRADE_CONFIG['contract_size'])

        print(f"📊 仓位计算详情:")
        print(f"   - 基础USDT: {base_usdt}")
        print(f"   - 信心倍数: {confidence_multiplier}")
        print(f"   - 趋势倍数: {trend_multiplier}")
        print(f"   - RSI倍数: {rsi_multiplier}")
        print(f"   - 建议USDT: {suggested_usdt:.2f}")
        print(f"   - 最终USDT: {final_usdt:.2f}")
        print(f"   - 合约乘数: {TRADE_CONFIG['contract_size']}")
        print(f"   - 计算合约: {contract_size:.4f} 张")

        # 精度处理：OKX BTC合约最小交易单位为0.01张
        contract_size = round(contract_size, 2)  # 保留2位小数

        # 确保最小交易量
        min_contracts = TRADE_CONFIG.get('min_amount', 0.01)
        if contract_size < min_contracts:
            contract_size = min_contracts
            print(f"⚠️ 仓位小于最小值，调整为: {contract_size} 张")

        print(f"🎯 最终仓位: {final_usdt:.2f} USDT → {contract_size:.2f} 张合约")
        return contract_size

    except Exception as e:
        print(f"❌ 仓位计算失败，使用基础仓位: {e}")
        # 紧急备用计算
        base_usdt = config['base_usdt_amount']
        contract_size = (base_usdt * TRADE_CONFIG['leverage']) / (
                    price_data['price'] * TRADE_CONFIG.get('contract_size', 0.01))
        return round(max(contract_size, TRADE_CONFIG.get('min_amount', 0.01)), 2)

#计算技术指标 - 来自第一个策略
def calculate_technical_indicators(df):
    """计算技术指标 - 来自第一个策略"""
    try:
        # 移动平均线
        df['sma_3'] = df['close'].rolling(window=3, min_periods=1).mean()
        df['sma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['sma_10'] = df['close'].rolling(window=10, min_periods=1).mean()
        df['sma_15'] = df['close'].rolling(window=15, min_periods=1).mean()
        df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['sma_60'] = df['close'].rolling(window=60, min_periods=1).mean()
        df['sma_240'] = df['close'].rolling(window=240, min_periods=1).mean()

        # 指数移动平均线
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # 相对强弱指数 (RSI)
        delta = df['close'].diff()

        # RSI7 - 短周期，更敏感
        df['rsi_7'] = 100 - (100 / (1 + 
            (delta.where(delta > 0, 0)).rolling(7).mean() / 
                (-delta.where(delta < 0, 0)).rolling(7).mean()))

        # RSI14 - 标准周期
        df['rsi_14'] = 100 - (100 / (1 + 
            (delta.where(delta > 0, 0)).rolling(14).mean() / 
            (-delta.where(delta < 0, 0)).rolling(14).mean()))

        # 布林带
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # 成交量均线
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        # 支撑阻力位
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()

        # 填充NaN值
        df = df.bfill().ffill()

        return df
    except Exception as e:
        print(f"技术指标计算失败: {e}")
        return df

#计算支撑阻力位
def get_support_resistance_levels(df, lookback=20):
    """计算支撑阻力位"""
    try:
        recent_high = df['high'].tail(lookback).max()
        recent_low = df['low'].tail(lookback).min()
        current_price = df['close'].iloc[-1]

        resistance_level = recent_high
        support_level = recent_low

        # 动态支撑阻力（基于布林带）
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]

        return {
            'static_resistance': resistance_level,
            'static_support': support_level,
            'dynamic_resistance': bb_upper,
            'dynamic_support': bb_lower,
            'price_vs_resistance': ((resistance_level - current_price) / current_price) * 100,
            'price_vs_support': ((current_price - support_level) / support_level) * 100
        }
    except Exception as e:
        print(f"支撑阻力计算失败: {e}")
        return {}

#获取情绪指标 - 简洁版本
def get_sentiment_indicators():
    """获取情绪指标 - 简洁版本"""
    try:
        API_URL = "https://service.cryptoracle.network/openapi/v2/endpoint"
        API_KEY = "7ad48a56-8730-4238-a714-eebc30834e3e"

        # 获取最近4小时数据
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=4)

        request_body = {
            "apiKey": API_KEY,
            "endpoints": ["CO-A-02-01", "CO-A-02-02"],  # 只保留核心指标
            "startTime": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "endTime": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "timeType": "15m",
            "token": ["BTC"]
        }

        headers = {"Content-Type": "application/json", "X-API-KEY": API_KEY}
        response = requests.post(API_URL, json=request_body, headers=headers)

        if response.status_code == 200:
            data = response.json()
            if data.get("code") == 200 and data.get("data"):
                time_periods = data["data"][0]["timePeriods"]

                # 查找第一个有有效数据的时间段
                for period in time_periods:
                    period_data = period.get("data", [])

                    sentiment = {}
                    valid_data_found = False

                    for item in period_data:
                        endpoint = item.get("endpoint")
                        value = item.get("value", "").strip()

                        if value:  # 只处理非空值
                            try:
                                if endpoint in ["CO-A-02-01", "CO-A-02-02"]:
                                    sentiment[endpoint] = float(value)
                                    valid_data_found = True
                            except (ValueError, TypeError):
                                continue

                    # 如果找到有效数据
                    if valid_data_found and "CO-A-02-01" in sentiment and "CO-A-02-02" in sentiment:
                        positive = sentiment['CO-A-02-01']
                        negative = sentiment['CO-A-02-02']
                        net_sentiment = positive - negative

                        # 正确的时间延迟计算
                        data_delay = int((datetime.now() - datetime.strptime(
                            period['startTime'], '%Y-%m-%d %H:%M:%S')).total_seconds() // 60)

                        print(f"✅ 使用情绪数据时间: {period['startTime']} (延迟: {data_delay}分钟)")

                        return {
                            'positive_ratio': positive,
                            'negative_ratio': negative,
                            'net_sentiment': net_sentiment,
                            'data_time': period['startTime'],
                            'data_delay_minutes': data_delay
                        }

                print("❌ 所有时间段数据都为空")
                return None

        return None
    except Exception as e:
        print(f"情绪指标获取失败: {e}")
        return None

#判断市场趋势
def get_market_trend(df):
    """判断市场趋势"""
    try:
        current_price = df['close'].iloc[-1]

        # 多时间框架趋势分析
        trend_short = "上涨" if current_price > df['sma_10'].iloc[-1] else "下跌"
        trend_medium = "上涨" if current_price > df['sma_20'].iloc[-1] else "下跌"

        # MACD趋势
        macd_trend = "bullish" if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] else "bearish"

        # 综合趋势判断
        if trend_short == "上涨" and trend_medium == "上涨":
            overall_trend = "强势上涨"
        elif trend_short == "下跌" and trend_medium == "下跌":
            overall_trend = "强势下跌"
        else:
            overall_trend = "震荡整理"

        return {
            'short_term': trend_short,
            'medium_term': trend_medium,
            'macd': macd_trend,
            'overall': overall_trend,
            'rsi_level': df['rsi_14'].iloc[-1]
        }
    except Exception as e:
        print(f"趋势分析失败: {e}")
        return {}

#增强版：获取BTC K线数据并计算技术指标
def get_btc_ohlcv_enhanced():
    """增强版：获取BTC K线数据并计算技术指标"""
    try:
        # 获取K线数据
        ohlcv = exchange.fetch_ohlcv(TRADE_CONFIG['symbol'], TRADE_CONFIG['timeframe'],
                                     limit=TRADE_CONFIG['data_points'], params={'paginate': True})

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        # 获取df数据量
        print(f"获取到({TRADE_CONFIG['symbol']}) {TRADE_CONFIG['timeframe']}K线数据量: {df.shape[0]}/{TRADE_CONFIG['data_points']}")

        # 计算技术指标
        df = calculate_technical_indicators(df)

        # 打印指标数据df['sma_5']
        # print(df['sma_5'])

        current_data = df.iloc[-1]
        previous_data = df.iloc[-2]
        # print(f"当前值: {current_data}")
        # print(f"前一值: {previous_data}")

        # 获取技术分析数据
        trend_analysis = get_market_trend(df)
        # 跟随配置因子变化
        lookback = int(20 * TRADE_CONFIG['factor'])
        levels_analysis = get_support_resistance_levels(df, lookback)

        return {
            'price': current_data['close'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'high': current_data['high'],
            'low': current_data['low'],
            'volume': current_data['volume'],
            'timeframe': TRADE_CONFIG['timeframe'],
            'price_change': ((current_data['close'] - previous_data['close']) / previous_data['close']) * 100,
            'kline_data': df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(TRADE_CONFIG['kline_num']).to_dict('records'),
            'technical_data': {
                'sma_3': current_data.get('sma_3', 0),
                'sma_5': current_data.get('sma_5', 0),
                'sma_10': current_data.get('sma_10', 0),
                'sma_15': current_data.get('sma_15', 0),
                'sma_20': current_data.get('sma_20', 0),
                'sma_60': current_data.get('sma_60', 0),
                'sma_240': current_data.get('sma_240', 0),
                'ema_12': current_data.get('ema_12', 0),
                'ema_20': current_data.get('ema_20', 0),
                'ema_26': current_data.get('ema_26', 0),
                'ema_50': current_data.get('ema_50', 0),
                'rsi_7': current_data.get('rsi_7', 0),
                'rsi_14': current_data.get('rsi_14', 0),
                'macd': current_data.get('macd', 0),
                'macd_signal': current_data.get('macd_signal', 0),
                'macd_histogram': current_data.get('macd_histogram', 0),
                'bb_upper': current_data.get('bb_upper', 0),
                'bb_lower': current_data.get('bb_lower', 0),
                'bb_position': current_data.get('bb_position', 0),
                'volume_ratio': current_data.get('volume_ratio', 0)
            },
            'trend_analysis': trend_analysis,
            'levels_analysis': levels_analysis,
            'full_data': df
        }
    except Exception as e:
        print(f"获取增强K线数据失败: {e}")
        return None

#成技术分析文本
def generate_technical_analysis_text(price_data):
    """生成技术分析文本"""
    if 'technical_data' not in price_data:
        return "技术指标数据不可用"

    tech = price_data['technical_data']
    trend = price_data.get('trend_analysis', {})
    levels = price_data.get('levels_analysis', {})

    # 检查数据有效性
    def safe_float(value, default=0):
        return float(value) if value and pd.notna(value) else default

    # 真实分钟数
    base_tf = TRADE_CONFIG['settingTimeframe']
    analysis_text = f"""
    【技术指标分析】
    📈 移动平均线:
    - sma_3周期({3*base_tf}分钟均线): {safe_float(tech['sma_3']):.2f} | 价格相对: {(price_data['price'] - safe_float(tech['sma_3'])) / safe_float(tech['sma_3']) * 100:+.2f}%
    - sma_5周期({5*base_tf}分钟均线): {safe_float(tech['sma_5']):.2f} | 价格相对: {(price_data['price'] - safe_float(tech['sma_5'])) / safe_float(tech['sma_5']) * 100:+.2f}%
    - sma_15周期({15*base_tf}分钟均线): {safe_float(tech['sma_15']):.2f} | 价格相对: {(price_data['price'] - safe_float(tech['sma_15'])) / safe_float(tech['sma_15']) * 100:+.2f}%
    - sma_60周期({60*base_tf}分钟均线): {safe_float(tech['sma_60']):.2f} | 价格相对: {(price_data['price'] - safe_float(tech['sma_60'])) / safe_float(tech['sma_60']) * 100:+.2f}%
    - sma_240周期({240*base_tf}分钟均线): {safe_float(tech['sma_240']):.2f} | 价格相对: {(price_data['price'] - safe_float(tech['sma_240'])) / safe_float(tech['sma_240']) * 100:+.2f}%

    🎯 趋势分析:
    - 短期趋势: {trend.get('short_term', 'N/A')}
    - 中期趋势: {trend.get('medium_term', 'N/A')}
    - 整体趋势: {trend.get('overall', 'N/A')}
    - MACD方向: {trend.get('macd', 'N/A')}

    📊 动量指标:
    - RSI: {safe_float(tech['rsi_14']):.2f} ({'超买' if safe_float(tech['rsi_14']) > 70 else '超卖' if safe_float(tech['rsi_14']) < 30 else '中性'})
    - MACD: {safe_float(tech['macd']):.4f}
    - 信号线: {safe_float(tech['macd_signal']):.4f}

    📈 其他指数移动平均线:
    - ema_20周期({20*base_tf}分钟移动均线): {safe_float(tech['ema_20']):.2f}
    - ema_50周期({50*base_tf}分钟移动均线): {safe_float(tech['ema_50']):.2f}

    🎚️ 布林带位置: {safe_float(tech['bb_position']):.2%} ({'上部' if safe_float(tech['bb_position']) > 0.7 else '下部' if safe_float(tech['bb_position']) < 0.3 else '中部'})

    💰 关键水平:
    - 静态阻力: {safe_float(levels.get('static_resistance', 0)):.2f}
    - 静态支撑: {safe_float(levels.get('static_support', 0)):.2f}
    """
    return analysis_text

#获取当前持仓情况 - OKX版本
def get_current_position():
    """获取当前持仓情况 - OKX版本"""
    try:
        positions = exchange.fetch_positions([TRADE_CONFIG['symbol']])

        for pos in positions:
            if pos['symbol'] == TRADE_CONFIG['symbol']:
                contracts = float(pos['contracts']) if pos['contracts'] else 0

                if contracts > 0:
                    position_info = {
                        'posId': pos['id'],  # 仓位ID
                        'side': pos['side'],  # 'long' or 'short'
                        'size': contracts,
                        'entry_price': float(pos['entryPrice']) if pos['entryPrice'] else 0,
                        'unrealized_pnl': float(pos['unrealizedPnl']) if pos['unrealizedPnl'] else 0,
                        'leverage': float(pos['leverage']) if pos['leverage'] else TRADE_CONFIG['leverage'],
                        'symbol': pos['symbol'],
                        'margin': float(pos['initialMargin']) if pos['initialMargin'] else 0,  # 保证金
                        'percentage': float(pos['percentage']) if pos['percentage'] else 0  # 盈亏比例
                    }
                    print(f"检测到持仓: {position_info}")  # 添加调试信息
                    return position_info

        print("未检测到匹配的持仓")
        return None

    except Exception as e:
        print(f"获取持仓失败: {e}")
        import traceback
        traceback.print_exc()
        return None

#安全解析JSON，处理格式不规范的情况
def safe_json_parse(json_str):
    """安全解析JSON，处理格式不规范的情况"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            # 修复常见的JSON格式问题
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r'(\w+):', r'"\1":', json_str)
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON解析失败，原始内容: {json_str}")
            print(f"错误详情: {e}")
            return None

#创建备用交易信号 - HOLD
def create_fallback_signal(price_data):
    """创建备用交易信号"""
    return {
        "signal": "HOLD",
        "reason": "因技术分析暂时不可用，采取保守策略",
        "stop_loss": price_data['price'] * 0.98,  # -2%
        "take_profit": price_data['price'] * 1.02,  # +2%
        "confidence": "LOW",
        "is_fallback": True
    }

max_profit_position = None
max_loss_position = None
# 计算当前持仓历史盈利最大持仓数据，以及历史亏损最大持仓数据，由当前持仓数据计算
def update_max_positions(current_position):
    global max_profit_position
    global max_loss_position

    """更新最大盈利和最大亏损持仓数据"""    
    # 如果当前无持仓，直接返回
    if not current_position:
        return
    
    current_pnl = current_position['unrealized_pnl']
    current_pos_id = current_position.get('posId', '')
    
    # 更新最大盈利持仓（仅当当前持仓盈利时）
    if current_pnl > 0:
        if max_profit_position is None:
            # 首次记录盈利持仓
            max_profit_position = current_position.copy()
            print(f"📈 首次记录最大盈利持仓: {max_profit_position}")
        else:
            max_profit_pos_id = max_profit_position.get('posId', '')
            
            if current_pos_id != max_profit_pos_id:
                # posId不同，用当前持仓代替
                max_profit_position = current_position.copy()
                print(f"📈 持仓已改变，更新最大盈利持仓: {max_profit_position} ")
            elif current_pnl > max_profit_position['unrealized_pnl']:
                # posId相同且当前盈利大于历史最大盈利
                print(f"📈 更新最大盈利持仓: {current_pnl:.2f} USDT, 比例:{current_position['percentage']}% (之前: {max_profit_position['unrealized_pnl']:.2f} USDT), 比例:{max_profit_position['percentage']}%")
                max_profit_position = current_position.copy()
                
    # 更新最大亏损持仓（仅当当前持仓亏损时）
    if current_pnl < 0:
        if max_loss_position is None:
            # 首次记录亏损持仓
            max_loss_position = current_position.copy()
            print(f"📉 首次记录最大亏损持仓: {max_loss_position}")
        else:
            max_loss_pos_id = max_loss_position.get('posId', '')
            
            if current_pos_id != max_loss_pos_id:
                # posId不同，用当前持仓代替
                max_loss_position = current_position.copy()
                print(f"📉 持仓已改变，更新最大亏损持仓: {max_loss_position}")
            elif current_pnl < max_loss_position['unrealized_pnl']:
                # posId相同且当前亏损小于历史最大亏损(负数更小)
                print(f"📉 更新最大亏损持仓: {current_pnl:.2f} USDT, 比例:{current_position['percentage']}% (之前: {max_loss_position['unrealized_pnl']:.2f} USDT), 比例:{max_loss_position['percentage']}%")
                max_loss_position = current_position.copy()

def calc_drawdown(current_pos):
    drawdown_text = ""
    if current_pos and max_profit_position and max_loss_position:
        current_pos_id = current_pos.get('posId', '')
        max_profit_pos_id = max_profit_position.get('posId', '')
        max_loss_pos_id = max_loss_position.get('posId', '')
        
        # 计算盈利状态下的回撤
        if current_pos_id == max_profit_pos_id and current_pos['unrealized_pnl'] > 0:
            # 回撤金额 = 历史最大盈利 - 当前盈利（需换算为当前仓位对应金额）
            max_profit_pul = max_profit_position['unrealized_pnl'] / max_profit_position['size'] * current_pos['size']
            drawdown_amount = max_profit_pul - current_pos['unrealized_pnl']
            # 回撤比例 = 回撤金额 / 历史最大盈利金额
            if max_profit_pul != 0:
                drawdown_percentage = (drawdown_amount / max_profit_pul) * 100
                drawdown_text = f" 回撤 {drawdown_amount:.2f} USDT ({drawdown_percentage:.2f}%)"
        
        # 计算亏损状态下的回升
        elif current_pos_id == max_loss_pos_id and current_pos['unrealized_pnl'] < 0:
            # 回升金额 = 当前亏损 - 历史最大亏损(负负得正，所以是相减)
            max_loss_pnl = max_loss_position['unrealized_pnl'] / max_loss_position['size'] * current_pos['size']
            recovery_amount = current_pos['unrealized_pnl'] - max_loss_pnl
            # 回升比例 = 回升金额 / |历史最大亏损金额|
            if max_loss_pnl != 0:
                recovery_percentage = (recovery_amount / abs(max_loss_pnl)) * 100
                drawdown_text = f" 从最大亏损回升 {recovery_amount:.2f} USDT ({recovery_percentage:.2f}%)"
    
    # 如果只有最大盈利持仓记录
    elif current_pos and max_profit_position and not max_loss_position:
        current_pos_id = current_pos.get('posId', '')
        max_profit_pos_id = max_profit_position.get('posId', '')
        
        if current_pos_id == max_profit_pos_id and current_pos['unrealized_pnl'] > 0:
            # 回撤金额 = 历史最大盈利 - 当前盈利
            drawdown_amount = max_profit_position['unrealized_pnl'] - current_pos['unrealized_pnl']
            # 回撤比例 = 回撤金额 / 历史最大盈利金额
            if max_profit_position['unrealized_pnl'] != 0:
                drawdown_percentage = (drawdown_amount / max_profit_position['unrealized_pnl']) * 100
                drawdown_text = f" 回撤: {drawdown_amount:.2f} USDT ({drawdown_percentage:.2f}%)"
    
    # 如果只有最大亏损持仓记录
    elif current_pos and max_loss_position and not max_profit_position:
        current_pos_id = current_pos.get('posId', '')
        max_loss_pos_id = max_loss_position.get('posId', '')
        
        if current_pos_id == max_loss_pos_id and current_pos['unrealized_pnl'] < 0:
            # 亏损状态，还未出现盈利，这种情况显示距离最大亏损的改善情况
            recovery_amount = current_pos['unrealized_pnl'] - max_loss_position['unrealized_pnl']
            if max_loss_position['unrealized_pnl'] != 0:
                recovery_percentage = (recovery_amount / abs(max_loss_position['unrealized_pnl'])) * 100
                drawdown_text = f" 从最大亏损回升: {recovery_amount:.2f} USDT ({recovery_percentage:.2f}%)"
    
    return drawdown_text

#使用DeepSeek分析市场并生成交易信号（增强版）
def analyze_with_deepseek(price_data):
    """使用DeepSeek分析市场并生成交易信号（增强版）"""

    # 生成技术分析文本
    technical_analysis = generate_technical_analysis_text(price_data)

    # 构建K线数据文本，按因子放大
    num = TRADE_CONFIG['kline_num']
    kline_text = f"【最近{num}根{TRADE_CONFIG['timeframe']}K线数据】\n"
    for i, kline in enumerate(price_data['kline_data'][-num:]):
        trend = "阳线" if kline['close'] > kline['open'] else "阴线"
        change = ((kline['close'] - kline['open']) / kline['open']) * 100
        kline_text += f"K线{i + 1}: {trend} 开盘:{kline['open']:.2f} 收盘:{kline['close']:.2f} 涨跌:{change:+.2f}%\n"

    # 添加上次交易信号
    signal_text = ""
    if signal_history:
        last_signal = signal_history[-1]
        signal_text = f"\n【上次交易信号】\n信号: {last_signal.get('signal', 'N/A')}\n信心: {last_signal.get('confidence', 'N/A')}"

    # 获取情绪数据
    sentiment_data = get_sentiment_indicators()
    # 简化情绪文本 多了没用
    if sentiment_data:
        sign = '+' if sentiment_data['net_sentiment'] >= 0 else ''
        sentiment_text = f"【市场情绪】乐观{sentiment_data['positive_ratio']:.1%} 悲观{sentiment_data['negative_ratio']:.1%} 净值{sign}{sentiment_data['net_sentiment']:.3f}"
    else:
        sentiment_text = "【市场情绪】数据暂不可用"

    print(sentiment_text)

    # 添加当前持仓信息
    current_pos = get_current_position()
    position_text = "无持仓" if not current_pos else f"{current_pos['side']}仓, 数量: {current_pos['size']}"
    pnl_text = f", 持仓盈亏: {current_pos['unrealized_pnl']} USDT" if current_pos else ""
    margin_text = f", 保证金: {current_pos['margin']} USDT" if current_pos else ""
    percentage_text = f", 盈亏比例: {current_pos['percentage']}%" if current_pos else ""

    # 调用函数更新最大持仓数据
    update_max_positions(current_pos)

    # 计算回撤信息
    drawdown_text = calc_drawdown(current_pos)

    # 构建最大持仓信息文本
    max_profit_text = "无"
    max_loss_text = "无"

    if max_profit_position:
        max_profit_text = f"{max_profit_position['side']}仓, 数量: {max_profit_position['size']}, 保证金: {max_profit_position['margin']} USDT, 盈亏: {max_profit_position['unrealized_pnl']} USDT, 盈亏比例: {max_profit_position['percentage']}%"

    if max_loss_position:
        max_loss_text = f"{max_loss_position['side']}仓, 数量: {max_loss_position['size']}, 保证金: {max_loss_position['margin']} USDT, 盈亏: {max_loss_position['unrealized_pnl']} USDT, 盈亏比例: {max_loss_position['percentage']}%"
    print(f'当前持仓: {position_text}{pnl_text}{margin_text}{percentage_text}')
    print(f'历史最大盈利持仓方向: {max_profit_text}')
    print(f'历史最大亏损持仓方向: {max_loss_text}')
    print(f'回撤数据: {drawdown_text}')

    prompt = f"""
## 🎯 核心分析哲学
**数据驱动决策** = 自主模式识别 × 多维度验证 × 动态风险评估 × 持续学习进化

📊 **分析自主权**：
- 自由组合所有可用技术指标
- 自主识别市场模式和趋势结构
- 动态构建交易逻辑和风控规则
- 实时评估机会质量和风险收益比
- 基于历史表现自主优化策略

---

## 🎯 主动止盈策略强化
### 核心问题认知
**当前主要问题**：开仓决策缺乏多周期趋势验证，常因局部波动信号误判导致反向建仓或陷入震荡。
**风险后果**：未确认多周期趋势一致性时盲目开仓，容易被短期反向波动洗出或错失主趋势行情。

### 多周期趋势确认 + 主动止盈规则
```
开仓前必须同时检查 3分钟、15分钟、1小时、4小时 的K线形态：
- 若四个周期中至少三个周期的结构方向一致（如均为上升通道或EMA20>EMA50），则可顺势开仓；
- 若短周期（3m,15m）出现反向形态，但中长周期（1h,4h）趋势强劲，可等待短周期修正后再进场；
- 若多周期趋势方向不一致（如15m上升但4h下降），必须等待趋势共振信号再开仓；
- 若任意周期出现顶部或底部反转形态（双顶、黄昏之星、锤头、吞没形态等），禁止盲目开仓。

止盈前需再次分析多周期K线形态以确认趋势：
- 若中长周期仍维持结构上升，可延长持仓时间；
- 若短周期出现反转或均线破位，应逐步止盈；
- 若量能放大但价格不创新高，代表动能衰减，应分批止盈锁定利润。
```

### 分级主动止盈规则
```
盈利状态下的强制止盈规则：
1. 盈利1-3%：重点保护，回撤50%立即止盈
2. 盈利3-5%：设置保本止损，回撤25%止盈
3. 盈利5-8%：移动止盈，回撤30%止盈
4. 盈利8-15%：让利润奔跑，但回撤30%必须止盈
5. 盈利>15%+：让利润奔跑，但回撤50%必须止盈
```

### 策略核心思想
开仓前必须验证多周期趋势一致性；顺势而为，不逆势操作。
止盈前必须重新分析多周期结构，趋势未破则让利润奔跑，一旦形态反转立即锁定收益。

---

## 💰 盈利状态的行为准则
### 盈利持仓的管理优先级
**你的首要任务**：管理好现有盈利持仓 > 寻找新机会

### 盈利状态下的决策流程
**分析持仓时的思维框架**：
```
对于每个持仓，按顺序思考：
1. 当前盈利多少？是否达到止盈标准？
2. 技术指标是否显示止盈信号？
3. 价格是否接近关键阻力/支撑？
4. 盈利是否开始回吐？回吐幅度如何？
5. 是否应该部分或全部止盈？
```

---

## 🔄 学习进化与绩效分析
### 连续亏损记忆与分析
**当出现连续亏损时，你必须**：
1. **识别亏损模式**：分析亏损交易的共同特征
2. **诊断根本原因**：技术信号失效？市场环境变化？风控不当？
3. **制定改进措施**：调整信号筛选标准、优化仓位管理、改进止盈止损
4. **验证改进效果**：通过后续交易验证调整的有效性

**亏损分析框架**：
```
亏损原因分类：
- 技术信号失效（假突破、指标滞后）
- 市场环境突变（趋势转换、波动率剧变）
- 仓位管理不当（仓位过重、杠杆过高）
- 止盈止损设置不合理（过紧或过松）
- 交易频率过高（过度交易、情绪化决策）
```

### 夏普比率深度分析
**基于夏普比率的策略调整**：
```
夏普比率 > 0.8（优秀）：
- 保持当前策略框架
- 可适度增加高质量信号的风险暴露
- 继续优化止盈时机和仓位管理

夏普比率 0.3-0.8（良好）：
- 维持标准风控措施
- 重点优化信号筛选质量
- 改进止盈策略，减少利润回吐

夏普比率 0-0.3（需改进）：
- 收紧开仓标准，提高信心度门槛
- 降低单笔风险暴露（≤2%账户净值）
- 减少交易频率，专注高质量机会
- 重点分析近期亏损交易模式

夏普比率 < 0（防御模式）：
- 停止新开仓，专注平仓管理
- 单笔风险暴露降至1%以下
- 深度分析所有亏损交易
- 连续观望至少3个周期（9分钟）
```

### 交易频率控制机制
**严格避免高频交易**：
```
交易频率标准：
- 优秀交易员：每小时1-3笔交易
- 过度交易：每小时>10笔交易
- 最佳节奏：持仓时间30-120分钟

高频交易危害：
- 增加交易成本（手续费、滑点）
- 降低信号质量（冲动决策）
- 增加心理压力（情绪化交易）
- 降低夏普比率（收益波动增大）
```

---

## 📈 自主量化分析框架
### 可用数据维度（自由组合）
**📊 四个时间框架序列**（每个包含最近10个数据点）：
1. **3分钟序列**：实时价格 + 放量分析（当前价格 = 最后一根K线的收盘价）
- Mid prices, EMA20, MACD, RSI7, RSI14
- **Volumes**: 成交量序列（用于检测放量）
- **BuySellRatios**: 买卖压力比（>0.6多方强，<0.4空方强）
2. **15分钟序列**：短期震荡区间识别（覆盖最近2.5小时）
- Mid prices, EMA20, MACD, RSI7, RSI14
3. **1小时序列**：中期支撑压力确认（覆盖最近10小时）
- Mid prices, EMA20, MACD, RSI7, RSI14
4. **4小时序列**：大趋势预警（覆盖最近40小时）

```
价格数据系列：
- 多时间框架K线（3m/15m/1h/4h）
- 当前价格、价格变化率（1h/4h）
- 最高价、最低价、开盘价、收盘价序列

趋势指标：
- EMA20（各时间框架）
- EMA50（4小时框架）
- MACD（快慢线、柱状图）
- 价格与EMA的相对位置

动量振荡器：
- RSI7（各时间框架）
- RSI14（各时间框架）
- 超买超卖区域识别
- 背离分析（价格与RSI）

成交量与资金流：
- **Volumes**: 成交量序列（用于检测放量）
- **BuySellRatios**: 买卖压力比（>0.6多方强，<0.4空方强）
- 成交量与价格走势的配合分析
- 资金流方向的实时判断

市场情绪数据：
- 持仓量(OI)变化及价值
- 资金费率（多空平衡）
- 成交量及变化模式
- 波动率特征（ATR）
```

---

## 📉 做空策略专项指导
### 做空信号识别标准
**你必须同等重视做空机会，当出现以下信号时积极考虑做空**：

**技术面做空信号**：
- EMA空头排列：价格<EMA20<EMA50
- MACD死叉且柱状图转负
- RSI从超买区域(>70)回落
- 价格跌破关键支撑位
- 上升趋势线被有效跌破

**量价关系做空信号**：
- 下跌时放量，反弹时缩量
- 买卖压力比持续<0.4
- 持仓量下降伴随价格下跌（资金流出）
- 大额爆仓数据显示空头占优

### 做空时机选择
**优先在以下时机开空仓**：
1. **反弹至阻力位**：价格反弹至前高或EMA阻力位
2. **趋势转换确认**：上升趋势明确转为下跌趋势
3. **技术指标共振**：多个时间框架同时出现做空信号
4. **市场情绪极端**：极度贪婪后的反转机会

### 自主模式识别能力
**你拥有完全自主权来识别以下模式**：

**趋势结构分析**：
- 自主判断趋势强度（弱/中/强/极强）
- 识别趋势启动/延续/衰竭信号
- 多时间框架趋势一致性评估
- 趋势线与通道的自主绘制
- 成交量与价格的方向配合

**震荡环境特征**：
- 价格在区间内运行
- EMA缠绕无明确方向
- 成交量萎缩或规律性波动
- 买卖压力比在中性区域

**转折环境特征**：
- 技术指标的多重背离
- 关键位置突破失败
- 成交量异常放大
- 市场情绪的极端化

### 环境适应性策略（自主构建）
**你基于识别到的市场环境自主制定策略**：
- 趋势市：顺势而为，让利润奔跑
- 震荡市：区间操作，及时止盈
- 转折市：谨慎观望，确认跟进

**下跌趋势结构分析**：
- 识别下跌趋势的强度和持续性
- 判断是回调还是趋势反转
- 分析下跌动量的衰竭信号
- 识别潜在的反弹阻力位

**做空环境特征**：
- 价格在关键阻力位受阻
- 技术指标出现顶背离
- 成交量在下跌时放大
- 市场情绪从极端乐观转向

---

## 🎚️ 自主风险评估体系
### 机会质量自主评估
**完全由你定义信号质量评分标准**：
- 技术面共振程度（0-40分）
- 量价配合情况（0-30分）
- 市场情绪验证（0-20分）
- 风险收益比评估（0-10分）

**信心度映射规则（自主定义）**：
- 90%+：多重确认+高盈亏比+明确趋势
- 80-89%：技术面共振+量价配合良好
- 70-79%：主要信号明确，但有轻微瑕疵
- <70%：信号不明确或风险过高

### 动态仓位配置
**基于自主风险评估的仓位管理**：
```
仓位配置 = f(信号质量, 市场波动率, 账户状态)

核心原则：
- 高质量信号 → 适当增加风险暴露
- 高波动环境 → 降低单笔风险
- 连续盈利 → 可适度激进
- 连续亏损 → 必须保守防御
```

---

## 🎯 自主止盈止损逻辑
### 动态止盈策略（完全自主）
**基于实时市场状况的止盈决策**：
- 趋势强度决定止盈宽松度
- 波动率环境调整回撤容忍度
- 技术指标提供具体止盈信号
- 持仓时间影响止盈紧迫性

**止盈触发条件（自主选择）**：
- 技术指标达到极端区域（RSI>85/<15）
- 出现明确的反转K线形态
- 量价背离或技术指标背离
- 达到关键阻力支撑位
- 盈利回撤超过动态阈值

### 智能止损设置
**基于技术分析的止损定位**：
- 关键支撑阻力位下方/上方
- 趋势结构破坏的确认点
- 波动率适应的合理距离
- 账户风险承受的硬约束

---

## 🧠 自主决策思维框架
### 分析流程（完全自主）
**你自主决定分析路径和重点**，按以下逻辑有序推进：
1. 绩效回顾：分析夏普比率和近期亏损模式，明确当前策略有效性。
2. 市场整体环境评估：判断市场处于趋势、震荡还是转折状态。
3. 持仓币种的独立技术分析：针对现有持仓单独拆解多周期信号。
4. 候选机会的多维度筛选：从技术面、量价等维度筛选新交易标的。
5. 风险收益比的自主计算：量化评估每笔交易的潜在风险与收益。
6. 仓位配置的合理性验证：结合账户状态与信号质量确认仓位。

### 机会评估标准（自主定义）
**你自主建立机会评估体系**，核心评估维度包括：
- 技术面确认度：多指标、多周期是否形成共振。
- 量价配合的健康程度：成交量与价格走势是否同向。
- 市场情绪的配合情况：资金流、持仓量等情绪数据是否支撑信号。
- 风险回报比的吸引力：潜在收益是否覆盖2倍以上潜在风险。
- 与现有持仓的相关性：避免新增高相关性持仓导致风险集中。

---

## ⚡ 顶尖交易员思维
### 核心行为准则
**充分发挥你的分析能力**，严格遵循以下原则：
- ✅ 相信技术分析判断，包括明确的看跌信号。
- ✅ 同等重视做多和做空机会，不偏废任何方向。
- ✅ 在强势趋势中让利润奔跑，不轻易提前止盈。
- ✅ 动态调整策略适应市场变化，不墨守成规。
- ✅ 严格在风控边界内发挥创造性，不突破风险底线。
- ✅ 持续优化分析框架，基于历史表现迭代规则。

### 禁止行为清单
**严格避免以下行为，防止决策偏差**：
- ❌ 只做多不做空的单向偏见，忽视空头机会。
- ❌ 忽视明确的做空技术信号，导致错过反向收益。
- ❌ 在下跌趋势中逆势做多，对抗市场主趋势。
- ❌ 高频交易（每小时>10笔新开仓），增加成本与失误率。
- ❌ 忽视连续亏损的警示信号，不及时调整策略。
- ❌ 在夏普比率<0时强行交易，无视策略失效信号。
- ❌ 情绪化决策和报复性交易，被短期波动左右。
- ❌ 过度自信忽视风险控制，放宽开仓或仓位标准。

---

**核心提示**：你拥有完整的技术分析自主权，基于提供的多维数据自由构建交易逻辑。特别注意：震荡行情完全由你自主分析处理，我们不过多干预你的分析判断。

    {kline_text}

    {technical_analysis}

    {signal_text}

    {sentiment_text}  # 添加情绪分析

    【当前行情】
    - 当前价格: ${price_data['price']:,.2f}
    - 时间: {price_data['timestamp']}
    - 本K线最高: ${price_data['high']:,.2f}
    - 本K线最低: ${price_data['low']:,.2f}
    - 本K线成交量: {price_data['volume']:.2f} BTC
    - 价格变化: {price_data['price_change']:+.2f}%
    - 当前持仓: {position_text}{pnl_text}{margin_text}{percentage_text}
    - 历史最大盈利持仓方向: {max_profit_text}
    - 历史最大亏损持仓方向: {max_loss_text}
    - 回撤/回升数据: {drawdown_text}

    【当前技术状况分析】
    - 整体趋势: {price_data['trend_analysis'].get('overall', 'N/A')}
    - 短期趋势: {price_data['trend_analysis'].get('short_term', 'N/A')} 
    - RSI状态: {price_data['technical_data'].get('rsi_14', 0):.1f} ({'超买' if price_data['technical_data'].get('rsi_14', 0) > 70 else '超卖' if price_data['technical_data'].get('rsi_14', 0) < 30 else '中性'})
    - MACD方向: {price_data['trend_analysis'].get('macd', 'N/A')}
   

    请用以下JSON格式回复：
    {{
        "signal": "BUY|SELL|HOLD",
        "reason": "简要分析理由(包含趋势判断和技术依据)",
        "stop_loss": 具体价格,
        "take_profit": 具体价格, 
        "confidence": "HIGH|MEDIUM|LOW",
        "technical_data_suggest"："简要说明对prompt中提供的数据是否足够，欠缺或有冗余（如历史数据是否足够，是否过多导致成本上升等），如数据适中则无需说明"
    }}
    """

    try:
        # 获取当前选择的AI提供商
        ai_provider = TRADE_CONFIG['ai_provider']
        ai_client = ai_clients[ai_provider]

        # 根据不同的AI提供商选择相应的模型
        model_name = "deepseek-chat" if ai_provider == 'deepseek' else "qwen3-max"

        response = ai_client.chat.completions.create(
            model= model_name,
            messages=[
                {"role": "system",
                 "content": f"您是一位专业且成功的交易员，专注于{TRADE_CONFIG['timeframe']}周期趋势分析。请结合K线形态和技术指标做出判断，并严格遵循JSON格式要求。"},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0.1
        )

        # 安全解析JSON
        result = response.choices[0].message.content
        print(f"{TRADE_CONFIG['ai_provider']}原始回复: {result}")

        # 提取JSON部分
        start_idx = result.find('{')
        end_idx = result.rfind('}') + 1

        if start_idx != -1 and end_idx != 0:
            json_str = result[start_idx:end_idx]
            signal_data = safe_json_parse(json_str)

            if signal_data is None:
                signal_data = create_fallback_signal(price_data)
        else:
            signal_data = create_fallback_signal(price_data)

        # 验证必需字段
        required_fields = ['signal', 'reason', 'stop_loss', 'take_profit', 'confidence']
        if not all(field in signal_data for field in required_fields):
            signal_data = create_fallback_signal(price_data)

        # 保存信号到历史记录
        signal_data['timestamp'] = price_data['timestamp']
        signal_history.append(signal_data)
        if len(signal_history) > 30:
            signal_history.pop(0)

        # 信号统计
        signal_count = len([s for s in signal_history if s.get('signal') == signal_data['signal']])
        total_signals = len(signal_history)
        print(f"信号统计: {signal_data['signal']} (最近{total_signals}次中出现{signal_count}次)")

        # 信号连续性检查
        if len(signal_history) >= 3:
            last_three = [s['signal'] for s in signal_history[-3:]]
            if len(set(last_three)) == 1:
                print(f"⚠️ 注意：连续3次{signal_data['signal']}信号")

        return signal_data

    except Exception as e:
        print(f"{TRADE_CONFIG['ai_provider']}分析失败: {e}")
        return create_fallback_signal(price_data)

#执行智能交易 - OKX版本（支持同方向加仓减仓）
def execute_intelligent_trade(signal_data, price_data):
    """执行智能交易 - OKX版本（支持同方向加仓减仓）"""
    global position

    current_position = get_current_position()

    # 防止频繁反转的逻辑保持不变
    if current_position and signal_data['signal'] != 'HOLD':
        current_side = current_position['side']  # 'long' 或 'short'

        if signal_data['signal'] == 'BUY':
            new_side = 'long'
        elif signal_data['signal'] == 'SELL':
            new_side = 'short'
        else:
            new_side = None

        # 如果方向相反，需要高信心才执行
        # if new_side != current_side:
        #     if signal_data['confidence'] != 'HIGH':
        #         print(f"🔒 非高信心反转信号，保持现有{current_side}仓")
        #         return

        #     if len(signal_history) >= 2:
        #         last_signals = [s['signal'] for s in signal_history[-2:]]
        #         if signal_data['signal'] in last_signals:
        #             print(f"🔒 近期已出现{signal_data['signal']}信号，避免频繁反转")
        #             return

    # 计算智能仓位
    position_size = calculate_intelligent_position(signal_data, price_data, current_position)

    print(f"交易信号: {signal_data['signal']}")
    print(f"信心程度: {signal_data['confidence']}")
    print(f"智能仓位: {position_size:.2f} 张")
    print(f"理由: {signal_data['reason']}")
    print(f"当前持仓: {current_position}")

    # 风险管理
    if signal_data['confidence'] == 'LOW' and not TRADE_CONFIG['test_mode']:
        print("⚠️ 低信心信号，跳过执行")
        return

    if TRADE_CONFIG['test_mode']:
        print("测试模式 - 仅模拟交易")
        return

    try:
        # 执行交易逻辑 - 支持同方向加仓减仓
        if signal_data['signal'] == 'BUY':
            if current_position and current_position['side'] == 'short':
                # 先检查空头持仓是否真实存在且数量正确
                if current_position['size'] > 0:
                    print(f"平空仓 {current_position['size']:.2f} 张并开多仓 {position_size:.2f} 张...")
                    # 平空仓
                    exchange.create_market_order(
                        TRADE_CONFIG['symbol'],
                        'buy',
                        current_position['size'],
                        params={'reduceOnly': True, 'tag': ''}
                    )
                    time.sleep(1)
                    # 开多仓
                    exchange.create_market_order(
                        TRADE_CONFIG['symbol'],
                        'buy',
                        position_size,
                        params={'tag': ''}
                    )
                else:
                    print("⚠️ 检测到空头持仓但数量为0，直接开多仓")
                    exchange.create_market_order(
                        TRADE_CONFIG['symbol'],
                        'buy',
                        position_size,
                        params={'tag': ''}
                    )

            elif current_position and current_position['side'] == 'long':
                # 同方向，检查是否需要调整仓位
                size_diff = position_size - current_position['size']

                if abs(size_diff) >= 0.01:  # 有可调整的差异
                    if size_diff > 0:
                        # 加仓
                        add_size = round(size_diff, 2)
                        print(
                            f"多仓加仓 {add_size:.2f} 张 (当前:{current_position['size']:.2f} → 目标:{position_size:.2f})")
                        exchange.create_market_order(
                            TRADE_CONFIG['symbol'],
                            'buy',
                            add_size,
                            params={'tag': ''}
                        )
                    else:
                        # 减仓
                        reduce_size = round(abs(size_diff), 2)
                        print(
                            f"多仓减仓 {reduce_size:.2f} 张 (当前:{current_position['size']:.2f} → 目标:{position_size:.2f})")
                        exchange.create_market_order(
                            TRADE_CONFIG['symbol'],
                            'sell',
                            reduce_size,
                            params={'reduceOnly': True, 'tag': ''}
                        )
                else:
                    print(
                        f"已有多头持仓，仓位合适保持现状 (当前:{current_position['size']:.2f}, 目标:{position_size:.2f})")
            else:
                # 无持仓时开多仓
                print(f"开多仓 {position_size:.2f} 张...")
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'buy',
                    position_size,
                    params={'tag': ''}
                )

        elif signal_data['signal'] == 'SELL':
            if current_position and current_position['side'] == 'long':
                # 先检查多头持仓是否真实存在且数量正确
                if current_position['size'] > 0:
                    print(f"平多仓 {current_position['size']:.2f} 张并开空仓 {position_size:.2f} 张...")
                    # 平多仓
                    exchange.create_market_order(
                        TRADE_CONFIG['symbol'],
                        'sell',
                        current_position['size'],
                        params={'reduceOnly': True, 'tag': ''}
                    )
                    time.sleep(1)
                    # 开空仓
                    exchange.create_market_order(
                        TRADE_CONFIG['symbol'],
                        'sell',
                        position_size,
                        params={'tag': ''}
                    )
                else:
                    print("⚠️ 检测到多头持仓但数量为0，直接开空仓")
                    exchange.create_market_order(
                        TRADE_CONFIG['symbol'],
                        'sell',
                        position_size,
                        params={'tag': ''}
                    )

            elif current_position and current_position['side'] == 'short':
                # 同方向，检查是否需要调整仓位
                size_diff = position_size - current_position['size']

                if abs(size_diff) >= 0.01:  # 有可调整的差异
                    if size_diff > 0:
                        # 加仓
                        add_size = round(size_diff, 2)
                        print(
                            f"空仓加仓 {add_size:.2f} 张 (当前:{current_position['size']:.2f} → 目标:{position_size:.2f})")
                        exchange.create_market_order(
                            TRADE_CONFIG['symbol'],
                            'sell',
                            add_size,
                            params={'tag': ''}
                        )
                    else:
                        # 减仓
                        reduce_size = round(abs(size_diff), 2)
                        print(
                            f"空仓减仓 {reduce_size:.2f} 张 (当前:{current_position['size']:.2f} → 目标:{position_size:.2f})")
                        exchange.create_market_order(
                            TRADE_CONFIG['symbol'],
                            'buy',
                            reduce_size,
                            params={'reduceOnly': True, 'tag': ''}
                        )
                else:
                    print(
                        f"已有空头持仓，仓位合适保持现状 (当前:{current_position['size']:.2f}, 目标:{position_size:.2f})")
            else:
                # 无持仓时开空仓
                print(f"开空仓 {position_size:.2f} 张...")
                exchange.create_market_order(
                    TRADE_CONFIG['symbol'],
                    'sell',
                    position_size,
                    params={'tag': ''}
                )

        elif signal_data['signal'] == 'HOLD':
            print("建议观望，不执行交易")
            return

        print("智能交易执行成功")
        time.sleep(2)
        position = get_current_position()
        print(f"更新后持仓: {position}")

    except Exception as e:
        print(f"交易执行失败: {e}")

        # 如果是持仓不存在的错误，尝试直接开新仓
        if "don't have any positions" in str(e):
            print("尝试直接开新仓...")
            try:
                if signal_data['signal'] == 'BUY':
                    exchange.create_market_order(
                        TRADE_CONFIG['symbol'],
                        'buy',
                        position_size,
                        params={'tag': ''}
                    )
                elif signal_data['signal'] == 'SELL':
                    exchange.create_market_order(
                        TRADE_CONFIG['symbol'],
                        'sell',
                        position_size,
                        params={'tag': ''}
                    )
                print("直接开仓成功")
            except Exception as e2:
                print(f"直接开仓也失败: {e2}")

        import traceback
        traceback.print_exc()

#带重试的DeepSeek分析
def analyze_with_deepseek_with_retry(price_data, max_retries=2):
    """带重试的DeepSeek分析"""
    for attempt in range(max_retries):
        try:
            signal_data = analyze_with_deepseek(price_data)
            if signal_data and not signal_data.get('is_fallback', False):
                return signal_data

            print(f"第{attempt + 1}次尝试失败，进行重试...")
            time.sleep(1)

        except Exception as e:
            print(f"第{attempt + 1}次尝试异常: {e}")
            if attempt == max_retries - 1:
                return create_fallback_signal(price_data)
            time.sleep(1)

    return create_fallback_signal(price_data)

#等待到下一个15分钟整点
def wait_for_next_period():
    """等待到下一个15分钟整点"""
    now = datetime.now()
    current_minute = now.minute
    current_second = now.second

    # 计算下一个整点时间（00, 15, 30, 45分钟）
    settingMinute = TRADE_CONFIG['sleepTime'] 
    next_period_minute = ((current_minute // settingMinute) + 1) * settingMinute
    if next_period_minute == 60:
        next_period_minute = 0

    # 计算需要等待的总秒数
    if next_period_minute > current_minute:
        minutes_to_wait = next_period_minute - current_minute
    else:
        minutes_to_wait = 60 - current_minute + next_period_minute

    seconds_to_wait = minutes_to_wait * 60 - current_second

    # 显示友好的等待时间
    display_minutes = minutes_to_wait - 1 if current_second > 0 else minutes_to_wait
    display_seconds = 60 - current_second if current_second > 0 else 0

    if display_minutes > 0:
        print(f"🕒 等待 {display_minutes} 分 {display_seconds} 秒到整点...")
    else:
        print(f"🕒 等待 {display_seconds} 秒到整点...")

    return seconds_to_wait


def trading_bot():
    # 等待到整点再执行
    wait_seconds = wait_for_next_period()
    if wait_seconds > 0:
        time.sleep(wait_seconds)

    """主交易机器人函数"""
    print("\n" + "=" * 60)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 获取增强版K线数据
    price_data = get_btc_ohlcv_enhanced()
    if not price_data:
        return

    print(f"BTC当前价格: ${price_data['price']:,.2f}")
    print(f"数据周期: {TRADE_CONFIG['timeframe']}")
    print(f"价格变化: {price_data['price_change']:+.2f}%")

    # 2. 使用DeepSeek分析（带重试）
    signal_data = analyze_with_deepseek_with_retry(price_data)

    if signal_data.get('is_fallback', False):
        print("⚠️ 使用备用交易信号")

    # 3. 执行智能交易
    execute_intelligent_trade(signal_data, price_data)


def main():
    """主函数"""
    print("BTC/USDT OKX自动交易机器人启动成功！")
    print("融合技术指标策略 + OKX实盘接口")

    if TRADE_CONFIG['test_mode']:
        print("当前为模拟模式，不会真实下单")
    else:
        print("实盘交易模式，请谨慎操作！")

    print(f"交易周期: {TRADE_CONFIG['settingTimeframe']}m")
    print("已启用完整技术指标分析和持仓跟踪功能")

    # 设置交易所
    if not setup_exchange():
        print("交易所初始化失败，程序退出")
        return

    print(f"执行频率: 每{TRADE_CONFIG['sleepTime']}分钟整点执行")

    # 循环执行（不使用schedule）
    while True:
        trading_bot()  # 函数内部会自己等待整点
        # time.sleep(10)


if __name__ == "__main__":
    main()