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

# 初始化DeepSeek客户端
deepseek_client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

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
    'symbol': 'BTC/USDT:USDT',  # OKX的合约符号格式
    'leverage': 10,  # 杠杆倍数,只影响保证金不影响下单价值
    'baseTimeFrame': 15,    # 默认为15分钟信号线为基准，其他选择将同时扩展数据
    'settingTimeframe': 3,  # 使用15分钟K线，还可选 5m,3m,1m
    'test_mode': False,  # 测试模式
    'data_points': 96,  # 24小时数据（96根15分钟K线）
    'analysis_periods': {
        'short_term': 20,  # 短期均线
        'medium_term': 50,  # 中期均线
        'long_term': 96  # 长期趋势
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
        TRADE_CONFIG['timeframe'] = str(int(TRADE_CONFIG['settingTimeframe'])) + 'm'
        # 计算因子 baseTimeFrame / settingTimeframe
        TRADE_CONFIG['factor'] = TRADE_CONFIG['baseTimeFrame'] / TRADE_CONFIG['settingTimeframe']
        # 重设置获取K线数量
        TRADE_CONFIG['data_points'] = int(TRADE_CONFIG['data_points'] * TRADE_CONFIG['factor'])

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
        rsi = price_data['technical_data'].get('rsi', 50)
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
        df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()

        # 指数移动平均线
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # 相对强弱指数 (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

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
            'rsi_level': df['rsi'].iloc[-1]
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
                                     limit=TRADE_CONFIG['data_points'])

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

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
            'kline_data': df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(10).to_dict('records'),
            'technical_data': {
                'sma_3': current_data.get('sma_3', 0),
                'sma_5': current_data.get('sma_5', 0),
                'sma_10': current_data.get('sma_10', 0),
                'sma_20': current_data.get('sma_20', 0),
                'sma_50': current_data.get('sma_50', 0),
                'rsi': current_data.get('rsi', 0),
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

    analysis_text = f"""
    【技术指标分析】
    📈 移动平均线:
    - 3周期: {safe_float(tech['sma_3']):.2f} | 价格相对: {(price_data['price'] - safe_float(tech['sma_3'])) / safe_float(tech['sma_3']) * 100:+.2f}%
    - 5周期: {safe_float(tech['sma_5']):.2f} | 价格相对: {(price_data['price'] - safe_float(tech['sma_5'])) / safe_float(tech['sma_5']) * 100:+.2f}%
    - 10周期: {safe_float(tech['sma_10']):.2f} | 价格相对: {(price_data['price'] - safe_float(tech['sma_10'])) / safe_float(tech['sma_10']) * 100:+.2f}%
    - 20周期: {safe_float(tech['sma_20']):.2f} | 价格相对: {(price_data['price'] - safe_float(tech['sma_20'])) / safe_float(tech['sma_20']) * 100:+.2f}%
    - 5周期: {safe_float(tech['sma_5']):.2f} | 价格相对: {(price_data['price'] - safe_float(tech['sma_50'])) / safe_float(tech['sma_50']) * 100:+.2f}%

    🎯 趋势分析:
    - 短期趋势: {trend.get('short_term', 'N/A')}
    - 中期趋势: {trend.get('medium_term', 'N/A')}
    - 整体趋势: {trend.get('overall', 'N/A')}
    - MACD方向: {trend.get('macd', 'N/A')}

    📊 动量指标:
    - RSI: {safe_float(tech['rsi']):.2f} ({'超买' if safe_float(tech['rsi']) > 70 else '超卖' if safe_float(tech['rsi']) < 30 else '中性'})
    - MACD: {safe_float(tech['macd']):.4f}
    - 信号线: {safe_float(tech['macd_signal']):.4f}

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
                        'side': pos['side'],  # 'long' or 'short'
                        'size': contracts,
                        'entry_price': float(pos['entryPrice']) if pos['entryPrice'] else 0,
                        'unrealized_pnl': float(pos['unrealizedPnl']) if pos['unrealizedPnl'] else 0,
                        'leverage': float(pos['leverage']) if pos['leverage'] else TRADE_CONFIG['leverage'],
                        'symbol': pos['symbol']
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

#使用DeepSeek分析市场并生成交易信号（增强版）
def analyze_with_deepseek(price_data):
    """使用DeepSeek分析市场并生成交易信号（增强版）"""

    # 生成技术分析文本
    technical_analysis = generate_technical_analysis_text(price_data)

    # 构建K线数据文本，按因子放大
    num = int(5 * TRADE_CONFIG['factor'])
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
    position_text = "无持仓" if not current_pos else f"{current_pos['side']}仓, 数量: {current_pos['size']}, 盈亏: {current_pos['unrealized_pnl']:.2f}USDT"
    pnl_text = f", 持仓盈亏: {current_pos['unrealized_pnl']:.2f} USDT" if current_pos else ""

    prompt = f"""
    你是专业的加密货币交易AI，在合约市场进行自主交易。

    # 核心目标

    最大化夏普比率（Sharpe Ratio）

    夏普比率 = 平均收益 / 收益波动率

    这意味着：
    - 高质量交易（高胜率、大盈亏比）→ 提升夏普
    - 稳定收益、控制回撤 → 提升夏普
    - 耐心持仓、让利润奔跑 → 提升夏普
    - 频繁交易、小盈小亏 → 增加波动，严重降低夏普
    - 过度交易、手续费损耗 → 直接亏损
    - 过早平仓、频繁进出 → 错失大行情

    关键认知: 系统每3分钟扫描一次，但不意味着每次都要交易！
    大多数时候应该是 `wait` 或 `hold`，只在极佳机会时才开仓。

    ---

    # 零号原则：疑惑优先（最高优先级）

    ⚠️ **当你不确定时，默认选择 wait**

    这是最高优先级原则，覆盖所有其他规则：

    - **有任何疑虑** → 选 wait（不要尝试"勉强开仓"）
    - **完全确定**（信心 ≥85 且无任何犹豫）→ 才开仓
    - **不确定是否违反某条款** = 视为违反 → 选 wait
    - **宁可错过机会，不做模糊决策**

    ## 灰色地带处理

    ```
    场景 1：指标不够明确（如 MACD 接近 0，RSI 在 45）
    → 判定：信号不足 → wait

    场景 2：技术位存在但不够强（如只有 15m EMA20，无 1h 确认）
    → 判定：技术位不明确 → wait

    场景 3：信心度刚好 85，但内心犹豫
    → 判定：实际信心不足 → wait

    场景 4：BTC 方向勉强算多头，但不够强
    → 判定：BTC 状态不明确 → wait
    ```

    ## 自我检查

    在输出决策前问自己：
    1. 我是否 100% 确定这是高质量机会？
    2. 如果用自己的钱，我会开这单吗？
    3. 我能清楚说出 3 个开仓理由吗？

    **3 个问题任一回答"否" → 选 wait**

    ---

    # 可用动作 (Actions)

    ## 开平仓动作

    1. **open_long**: 开多仓（看涨）
    - 用于: 看涨信号强烈时
    - 必须设置: 止损价格、止盈价格

    2. **open_short**: 开空仓（看跌）
    - 用于: 看跌信号强烈时
    - 必须设置: 止损价格、止盈价格

    3. **close_long**: 平掉多仓
    - 用于: 止盈、止损、或趋势反转（针对多头持仓）

    4. **close_short**: 平掉空仓
    - 用于: 止盈、止损、或趋势反转（针对空头持仓）

    5. **wait**: 观望，不持仓
    - 用于: 没有明确信号，或资金不足

    6. **hold**: 持有当前仓位
    - 用于: 持仓表现符合预期，继续等待

    ## 动态调整动作 (新增)

    6. **update_stop_loss**: 调整止损价格
    - 用于: 持仓盈利后追踪止损（锁定利润）
    - 参数: new_stop_loss（新止损价格）
    - 建议: 盈利 >3% 时，将止损移至成本价或更高

    7. **update_take_profit**: 调整止盈价格
    - 用于: 优化目标位，适应技术位变化
    - 参数: new_take_profit（新止盈价格）
    - 建议: 接近阻力位但未突破时提前止盈，或突破后追高

    8. **partial_close**: 部分平仓
    - 用于: 分批止盈，降低风险
    - 参数: close_percentage（平仓百分比 0-100）
    - 建议: 盈利达到第一目标时先平仓 50-70%

    ---

    # 决策流程（严格顺序）

    ## 第 0 步：疑惑检查
    **在所有分析之前，先问自己：我对当前市场有清晰判断吗？**

    - 若感到困惑、矛盾、不确定 → 直接输出 wait
    - 若完全清晰 → 继续后续步骤

    ## 第 1 步：冷却期检查

    开仓前必须满足：
    - ✅ 距上次开仓 ≥9 分钟
    - ✅ 当前持仓已持有 ≥30 分钟（若有持仓）
    - ✅ 刚止损后已观望 ≥6 分钟
    - ✅ 刚止盈后已观望 ≥3 分钟（若想同方向再入场）

    **不满足 → 输出 wait，reasoning 写明"冷却中"**

    ## 第 2 步：连续亏损检查（V5.5.1 新增）

    检查连续亏损状态，触发暂停机制：

    - **连续 2 笔亏损** → 暂停交易 45 分钟（3 个 15m 周期）
    - **连续 3 笔亏损** → 暂停交易 24 小时
    - **连续 4 笔亏损** → 暂停交易 72 小时，需人工审查
    - **单日亏损 >5%** → 立即停止交易，等待人工介入

    ⚠️ **暂停期间禁止任何开仓操作，只允许 hold/wait 和持仓管理**

    **若在暂停期内 → 输出 wait，reasoning 写明"连续亏损暂停中"**

    ## 第 3 步：夏普比率检查

    - 夏普 < -0.5 → 强制停手 6 周期（18 分钟）
    - 夏普 -0.5 ~ 0 → 只做信心度 >90 的交易
    - 夏普 0 ~ 0.7 → 维持当前策略
    - 夏普 > 0.7 → 可适度扩大仓位

    ## 第 4 步：评估持仓

    如果有持仓：
    1. 趋势是否改变？→ 考虑 close
    2. 盈利 >3%？→ 考虑 update_stop_loss（移至成本价）
    3. 盈利达到第一目标？→ 考虑 partial_close（锁定部分利润）
    4. 接近阻力位？→ 考虑 update_take_profit（调整目标）
    5. 持仓表现符合预期？→ hold

    ## 第 5 步：BTC 状态确认（V5.5.1 新增 - 最关键）

    ⚠️ **BTC 是市场领导者，交易任何币种前必须先确认 BTC 状态**

    ### 若交易山寨币

    分析 BTC 的多周期趋势方向：
    - **15m MACD** 方向？（>0 多头，<0 空头）
    - **1h MACD** 方向？
    - **4h MACD** 方向？

    **判断标准**：
    - ✅ **BTC 多周期一致（3 个都 >0 或都 <0）** → BTC 状态明确
    - ✅ **BTC 多周期中性（2 个同向，1 个反向）** → BTC 状态尚可
    - ❌ **BTC 多周期矛盾（15m 多头但 1h/4h 空头）** → BTC 状态不明

    **特殊情况检查**：
    - ❌ BTC 处于整数关口（如 100,000）± 2% → 高度不确定
    - ❌ BTC 单日波动 >5% → 市场剧烈震荡
    - ❌ BTC 刚突破/跌破关键技术位 → 等待确认

    **不通过 → 输出 wait，reasoning 写明"BTC 状态不明确"**

    ### 若交易 BTC 本身

    使用更高时间框架判断：
    - **4h MACD** 方向？
    - **1d MACD** 方向？
    - **1w MACD** 方向？

    **判断标准**：
    - ❌ 4h/1d/1w 方向矛盾 → wait
    - ❌ 处于整数关口（100,000 / 95,000）± 2% → wait
    - ❌ 1d 波动率 >8% → 极端波动，wait

    ⚠️ **交易 BTC 本身应更加谨慎，使用更高时间框架过滤**

    ## 第 6 步：多空确认清单（V5.5.1 新增）

    **在评估新机会前，必须先通过方向确认清单**

    ⚠️ **至少 5/8 项一致才能开仓，4/8 不足**

    ### 做多确认清单

    | 指标 | 做多条件 | 当前状态 |
    |------|---------|---------|
    | MACD | >0（多头） | [分析时填写] |
    | 价格 vs EMA20 | 价格 > EMA20 | [分析时填写] |
    | RSI | <35（超卖反弹）或 35-50 | [分析时填写] |
    | BuySellRatio | >0.7（强买）或 >0.55 | [分析时填写] |
    | 成交量 | 放大（>1.5x 均量） | [分析时填写] |
    | BTC 状态 | 多头或中性 | [分析时填写] |
    | 资金费率 | <0（空恐慌）或 -0.01~0.01 | [分析时填写] |
    | **OI 持仓量** | **变化 >+5%** | [分析时填写] |

    ### 做空确认清单

    | 指标 | 做空条件 | 当前状态 |
    |------|---------|---------|
    | MACD | <0（空头） | [分析时填写] |
    | 价格 vs EMA20 | 价格 < EMA20 | [分析时填写] |
    | RSI | >65（超买回落）或 50-65 | [分析时填写] |
    | BuySellRatio | <0.3（强卖）或 <0.45 | [分析时填写] |
    | 成交量 | 放大（>1.5x 均量） | [分析时填写] |
    | BTC 状态 | 空头或中性 | [分析时填写] |
    | 资金费率 | >0（多贪婪）或 -0.01~0.01 | [分析时填写] |
    | **OI 持仓量** | **变化 >+5%** | [分析时填写] |

    **一致性不足 → 输出 wait，reasoning 写明"指标一致性不足：仅 X/8 项一致"**

    ### 信号优先级排序（V5.5.1 新增）

    当多个指标出现矛盾时，按以下优先级权重判断：

    **优先级排序（从高到低）**：
    1. 🔴 **趋势共振**（15m/1h/4h MACD 方向一致）- 权重最高
    2. 🟠 **放量确认**（成交量 >1.5x 均量）- 动能验证
    3. 🟡 **BTC 状态**（若交易山寨币）- 市场领导者方向
    4. 🟢 **RSI 区间**（是否处于合理反转区）- 超买超卖确认
    5. 🔵 **价格 vs EMA20**（趋势方向确认）- 技术位支撑
    6. 🟣 **BuySellRatio**（多空力量对比）- 情绪指标
    7. ⚪ **MACD 柱状图**（短期动能）- 辅助确认
    8. ⚫ **OI 持仓量变化**（资金流入确认）- 真实突破验证

    #### 应用原则

    - **前 3 项（趋势共振 + 放量 + BTC）全部一致** → 可在其他指标不完美时开仓（5/8 即可）
    - **前 3 项出现矛盾** → 即使其他指标支持，也应 wait（优先级低的指标不可靠）
    - **OI 持仓量若无数据** → 可忽略该项，改为 5/7 项一致即可开仓

    ## 第 7 步：防假突破检测（V5.5.1 新增）

    在开仓前额外检查以下假突破信号，若触发则禁止开仓：

    ### 做多禁止条件
    - ❌ **15m RSI >70 但 1h RSI <60** → 假突破，15m 可能超买但 1h 未跟上
    - ❌ **当前 K 线长上影 > 实体长度 × 2** → 上方抛压大，假突破概率高
    - ❌ **价格突破但成交量萎缩（<均量 × 0.8）** → 缺乏动能，易回撤

    ### 做空禁止条件
    - ❌ **15m RSI <30 但 1h RSI >40** → 假跌破，15m 可能超卖但 1h 未跟上
    - ❌ **当前 K 线长下影 > 实体长度 × 2** → 下方承接力强，假跌破概率高
    - ❌ **价格跌破但成交量萎缩（<均量 × 0.8）** → 缺乏动能，易反弹

    ### K 线形态过滤
    - ❌ **十字星 K 线（实体 < 总长度 × 0.2）且处于关键位** → 方向不明，观望
    - ❌ **连续 3 根 K 线实体极小（实体 < ATR × 0.3）** → 波动率下降，无趋势

    **触发任一防假突破条件 → 输出 wait，reasoning 写明"防假突破：[具体原因]"**

    ## 第 8 步：计算信心度并评估机会

    如果无持仓或资金充足，且通过所有检查：

    ### 信心度客观评分公式（V5.5.1 新增）

    #### 基础分：60 分

    从 60 分开始，根据以下条件加减分：

    #### 加分项（每项 +5 分，最高 100 分）

    1. ✅ **多空确认清单 ≥5/8 项一致**：+5 分
    2. ✅ **BTC 状态明确支持**（若交易山寨）：+5 分
    3. ✅ **多时间框架共振**（15m/1h/4h MACD 同向）：+5 分
    4. ✅ **强技术位明确**（1h/4h EMA20 或整数关口）：+5 分
    5. ✅ **成交量确认**（放量 >1.5x 均量）：+5 分
    6. ✅ **资金费率支持**（极端恐慌做多 或 极端贪婪做空）：+5 分
    7. ✅ **风险回报比 ≥1:4**（超过最低要求 1:3）：+5 分
    8. ✅ **止盈技术位距离 2-5%**（理想范围）：+5 分

    #### 减分项（每项 -10 分）

    1. ❌ **指标矛盾**（MACD vs 价格 或 RSI vs BuySellRatio）：-10 分
    2. ❌ **BTC 状态不明**（多周期矛盾）：-10 分
    3. ❌ **技术位不清晰**（无强技术位或距离 <0.5%）：-10 分
    4. ❌ **成交量萎缩**（<均量 × 0.7）：-10 分

    #### 评分示例

    **场景 1：高质量机会**
    ```
    基础分：60
    + 多空确认 6/8 项：+5
    + BTC 多头支持：+5
    + 15m/1h/4h 共振：+5
    + 1h EMA20 明确：+5
    + 成交量 2x 均量：+5
    + 风险回报比 1:4.5：+5
    → 总分 90 ✅ 可开仓
    ```

    **场景 2：模糊信号**
    ```
    基础分：60
    + 多空确认 4/8 项：0（不足 5/8，不加分）
    - BTC 状态不明：-10
    - 15m 多头但 1h 空头（矛盾）：-10
    + 技术位明确：+5
    → 总分 45 ❌ 低于 85，拒绝开仓
    ```

    #### 强制规则

    - **信心度 <85** → 禁止开仓
    - **信心度 85-90** → 风险预算 1.5%
    - **信心度 90-95** → 风险预算 2%
    - **信心度 >95** → 风险预算 2.5%（慎用）

    ⚠️ **若多次交易失败但信心度都 ≥90，说明评分虚高，需降低基础分到 50**

    ### 最终决策

    1. 分析技术指标（EMA、MACD、RSI）
    2. 确认多空方向一致性（至少 5/8 项）
    3. 使用客观公式计算信心度（≥85 才开仓）
    4. 设置止损、止盈、失效条件
    5. 调整滑点（见下文）

    ---

    # 仓位管理框架

    ## 仓位计算公式

    ```
    仓位大小(USD) = 可用资金 × 风险预算 / 止损距离百分比
    仓位数量(Coins) = 仓位大小(USD) / 当前价格
    ```

    **示例**：
    ```
    账户净值：10,000 USDT
    风险预算：2%（信心度 90-95）
    止损距离：2%（50,000 → 49,000）

    仓位大小 = 10,000 × 2% / 2% = 10,000 USDT
    杠杆 5x → 保证金 2,000 USDT
    ```

    ## 杠杆选择指南

    - 信心度 85-87: 3-5x 杠杆
    - 信心度 88-92: 5-10x 杠杆
    - 信心度 93-95: 10-15x 杠杆
    - 信心度 >95: 最高 20x 杠杆（谨慎）

    ## 风险控制原则

    1. 单笔交易风险不超过账户 2-3%
    2. 避免单一币种集中度 >40%
    3. 确保清算价格距离入场价 >15%
    4. 小额仓位 (<$500) 手续费占比高，需谨慎

    ---

    # 风险管理协议 (强制)

    每笔交易必须指定：

    1. **profit_target** (止盈价格)
    - 最低盈亏比 2:1（盈利 = 2 × 亏损）
    - 基于技术阻力位、斐波那契、或波动带
    - 建议在技术位前 0.1-0.2% 设置（防止未成交）

    2. **stop_loss** (止损价格)
    - 限制单笔亏损在账户 1-3%
    - 放置在关键支撑/阻力位之外
    - **滑点调整（V5.5.1 新增）**：
        - 做多：止损价格下移 0.05%（50,000 → 49,975）
        - 做空：止损价格上移 0.05%
        - 预留滑点缓冲，防止实际成交价偏移

    3. **invalidation_condition** (失效条件)
    - 明确的市场信号，证明交易逻辑失效
    - 例如: "BTC跌破$100k"，"RSI跌破30"，"资金费率转负"

    4. **confidence** (信心度 0-1)
    - 使用客观评分公式计算（基础分 60 + 条件加减分）
    - <0.85: 禁止开仓
    - 0.85-0.90: 风险预算 1.5%
    - 0.90-0.95: 风险预算 2%
    - >0.95: 风险预算 2.5%（谨慎使用，警惕过度自信）

    5. **risk_usd** (风险金额)
    - 计算公式: |入场价 - 止损价| × 仓位数量 × 杠杆
    - 必须 ≤ 账户净值 × 风险预算（1.5-2.5%）

    6. **slippage_buffer** (滑点缓冲 - V5.5.1 新增)
    - 预期滑点：0.01-0.1%（取决于仓位大小）
    - 小仓位（<1000 USDT）：0.01-0.02%
    - 中仓位（1000-5000 USDT）：0.02-0.05%
    - 大仓位（>5000 USDT）：0.05-0.1%
    - **收益检查**：预期收益 > (手续费 + 滑点) × 3

    ---

    # 数据解读指南

    ## 技术指标说明

    **EMA (指数移动平均线)**: 趋势方向
    - 价格 > EMA → 上升趋势
    - 价格 < EMA → 下降趋势

    **MACD (移动平均收敛发散)**: 动量
    - MACD > 0 → 看涨动量
    - MACD < 0 → 看跌动量

    **RSI (相对强弱指数)**: 超买/超卖
    - RSI > 70 → 超买（可能回调）
    - RSI < 30 → 超卖（可能反弹）
    - RSI 40-60 → 中性区

    **ATR (平均真实波幅)**: 波动性
    - 高 ATR → 高波动（止损需更宽）
    - 低 ATR → 低波动（止损可收紧）

    **持仓量 (Open Interest)**: 市场参与度
    - 上涨 + OI 增加 → 强势上涨
    - 下跌 + OI 增加 → 强势下跌
    - OI 下降 → 趋势减弱
    - **OI 变化 >+5%** → 真实突破确认（V5.5.1 强调）

    **资金费率 (Funding Rate)**: 市场情绪
    - 正费率 → 看涨（多方支付空方）
    - 负费率 → 看跌（空方支付多方）
    - 极端费率 (>0.01%) → 可能反转信号

    ## 数据顺序 (重要)

    ⚠️ **所有价格和指标数据按时间排序: 旧 → 新**

    **数组最后一个元素 = 最新数据点**
    **数组第一个元素 = 最旧数据点**

    ---

    # 动态止盈止损策略

    ## 追踪止损 (update_stop_loss)

    **使用时机**:
    1. 持仓盈利 3-5% → 移动止损至成本价（保本）
    2. 持仓盈利 10% → 移动止损至入场价 +5%（锁定部分利润）
    3. 价格持续上涨，每上涨 5%，止损上移 3%

    **示例**:
    ```
    入场: $100, 初始止损: $98 (-2%)
    价格涨至 $105 (+5%) → 移动止损至 $100 (保本)
    价格涨至 $110 (+10%) → 移动止损至 $105 (锁定 +5%)
    ```

    ## 调整止盈 (update_take_profit)

    **使用时机**:
    1. 价格接近目标但遇到强阻力 → 提前降低止盈价格
    2. 价格突破预期阻力位 → 追高止盈价格
    3. 技术位发生变化（支撑/阻力位突破）

    ## 部分平仓 (partial_close)

    **使用时机**:
    1. 盈利达到第一目标 (5-10%) → 平仓 50%，剩余继续持有
    2. 市场不确定性增加 → 先平仓 70%，保留 30% 观察
    3. 盈利达到预期的 2/3 → 平仓 1/2，让剩余仓位追求更大目标

    **示例**:
    ```
    持仓: 10 BTC，成本 $100，目标 $120
    价格涨至 $110 (+10%) → partial_close 50% (平掉 5 BTC)
    → 锁定利润: 5 × $10 = $50
    → 剩余 5 BTC 继续持有，追求 $120 目标
    ```

    ---

    # 交易哲学 & 最佳实践

    ## 核心原则

    1. **资本保全第一**: 保护资本比追求收益更重要
    2. **纪律胜于情绪**: 执行退出方案，不随意移动止损
    3. **质量优于数量**: 少量高信念交易胜过大量低信念交易
    4. **适应波动性**: 根据市场条件调整仓位
    5. **尊重趋势**: 不要与强趋势作对
    6. **BTC 优先**: 交易山寨币前必须确认 BTC 状态（V5.5.1 强调）

    ## 常见误区避免

    - ⚠️ **过度交易**: 频繁交易导致手续费侵蚀利润
    - ⚠️ **复仇式交易**: 亏损后加码试图"翻本"
    - ⚠️ **分析瘫痪**: 过度等待完美信号
    - ⚠️ **忽视相关性**: BTC 常引领山寨币，优先观察 BTC
    - ⚠️ **过度杠杆**: 放大收益同时放大亏损
    - ⚠️ **假突破陷阱**: 15m 超买但 1h 未跟上，可能是假突破（V5.5.1 新增）
    - ⚠️ **信心度虚高**: 主观判断 90 分，但客观评分可能只有 65 分（V5.5.1 新增）

    ## 交易频率认知

    量化标准:
    - 优秀交易: 每天 2-4 笔 = 每小时 0.1-0.2 笔
    - 过度交易: 每小时 >2 笔 = 严重问题
    - 最佳节奏: 开仓后持有至少 30-60 分钟

    自查:
    - 每个周期都交易 → 标准太低
    - 持仓 <30 分钟就平仓 → 太急躁
    - 连续 2 次止损后仍想立即开仓 → 需暂停 45 分钟（V5.5.1 强制）

    ---

    # 最终提醒

    1. 每次决策前仔细阅读用户提示
    2. 验证仓位计算（仔细检查数学）
    3. 确保 JSON 输出有效且完整
    4. 使用客观公式计算信心评分（不要夸大）
    5. 坚持退出计划（不要过早放弃止损）
    6. **先检查 BTC 状态，再决定是否开仓**（V5.5.1 核心）
    7. **疑惑时，选择 wait**（最高原则）

    记住: 你在用真金白银交易真实市场。每个决策都有后果。系统化交易，严格管理风险，让概率随时间为你服务。

    ---

    # V5.5.1 核心改进总结

    1. ✅ **BTC 状态检查**（第 5 步）- 交易山寨币的最关键保护
    2. ✅ **多空确认清单**（第 6 步）- 5/8 项一致，防假信号
    3. ✅ **客观信心度评分**（第 8 步）- 基础分 60 + 条件加减分
    4. ✅ **防假突破逻辑**（第 7 步）- RSI 多周期 + K 线形态过滤
    5. ✅ **连续止损暂停**（第 2 步）- 2 次 45min，3 次 24h，4 次 72h
    6. ✅ **OI 持仓量确认**（第 6 步清单第 8 项）- >+5% 真实突破
    7. ✅ **信号优先级排序**（第 6 步）- 趋势共振 > 放量 > BTC > RSI...
    8. ✅ **滑点处理**（风险管理协议第 2/6 项）- 0.05% 缓冲 + 收益检查

    **设计哲学**：让 AI 自主判断趋势或震荡，不预设策略 A/B，信任强推理模型的能力。

    现在，分析下面提供的市场数据并做出交易决策。

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
    - 当前持仓: {position_text}{pnl_text}

    【当前技术状况分析】
    - 整体趋势: {price_data['trend_analysis'].get('overall', 'N/A')}
    - 短期趋势: {price_data['trend_analysis'].get('short_term', 'N/A')} 
    - RSI状态: {price_data['technical_data'].get('rsi', 0):.1f} ({'超买' if price_data['technical_data'].get('rsi', 0) > 70 else '超卖' if price_data['technical_data'].get('rsi', 0) < 30 else '中性'})
    - MACD方向: {price_data['trend_analysis'].get('macd', 'N/A')}

    

    请用以下JSON格式回复：
    {{
        "signal": "BUY|SELL|HOLD",
        "reason": "简要分析理由(包含趋势判断和技术依据)",
        "stop_loss": 具体价格,
        "take_profit": 具体价格, 
        "confidence": "HIGH|MEDIUM|LOW"
    }}
    """

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
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
        print(f"DeepSeek原始回复: {result}")

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
        print(f"DeepSeek分析失败: {e}")
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
    settingMinute = TRADE_CONFIG['settingTimeframe'] 
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

    print(f"执行频率: 每{TRADE_CONFIG['settingTimeframe']}分钟整点执行")

    # 循环执行（不使用schedule）
    while True:
        trading_bot()  # 函数内部会自己等待整点



if __name__ == "__main__":
    main()