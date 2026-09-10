"""
Broker 介面抽象層
================
預留群益 API 串接結構，目前僅實作回測用模擬 Broker。

架構：
  BrokerInterface (ABC)
    ├── BacktestBroker   → 使用歷史資料回測
    └── CapitalBroker    → 群益 API（未來實作）
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


# ============================================================
# 行情資料結構
# ============================================================
@dataclass
class Quote:
    """即時/歷史報價"""
    ticker: str
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: int = 0


@dataclass
class OrderResult:
    """下單結果"""
    order_id: str
    ticker: str
    side: str        # 'BUY' or 'SELL'
    qty: int         # 口數
    price: float     # 成交價
    status: str      # 'FILLED', 'REJECTED', 'PENDING'
    message: str = ''
    fee: float = 0.0
    tax: float = 0.0


@dataclass
class Position:
    """持倉資訊"""
    ticker: str
    side: str        # 'LONG' or 'SHORT'
    qty: int
    avg_price: float
    unrealized_pnl: float = 0.0
    margin_used: float = 0.0


@dataclass
class MarginInfo:
    """保證金資訊"""
    total_equity: float
    available_margin: float
    used_margin: float
    maintenance_margin: float


# ============================================================
# 抽象 Broker 介面
# ============================================================
class BrokerInterface(ABC):
    """
    Broker 介面 — 定義報價取得、下單、持倉查詢等方法。
    回測模式與實盤模式共用同一介面，方便切換。
    """

    @abstractmethod
    def get_quote(self, ticker: str) -> Optional[Quote]:
        """取得指定標的的最新報價"""
        pass

    @abstractmethod
    def get_historical_data(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """取得歷史資料 (OHLCV)"""
        pass

    @abstractmethod
    def place_order(self, ticker: str, side: str, qty: int, price: float = 0.0) -> OrderResult:
        """
        下單
        side: 'BUY' or 'SELL'
        qty: 口數
        price: 限價，0 表示市價
        """
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """查詢目前所有持倉"""
        pass

    @abstractmethod
    def get_margin_info(self) -> MarginInfo:
        """查詢保證金帳戶資訊"""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """取消委託單"""
        pass


# ============================================================
# 回測用模擬 Broker
# ============================================================
class BacktestBroker(BrokerInterface):
    """
    回測模擬 Broker：使用歷史 CSV 資料模擬交易。
    """

    def __init__(self, price_csv: str, initial_capital: float = 1_000_000):
        self.prices = pd.read_csv(price_csv, index_col=0, parse_dates=True)
        self.prices = self.prices.ffill().bfill()
        self.capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self._current_date: Optional[pd.Timestamp] = None
        self._order_counter = 0

    def set_current_date(self, date: pd.Timestamp):
        """設定模擬當前日期"""
        self._current_date = date

    def get_quote(self, ticker: str) -> Optional[Quote]:
        if self._current_date is None or ticker not in self.prices.columns:
            return None
        if self._current_date not in self.prices.index:
            return None
        close = self.prices.loc[self._current_date, ticker]
        return Quote(
            ticker=ticker,
            timestamp=self._current_date,
            open=close, high=close, low=close, close=close
        )

    def get_historical_data(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        if ticker not in self.prices.columns:
            return pd.DataFrame()
        data = self.prices[[ticker]].loc[start:end].copy()
        data.columns = ['Close']
        return data

    def place_order(self, ticker: str, side: str, qty: int, price: float = 0.0) -> OrderResult:
        self._order_counter += 1
        quote = self.get_quote(ticker)
        if quote is None:
            return OrderResult(
                order_id=f"BT-{self._order_counter}",
                ticker=ticker, side=side, qty=qty, price=0,
                status='REJECTED', message='無法取得報價'
            )
        fill_price = quote.close if price == 0 else price
        return OrderResult(
            order_id=f"BT-{self._order_counter}",
            ticker=ticker, side=side, qty=qty, price=fill_price,
            status='FILLED', message='模擬成交'
        )

    def get_positions(self) -> List[Position]:
        return list(self.positions.values())

    def get_margin_info(self) -> MarginInfo:
        used = sum(p.margin_used for p in self.positions.values())
        return MarginInfo(
            total_equity=self.capital,
            available_margin=self.capital - used,
            used_margin=used,
            maintenance_margin=used * 0.75
        )

    def cancel_order(self, order_id: str) -> bool:
        return True  # 回測模式直接成功


# ============================================================
# 群益 API Broker（Stub，未來實作）
# ============================================================
class CapitalBroker(BrokerInterface):
    """
    群益證券 API Broker

    未來串接時需要：
    1. 安裝群益 Python API SDK
    2. 設定帳號密碼與憑證
    3. 實作 WebSocket 即時報價
    4. 實作 REST/COM 下單介面

    群益 API 文件：https://www.capital.com.tw/Service2/download/api.asp

    使用範例（未來）:
        broker = CapitalBroker(
            account='YOUR_ACCOUNT',
            password='YOUR_PASSWORD',
            cert_path='path/to/cert.pfx'
        )
        broker.connect()
        quote = broker.get_quote('2330.TW')
        result = broker.place_order('2330.TW', 'BUY', 1)
    """

    def __init__(self, account: str = '', password: str = '', cert_path: str = ''):
        self.account = account
        self.password = password
        self.cert_path = cert_path
        self._connected = False
        # TODO: 初始化群益 API SDK

    def connect(self):
        """連線到群益交易伺服器"""
        # TODO: 實作群益 API 連線
        # import comtypes  # 群益 API 使用 COM
        # self._api = ...
        raise NotImplementedError("群益 API 尚未實作，請先使用 BacktestBroker 進行回測。")

    def get_quote(self, ticker: str) -> Optional[Quote]:
        # TODO: 透過群益 API 取得即時報價
        raise NotImplementedError("群益 API 尚未實作")

    def get_historical_data(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        # TODO: 透過群益 API 或 yfinance 取得歷史資料
        raise NotImplementedError("群益 API 尚未實作")

    def place_order(self, ticker: str, side: str, qty: int, price: float = 0.0) -> OrderResult:
        # TODO: 透過群益 API 下單
        # 股票期貨下單需要：
        #   - 商品代碼（如 CDF00 = 台積電期貨近月）
        #   - 買賣方向
        #   - 口數
        #   - 價格類型（市價/限價）
        raise NotImplementedError("群益 API 尚未實作")

    def get_positions(self) -> List[Position]:
        # TODO: 查詢群益帳戶持倉
        raise NotImplementedError("群益 API 尚未實作")

    def get_margin_info(self) -> MarginInfo:
        # TODO: 查詢群益保證金帳戶
        raise NotImplementedError("群益 API 尚未實作")

    def cancel_order(self, order_id: str) -> bool:
        # TODO: 取消群益委託單
        raise NotImplementedError("群益 API 尚未實作")
