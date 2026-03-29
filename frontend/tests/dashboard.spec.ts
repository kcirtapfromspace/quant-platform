import { expect, test, type Page } from '@playwright/test';

type MockFrame = {
  type: string;
  data: unknown;
  delayMs?: number;
};

const DEFAULT_QUOTES = [
  {
    symbol: 'AAPL',
    price: 189.42,
    change: 2.15,
    changePercent: 1.15,
    high: 190.12,
    low: 186.8,
    open: 187.01,
    previousClose: 187.27,
    volume: 12_400_000,
    timestamp: 1_711_000_000,
  },
  {
    symbol: 'NVDA',
    price: 921.13,
    change: 14.8,
    changePercent: 1.63,
    high: 925.4,
    low: 903.5,
    open: 906.3,
    previousClose: 906.33,
    volume: 18_200_000,
    timestamp: 1_711_000_000,
  },
];

const PORTFOLIO = {
  cash: 245_000,
  equity: 760_452.18,
  totalValue: 1_005_452.18,
  dailyPnl: 12_481.24,
  dailyPnlPercent: 1.26,
  positions: [
    {
      symbol: 'AAPL',
      quantity: 250,
      avgCost: 181.25,
      currentPrice: 189.42,
      marketValue: 47_355,
      unrealizedPnl: 2_042.5,
      unrealizedPnlPercent: 4.51,
    },
    {
      symbol: 'NVDA',
      quantity: 80,
      avgCost: 884.5,
      currentPrice: 921.13,
      marketValue: 73_690.4,
      unrealizedPnl: 2_930.4,
      unrealizedPnlPercent: 4.14,
    },
  ],
};

const ORDERS = [
  {
    id: 'ord-1',
    symbol: 'AAPL',
    side: 'buy',
    type: 'limit',
    quantity: 100,
    limitPrice: 188.5,
    fillPrice: 188.42,
    status: 'filled',
    createdAt: 1_711_000_100,
    filledAt: 1_711_000_160,
  },
  {
    id: 'ord-2',
    symbol: 'NVDA',
    side: 'sell',
    type: 'market',
    quantity: 20,
    status: 'pending',
    createdAt: 1_711_000_220,
  },
];

const STRATEGIES = [
  {
    strategy_key: 'mom_aapl',
    name: 'Alpha Momentum',
    status: 'active',
    regime: 'bull',
    signal_confidence: 0.82,
    daily_pnl: 4200,
    positions: 3,
    category: 'Time-series',
  },
  {
    strategy_key: 'value_nvda',
    name: 'Beta Value',
    status: 'paper',
    regime: 'sideways',
    signal_confidence: 0.61,
    daily_pnl: -850,
    positions: 2,
    category: 'Factor',
  },
  {
    strategy_key: 'stat_arb',
    name: 'Gamma Spread',
    status: 'backtesting',
    regime: 'bear',
    signal_confidence: 0.48,
    daily_pnl: 325,
    positions: 1,
    category: 'Cross-sectional',
  },
];

const RISK_SNAPSHOT = {
  var95: 0.018,
  var99: 0.029,
  drawdown: 0.041,
  maxDrawdown: 0.127,
  circuitBreakerArmed: false,
  positionLimitUtilization: [
    { symbol: 'AAPL', utilization: 0.52 },
    { symbol: 'NVDA', utilization: 0.74 },
  ],
};

const HISTORY = [
  { time: 1_710_000_000, open: 180.1, high: 181.4, low: 179.8, close: 181.1, volume: 1_100_000 },
  { time: 1_710_086_400, open: 181.1, high: 183.2, low: 180.5, close: 182.7, volume: 1_280_000 },
  { time: 1_710_172_800, open: 182.7, high: 184.5, low: 181.9, close: 183.9, volume: 1_340_000 },
  { time: 1_710_259_200, open: 183.9, high: 186.2, low: 183.1, close: 185.8, volume: 1_410_000 },
  { time: 1_710_345_600, open: 185.8, high: 189.9, low: 185.2, close: 189.4, volume: 1_580_000 },
];

const NVDA_HISTORY = [
  { time: 1_710_000_000, open: 890, high: 901, low: 886, close: 899, volume: 2_100_000 },
  { time: 1_710_086_400, open: 899, high: 910, low: 894, close: 906, volume: 2_340_000 },
  { time: 1_710_172_800, open: 906, high: 922, low: 904, close: 918, volume: 2_510_000 },
  { time: 1_710_259_200, open: 918, high: 926, low: 912, close: 923, volume: 2_760_000 },
  { time: 1_710_345_600, open: 923, high: 930, low: 919, close: 921, volume: 2_880_000 },
];

const PORTFOLIO_HISTORY = [
  { time: 1_710_000_000, value: 950_000 },
  { time: 1_710_086_400, value: 963_500 },
  { time: 1_710_172_800, value: 978_200 },
  { time: 1_710_259_200, value: 991_750 },
  { time: 1_710_345_600, value: 1_005_452.18 },
];

const DEFAULT_WS_FRAMES: MockFrame[] = [
  { type: 'portfolio', data: PORTFOLIO, delayMs: 5 },
  { type: 'quote', data: DEFAULT_QUOTES[0], delayMs: 10 },
  { type: 'quote', data: DEFAULT_QUOTES[1], delayMs: 12 },
  {
    type: 'orderbook',
    data: {
      symbol: 'AAPL',
      bids: [
        { price: 189.4, size: 1200 },
        { price: 189.35, size: 950 },
      ],
      asks: [
        { price: 189.45, size: 1100 },
        { price: 189.5, size: 1500 },
      ],
    },
    delayMs: 15,
  },
  {
    type: 'orderbook',
    data: {
      symbol: 'NVDA',
      bids: [
        { price: 921.1, size: 600 },
        { price: 920.8, size: 540 },
      ],
      asks: [
        { price: 921.3, size: 580 },
        { price: 921.6, size: 710 },
      ],
    },
    delayMs: 18,
  },
  {
    type: 'ohlcv',
    data: {
      time: 1_710_432_000,
      open: 188.9,
      high: 189.8,
      low: 188.1,
      close: 189.4,
      volume: 1_660_000,
    },
    delayMs: 20,
  },
  {
    type: 'fill',
    data: {
      id: 'ord-3',
      symbol: 'MSFT',
      side: 'buy',
      type: 'limit',
      quantity: 50,
      limitPrice: 411.5,
      fillPrice: 411.2,
      status: 'filled',
      createdAt: 1_711_000_300,
      filledAt: 1_711_000_360,
    },
    delayMs: 22,
  },
  { type: 'strategy_state', data: STRATEGIES[0], delayMs: 25 },
  { type: 'strategy_state', data: STRATEGIES[1], delayMs: 30 },
  { type: 'strategy_state', data: STRATEGIES[2], delayMs: 35 },
];

async function installMockWebSocket(page: Page, frames: MockFrame[] = DEFAULT_WS_FRAMES) {
  await page.addInitScript((initialFrames: MockFrame[]) => {
    type ListenerMap = Record<string, Set<(event: Event | MessageEvent) => void>>;

    class MockWebSocket {
      static instances: MockWebSocket[] = [];
      static CONNECTING = 0;
      static OPEN = 1;
      static CLOSING = 2;
      static CLOSED = 3;

      url: string;
      readyState = MockWebSocket.CONNECTING;
      onopen: ((event: Event) => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      onclose: ((event: Event) => void) | null = null;
      onerror: ((event: Event) => void) | null = null;
      private listeners: ListenerMap = {
        open: new Set(),
        message: new Set(),
        close: new Set(),
        error: new Set(),
      };

      constructor(url: string) {
        this.url = url;
        MockWebSocket.instances.push(this);
        window.setTimeout(() => {
          this.readyState = MockWebSocket.OPEN;
          this.dispatch('open', new Event('open'));

          initialFrames.forEach((frame, index) => {
            window.setTimeout(() => {
              if (this.readyState !== MockWebSocket.OPEN) return;
              this.dispatch(
                'message',
                new MessageEvent('message', { data: JSON.stringify({ type: frame.type, data: frame.data }) }),
              );
            }, frame.delayMs ?? index * 10);
          });
        }, 0);
      }

      addEventListener(type: string, handler: (event: Event | MessageEvent) => void) {
        this.listeners[type]?.add(handler);
      }

      removeEventListener(type: string, handler: (event: Event | MessageEvent) => void) {
        this.listeners[type]?.delete(handler);
      }

      send() {}

      close() {
        this.readyState = MockWebSocket.CLOSED;
        this.dispatch('close', new Event('close'));
      }

      dispatch(type: string, event: Event | MessageEvent) {
        if (type === 'open') this.onopen?.(event as Event);
        if (type === 'message') this.onmessage?.(event as MessageEvent);
        if (type === 'close') this.onclose?.(event as Event);
        if (type === 'error') this.onerror?.(event as Event);
        this.listeners[type]?.forEach((listener) => listener(event));
      }
    }

    Object.assign(window, {
      WebSocket: MockWebSocket,
      __pushMockWsFrame(frame: MockFrame) {
        const message = new MessageEvent('message', {
          data: JSON.stringify({ type: frame.type, data: frame.data }),
        });
        MockWebSocket.instances.forEach((instance) => {
          if (instance.readyState === MockWebSocket.OPEN) {
            instance.dispatch('message', message);
          }
        });
      },
    });
  }, frames);
}

async function mockApi(page: Page) {
  await page.route('**/api/v1/**', async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname.replace('/api/v1', '');

    const jsonResponse = (payload: unknown) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(payload),
      });

    if (path.startsWith('/history/')) {
      const symbol = path.split('/').pop();
      return jsonResponse(symbol === 'NVDA' ? NVDA_HISTORY : HISTORY);
    }

    switch (path) {
      case '/watchlist':
        return jsonResponse(['AAPL', 'NVDA', 'MSFT']);
      case '/orders':
        if (route.request().method() === 'DELETE') {
          return route.fulfill({ status: 204, body: '' });
        }
        return jsonResponse(ORDERS);
      case '/strategies':
        return jsonResponse(STRATEGIES);
      case '/portfolio/history':
        return jsonResponse(PORTFOLIO_HISTORY);
      case '/risk/snapshot':
        return jsonResponse(RISK_SNAPSHOT);
      case '/market/ohlcv':
        return jsonResponse(url.searchParams.get('symbol') === 'NVDA' ? NVDA_HISTORY : HISTORY);
      default:
        return route.fulfill({
          status: 404,
          contentType: 'application/json',
          body: JSON.stringify({ error: `Unhandled mock route for ${path}` }),
        });
    }
  });
}

async function gotoDashboard(page: Page) {
  await installMockWebSocket(page);
  await mockApi(page);
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'QUA Trading' })).toBeVisible();
  await expect(page.getByText('Live', { exact: true })).toBeVisible();
}

async function openSidebarView(page: Page, label: string) {
  await page.getByRole('button', { name: label }).click();
}

test('live trading updates quotes over websocket without a refresh', async ({ page }) => {
  await gotoDashboard(page);
  const aaplWatchlistRow = page.getByRole('button', { name: /AAPL Vol/ });

  await expect(page.getByRole('heading', { name: 'Watchlist' })).toBeVisible();
  await expect(aaplWatchlistRow).toBeVisible();
  await expect(aaplWatchlistRow).toContainText('$189.42');

  await page.evaluate(() => {
    (window as Window & { __pushMockWsFrame: (frame: MockFrame) => void }).__pushMockWsFrame({
      type: 'quote',
      data: {
        symbol: 'AAPL',
        price: 190.1,
        change: 2.83,
        changePercent: 1.51,
        high: 190.4,
        low: 186.8,
        open: 187.01,
        previousClose: 187.27,
        volume: 12_700_000,
        timestamp: 1_711_000_500,
      },
    });
  });

  await expect(aaplWatchlistRow).toContainText('$190.10');
});

test('portfolio dashboard renders equity and positions data', async ({ page }) => {
  await gotoDashboard(page);
  await openSidebarView(page, 'Portfolio');
  const positionsTable = page
    .getByRole('table')
    .filter({ has: page.getByRole('columnheader', { name: 'Symbol' }) });

  await expect(page.getByText('Equity Curve (Daily NAV)')).toBeVisible();
  await expect(page.getByText('Exposure by Symbol')).toBeVisible();
  await expect(positionsTable).toContainText('AAPL');
  await expect(positionsTable).toContainText('NVDA');
});

test('market data page loads charts and the symbol selector switches context', async ({ page }) => {
  await gotoDashboard(page);
  await openSidebarView(page, 'Market Data');

  await expect(page.getByText('AAPL · 5m')).toBeVisible();
  await expect(page.getByText('Order Book')).toBeVisible();

  const symbolInput = page.getByPlaceholder('Symbol…');
  await symbolInput.fill('NV');
  await page.getByRole('button', { name: 'NVDA' }).click();

  await expect(page.getByText('NVDA · 5m')).toBeVisible();
  await expect(page.getByText('Order Book')).toBeVisible();
  await expect(page.getByText('NVDA').first()).toBeVisible();
});

test('strategy monitor shows live strategy signals', async ({ page }) => {
  await gotoDashboard(page);
  await openSidebarView(page, 'Strategy Monitor');

  await expect(page.getByText('Active Strategies')).toBeVisible();
  await expect(page.getByText('Alpha Momentum')).toBeVisible();
  await expect(page.getByText('Beta Value')).toBeVisible();
  await expect(page.getByText('Signal Confidence').first()).toBeVisible();
});

test('risk dashboard renders the risk snapshot cards and tables', async ({ page }) => {
  await gotoDashboard(page);
  await openSidebarView(page, 'Risk');

  await expect(page.getByText('VaR 95%')).toBeVisible();
  await expect(page.getByText('VaR 99%')).toBeVisible();
  await expect(page.getByText('Portfolio Drawdown')).toBeVisible();
  await expect(page.getByText('Position Limit Utilization')).toBeVisible();
  await expect(page.getByText('AAPL').first()).toBeVisible();
});

test('trade blotter loads order history and slippage metrics', async ({ page }) => {
  await gotoDashboard(page);
  await openSidebarView(page, 'Trade Blotter');

  await expect(page.getByRole('heading', { name: 'Trade Blotter' })).toBeVisible();
  await expect(page.getByText('Avg Slippage')).toBeVisible();
  await expect(page.locator('table')).toContainText('AAPL');
  await expect(page.locator('table')).toContainText('filled');
  await expect(page.locator('table')).toContainText('pending');
});
