CREATE TABLE IF NOT EXISTS paper_bankroll (
    id BIGSERIAL PRIMARY KEY,
    bankroll NUMERIC(10,2) NOT NULL DEFAULT 500.00,
    total_pnl NUMERIC(10,2) NOT NULL DEFAULT 0.00,
    last_updated TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO paper_bankroll (bankroll, total_pnl)
SELECT 500.00, 0.00
WHERE NOT EXISTS (SELECT 1 FROM paper_bankroll);

CREATE TABLE IF NOT EXISTS paper_trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    kalshi_market_id TEXT NOT NULL,
    sport TEXT,
    game_date DATE,
    home_team TEXT,
    away_team TEXT,
    side TEXT NOT NULL DEFAULT 'yes',
    count INTEGER NOT NULL,
    fill_price NUMERIC(6,4) NOT NULL,
    bet_usd NUMERIC(10,2) NOT NULL,
    fee NUMERIC(10,4) NOT NULL,
    model_prob NUMERIC(6,4),
    market_prob NUMERIC(6,4),
    net_edge NUMERIC(6,4),
    status TEXT NOT NULL DEFAULT 'open',
    settlement_pnl NUMERIC(10,2),
    settled_at TIMESTAMPTZ,
    run_date DATE
);

CREATE INDEX IF NOT EXISTS idx_paper_trades_status ON paper_trades(status);
CREATE INDEX IF NOT EXISTS idx_paper_trades_run_date ON paper_trades(run_date);
