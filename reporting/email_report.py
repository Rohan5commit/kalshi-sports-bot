"""
reporting/email_report.py — Builds and sends the daily HTML email report via SMTP.
"""
import os
import smtplib
import traceback
from datetime import date, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from config import (
    REPORT_FROM_EMAIL, REPORT_TO_EMAIL, SPORTS,
    SMTP_DEFAULT_HOST, SMTP_DEFAULT_PORT,
)
from db.supabase_client import (
    get_todays_predictions, get_todays_trades, get_open_trades,
    get_todays_errors, get_todays_threshold_events, log_error,
)
from trading.kelly import compute_pnl


def _fmt_pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def _fmt_usd(val: float) -> str:
    return f"${val:.2f}"


def _get_dropped_rows_summary(run_date: date) -> dict:
    try:
        from data.data_sync_validator import get_dropped_rows_summary
        return get_dropped_rows_summary(run_date)
    except Exception:
        return {}


def _build_html(run_date: date) -> str:
    predictions = get_todays_predictions(run_date)
    trades = get_todays_trades(run_date)
    open_positions = get_open_trades()
    errors = get_todays_errors(run_date)
    threshold_events = get_todays_threshold_events(run_date)
    dropped_summary = _get_dropped_rows_summary(run_date)

    pnl = compute_pnl(trades)
    bets = [t for t in trades]
    skipped = [p for p in predictions if p["decision"].startswith("skip")]
    sports_covered = list({p["sport"] for p in predictions})
    total_dropped = sum(dropped_summary.values())
    pnl_class = "pnl-pos" if pnl >= 0 else "pnl-neg"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: Arial, sans-serif; font-size: 14px; color: #222; max-width: 820px; margin: 0 auto; padding: 20px; }}
  h1 {{ color: #1a237e; border-bottom: 2px solid #1a237e; padding-bottom: 8px; }}
  h2 {{ color: #283593; margin-top: 28px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
  th {{ background: #283593; color: white; padding: 8px 12px; text-align: left; font-size: 12px; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid #e0e0e0; font-size: 13px; }}
  tr:nth-child(even) {{ background: #f5f5f5; }}
  .stat-box {{ display: inline-block; background: #e8eaf6; border-radius: 8px; padding: 12px 20px; margin: 8px; text-align: center; }}
  .stat-val {{ font-size: 24px; font-weight: bold; color: #1a237e; }}
  .stat-label {{ font-size: 11px; color: #555; margin-top: 4px; }}
  .pnl-pos {{ color: #2e7d32; font-weight: bold; }}
  .pnl-neg {{ color: #c62828; font-weight: bold; }}
  .dq-ok {{ color: #2e7d32; }}
  .dq-warn {{ color: #f57c00; font-weight: bold; }}
  pre {{ background: #f5f5f5; padding: 10px; border-radius: 4px; font-size: 11px; overflow-x: auto; }}
</style>
</head>
<body>
<h1>Kalshi Sports Bot - Daily Report</h1>
<p><strong>Date:</strong> {run_date.strftime("%A, %B %d, %Y")} &nbsp;|&nbsp; <strong>Sports:</strong> {", ".join(sports_covered) or "None"}</p>

<div style="margin-top:16px;">
  <div class="stat-box"><div class="stat-val">{len(bets)}</div><div class="stat-label">Trades Placed</div></div>
  <div class="stat-box"><div class="stat-val">{len(skipped)}</div><div class="stat-label">Skipped</div></div>
  <div class="stat-box"><div class="stat-val">{len(open_positions)}</div><div class="stat-label">Open Positions</div></div>
  <div class="stat-box"><div class="stat-val {pnl_class}">{_fmt_usd(pnl)}</div><div class="stat-label">Demo P&amp;L</div></div>
</div>
"""

    html += "<h2>Trades Placed</h2>"
    if bets:
        html += "<table><tr><th>Sport</th><th>Matchup</th><th>Side</th><th>Stake</th><th>Model Prob</th><th>Edge</th><th>Status</th></tr>"
        for t in bets:
            html += (f"<tr><td>{t.get('sport','')}</td>"
                     f"<td>{t.get('home_team','')} vs {t.get('away_team','')}</td>"
                     f"<td>{t.get('side','').upper()}</td>"
                     f"<td>{_fmt_usd(t.get('bet_size_usd', 0))}</td>"
                     f"<td>{_fmt_pct(t.get('final_prob', 0))}</td>"
                     f"<td>{_fmt_pct(t.get('edge', 0))}</td>"
                     f"<td>{t.get('status','')}</td></tr>")
        html += "</table>"
    else:
        html += "<p><em>No trades placed today.</em></p>"

    html += "<h2>Evaluated but Skipped</h2>"
    if skipped:
        html += "<table><tr><th>Sport</th><th>Matchup</th><th>Model Prob</th><th>Kalshi Implied</th><th>Edge</th><th>Reason</th></tr>"
        for p in skipped:
            reason_map = {"skip_no_market": "No Kalshi market", "skip": "Low edge / abstention"}
            reason = reason_map.get(p.get("decision", "skip"), p.get("decision", ""))
            html += (f"<tr><td>{p.get('sport','')}</td>"
                     f"<td>{p.get('home_team','')} vs {p.get('away_team','')}</td>"
                     f"<td>{_fmt_pct(p.get('final_prob', 0))}</td>"
                     f"<td>{_fmt_pct(p.get('kalshi_implied', 0))}</td>"
                     f"<td>{_fmt_pct(p.get('edge', 0))}</td>"
                     f"<td>{reason}</td></tr>")
        html += "</table>"
    else:
        html += "<p><em>No skipped markets today.</em></p>"

    html += "<h2>Open Positions</h2>"
    if open_positions:
        html += "<table><tr><th>Sport</th><th>Matchup</th><th>Market ID</th><th>Side</th><th>Stake</th></tr>"
        for pos in open_positions:
            html += (f"<tr><td>{pos.get('sport','')}</td>"
                     f"<td>{pos.get('home_team','')} vs {pos.get('away_team','')}</td>"
                     f"<td>{pos.get('kalshi_market_id','')}</td>"
                     f"<td>{pos.get('side','')}</td>"
                     f"<td>{_fmt_usd(pos.get('bet_size_usd', 0))}</td></tr>")
        html += "</table>"
    else:
        html += "<p><em>No open positions.</em></p>"

    pnl_class2 = "pnl-pos" if pnl >= 0 else "pnl-neg"
    html += f"<h2>Running Demo P&amp;L</h2><p class='{pnl_class2}' style='font-size:20px;'>{_fmt_usd(pnl)}</p>"

    # ── Data Quality section ───────────────────────────────────────────────────
    html += "<h2>Data Quality</h2>"
    dq_class = "dq-warn" if total_dropped > 0 else "dq-ok"
    html += (f"<p class='{dq_class}'><strong>Dropped rows today: {total_dropped}</strong></p>")
    if dropped_summary:
        html += "<table><tr><th>Failed Check</th><th>Rows Dropped</th></tr>"
        for check, count in sorted(dropped_summary.items(), key=lambda x: -x[1]):
            html += f"<tr><td>{check}</td><td>{count}</td></tr>"
        html += "</table>"
        html += ("<p style='font-size:12px;color:#777;'>Rows are dropped when temporal "
                 "sync rules are violated (lookahead bias prevention). "
                 "Full audit in <code>dropped_rows</code> Supabase table.</p>")
    else:
        html += "<p class='dq-ok'>No rows dropped &mdash; all sync checks passed.</p>"

    if threshold_events:
        html += "<h2>Threshold Adjustments</h2><ul>"
        for ev in threshold_events:
            html += (f"<li><strong>{ev.get('event_type')}</strong>: "
                     f"{_fmt_pct(ev.get('old_value',0))} → {_fmt_pct(ev.get('new_value',0))} "
                     f"— {ev.get('reason','')}</li>")
        html += "</ul>"

    html += "<h2>Errors</h2>"
    if errors:
        for err in errors:
            html += (f"<div style='border-left:4px solid #c62828;padding:8px 16px;margin:8px 0;background:#fff8f8;'>"
                     f"<p><strong>Context:</strong> {err.get('context','')}</p>"
                     f"<p><strong>Error:</strong> {err.get('error_msg','')}</p>"
                     f"<pre>{err.get('traceback','')}</pre></div>")
    else:
        html += "<p style='color:#2e7d32;'>No errors today.</p>"

    html += "</body></html>"
    return html


def send_daily_report(run_date: Optional[date] = None):
    if run_date is None:
        run_date = datetime.utcnow().date()

    try:
        html_body = _build_html(run_date)
    except Exception as exc:
        log_error(context="email_report.build_html", error_msg=str(exc),
                  tb=traceback.format_exc(), run_date=run_date)
        html_body = f"<p>Error building report: {exc}</p>"

    smtp_host = os.environ.get("SMTP_HOST", SMTP_DEFAULT_HOST)
    smtp_port = int(os.environ.get("SMTP_PORT", SMTP_DEFAULT_PORT))
    smtp_user = os.environ["SMTP_USER"]
    smtp_password = os.environ["SMTP_PASSWORD"]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Kalshi Sports Bot Report — {run_date}"
    msg["From"] = smtp_user
    msg["To"] = REPORT_TO_EMAIL
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, [REPORT_TO_EMAIL], msg.as_string())
        print(f"Email sent via {smtp_host}:{smtp_port} to {REPORT_TO_EMAIL}")
    except Exception as exc:
        log_error(context="email_report.send", error_msg=str(exc),
                  tb=traceback.format_exc(), run_date=run_date)
        raise
