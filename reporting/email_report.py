"""
reporting/email_report.py — Builds and sends the daily HTML email report via SendGrid.
"""
import os
import traceback
from datetime import date, datetime
from typing import Optional

import sendgrid
from sendgrid.helpers.mail import Mail, Content

from config import REPORT_FROM_EMAIL, REPORT_TO_EMAIL, SPORTS
from db.supabase_client import (
    get_todays_predictions, get_todays_trades, get_open_trades,
    get_todays_errors, get_todays_threshold_events, log_error,
)
from trading.kalshi_client import get_implied_probability, get_market
from trading.kelly import compute_pnl


def _fmt_pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def _fmt_usd(val: float) -> str:
    return f"${val:.2f}"


def _build_html(run_date: date) -> str:
    predictions = get_todays_predictions(run_date)
    trades = get_todays_trades(run_date)
    open_positions = get_open_trades()
    errors = get_todays_errors(run_date)
    threshold_events = get_todays_threshold_events(run_date)

    pnl = compute_pnl(trades)
    bets = [t for t in trades]
    skipped = [p for p in predictions if p["decision"].startswith("skip")]

    # Group predictions by sport
    sports_covered = list({p["sport"] for p in predictions})

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: Arial, sans-serif; font-size: 14px; color: #222; max-width: 800px; margin: 0 auto; padding: 20px; }}
  h1 {{ color: #1a237e; border-bottom: 2px solid #1a237e; padding-bottom: 8px; }}
  h2 {{ color: #283593; margin-top: 28px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
  th {{ background: #283593; color: white; padding: 8px 12px; text-align: left; font-size: 12px; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid #e0e0e0; font-size: 13px; }}
  tr:nth-child(even) {{ background: #f5f5f5; }}
  .badge-bet {{ background: #2e7d32; color: white; border-radius: 4px; padding: 2px 8px; }}
  .badge-skip {{ background: #9e9e9e; color: white; border-radius: 4px; padding: 2px 8px; }}
  .badge-error {{ background: #c62828; color: white; border-radius: 4px; padding: 2px 8px; }}
  .stat-box {{ display: inline-block; background: #e8eaf6; border-radius: 8px; padding: 12px 20px; margin: 8px; text-align: center; }}
  .stat-val {{ font-size: 24px; font-weight: bold; color: #1a237e; }}
  .stat-label {{ font-size: 11px; color: #555; margin-top: 4px; }}
  .pnl-pos {{ color: #2e7d32; font-weight: bold; }}
  .pnl-neg {{ color: #c62828; font-weight: bold; }}
  pre {{ background: #f5f5f5; padding: 10px; border-radius: 4px; font-size: 11px; overflow-x: auto; }}
  .section {{ margin-top: 32px; }}
</style>
</head>
<body>
<h1>🤖 Kalshi Sports Bot — Daily Report</h1>
<p><strong>Date:</strong> {run_date.strftime("%A, %B %d, %Y")} &nbsp;|&nbsp; <strong>Sports:</strong> {", ".join(sports_covered) or "None"}</p>

<div class="section">
  <div class="stat-box"><div class="stat-val">{len(bets)}</div><div class="stat-label">Trades Placed</div></div>
  <div class="stat-box"><div class="stat-val">{len(skipped)}</div><div class="stat-label">Markets Skipped</div></div>
  <div class="stat-box"><div class="stat-val">{len(open_positions)}</div><div class="stat-label">Open Positions</div></div>
  <div class="stat-box"><div class="stat-val class="{"pnl-pos" if pnl >= 0 else "pnl-neg"}">{_fmt_usd(pnl)}</div><div class="stat-label">Demo P&amp;L</div></div>
</div>
"""

    # Trades placed
    html += '<h2>Trades Placed</h2>'
    if bets:
        html += """<table>
<tr><th>Sport</th><th>Matchup</th><th>Side</th><th>Stake</th><th>Model Prob</th><th>Edge</th><th>Status</th></tr>"""
        for t in bets:
            html += f"""<tr>
<td>{t.get("sport","")}</td>
<td>{t.get("home_team","")} vs {t.get("away_team","")}</td>
<td>{t.get("side","").upper()}</td>
<td>{_fmt_usd(t.get("bet_size_usd", 0))}</td>
<td>{_fmt_pct(t.get("final_prob", 0))}</td>
<td>{_fmt_pct(t.get("edge", 0))}</td>
<td>{t.get("status","")}</td>
</tr>"""
        html += "</table>"
    else:
        html += "<p><em>No trades placed today.</em></p>"

    # Skipped markets
    html += '<h2>Evaluated but Skipped Markets</h2>'
    if skipped:
        html += """<table>
<tr><th>Sport</th><th>Matchup</th><th>Model Prob</th><th>Kalshi Implied</th><th>Edge</th><th>Reason</th></tr>"""
        for p in skipped:
            reason_map = {
                "skip_no_market": "No Kalshi market found",
                "skip": "Insufficient edge / abstention band",
            }
            reason = reason_map.get(p.get("decision", "skip"), p.get("decision", ""))
            html += f"""<tr>
<td>{p.get("sport","")}</td>
<td>{p.get("home_team","")} vs {p.get("away_team","")}</td>
<td>{_fmt_pct(p.get("final_prob", 0))}</td>
<td>{_fmt_pct(p.get("kalshi_implied", 0))}</td>
<td>{_fmt_pct(p.get("edge", 0))}</td>
<td>{reason}</td>
</tr>"""
        html += "</table>"
    else:
        html += "<p><em>No skipped markets today.</em></p>"

    # Open positions
    html += '<h2>Open Positions</h2>'
    if open_positions:
        html += """<table>
<tr><th>Sport</th><th>Matchup</th><th>Market ID</th><th>Side</th><th>Stake</th></tr>"""
        for pos in open_positions:
            html += f"""<tr>
<td>{pos.get("sport","")}</td>
<td>{pos.get("home_team","")} vs {pos.get("away_team","")}</td>
<td>{pos.get("kalshi_market_id","")}</td>
<td>{pos.get("side","")}</td>
<td>{_fmt_usd(pos.get("bet_size_usd", 0))}</td>
</tr>"""
        html += "</table>"
    else:
        html += "<p><em>No open positions.</em></p>"

    # P&L summary
    html += f'<h2>Running Demo P&amp;L</h2>'
    pnl_class = "pnl-pos" if pnl >= 0 else "pnl-neg"
    html += f'<p class="{pnl_class}" style="font-size:20px;">{_fmt_usd(pnl)}</p>'

    # Threshold events
    if threshold_events:
        html += '<h2>Threshold Adjustment Events</h2><ul>'
        for ev in threshold_events:
            html += f'<li><strong>{ev.get("event_type")}</strong>: {_fmt_pct(ev.get("old_value",0))} → {_fmt_pct(ev.get("new_value",0))} — {ev.get("reason","")}</li>'
        html += '</ul>'

    # Errors
    html += '<h2>Errors</h2>'
    if errors:
        for err in errors:
            html += f"""
<div style="border-left: 4px solid #c62828; padding: 8px 16px; margin: 8px 0; background:#fff8f8;">
  <p><strong>Context:</strong> {err.get("context","")}</p>
  <p><strong>Error:</strong> {err.get("error_msg","")}</p>
  <pre>{err.get("traceback","")}</pre>
</div>"""
    else:
        html += "<p style='color:#2e7d32;'>✓ No errors today.</p>"

    html += "</body></html>"
    return html


def send_daily_report(run_date: Optional[date] = None):
    """Build and send the daily HTML email via SendGrid."""
    if run_date is None:
        run_date = datetime.utcnow().date()

    try:
        html_body = _build_html(run_date)
    except Exception as exc:
        log_error(
            context="email_report.build_html",
            error_msg=str(exc),
            tb=traceback.format_exc(),
            run_date=run_date,
        )
        html_body = f"<p>Error building report: {exc}</p>"

    try:
        sg = sendgrid.SendGridAPIClient(api_key=os.environ["SENDGRID_API_KEY"])
        mail = Mail(
            from_email=REPORT_FROM_EMAIL,
            to_emails=REPORT_TO_EMAIL,
            subject=f"Kalshi Sports Bot Report — {run_date}",
            html_content=html_body,
        )
        response = sg.send(mail)
        print(f"Email sent: status {response.status_code}")
    except Exception as exc:
        log_error(
            context="email_report.send",
            error_msg=str(exc),
            tb=traceback.format_exc(),
            run_date=run_date,
        )
        raise
