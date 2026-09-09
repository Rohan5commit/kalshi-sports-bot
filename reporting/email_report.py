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
    PAPER_BANKROLL_START,
)
from db.supabase_client import (
    get_todays_predictions, get_todays_errors, get_todays_threshold_events, log_error,
)


def _fmt_pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def _fmt_usd(val: float) -> str:
    return f"${val:.2f}"


def _build_html(run_date: date) -> str:
    reconcile_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    try:
        from trading.reconcile import reconcile_open_trades
        reconcile_open_trades()
        reconcile_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pass

    predictions = get_todays_predictions(run_date)
    errors = get_todays_errors(run_date)
    threshold_events = get_todays_threshold_events(run_date)
    sports_covered = list({p["sport"] for p in predictions})

    # Paper trading data
    try:
        from trading.paper_ledger import get_bankroll
        from db.supabase_client import _get_client, _retry
        bankroll = get_bankroll()

        def _fetch_paper():
            sb = _get_client()
            return sb.table("paper_trades").select("*").order("created_at", desc=True).execute().data or []

        def _fetch_today_paper():
            sb = _get_client()
            return sb.table("paper_trades").select("*").eq("run_date", str(run_date)).execute().data or []

        all_paper_trades = _retry(_fetch_paper)
        todays_paper_trades = _retry(_fetch_today_paper)

        settled = [t for t in all_paper_trades if t.get("status") in ("won", "lost")]
        total_pnl = sum(float(t.get("settlement_pnl") or 0) for t in settled)
        wins = sum(1 for t in settled if t.get("status") == "won")
        losses = sum(1 for t in settled if t.get("status") == "lost")
        win_rate = wins / len(settled) if settled else 0.0
        pnl_pct = (total_pnl / PAPER_BANKROLL_START) * 100
    except Exception as exc:
        bankroll = PAPER_BANKROLL_START
        todays_paper_trades = []
        all_paper_trades = []
        settled = []
        total_pnl = 0.0
        wins = losses = 0
        win_rate = 0.0
        pnl_pct = 0.0

    pnl_class = "pnl-pos" if total_pnl >= 0 else "pnl-neg"

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
  .stat-box {{ display: inline-block; background: #e8eaf6; border-radius: 8px; padding: 12px 20px; margin: 8px; text-align: center; min-width: 100px; }}
  .stat-val {{ font-size: 24px; font-weight: bold; color: #1a237e; }}
  .stat-label {{ font-size: 11px; color: #555; margin-top: 4px; }}
  .pnl-pos {{ color: #2e7d32; font-weight: bold; }}
  .pnl-neg {{ color: #c62828; font-weight: bold; }}
  pre {{ background: #f5f5f5; padding: 10px; border-radius: 4px; font-size: 11px; overflow-x: auto; }}
  .paper-badge {{ background: #e3f2fd; color: #1565c0; font-size: 10px; padding: 2px 6px; border-radius: 4px; font-weight: bold; }}
</style>
</head>
<body>
<h1>Kalshi Sports Bot — Daily Report <span class="paper-badge">PAPER TRADING</span></h1>
<p><strong>Date:</strong> {run_date.strftime("%A, %B %d, %Y")} &nbsp;|&nbsp;
   <strong>Sports:</strong> {", ".join(sports_covered) or "None"} &nbsp;|&nbsp;
   <strong>Bankroll:</strong> {_fmt_usd(bankroll)}</p>

<div style="margin-top:16px;">
  <div class="stat-box"><div class="stat-val">{len(todays_paper_trades)}</div><div class="stat-label">Fills Today</div></div>
  <div class="stat-box"><div class="stat-val">{len(settled)}</div><div class="stat-label">Settled All-Time</div></div>
  <div class="stat-box"><div class="stat-val">{wins}W / {losses}L</div><div class="stat-label">Win / Loss</div></div>
  <div class="stat-box"><div class="stat-val">{_fmt_pct(win_rate)}</div><div class="stat-label">Win Rate</div></div>
  <div class="stat-box"><div class="stat-val class='{pnl_class}'">{_fmt_usd(total_pnl)}</div><div class="stat-label">Total P&amp;L</div></div>
  <div class="stat-box"><div class="stat-val {pnl_class}">{pnl_pct:+.2f}%</div><div class="stat-label">Return (vs {_fmt_usd(PAPER_BANKROLL_START)})</div></div>
</div>
"""

    html += "<h2>Today's Paper Fills</h2>"
    if todays_paper_trades:
        html += ("<table><tr><th>Sport</th><th>Matchup</th><th>Side</th>"
                 "<th>Contracts</th><th>Fill Price</th><th>Stake</th>"
                 "<th>Edge</th><th>Status</th></tr>")
        for t in todays_paper_trades:
            away = t.get("away_team", "")
            home = t.get("home_team", "")
            matchup = f"{away} @ {home}" if away and home else t.get("kalshi_market_id", "")
            status = t.get("status", "open")
            pnl_str = ""
            if status in ("won", "lost"):
                pnl_val = float(t.get("settlement_pnl") or 0)
                pnl_str = f" ({'+' if pnl_val >= 0 else ''}{pnl_val:.2f})"
            html += (f"<tr><td>{t.get('sport','')}</td>"
                     f"<td>{matchup}</td>"
                     f"<td>{t.get('side','').upper()}</td>"
                     f"<td>{t.get('count','')}</td>"
                     f"<td>{float(t.get('fill_price',0)):.4f}</td>"
                     f"<td>{_fmt_usd(float(t.get('bet_usd',0)))}</td>"
                     f"<td>{_fmt_pct(float(t.get('net_edge',0)))}</td>"
                     f"<td>{status.upper()}{pnl_str}</td></tr>")
        html += "</table>"
    else:
        html += "<p><em>No paper fills today.</em></p>"

    html += "<h2>All Settled Positions</h2>"
    if settled:
        html += ("<table><tr><th>Date</th><th>Sport</th><th>Matchup</th><th>Side</th>"
                 "<th>Fill Price</th><th>Stake</th><th>Result</th><th>P&amp;L</th></tr>")
        for t in settled[-20:]:  # show last 20
            away = t.get("away_team", "")
            home = t.get("home_team", "")
            matchup = f"{away} @ {home}" if away and home else t.get("kalshi_market_id", "")
            pnl_val = float(t.get("settlement_pnl") or 0)
            pnl_col = "pnl-pos" if pnl_val >= 0 else "pnl-neg"
            html += (f"<tr><td>{str(t.get('run_date',''))[:10]}</td>"
                     f"<td>{t.get('sport','')}</td>"
                     f"<td>{matchup}</td>"
                     f"<td>{t.get('side','').upper()}</td>"
                     f"<td>{float(t.get('fill_price',0)):.4f}</td>"
                     f"<td>{_fmt_usd(float(t.get('bet_usd',0)))}</td>"
                     f"<td>{t.get('status','').upper()}</td>"
                     f"<td class='{pnl_col}'>{'+' if pnl_val >= 0 else ''}{_fmt_usd(pnl_val)}</td></tr>")
        html += "</table>"
    else:
        html += "<p><em>No settled positions yet.</em></p>"

    html += "<h2>Today's Predictions</h2>"
    if predictions:
        html += "<table><tr><th>Sport</th><th>Matchup</th><th>Vegas Prob</th><th>Kalshi</th><th>Edge</th><th>Decision</th></tr>"
        for p in predictions:
            away = p.get("away_team", "")
            home = p.get("home_team", "")
            matchup = f"{away} @ {home}"
            edge_val = float(p.get("edge") or 0)
            edge_col = "pnl-pos" if edge_val > 0 else ""
            html += (f"<tr><td>{p.get('sport','')}</td>"
                     f"<td>{matchup}</td>"
                     f"<td>{_fmt_pct(float(p.get('final_prob',0)))}</td>"
                     f"<td>{_fmt_pct(float(p.get('kalshi_implied',0)))}</td>"
                     f"<td class='{edge_col}'>{edge_val:+.3f}</td>"
                     f"<td>{p.get('decision','')}</td></tr>")
        html += "</table>"
    else:
        html += "<p><em>No predictions logged today.</em></p>"

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

    html += f"<p style='color:#888;font-size:11px;margin-top:32px;border-top:1px solid #eee;padding-top:8px;'>Reconciled at {reconcile_ts}</p>"
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
    msg["Subject"] = f"Kalshi Sports Bot — {run_date} [Paper]"
    msg["To"] = REPORT_TO_EMAIL
    msg["From"] = smtp_user
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
