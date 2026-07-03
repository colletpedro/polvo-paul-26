"""Gate da Camada 3 do Modo 2 — validação cruzada StatsBomb × SofaScore.

SPEC_MARKETS.md §2.3 e §10 passo 1.

Papel (pós-passo 0, arquitetura já é SofaScore-only): StatsBomb é o **ground truth**
de evento; este script confirma que as contagens do SofaScore (fonte operacional) são
fidedignas ANTES de confiar no histórico dela para treino. Cumpre duas funções do §2.3:

  1. GATE DE PARIDADE DE DEFINIÇÃO — se, no mesmo jogo, StatsBomb deriva 5 escanteios e
     SofaScore reporta 6, existe divergência de definição a tratar. (SPEC.md §2.4 aplicado
     ENTRE fontes.)
  2. PISO DE ERRO DE MEDIÇÃO — a discrepância média entre as fontes nos mesmos jogos é o
     piso de erro do Modo 2: nenhum modelo prevê contagem com precisão maior que as fontes.

NÃO treina modelo. NÃO coleta dataset. Só casa os jogos do overlap (Euro 2024 e Copa
América 2024, presentes nas DUAS fontes), compara as contagens por time e mede a discrepância.

Derivação StatsBomb (ground truth), por time:
  escanteio = evento Pass com pass.type.name == "Corner"  (team = quem cobra)
  ⚠ CARTÕES vêm de DOIS tipos de evento (descoberto POR este gate — a suposição inicial
    "cartão = só Bad Behaviour" subcontava ~70% dos amarelos):
      · Bad Behaviour  → bad_behaviour.card.name   (má conduta SEM falta: dissídio etc.)
      · Foul Committed → foul_committed.card.name   (cartão associado a uma falta — maioria)
    Paridade com a definição do SofaScore (validada empiricamente neste script):
      amarelo  = Yellow Card + Second Yellow   (todo 2º amarelo é um amarelo MOSTRADO)
      vermelho = Red Card + Second Yellow       (todo 2º amarelo/vermelho direto = expulsão)

Extração SofaScore (operacional), de /event/{id}/statistics (período ALL):
  cornerKicks / yellowCards / redCards, home & away.
  ⚠ yellowCards/redCards são ausentes quando o valor é 0 — tratados como 0.

Acesso SofaScore: curl_cffi (bypass de TLS fingerprint / Varnish), padrão validado em
check_sofascore_coverage.py e check_sofascore_depth.py. Rate limit: sleep 3 s (§2.2).

Uso:
  .venv/bin/python src/evaluate/cross_validation.py
  .venv/bin/python src/evaluate/cross_validation.py --sleep 3 --impersonate chrome124

Flags:
  --sleep S        Segundos entre requisições SofaScore (default: 3.0 — rate limit §2.2).
  --impersonate I  Perfil curl_cffi (default: chrome124).
  --date-tol D     Tolerância em dias ao casar por data (default: 1).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.collect.statsbomb import derive_counts, fetch_events, fetch_matches
# Reuso READ-ONLY do coletor (mesma chamada de event list; extração canônica de meta).
from src.collect import sofascore as _sofa_mod
from src.collect.sofascore import (
    fetch_historical_counts as _sofa_cache_df,
    _iter_finished_events as _sofa_raw_events,
    _event_meta_fields as _sofa_meta_fields,
)

_BASE = "https://api.sofascore.com/api/v1"

# ---------------------------------------------------------------------------
# Overlap: competições presentes nas DUAS fontes (§2.3).
#   sb = (competition_id, season_id) no StatsBomb Open Data.
#   sofa_tid = uniqueTournament id no SofaScore (confirmado em check_sofascore_depth).
#   year = ano da season a casar no SofaScore.
# ---------------------------------------------------------------------------
_OVERLAP: list[dict[str, Any]] = [
    {"label": "UEFA Euro 2024",   "sb": (55, 282),  "sofa_tid": 1,   "year": 2024},
    {"label": "Copa América 2024", "sb": (223, 282), "sofa_tid": 133, "year": 2024},
]

# ---------------------------------------------------------------------------
# Normalização de nomes de seleção + alias entre fontes.
# StatsBomb e SofaScore divergem em alguns nomes (acentos, exônimos). Mapeia
# variantes normalizadas para uma forma canônica. Chave e valor já normalizados
# (ascii, minúsculo, sem pontuação).
# ---------------------------------------------------------------------------
_ALIAS: dict[str, str] = {
    "turkiye": "turkey",
    "czechia": "czech republic",
    "usa": "united states",
    "united states of america": "united states",
    "korea republic": "south korea",
    "republic of korea": "south korea",
    "ir iran": "iran",
    "bosnia herzegovina": "bosnia and herzegovina",
}


def _norm(name: str) -> str:
    """Normaliza um nome de seleção: ascii, minúsculo, sem pontuação, + alias."""
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    s = "".join(ch if ch.isalnum() or ch == " " else " " for ch in s.lower())
    s = " ".join(s.split())
    return _ALIAS.get(s, s)


# ---------------------------------------------------------------------------
# Cliente HTTP SofaScore — curl_cffi (padrão validado em check_sofascore_*.py)
# ---------------------------------------------------------------------------
_SESSION: Any = None
_COOKIE = os.environ.get("SOFASCORE_COOKIE", "").strip()


def _build_headers() -> dict[str, str]:
    h = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Referer": "https://www.sofascore.com/",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://www.sofascore.com",
    }
    if _COOKIE:
        h["Cookie"] = _COOKIE
    return h


def _init_client(impersonate: str = "chrome124") -> str:
    global _SESSION
    try:
        from curl_cffi import requests as cffi_requests
        _SESSION = cffi_requests.Session(impersonate=impersonate)
        return "curl_cffi"
    except ImportError:
        return "cookie" if _COOKIE else "urllib"


def _get(path: str, sleep_s: float = 0.0) -> tuple[int, dict]:
    """GET; retorna (status_code, json_body). sleep_s ANTES da chamada (rate limit)."""
    if sleep_s > 0:
        time.sleep(sleep_s)
    url = f"{_BASE}/{path.lstrip('/')}"
    if _SESSION is not None:
        try:
            resp = _SESSION.get(url, headers=_build_headers(), timeout=20)
            try:
                body = resp.json()
            except Exception:
                body = {}
            return resp.status_code, body
        except Exception as e:
            return -1, {"_exception": str(e)}
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, headers=_build_headers())
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode())
        except Exception:
            body = {}
        return e.code, body
    except Exception as e:
        return -1, {"_exception": str(e)}


def _year_of(season_name: str) -> int | None:
    """Extrai o MAIOR ano de um nome/label de season ('2018/2019' -> 2019)."""
    years = [int(y) for y in re.findall(r"(?:19|20)\d{2}", season_name or "")]
    return max(years) if years else None


# ---------------------------------------------------------------------------
# Lado StatsBomb (ground truth)
# ---------------------------------------------------------------------------

def _sb_meta(cid: int, sid: int) -> list[dict]:
    """Metadados dos jogos StatsBomb (sem baixar eventos ainda — barato/cache)."""
    out = []
    for m in fetch_matches(cid, sid):
        home = m["home_team"]["home_team_name"]
        away = m["away_team"]["away_team_name"]
        out.append({
            "match_id": m["match_id"],
            "date": m["match_date"],  # 'YYYY-MM-DD'
            "home": home,
            "away": away,
            "key": frozenset({_norm(home), _norm(away)}),
        })
    return out


def _sb_counts(match_id: int, home: str, away: str) -> dict[str, int]:
    """Escanteios/cartões por time de um jogo StatsBomb, alinhados a home/away.

    Delega a derivação à função canônica statsbomb.derive_counts (fonte única da
    regra Bad Behaviour + Foul Committed / Second Yellow).
    """
    per_team = derive_counts(fetch_events(match_id))
    zero = {"corners": 0, "yellow": 0, "red": 0}
    h = per_team.get(home, zero)
    a = per_team.get(away, zero)
    return {
        "home_corners": h["corners"], "away_corners": a["corners"],
        "home_yellow": h["yellow"],   "away_yellow": a["yellow"],
        "home_red": h["red"],         "away_red": a["red"],
    }


# ---------------------------------------------------------------------------
# Lado SofaScore (operacional)
# ---------------------------------------------------------------------------

def _sofa_season_id(tid: int, year: int, sleep_s: float) -> tuple[int | None, str]:
    """Localiza o season id do SofaScore para o ano pedido."""
    status, body = _get(f"unique-tournament/{tid}/seasons", sleep_s)
    if status != 200:
        return None, f"seasons HTTP {status}"
    for s in body.get("seasons", []):
        if _year_of(s.get("year") or s.get("name") or "") == year:
            return s["id"], s.get("name", "?")
    return None, "sem season para o ano"


def _sofa_events(tid: int, sid: int, sleep_s: float, max_pages: int = 6) -> list[dict]:
    """Todos os eventos FINISHED de uma season (pagina até esgotar)."""
    evs: list[dict] = []
    for page in range(max_pages):
        status, body = _get(
            f"unique-tournament/{tid}/season/{sid}/events/last/{page}", sleep_s)
        if status != 200:
            break
        page_evs = body.get("events", [])
        if not page_evs:
            break
        evs.extend(page_evs)
        if not body.get("hasNextPage"):
            break
    finished = []
    for e in evs:
        if e.get("status", {}).get("type") != "finished":
            continue
        ts = e.get("startTimestamp")
        d = (datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
             if ts else "")
        home = e.get("homeTeam", {}).get("name", "?")
        away = e.get("awayTeam", {}).get("name", "?")
        finished.append({
            "id": e["id"], "date": d, "home": home, "away": away,
            "key": frozenset({_norm(home), _norm(away)}),
        })
    return finished


def _sofa_counts(event_id: int, sleep_s: float) -> dict[str, int] | None:
    """Extrai escanteios/cartões por time do /statistics (ausência = 0)."""
    status, body = _get(f"event/{event_id}/statistics", sleep_s)
    if status != 200:
        return None
    stats = body.get("statistics")
    if not stats:
        return None
    vals: dict[str, dict] = {}
    for period in stats:
        if period.get("period") != "ALL":
            continue
        for g in period.get("groups", []):
            for item in g.get("statisticsItems", []):
                k = item.get("key", "")
                if k in ("cornerKicks", "yellowCards", "redCards"):
                    vals[k] = {"home": item.get("home"), "away": item.get("away")}

    def num(k: str, side: str) -> int:
        v = vals.get(k, {}).get(side)
        if v is None:  # ausente quando 0 (cartões) — tratar como 0
            return 0
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return 0

    return {
        "home_corners": num("cornerKicks", "home"), "away_corners": num("cornerKicks", "away"),
        "home_yellow": num("yellowCards", "home"),  "away_yellow": num("yellowCards", "away"),
        "home_red": num("redCards", "home"),        "away_red": num("redCards", "away"),
    }


# ---------------------------------------------------------------------------
# Casamento de jogos
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> date | None:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _match_games(
    sb_games: list[dict], sofa_games: list[dict], date_tol: int
) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    """Casa por (times + data). Retorna (pares, sb_nao_casados, sofa_nao_casados).

    Casa pelo par de times (não-ordenado, normalizado) e desempata pela data mais
    próxima dentro de `date_tol` dias. Cada evento SofaScore é usado no máximo uma vez.
    """
    sofa_by_key: dict[frozenset, list[dict]] = defaultdict(list)
    for g in sofa_games:
        sofa_by_key[g["key"]].append(g)

    pairs: list[tuple[dict, dict]] = []
    unmatched_sb: list[dict] = []
    used_sofa: set[int] = set()

    for sb in sb_games:
        sb_d = _parse_date(sb["date"])
        candidates = [c for c in sofa_by_key.get(sb["key"], []) if c["id"] not in used_sofa]
        if not candidates:
            # Distingue "times não casaram" de "só a data divergiu"
            if sb["key"] in sofa_by_key:
                sb["_reason"] = "par de times casa, mas todos os candidatos já usados"
            else:
                sb["_reason"] = "sem evento SofaScore com os mesmos times (alias?)"
            unmatched_sb.append(sb)
            continue

        def _daydiff(c: dict) -> int:
            cd = _parse_date(c["date"])
            if sb_d is None or cd is None:
                return 999
            return abs((cd - sb_d).days)

        best = min(candidates, key=_daydiff)
        if _daydiff(best) > date_tol:
            sb["_reason"] = (
                f"times casam mas data divergiu >{date_tol}d "
                f"(SB={sb['date']} vs Sofa={best['date']})"
            )
            unmatched_sb.append(sb)
            continue

        pairs.append((sb, best))
        used_sofa.add(best["id"])

    unmatched_sofa = [g for g in sofa_games if g["id"] not in used_sofa]
    return pairs, unmatched_sb, unmatched_sofa


def _aligned_diffs(
    sb: dict, sofa: dict, sb_counts: dict, sofa_counts: dict
) -> dict[str, Any]:
    """Compara par a par, alinhando home/away entre as fontes pelos nomes.

    Retorna diffs absolutos por time para escanteios, amarelos e vermelhos, além
    dos valores brutos alinhados para a tabela.
    """
    # Orientação: o home do SofaScore é o mesmo time que o home do StatsBomb?
    same = _norm(sofa["home"]) == _norm(sb["home"])

    def sofa_val(field_home: str, field_away: str) -> tuple[int, int]:
        # Retorna (valor alinhado ao home SB, valor alinhado ao away SB)
        if same:
            return sofa_counts[field_home], sofa_counts[field_away]
        return sofa_counts[field_away], sofa_counts[field_home]

    so_ch, so_ca = sofa_val("home_corners", "away_corners")
    so_yh, so_ya = sofa_val("home_yellow", "away_yellow")
    so_rh, so_ra = sofa_val("home_red", "away_red")

    return {
        "same_orientation": same,
        "corners": {
            "home": (sb_counts["home_corners"], so_ch),
            "away": (sb_counts["away_corners"], so_ca),
        },
        "yellow": {
            "home": (sb_counts["home_yellow"], so_yh),
            "away": (sb_counts["away_yellow"], so_ya),
        },
        "red": {
            "home": (sb_counts["home_red"], so_rh),
            "away": (sb_counts["away_red"], so_ra),
        },
    }


# ---------------------------------------------------------------------------
# Estatística de discrepância
# ---------------------------------------------------------------------------

def _summ(diffs: list[int]) -> dict[str, float]:
    if not diffs:
        return {"mean": 0.0, "max": 0, "n": 0, "exact_pct": 0.0}
    exact = sum(1 for d in diffs if d == 0)
    return {
        "mean": sum(diffs) / len(diffs),
        "max": max(diffs),
        "n": len(diffs),
        "exact_pct": 100.0 * exact / len(diffs),
    }


# ---------------------------------------------------------------------------
# Execução por competição
# ---------------------------------------------------------------------------

def run_overlap(spec: dict, sleep_s: float, date_tol: int) -> dict:
    label = spec["label"]
    cid, sid = spec["sb"]
    tid, year = spec["sofa_tid"], spec["year"]

    print("\n" + "=" * 78)
    print(f"OVERLAP — {label}   (StatsBomb {cid}/{sid}  ×  SofaScore uT={tid} ano={year})")
    print("=" * 78)

    # --- StatsBomb: metadados (barato) ---
    sb_games = _sb_meta(cid, sid)
    print(f"  StatsBomb: {len(sb_games)} jogos.")

    # --- SofaScore: season + eventos FINISHED ---
    sofa_sid, sofa_sname = _sofa_season_id(tid, year, sleep_s)
    if sofa_sid is None:
        print(f"  SofaScore: season {year} NÃO localizada ({sofa_sname}). Overlap abortado.")
        return {"label": label, "status": "no_sofa_season", "pairs": []}
    print(f"  SofaScore: season '{sofa_sname}' (sid={sofa_sid}) — buscando eventos...")
    sofa_games = _sofa_events(tid, sofa_sid, sleep_s)
    print(f"  SofaScore: {len(sofa_games)} eventos FINISHED.")

    # --- Casamento por times + data ---
    pairs, unmatched_sb, unmatched_sofa = _match_games(sb_games, sofa_games, date_tol)
    print(f"  Casados: {len(pairs)}  |  SB sem par: {len(unmatched_sb)}  |  "
          f"Sofa sem par: {len(unmatched_sofa)}")

    # --- Contagens + comparação só nos casados ---
    print(f"\n  {'data':<11} {'jogo':<34} {'escanteios':>16} {'amarelos':>12} {'verm':>10}")
    print(f"  {'':<11} {'(SB→Sofa por time)':<34} "
          f"{'SB / Sofa':>16} {'SB / Sofa':>12} {'SB/Sofa':>10}")
    print("  " + "─" * 84)

    rows = []
    for sb, sofa in pairs:
        sb_counts = _sb_counts(sb["match_id"], sb["home"], sb["away"])
        sofa_counts = _sofa_counts(sofa["id"], sleep_s)
        if sofa_counts is None:
            print(f"  {sb['date']:<11} {sb['home'][:15]}×{sb['away'][:15]:<16}  "
                  f"→ SofaScore /statistics vazio (evento {sofa['id']}); ignorado.")
            continue
        d = _aligned_diffs(sb, sofa, sb_counts, sofa_counts)
        rows.append({"sb": sb, "sofa": sofa, "d": d})

        cc = d["corners"]; yy = d["yellow"]; rr = d["red"]
        game = f"{sb['home'][:15]}×{sb['away'][:15]}"
        corner_s = f"{cc['home'][0]}+{cc['away'][0]} / {cc['home'][1]}+{cc['away'][1]}"
        yellow_s = f"{yy['home'][0]+yy['away'][0]} / {yy['home'][1]+yy['away'][1]}"
        red_s = f"{rr['home'][0]+rr['away'][0]} / {rr['home'][1]+rr['away'][1]}"
        flag = "" if d["same_orientation"] else "  [home/away invertido entre fontes]"
        print(f"  {sb['date']:<11} {game:<34} {corner_s:>16} {yellow_s:>12} {red_s:>10}{flag}")

    return {
        "label": label,
        "status": "ok",
        "pairs": rows,
        "n_sb": len(sb_games),
        "n_sofa": len(sofa_games),
        "unmatched_sb": unmatched_sb,
        "unmatched_sofa": unmatched_sofa,
    }


# ---------------------------------------------------------------------------
# Resumo agregado (o GATE)
# ---------------------------------------------------------------------------

def _print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("RESUMO — GATE DA CAMADA 3 (§2.3): discrepância StatsBomb × SofaScore")
    print("=" * 78)

    # Contagem de casamento
    total_pairs = sum(len(r["pairs"]) for r in results if r["status"] == "ok")
    print("\n[CASAMENTO DE JOGOS]")
    for r in results:
        if r["status"] != "ok":
            print(f"  {r['label']:<22}  {r['status']}")
            continue
        print(f"  {r['label']:<22}  casados={len(r['pairs'])}  "
              f"(SB={r['n_sb']}, Sofa={r['n_sofa']}, "
              f"SB sem par={len(r['unmatched_sb'])}, Sofa sem par={len(r['unmatched_sofa'])})")
    print(f"  {'TOTAL':<22}  casados={total_pairs}")

    # Motivos de não-casamento
    any_unmatched = any(
        r["status"] == "ok" and (r["unmatched_sb"] or r["unmatched_sofa"]) for r in results
    )
    if any_unmatched:
        print("\n[JOGOS NÃO CASADOS — por quê]")
        for r in results:
            if r["status"] != "ok":
                continue
            for sb in r["unmatched_sb"]:
                print(f"  [{r['label']}] SB  {sb['date']} {sb['home']}×{sb['away']}"
                      f"  → {sb.get('_reason', '?')}")
            for so in r["unmatched_sofa"]:
                print(f"  [{r['label']}] Sofa {so['date']} {so['home']}×{so['away']}"
                      f"  → sem jogo StatsBomb correspondente")

    # Discrepância agregada (por time = 2 observações por jogo)
    corner_diffs: list[int] = []
    yellow_diffs: list[int] = []
    red_diffs: list[int] = []
    card_diffs: list[int] = []       # amarelos+vermelhos por time
    exact_games = 0
    total_games = 0

    for r in results:
        if r["status"] != "ok":
            continue
        for row in r["pairs"]:
            d = row["d"]
            total_games += 1
            game_exact = True
            for side in ("home", "away"):
                dc = abs(d["corners"][side][0] - d["corners"][side][1])
                dy = abs(d["yellow"][side][0] - d["yellow"][side][1])
                dr = abs(d["red"][side][0] - d["red"][side][1])
                sb_cards = d["yellow"][side][0] + d["red"][side][0]
                so_cards = d["yellow"][side][1] + d["red"][side][1]
                dcard = abs(sb_cards - so_cards)
                corner_diffs.append(dc)
                yellow_diffs.append(dy)
                red_diffs.append(dr)
                card_diffs.append(dcard)
                if dc or dy or dr:
                    game_exact = False
            if game_exact:
                exact_games += 1

    print("\n[DISCREPÂNCIA — diferença absoluta por time (2 obs/jogo)]")
    if total_games == 0:
        print("  Nenhum jogo casado com /statistics — impossível medir. GATE INCONCLUSIVO.")
        return

    def _line(name: str, s: dict) -> None:
        print(f"  {name:<24} média={s['mean']:.2f}  máx={s['max']}  "
              f"exato={s['exact_pct']:.0f}%  (n={s['n']} obs)")

    _line("escanteios", _summ(corner_diffs))
    _line("cartões amarelos", _summ(yellow_diffs))
    _line("cartões vermelhos", _summ(red_diffs))
    _line("cartões (amar+verm)", _summ(card_diffs))

    print(f"\n  Jogos com TODAS as contagens idênticas (ambos os times): "
          f"{exact_games}/{total_games}  ({100.0*exact_games/total_games:.0f}%)")

    # Veredito do gate
    cs = _summ(corner_diffs)
    ks = _summ(card_diffs)
    print("\n[VEREDITO DO GATE — §2.3 / §10 passo 1]")
    print("  1. Paridade de definição: a diferença média por time é o resíduo de definição.")
    print(f"     escanteios: {cs['mean']:.2f}/time   cartões: {ks['mean']:.2f}/time")
    print("  2. Piso de erro de medição (§6.2 item 2): nenhum modelo do Modo 2 deve ser")
    print("     cobrado por precisão MAIOR que estes valores — é o ruído inter-fonte.")
    print(f"     PISO escanteios = {cs['mean']:.2f} | PISO cartões = {ks['mean']:.2f} (dif. abs. média/time).")
    print("  Interpretação: divergência baixa e consistente → SofaScore fidedigno p/ treino.")
    print("  Divergência alta/sistemática → investigar definição antes de qualquer fit() (§10 passo 4).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# ===========================================================================
# AUDITORIA DO CACHE DO COLETOR (--check-cache) — 3 pernas, escopos separados.
#
#   PERNA 1  cache == API   : fidelidade de escrita do coletor. Todos os 394 jogos.
#                             Igualdade EXATA (tolerância ZERO). Campos da EVENT LIST
#                             (8 req/season): *_score, *_goals_reg, round_name,
#                             cup_round_type, round_num. Com --full-stats, também
#                             refaz /statistics por jogo (394 req) p/ escanteios/cartões.
#   PERNA 2  API ~ StatsBomb : o gate que já existe (run_overlap + _print_summary).
#                             NÃO alterado. Escopo: 83 casados. Tolerância ~1%.
#   PERNA 3  cache ~ SB      : os 83 casados, cache × StatsBomb, MESMA tolerância da
#                             perna 2. Offline (usa o cache + eventos StatsBomb cacheados).
# ===========================================================================

# Campos que provêm da EVENT LIST (comparáveis com 8 req/season).
_LEG1_EVENT_FIELDS = [
    "home_score", "away_score", "home_goals_reg", "away_goals_reg",
    "round_name", "cup_round_type", "round_num",
]
# Campos que provêm de /event/{id}/statistics (só com --full-stats; 1 req/jogo).
_LEG1_STAT_FIELDS = [
    "home_corners", "away_corners", "home_yellow", "away_yellow", "home_red", "away_red",
]


def _cell_eq(cache_v: Any, api_v: Any) -> bool:
    """Igualdade exata tolerante a None/NaN e int-vs-float. String compara literal."""
    def norm(x: Any) -> Any:
        if x is None:
            return None
        if isinstance(x, float) and x != x:  # NaN
            return None
        return x
    a, b = norm(cache_v), norm(api_v)
    if a is None or b is None:
        return a is None and b is None
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def _api_event_view(ev: dict) -> dict:
    """Reconstrói, do OBJETO DE EVENTO da API, os campos que o coletor grava."""
    meta = _sofa_meta_fields(ev)  # home_goals_reg/away_goals_reg/round_name/cup_round_type/round_num
    return {
        "home_score": ev.get("homeScore", {}).get("current"),
        "away_score": ev.get("awayScore", {}).get("current"),
        "home_goals_reg": meta["home_goals_reg"],
        "away_goals_reg": meta["away_goals_reg"],
        "round_name": meta["round_name"],
        "cup_round_type": meta["cup_round_type"],
        "round_num": meta["round_num"],
    }


def leg1_cache_vs_api(sleep_s: float, full_stats: bool) -> dict:
    """PERNA 1 — cache hist_*.jsonl == API, campo a campo, tolerância ZERO."""
    df = _sofa_cache_df()
    checked = exact = 0
    divergences: list[dict] = []
    et_games: list[dict] = []
    unrecoverable: list[dict] = []

    for (tid, sid), g in df.groupby(["tournament_id", "season_id"]):
        events = _sofa_raw_events(int(tid), int(sid), sleep_s)   # 8 req no total (1-2/season)
        by_id = {int(e["id"]): e for e in events}
        print(f"    season sid={int(sid)}: {len(g)} jogos no cache, "
              f"{len(by_id)} na event list da API", flush=True)
        for _, row in g.iterrows():
            eid = int(row["event_id"])
            ev = by_id.get(eid)
            if ev is None:
                unrecoverable.append({"edition": row["season"], "date": row["date"],
                                      "home": row["home_team"], "away": row["away_team"],
                                      "event_id": eid})
                continue
            checked += 1
            view = _api_event_view(ev)
            diffs = [(f, row[f], view[f]) for f in _LEG1_EVENT_FIELDS
                     if not _cell_eq(row[f], view[f])]
            if full_stats:
                sc = _sofa_counts(eid, sleep_s)   # 1 req/jogo — /statistics
                if sc is None:
                    diffs.append(("statistics", "cache=preenchido", "API=vazio"))
                else:
                    diffs += [(f, row[f], sc[f]) for f in _LEG1_STAT_FIELDS
                              if not _cell_eq(row[f], sc[f])]
            if diffs:
                divergences.append({"edition": row["season"], "date": row["date"],
                                    "home": row["home_team"], "away": row["away_team"],
                                    "event_id": eid, "diffs": diffs})
            else:
                exact += 1
            # ET: placar final difere do 90min em algum lado (é onde o backfill atua).
            if not (_cell_eq(row["home_score"], row["home_goals_reg"])
                    and _cell_eq(row["away_score"], row["away_goals_reg"])):
                score_ok = not any(f in ("home_score", "away_score",
                                         "home_goals_reg", "away_goals_reg")
                                   for f, _, _ in diffs)
                et_games.append({
                    "edition": row["season"], "date": row["date"],
                    "home": row["home_team"], "away": row["away_team"],
                    "cache_score": (row["home_score"], row["away_score"]),
                    "cache_reg": (row["home_goals_reg"], row["away_goals_reg"]),
                    "api_score": (view["home_score"], view["away_score"]),
                    "api_reg": (view["home_goals_reg"], view["away_goals_reg"]),
                    "score_ok": score_ok,
                })
    return {"checked": checked, "exact": exact, "divergences": divergences,
            "et_games": et_games, "unrecoverable": unrecoverable, "full_stats": full_stats}


def _cache_sofa_games(df_edition) -> list[dict]:
    """Constrói 'jogos SofaScore' a partir do CACHE (mesmo formato do join)."""
    games = []
    for _, r in df_edition.iterrows():
        home, away = r["home_team"], r["away_team"]
        games.append({
            "id": int(r["event_id"]), "date": r["date"], "home": home, "away": away,
            "key": frozenset({_norm(home), _norm(away)}),
            "_counts": {
                "home_corners": int(r["home_corners"]), "away_corners": int(r["away_corners"]),
                "home_yellow": int(r["home_yellow"]), "away_yellow": int(r["away_yellow"]),
                "home_red": int(r["home_red"]), "away_red": int(r["away_red"]),
            },
        })
    return games


def leg3_cache_vs_statsbomb(date_tol: int) -> dict:
    """PERNA 3 — os 83 casados, CACHE × StatsBomb, mesma tolerância da perna 2. Offline."""
    df = _sofa_cache_df()
    corner_d: list[int] = []
    card_d: list[int] = []
    exact_games = total = 0
    per_edition: list[dict] = []
    for spec in _OVERLAP:
        cid, sid = spec["sb"]
        sb_games = _sb_meta(cid, sid)
        ed = df[(df["tournament_id"] == spec["sofa_tid"]) & (df["season_year"] == spec["year"])]
        sofa_games = _cache_sofa_games(ed)
        pairs, un_sb, un_so = _match_games(sb_games, sofa_games, date_tol)  # MESMO join
        for sb, so in pairs:
            sb_counts = _sb_counts(sb["match_id"], sb["home"], sb["away"])
            d = _aligned_diffs(sb, so, sb_counts, so["_counts"])
            total += 1
            game_exact = True
            for side in ("home", "away"):
                dc = abs(d["corners"][side][0] - d["corners"][side][1])
                dy = abs(d["yellow"][side][0] - d["yellow"][side][1])
                dr = abs(d["red"][side][0] - d["red"][side][1])
                sb_cards = d["yellow"][side][0] + d["red"][side][0]
                so_cards = d["yellow"][side][1] + d["red"][side][1]
                corner_d.append(dc)
                card_d.append(abs(sb_cards - so_cards))
                if dc or dy or dr:
                    game_exact = False
            if game_exact:
                exact_games += 1
        per_edition.append({"label": spec["label"], "pairs": len(pairs),
                            "un_sb": len(un_sb), "un_so": len(un_so)})
    return {"corner_d": corner_d, "card_d": card_d, "exact_games": exact_games,
            "total": total, "per_edition": per_edition}


def run_cache_audit(sleep_s: float, date_tol: int, full_stats: bool,
                    impersonate: str = "chrome124") -> None:
    """Orquestra as 3 pernas, com escopos e vereditos SEPARADOS."""
    # O coletor tem seu PRÓPRIO _SESSION curl_cffi (separado do deste módulo) —
    # a PERNA 1 usa a event list VIA sofascore, então inicializa a sessão dele.
    _sofa_mod._ensure_client(impersonate)
    # ---------------- PERNA 1 ----------------
    print("\n" + "#" * 78)
    print("PERNA 1 — cache hist_*.jsonl == API  (fidelidade de escrita; tolerância ZERO)")
    print("#" * 78)
    stat_note = ("event list + /statistics (394 req)" if full_stats
                 else "SÓ event list (8 req); escanteios/cartões cobertos pela PERNA 3")
    print(f"  Campos comparados: {stat_note}")
    r1 = leg1_cache_vs_api(sleep_s, full_stats)
    print(f"\n  Jogos checados: {r1['checked']}   exatos: {r1['exact']}   "
          f"divergências: {len(r1['divergences'])}")
    if r1["divergences"]:
        print("  DIVERGÊNCIAS (campo: cache vs API):")
        for dvg in r1["divergences"]:
            fields = "; ".join(f"{f}: {c!r}≠{a!r}" for f, c, a in dvg["diffs"])
            print(f"    ✗ [{dvg['edition']}] {dvg['date']} {dvg['home']}×{dvg['away']} "
                  f"(ev={dvg['event_id']}): {fields}")
    else:
        print("  ✓ zero divergência nos campos comparados.")

    # -------------- PERNA 1 — subseção ET --------------
    print("\n  ── PERNA 1 · subseção ET (placar final ≠ 90min; onde o backfill atua) ──")
    if not r1["et_games"]:
        print("    (nenhum jogo ET/pós-90 no cache)")
    else:
        print(f"    {len(r1['et_games'])} jogos com ação pós-90min:")
        print(f"    {'edição':<22} {'jogo':<28} {'cache_score':>11} {'cache_reg':>10} "
              f"{'api_reg':>9} {'ok?':>4}")
        for e in r1["et_games"]:
            jogo = f"{str(e['home'])[:13]}×{str(e['away'])[:13]}"
            cs = f"{e['cache_score'][0]}-{e['cache_score'][1]}"
            cr = f"{e['cache_reg'][0]}-{e['cache_reg'][1]}"
            ar = f"{e['api_reg'][0]}-{e['api_reg'][1]}"
            print(f"    {str(e['edition'])[:22]:<22} {jogo:<28} {cs:>11} {cr:>10} "
                  f"{ar:>9} {'✓' if e['score_ok'] else '✗':>4}")

    # ---------------- PERNA 2 (gate existente, INALTERADO) ----------------
    print("\n" + "#" * 78)
    print("PERNA 2 — API ~ StatsBomb  (gate existente; NÃO alterado; tolerância ~1%)")
    print("#" * 78)
    leg2 = [run_overlap(spec, sleep_s, date_tol) for spec in _OVERLAP]
    _print_summary(leg2)

    # ---------------- PERNA 3 (derivada, offline) ----------------
    print("\n" + "#" * 78)
    print("PERNA 3 — cache ~ StatsBomb nos 83  (mesma tolerância da perna 2; offline)")
    print("#" * 78)
    r3 = leg3_cache_vs_statsbomb(date_tol)
    for pe in r3["per_edition"]:
        print(f"  {pe['label']:<22} casados={pe['pairs']}  "
              f"(SB sem par={pe['un_sb']}, cache sem par={pe['un_so']})")
    if r3["total"]:
        cs = _summ(r3["corner_d"])
        ks = _summ(r3["card_d"])
        print(f"  escanteios: média={cs['mean']:.2f} máx={cs['max']} exato={cs['exact_pct']:.0f}%  "
              f"(n={cs['n']})")
        print(f"  cartões:    média={ks['mean']:.2f} máx={ks['max']} exato={ks['exact_pct']:.0f}%  "
              f"(n={ks['n']})")
        print(f"  jogos idênticos (ambos times): {r3['exact_games']}/{r3['total']} "
              f"({100*r3['exact_games']/r3['total']:.0f}%)")

    # ---------------- Irrecuperáveis ----------------
    print("\n" + "#" * 78)
    print("JOGOS IRRECUPERÁVEIS VIA API (evento sumiu da event list atual)")
    print("#" * 78)
    if not r1["unrecoverable"]:
        print("  ✓ nenhum — todos os 394 jogos do cache foram reencontrados na API.")
    else:
        print(f"  ⚠ {len(r1['unrecoverable'])} jogos do cache NÃO encontrados na API "
              "(insumo p/ decisão de versionamento do corpus):")
        for u in r1["unrecoverable"]:
            print(f"    [{u['edition']}] {u['date']} {u['home']}×{u['away']} (ev={u['event_id']})")

    # ---------------- Vereditos POR PERNA (sem agregar) ----------------
    print("\n" + "=" * 78)
    print("VEREDITO POR PERNA (cada uma responde uma pergunta diferente)")
    print("=" * 78)
    fields_scope = "todos os campos" if full_stats else "campos da event list (escanteio/cartão via PERNA 3)"
    v1 = "PASSOU" if not r1["divergences"] and not r1["unrecoverable"] else "FALHOU"
    print(f"  PERNA 1 (cache==API, escrita): {v1} — {r1['exact']}/{r1['checked']} exatos, "
          f"{len(r1['divergences'])} divergências ({fields_scope}).")
    print(f"  PERNA 2 (API~StatsBomb):       ver bloco acima (referência 97% escanteios / 99% cartões).")
    if r3["total"]:
        print(f"  PERNA 3 (cache~StatsBomb):     escanteios {_summ(r3['corner_d'])['exact_pct']:.0f}% / "
              f"cartões {_summ(r3['card_d'])['exact_pct']:.0f}% exatos nos {r3['total']} — "
              "deve espelhar a perna 2 (transitividade).")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Gate da Camada 3 (Modo 2): validação cruzada StatsBomb × SofaScore.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--sleep", type=float, default=3.0, metavar="S",
                   help="Segundos entre requisições SofaScore (default: 3.0 — rate limit §2.2).")
    p.add_argument("--impersonate", default="chrome124", metavar="I",
                   help="Perfil curl_cffi (default: chrome124).")
    p.add_argument("--date-tol", type=int, default=1, metavar="D",
                   help="Tolerância em dias ao casar por data (default: 1).")
    p.add_argument("--check-cache", action="store_true",
                   help="Audita o cache hist_*.jsonl do coletor (3 pernas separadas).")
    p.add_argument("--full-stats", action="store_true",
                   help="PERNA 1: também refaz /statistics por jogo (394 req) p/ checar\n"
                        "escanteios/cartões exatos. Default: só campos da event list (8 req).")
    args = p.parse_args()

    client = _init_client(args.impersonate)
    print(f"Cliente HTTP SofaScore: {client}", end="")
    if client != "curl_cffi":
        print("  ⚠ curl_cffi ausente — TLS fingerprint bloqueará (403). Instale curl_cffi.")
        if client == "urllib" and not _COOKIE:
            sys.exit(1)
    else:
        print(f"  (impersonate={args.impersonate})")

    st, _ = _get("unique-tournament/1/seasons", 0)
    print(f"Sanity /unique-tournament/1/seasons → HTTP {st}")
    if st != 200:
        print("  ✗ Acesso SofaScore negado — abortando. (Confirme curl_cffi / cookie.)")
        sys.exit(1)

    if args.check_cache:
        run_cache_audit(args.sleep, args.date_tol, args.full_stats, args.impersonate)
        return

    results = [run_overlap(spec, args.sleep, args.date_tol) for spec in _OVERLAP]
    _print_summary(results)


if __name__ == "__main__":
    main()
