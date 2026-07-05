"""Coletor operacional SofaScore — Modo 2, passo 2 do SPEC_MARKETS.md §10.

Fonte primária de treino histórico E de inferência ao vivo na WC2026 (§2.2). Dois
pipelines DISTINTOS, conforme §2.7:

  PIPELINE HISTÓRICO  — coleta ÚNICA, cache permanente, RETOMÁVEL. Jogos antigos não
                        mudam: coleta-se uma vez e nunca mais. Cache incremental POR JOGO
                        (JSONL append) — se o processo cair, reinicia e PULA o que já está
                        em cache, continuando de onde parou.
  PIPELINE AO VIVO    — WC2026 (uID=16, season 2026). Atualização recorrente: re-coleta os
                        jogos FINISHED novos a cada execução, sem re-baixar os já cacheados.

Acesso: `api.sofascore.com/api/v1` via curl_cffi (impersonação de TLS do Chrome — bypass
do TLS fingerprinting / Varnish). Padrão validado em check_sofascore_coverage.py /
check_sofascore_depth.py. Rate limit: sleep 3 s (§2.2, medido — não assumir 25–30 s).

⚠ ANTI-FIBA (§2.2 requisito 2): nomes de torneio colidem entre esportes (WCQ ↔ basquete
  FIBA). Antes de coletar, valida-se category.sport.name == "Football" por ID.

⚠ Presença condicional (§2.2): yellowCards/redCards são ausentes no /statistics quando o
  valor é 0 → tratados como 0. expectedGoals só existe a partir de ~2020 → nullable.

NÃO deriva features (isso é o passo 3, counts.py). Só coleta e cacheia contagens BRUTAS
por jogo (home/away) + metadados: times, data, torneio, temporada, placar (`*_score`,
INCLUI pênaltis), gols de 90min (`*_goals_reg` = normaltime, SEM pênaltis) e fase
(`round_name`/`cup_round_type` de roundInfo). Cache antigo é enriquecido via --backfill-meta.

Uso:
  .venv/bin/python src/collect/sofascore.py --historical   # coleta o corpus (retomável)
  .venv/bin/python src/collect/sofascore.py --live         # só WC2026 (inferência)
  .venv/bin/python src/collect/sofascore.py --summary      # resumo do que já está em cache

Flags:
  --historical / --live / --summary
  --min-year Y   Ignora temporadas anteriores a Y no histórico (default: sem filtro).
  --sleep S      Segundos entre requisições (default: 3.0 — rate limit §2.2).
  --impersonate I  Perfil curl_cffi (default: chrome124).

Import:
  from src.collect.sofascore import fetch_historical_counts, fetch_live_counts
  from src.collect.sofascore import collect_historical, collect_live
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

_BASE = "https://api.sofascore.com/api/v1"
_ROOT = Path(__file__).parents[2]
_CACHE_DIR = _ROOT / "data" / "raw" / "sofascore"

# ---------------------------------------------------------------------------
# Torneios do corpus histórico — IDs de unique-tournament CONFIRMADOS (§2.2 passo 0).
# ---------------------------------------------------------------------------
class _Tour(NamedTuple):
    id: int
    label: str


_HISTORICAL_TOURNAMENTS: list[_Tour] = [
    _Tour(16,    "FIFA World Cup"),
    _Tour(1,     "UEFA Euro"),
    _Tour(133,   "Copa America"),
    _Tour(270,   "AFCON"),
    _Tour(246,   "AFC Asian Cup"),
    _Tour(140,   "CONCACAF Gold Cup"),
    _Tour(10783, "UEFA Nations League"),
]
# Amistosos (851) são BÔNUS condicional (~20% cobertura pré-2021 + viés de intensidade,
# §2.2 requisito 3) — fora do corpus base. Coletar via --tournaments se desejado no futuro.

# ---------------------------------------------------------------------------
# Escopo CONTROLADO editável — rodada dirigida (NÃO o corpus inteiro).
# Mapeia unique-tournament id → {season_id: rótulo}. Filtra por season_id (não por
# ano) porque o SofaScore guarda o ANO EM QUE FOI JOGADO: ex.: "Euro 2020" tem
# year='2021' (sid 26542). season_id é global e inequívoco. Editar aqui p/ outras rodadas.
# ---------------------------------------------------------------------------
_CONTROLLED_SCOPE: dict[int, dict[int, str]] = {
    16:  {15586: "World Cup 2018",       41087: "World Cup 2022"},
    1:   {26542: "Euro 2020",            56953: "Euro 2024"},
    133: {26681: "Copa America 2021",    57114: "Copa America 2024"},
    270: {38181: "AFCON 2021",           56021: "AFCON 2023"},
    # Densificação do corpus (jogos entre-torneios, §10 fase 2) — janelas N=10 hoje
    # esticam da Euro2024 até a WC2022; NL/AsianCup/GoldCup rejuvenescem a forma recente.
    246: {16919: "Asian Cup 2019",       51384: "Asian Cup 2023"},
    140: {23156: "Gold Cup 2019",        36683: "Gold Cup 2021",
          50492: "Gold Cup 2023",       72840: "Gold Cup 2025"},
    10783: {16000: "Nations League 18/19", 27595: "Nations League 20/21",
            40128: "Nations League 22/23", 58337: "Nations League 24/25"},
}

_WC_TOURNAMENT_ID = 16
_WC_LIVE_YEAR = 2026

# Chaves de /statistics coletadas (contagens brutas do Modo 2).
_STAT_KEYS = ("cornerKicks", "yellowCards", "redCards", "ballPossession", "expectedGoals")

# ---------------------------------------------------------------------------
# Cliente HTTP — curl_cffi (obrigatório: bypass de TLS fingerprint / Varnish)
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


def _ensure_client(impersonate: str = "chrome124") -> str:
    """Inicializa o cliente uma vez (idempotente). Retorna o modo do cliente."""
    global _SESSION
    if _SESSION is not None:
        return "curl_cffi"
    return _init_client(impersonate)


def _get(path: str, sleep_s: float = 0.0) -> tuple[int, dict]:
    """GET; retorna (status_code, json_body). sleep_s ANTES da chamada (rate limit §2.2)."""
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


# ---------------------------------------------------------------------------
# Descoberta: validação de esporte + seasons + eventos
# ---------------------------------------------------------------------------

def _year_of(season_name: str) -> int | None:
    """MAIOR ano de um nome/label de season ('2018/2019' -> 2019); aceita 'YY/YY'."""
    s = season_name or ""
    years = [int(y) for y in re.findall(r"(?:19|20)\d{2}", s)]
    if not years:
        yy = re.findall(r"\b(\d{2})\b", s)
        years = [2000 + int(y) for y in yy]
    return max(years) if years else None


def _validate_football(tid: int, sleep_s: float) -> tuple[bool, str, str]:
    """Confirma category.sport.name == 'Football' por ID (anti-FIBA §2.2 req.2).

    Retorna (is_football, sport_name, tournament_name).
    """
    st, body = _get(f"unique-tournament/{tid}", sleep_s)
    if st != 200:
        return False, f"HTTP {st}", ""
    ut = body.get("uniqueTournament", {}) or {}
    sport = ((ut.get("category", {}) or {}).get("sport", {}) or {}).get("name", "")
    return sport == "Football", sport or "?", ut.get("name", "?")


def _get_seasons(tid: int, sleep_s: float) -> list[dict]:
    st, body = _get(f"unique-tournament/{tid}/seasons", sleep_s)
    return body.get("seasons", []) if st == 200 else []


def _iter_finished_events(tid: int, sid: int, sleep_s: float, max_pages: int = 30) -> list[dict]:
    """Todos os eventos FINISHED de uma season, paginando /events/last/{page}."""
    evs: list[dict] = []
    for page in range(max_pages):
        st, body = _get(f"unique-tournament/{tid}/season/{sid}/events/last/{page}", sleep_s)
        if st != 200:
            break
        page_evs = body.get("events", [])
        if not page_evs:
            break
        evs.extend(page_evs)
        if not body.get("hasNextPage"):
            break
    return [e for e in evs if e.get("status", {}).get("type") == "finished"]


# ---------------------------------------------------------------------------
# Extração de contagens + montagem da linha
# ---------------------------------------------------------------------------

def _extract_stats(stat_body: dict) -> dict[str, dict]:
    """{key: {home, away}} das chaves-alvo no período ALL. Vazio se sem statistics."""
    stats = stat_body.get("statistics")
    out: dict[str, dict] = {}
    if not stats:
        return out
    for period in stats:
        if period.get("period") != "ALL":
            continue
        for g in period.get("groups", []):
            for item in g.get("statisticsItems", []):
                k = item.get("key", "")
                if k in _STAT_KEYS:
                    out[k] = {"home": item.get("home"), "away": item.get("away")}
    return out


def _to_int(v: Any) -> int:
    """Coerção p/ contagem inteira; None/ausente = 0 (presença condicional §2.2)."""
    if v is None:
        return 0
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def _to_float(v: Any) -> float | None:
    """Coerção p/ float (posse '55%' → 55.0, xG); None quando ausente/inválido."""
    if v is None:
        return None
    if isinstance(v, str):
        v = v.replace("%", "").strip()
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _regulation_goals(score: dict) -> int | None:
    """Gols de 90min (normaltime), EXCLUINDO prorrogação e pênaltis.

    ⚠ homeScore.current INCLUI pênaltis em mata-mata (ex.: 0-0 decidido nos pênaltis vira
      4-2). Para gols como target/feature, usar 'normaltime' (period1+period2). Fallback a
      'current' só se normaltime ausente (jogo sem essa subchave).
    """
    if not isinstance(score, dict):
        return None
    nt = score.get("normaltime")
    return nt if nt is not None else score.get("current")


def _event_meta_fields(ev: dict) -> dict:
    """Campos derivados do OBJETO DE EVENTO (event list) — sem custo extra de requisição:
    gols regulamentares (normaltime) + fase (roundInfo). Usado por _build_row e pelo backfill."""
    ri = ev.get("roundInfo") or {}
    return {
        "home_goals_reg": _regulation_goals(ev.get("homeScore", {})),
        "away_goals_reg": _regulation_goals(ev.get("awayScore", {})),
        "round_name": ri.get("name"),
        "cup_round_type": ri.get("cupRoundType"),
        "round_num": ri.get("round"),
    }


def _build_row(tour: _Tour, sid: int, sname: str, syear: int | None,
               ev: dict, stats: dict) -> dict:
    """Uma linha por JOGO (home/away lado a lado) + metadados. has_stats indica cobertura."""
    has_stats = bool(stats)
    ts = ev.get("startTimestamp")
    date = (datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
            if ts else None)

    def si(key: str, side: str) -> int | None:
        return _to_int(stats.get(key, {}).get(side)) if has_stats else None

    def sf(key: str, side: str) -> float | None:
        return _to_float(stats.get(key, {}).get(side)) if has_stats else None

    return {
        "tournament_id": tour.id,
        "tournament": tour.label,
        "season_id": sid,
        "season": sname,
        "season_year": syear,
        "event_id": ev["id"],
        "date": date,
        "start_timestamp": ts,
        "home_team": ev.get("homeTeam", {}).get("name"),
        "away_team": ev.get("awayTeam", {}).get("name"),
        "home_score": ev.get("homeScore", {}).get("current"),  # ⚠ inclui pênaltis em mata-mata
        "away_score": ev.get("awayScore", {}).get("current"),
        **_event_meta_fields(ev),  # home_goals_reg/away_goals_reg (90min) + fase (roundInfo)
        "home_corners": si("cornerKicks", "home"), "away_corners": si("cornerKicks", "away"),
        "home_yellow": si("yellowCards", "home"),  "away_yellow": si("yellowCards", "away"),
        "home_red": si("redCards", "home"),        "away_red": si("redCards", "away"),
        "home_possession": sf("ballPossession", "home"), "away_possession": sf("ballPossession", "away"),
        "home_xg": sf("expectedGoals", "home"),    "away_xg": sf("expectedGoals", "away"),
        "has_stats": has_stats,
    }


# ---------------------------------------------------------------------------
# Cache incremental por jogo (JSONL append — crash-safe e retomável)
# ---------------------------------------------------------------------------

def _hist_path(tid: int, sid: int) -> Path:
    return _CACHE_DIR / f"hist_{tid}_{sid}.jsonl"


def _live_path() -> Path:
    return _CACHE_DIR / f"live_wc2026.jsonl"


def _cached_event_ids(path: Path) -> set[int]:
    """IDs de evento já persistidos no JSONL (para pular na retomada)."""
    if not path.exists():
        return set()
    ids: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.add(int(json.loads(line)["event_id"]))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return ids


def _append_row(path: Path, row: dict) -> None:
    """Anexa uma linha ao JSONL e faz flush imediato (persistência por jogo)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# ---------------------------------------------------------------------------
# PIPELINE HISTÓRICO (coleta única, retomável)
# ---------------------------------------------------------------------------

def _collect_season(tour: _Tour, sid: int, sname: str, syear: int | None,
                    sleep_s: float) -> tuple[int, int]:
    """Coleta uma (torneio, temporada), pulando jogos já em cache. Retorna (novos, total)."""
    path = _hist_path(tour.id, sid)
    done = _cached_event_ids(path)
    events = _iter_finished_events(tour.id, sid, sleep_s)
    todo = [e for e in events if int(e["id"]) not in done]

    print(f"    season '{sname}' (ano={syear}, sid={sid}): {len(events)} FINISHED, "
          f"{len(done)} em cache, {len(todo)} a coletar")
    for i, ev in enumerate(todo, 1):
        st, body = _get(f"event/{ev['id']}/statistics", sleep_s)
        stats = _extract_stats(body) if st == 200 else {}
        row = _build_row(tour, sid, sname, syear, ev, stats)
        _append_row(path, row)
        cov = "stats" if row["has_stats"] else f"SEM stats(HTTP {st})"
        print(f"      [{i}/{len(todo)}] ev={ev['id']} "
              f"{str(row['home_team'])[:14]:<14} {row['home_score']}-{row['away_score']} "
              f"{str(row['away_team'])[:14]:<14} → {cov}", flush=True)
    return len(todo), len(events)


def collect_historical(tours: list[_Tour] | None = None, sleep_s: float = 3.0,
                       min_year: int | None = None, impersonate: str = "chrome124",
                       season_ids: set[int] | None = None) -> None:
    """Coleta o corpus histórico de contagens. RETOMÁVEL — pula tudo que já está em cache.

    season_ids: se fornecido, coleta APENAS as temporadas com esses ids (rodada dirigida).
    min_year:   se fornecido, ignora temporadas com ano < min_year.
    """
    mode = _ensure_client(impersonate)
    if mode != "curl_cffi" and not _COOKIE:
        print("✗ curl_cffi ausente — TLS fingerprint bloqueará (403). Instale curl_cffi.")
        return
    tours = tours or _HISTORICAL_TOURNAMENTS

    st, _ = _get("unique-tournament/16/seasons", 0)
    if st != 200:
        print(f"✗ Acesso SofaScore negado (HTTP {st}) — abortando.")
        return

    print(f"\n=== PIPELINE HISTÓRICO === cliente={mode}  torneios={len(tours)}  "
          f"min_year={min_year or 'sem filtro'}  "
          f"season_ids={'|'.join(map(str, sorted(season_ids))) if season_ids else 'todos'}  "
          f"sleep={sleep_s}s\n")

    t0 = time.monotonic()
    total_new = 0
    for tour in tours:
        is_fb, sport, tname = _validate_football(tour.id, sleep_s)
        if not is_fb:
            print(f"  [SKIP] {tour.label} (id={tour.id}): sport='{sport}' != Football "
                  f"(anti-FIBA §2.2).")
            continue
        print(f"  [{tour.label}] id={tour.id} '{tname}' — sport OK (Football)")
        seasons = _get_seasons(tour.id, sleep_s)
        if not seasons:
            print(f"    (sem seasons expostas)")
            continue
        for s in seasons:
            sid = s.get("id")
            sname = s.get("name", "?")
            syear = _year_of(s.get("year") or sname)
            if sid is None:
                continue
            if season_ids is not None and sid not in season_ids:
                continue
            if min_year is not None and syear is not None and syear < min_year:
                continue
            new, _tot = _collect_season(tour, sid, sname, syear, sleep_s)
            total_new += new

    elapsed = time.monotonic() - t0
    print(f"\n=== HISTÓRICO CONCLUÍDO === {total_new} jogos novos coletados nesta execução "
          f"em {elapsed/60:.1f} min ({elapsed:.0f} s).")
    print(f"    Cache: {_CACHE_DIR}/hist_*.jsonl  (consolidar via fetch_historical_counts())")


# ---------------------------------------------------------------------------
# PIPELINE AO VIVO (WC2026 — atualização recorrente)
# ---------------------------------------------------------------------------

def collect_live(sleep_s: float = 3.0, impersonate: str = "chrome124") -> int:
    """Coleta /statistics dos jogos FINISHED da WC2026 (uID=16, season 2026).

    Re-coleta apenas os jogos novos a cada execução (pula os já em cache). Retorna o
    nº de jogos novos coletados.
    """
    mode = _ensure_client(impersonate)
    if mode != "curl_cffi" and not _COOKIE:
        print("✗ curl_cffi ausente — TLS fingerprint bloqueará (403). Instale curl_cffi.")
        return 0

    is_fb, sport, tname = _validate_football(_WC_TOURNAMENT_ID, sleep_s)
    if not is_fb:
        print(f"✗ uID={_WC_TOURNAMENT_ID}: sport='{sport}' != Football — abortando.")
        return 0

    seasons = _get_seasons(_WC_TOURNAMENT_ID, sleep_s)
    season = next((s for s in seasons
                   if _year_of(s.get("year") or s.get("name")) == _WC_LIVE_YEAR), None)
    if season is None:
        print(f"✗ Season {_WC_LIVE_YEAR} de '{tname}' ainda não exposta no SofaScore.")
        return 0

    sid, sname = season["id"], season.get("name", "?")
    tour = _Tour(_WC_TOURNAMENT_ID, "FIFA World Cup")
    print(f"\n=== PIPELINE AO VIVO === WC{_WC_LIVE_YEAR} '{sname}' (sid={sid})  "
          f"cliente={mode}  sleep={sleep_s}s")

    path = _live_path()
    done = _cached_event_ids(path)
    events = _iter_finished_events(_WC_TOURNAMENT_ID, sid, sleep_s)
    todo = [e for e in events if int(e["id"]) not in done]
    print(f"  {len(events)} FINISHED, {len(done)} em cache, {len(todo)} novos a coletar")

    for i, ev in enumerate(todo, 1):
        st, body = _get(f"event/{ev['id']}/statistics", sleep_s)
        stats = _extract_stats(body) if st == 200 else {}
        row = _build_row(tour, sid, sname, _WC_LIVE_YEAR, ev, stats)
        _append_row(path, row)
        cov = "stats" if row["has_stats"] else f"SEM stats(HTTP {st})"
        print(f"    [{i}/{len(todo)}] ev={ev['id']} "
              f"{str(row['home_team'])[:14]:<14} {row['home_score']}-{row['away_score']} "
              f"{str(row['away_team'])[:14]:<14} → {cov}", flush=True)

    print(f"=== AO VIVO CONCLUÍDO === {len(todo)} jogos novos.")
    return len(todo)


# ---------------------------------------------------------------------------
# Backfill de metadados de evento (gols regulamentares + fase) no cache existente
# ---------------------------------------------------------------------------

def backfill_event_meta(sleep_s: float = 3.0, impersonate: str = "chrome124") -> None:
    """Enriquece o cache histórico já coletado com gols de 90min (normaltime) e fase.

    Esses campos vêm do OBJETO DE EVENTO (event list), não do /statistics — então o
    backfill re-busca apenas as listas de eventos por season (barato, ~2 req/season), SEM
    re-baixar as contagens já cacheadas. Idempotente: pula arquivos já enriquecidos.
    """
    mode = _ensure_client(impersonate)
    if mode != "curl_cffi" and not _COOKIE:
        print("✗ curl_cffi ausente — TLS fingerprint bloqueará (403).")
        return
    for path in sorted(_CACHE_DIR.glob("hist_*.jsonl")):
        rows = _read_rows(path)
        if not rows:
            continue
        if all("home_goals_reg" in r for r in rows):
            print(f"  {path.name}: já enriquecido ({len(rows)} jogos) — pulando")
            continue
        parts = path.stem.split("_")  # hist_{tid}_{sid}
        if len(parts) != 3:
            print(f"  {path.name}: nome inesperado — pulando")
            continue
        tid, sid = int(parts[1]), int(parts[2])
        events = _iter_finished_events(tid, sid, sleep_s)
        by_id = {int(e["id"]): e for e in events}
        n_upd = 0
        for r in rows:
            ev = by_id.get(int(r["event_id"]))
            if ev is not None:
                r.update(_event_meta_fields(ev))
                n_upd += 1
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
        print(f"  {path.name}: {n_upd}/{len(rows)} jogos enriquecidos (gols 90min + fase)")
    print("Backfill concluído.")


# ---------------------------------------------------------------------------
# API pública de consolidação
# ---------------------------------------------------------------------------

def fetch_historical_counts() -> "pd.DataFrame":  # noqa: F821
    """Consolida todo o cache histórico (hist_*.jsonl) em um DataFrame de contagens brutas."""
    import pandas as pd
    rows: list[dict] = []
    for p in sorted(_CACHE_DIR.glob("hist_*.jsonl")):
        rows.extend(_read_rows(p))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["date", "event_id"]).reset_index(drop=True)
    return df


def fetch_live_counts() -> "pd.DataFrame":  # noqa: F821
    """Consolida o cache ao vivo (WC2026) em um DataFrame de contagens brutas."""
    import pandas as pd
    df = pd.DataFrame(_read_rows(_live_path()))
    if not df.empty:
        df = df.sort_values(["date", "event_id"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Resumo do cache (sem rede)
# ---------------------------------------------------------------------------

def _print_summary() -> None:
    df = fetch_historical_counts()
    print("\n=== RESUMO DO CACHE HISTÓRICO ===")
    if df.empty:
        print("  (vazio — rode --historical)")
    else:
        n = len(df)
        with_stats = int(df["has_stats"].sum())
        print(f"  {n} jogos em cache · {with_stats} com /statistics ({with_stats/n:.0%})")
        print(f"  {'torneio':<22} {'jogos':>6} {'c/stats':>7} {'anos'}")
        print("  " + "─" * 60)
        for tour, g in df.groupby("tournament"):
            years = sorted({int(y) for y in g["season_year"].dropna().unique()})
            span = f"{years[0]}–{years[-1]}" if years else "?"
            print(f"  {str(tour):<22} {len(g):>6} {int(g['has_stats'].sum()):>7} {span}")
    live = fetch_live_counts()
    print("\n=== RESUMO DO CACHE AO VIVO (WC2026) ===")
    if live.empty:
        print("  (vazio — rode --live)")
    else:
        print(f"  {len(live)} jogos · {int(live['has_stats'].sum())} com /statistics")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Coletor operacional SofaScore (Modo 2, §10 passo 2). "
                    "Retomada implícita: sempre pula jogos já em cache.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--historical", action="store_true",
                   help="Coleta o corpus histórico (retomável).")
    p.add_argument("--live", action="store_true",
                   help="Coleta só a WC2026 (inferência ao vivo).")
    p.add_argument("--summary", action="store_true",
                   help="Só imprime o resumo do cache (sem rede).")
    p.add_argument("--backfill-meta", action="store_true",
                   help="Enriquece o cache existente com gols de 90min + fase (re-busca\n"
                        "só as listas de eventos; idempotente).")
    p.add_argument("--scope", choices=["controlled"], default=None,
                   help="Rodada dirigida: usa o escopo hardcoded _CONTROLLED_SCOPE\n"
                        "(WC 2018/2022, Euro 2020/2024, CopaAm 2021/2024, AFCON 2021/2023).")
    p.add_argument("--tournaments", metavar="IDS", default=None,
                   help="Restringe o histórico a estes unique-tournament ids (csv). Ex: 16,1,133,270")
    p.add_argument("--season-ids", metavar="IDS", default=None,
                   help="Restringe o histórico a estes season ids (csv). Ex: 15586,41087")
    p.add_argument("--min-year", type=int, default=None, metavar="Y",
                   help="Ignora temporadas < Y no histórico (default: sem filtro).")
    p.add_argument("--sleep", type=float, default=3.0, metavar="S",
                   help="Segundos entre requisições (default: 3.0 — rate limit §2.2).")
    p.add_argument("--impersonate", default="chrome124", metavar="I",
                   help="Perfil curl_cffi (default: chrome124).")
    args = p.parse_args()

    if not (args.historical or args.live or args.summary or args.backfill_meta):
        p.print_help()
        print("\n  Nada a fazer: passe --historical, --live, --backfill-meta ou --summary.")
        return

    if args.backfill_meta:
        backfill_event_meta(sleep_s=args.sleep, impersonate=args.impersonate)

    if args.historical:
        tours: list[_Tour] | None = None
        season_ids: set[int] | None = None
        if args.scope == "controlled":
            want_ids = set(_CONTROLLED_SCOPE)
            tours = [t for t in _HISTORICAL_TOURNAMENTS if t.id in want_ids]
            season_ids = {sid for m in _CONTROLLED_SCOPE.values() for sid in m}
        else:
            if args.tournaments:
                want = {int(x) for x in args.tournaments.split(",") if x.strip()}
                tours = [t for t in _HISTORICAL_TOURNAMENTS if t.id in want]
            if args.season_ids:
                season_ids = {int(x) for x in args.season_ids.split(",") if x.strip()}
        collect_historical(tours=tours, sleep_s=args.sleep, min_year=args.min_year,
                           impersonate=args.impersonate, season_ids=season_ids)
    if args.live:
        collect_live(sleep_s=args.sleep, impersonate=args.impersonate)
    if args.summary:
        _print_summary()


if __name__ == "__main__":
    main()
