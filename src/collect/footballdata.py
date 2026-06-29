"""Coletor Tier 1 — football-data.org (free) para WC2026.

Endpoints:
  GET /v4/competitions/WC/matches    → fixtures + resultados (104 partidas)
  GET /v4/competitions/WC/standings  → classificação dos 12 grupos

Cache local: data/raw/footballdata/wc{season}_{resource}.json
Rate limit (free tier): 10 req/min → throttle mínimo de 6,5 s entre chamadas.

Variável de ambiente (obrigatória):
  FOOTBALLDATA_API_KEY — token do football-data.org

Uso rápido:
  python src/collect/footballdata.py            # busca e exibe resumo (usa cache 24 h)
  python src/collect/footballdata.py --force    # ignora cache, refaz requisições

Import:
  from src.collect.footballdata import fetch_matches, fetch_standings
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_BASE_URL = "https://api.football-data.org/v4"
_COMPETITION = "WC"
_DEFAULT_SEASON = 2026
# Free tier: 10 req/min. 6,5 s garante margem mesmo com latência variável.
_MIN_DELAY_S = 6.5
_ROOT = Path(__file__).parents[2]
_CACHE_DIR = _ROOT / "data" / "raw" / "footballdata"

# Throttle de nível de módulo: compartilhado entre todas as chamadas na sessão.
_last_request_ts: float = 0.0


# ---------------------------------------------------------------------------
# Infraestrutura HTTP + cache
# ---------------------------------------------------------------------------


def _api_token() -> str:
    token = os.environ.get("FOOTBALLDATA_API_KEY", "").strip()
    if not token:
        sys.exit("ERRO: defina FOOTBALLDATA_API_KEY antes de usar o coletor.")
    return token


def _throttle() -> None:
    global _last_request_ts
    elapsed = time.monotonic() - _last_request_ts
    if elapsed < _MIN_DELAY_S:
        time.sleep(_MIN_DELAY_S - elapsed)
    _last_request_ts = time.monotonic()


def _get(path: str, params: dict | None = None) -> dict:
    """GET autenticado contra a API; aplica throttle; retorna JSON parseado."""
    _throttle()
    query = f"?{urllib.parse.urlencode(params)}" if params else ""
    url = f"{_BASE_URL}/{path.lstrip('/')}{query}"
    req = urllib.request.Request(url, headers={"X-Auth-Token": _api_token()})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:400]
        raise RuntimeError(f"HTTP {e.code} em /{path}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Erro de rede em /{path}: {e.reason}") from e


def _cache_path(resource: str, season: int) -> Path:
    return _CACHE_DIR / f"wc{season}_{resource}.json"


def _load_cache(resource: str, season: int, max_age_hours: float) -> dict | None:
    """Retorna dados do cache se existirem e estiverem dentro do TTL; senão None."""
    path = _cache_path(resource, season)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    cached_at = raw.get("_cached_at")
    if not cached_at:
        return None
    age_s = (datetime.now(timezone.utc) - datetime.fromisoformat(cached_at)).total_seconds()
    if age_s > max_age_hours * 3600:
        return None
    # Remove metadado interno antes de retornar — dado idêntico ao retorno live.
    return {k: v for k, v in raw.items() if k != "_cached_at"}


def _save_cache(resource: str, season: int, data: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {**data, "_cached_at": datetime.now(timezone.utc).isoformat()}
    _cache_path(resource, season).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def fetch_matches(
    season: int = _DEFAULT_SEASON,
    *,
    use_cache: bool = True,
    cache_ttl_hours: float = 24.0,
) -> dict:
    """Retorna todos os jogos da WC para a temporada indicada.

    Resposta: dict com chave ``matches`` — lista de até 104 dicts, cada um com:
      id, stage, matchday, group, utcDate, status,
      homeTeam {id, name, tla}, awayTeam {id, name, tla},
      score {winner, duration, fullTime {home, away}, halfTime {home, away}},
      venue.

    Estágios esperados: GROUP_STAGE (72) · LAST_32 (16) · LAST_16 (8) ·
    QUARTER_FINALS (4) · SEMI_FINALS (2) · THIRD_PLACE (1) · FINAL (1).
    """
    resource = "matches"
    if use_cache:
        cached = _load_cache(resource, season, cache_ttl_hours)
        if cached is not None:
            return cached
    data = _get(f"competitions/{_COMPETITION}/matches", {"season": season})
    _save_cache(resource, season, data)
    return data


def fetch_standings(
    season: int = _DEFAULT_SEASON,
    *,
    use_cache: bool = True,
    cache_ttl_hours: float = 24.0,
) -> dict:
    """Retorna a classificação dos grupos da WC para a temporada indicada.

    Resposta: dict com chave ``standings`` — lista de grupos, cada um com:
      stage, type, group,
      table: [{position, team {id, name, tla}, playedGames, form,
               won, draw, lost, points, goalsFor, goalsAgainst, goalDifference}]
    """
    resource = "standings"
    if use_cache:
        cached = _load_cache(resource, season, cache_ttl_hours)
        if cached is not None:
            return cached
    data = _get(f"competitions/{_COMPETITION}/standings", {"season": season})
    _save_cache(resource, season, data)
    return data


# ---------------------------------------------------------------------------
# CLI de verificação / recarga de cache
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Coleta dados WC2026 do football-data.org e armazena em cache local."
    )
    parser.add_argument("--season", type=int, default=_DEFAULT_SEASON)
    parser.add_argument(
        "--force", action="store_true", help="Ignora cache existente e recarrega da API."
    )
    args = parser.parse_args()

    use_cache = not args.force

    print(f"=== Fixtures WC{args.season} ===")
    matches_data = fetch_matches(args.season, use_cache=use_cache)
    all_matches = matches_data.get("matches", [])
    by_stage: dict[str, int] = {}
    for m in all_matches:
        s = m.get("stage", "UNKNOWN")
        by_stage[s] = by_stage.get(s, 0) + 1
    for stage, count in sorted(by_stage.items()):
        print(f"  {stage}: {count}")
    print(f"  Total: {len(all_matches)} partidas")
    cache_file = _cache_path("matches", args.season)
    print(f"  Cache: {cache_file}\n")

    print(f"=== Standings WC{args.season} ===")
    standings_data = fetch_standings(args.season, use_cache=use_cache)
    groups = standings_data.get("standings", [])
    for g in groups:
        group_name = g.get("group", "?")
        table = g.get("table", [])
        teams = ", ".join(row["team"]["tla"] for row in table)
        print(f"  {group_name}: {teams}")
    print(f"  Total: {len(groups)} grupos")
    cache_file = _cache_path("standings", args.season)
    print(f"  Cache: {cache_file}")
