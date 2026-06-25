"""Coletor de treino histórico — StatsBomb Open Data (GitHub, gratuito).

Baixa dados das competições de seleções definidas em SPEC.md §2.2 e §2.6:
  WC2022 · WC2018 · Euro 2024 · Euro 2020 · Copa América 2024

Fonte: https://github.com/statsbomb/open-data (sem API key, sem rate limit)
Cache local: data/raw/statsbomb/

⚠️  PARIDADE TREINO↔INFERÊNCIA (SPEC.md §2.4 — DECISÃO MAIS IMPORTANTE DO PROJETO)
    Este coletor expõe dados brutos incluindo xG, posse e eventos ricos do StatsBomb.
    A camada de features (src/features/) é responsável por NÃO usar esses campos no
    modelo Tier 1 — apenas gols + contexto. Usar xG do StatsBomb como feature de treino
    enquanto a inferência (WC2026) não tem xG garante distribution shift silencioso.

Uso:
  python src/collect/statsbomb.py            # lista competições e contagem de partidas
  python src/collect/statsbomb.py --download # baixa matches para cache local
  python src/collect/statsbomb.py --force    # ignora cache, refaz download

Import:
  from src.collect.statsbomb import fetch_all_training_matches
  from src.collect.statsbomb import fetch_matches, fetch_events
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
_ROOT = Path(__file__).parents[2]
_CACHE_DIR = _ROOT / "data" / "raw" / "statsbomb"

# IDs verificados em 2026-06-24 a partir de competitions.json do StatsBomb Open Data.
# Alterar apenas se o repositório StatsBomb adicionar novas temporadas relevantes.
class _Comp(NamedTuple):
    competition_id: int
    season_id: int
    label: str


TRAINING_COMPETITIONS: list[_Comp] = [
    _Comp(43, 106, "FIFA World Cup 2022"),
    _Comp(43,   3, "FIFA World Cup 2018"),
    _Comp(55, 282, "UEFA Euro 2024"),
    _Comp(55,  43, "UEFA Euro 2020"),
    _Comp(223, 282, "Copa America 2024"),
]


# ---------------------------------------------------------------------------
# Infraestrutura HTTP + cache
# ---------------------------------------------------------------------------

# Throttle suave para não sobrecarregar o CDN do GitHub em downloads em massa.
_MIN_DELAY_S = 0.3
_last_request_ts: float = 0.0


def _throttle() -> None:
    global _last_request_ts
    elapsed = time.monotonic() - _last_request_ts
    if elapsed < _MIN_DELAY_S:
        time.sleep(_MIN_DELAY_S - elapsed)
    _last_request_ts = time.monotonic()


def _fetch_url(url: str) -> bytes:
    _throttle()
    req = urllib.request.Request(url, headers={"User-Agent": "polvo-paul-26/statsbomb-collector"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {url}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Erro de rede: {url} — {e.reason}") from e


def _cache_path(name: str) -> Path:
    return _CACHE_DIR / name


def _load_cache(name: str) -> list | dict | None:
    p = _cache_path(name)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _save_cache(name: str, data: list | dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(name).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def fetch_competitions(*, use_cache: bool = True) -> list[dict]:
    """Retorna todas as competições disponíveis no StatsBomb Open Data.

    O resultado inclui competition_id, season_id, competition_name, season_name,
    competition_gender, competition_youth, competition_international.
    """
    name = "competitions.json"
    if use_cache:
        cached = _load_cache(name)
        if cached is not None:
            return cached
    data = json.loads(_fetch_url(f"{_BASE_URL}/competitions.json"))
    _save_cache(name, data)
    return data


def fetch_matches(
    competition_id: int,
    season_id: int,
    *,
    use_cache: bool = True,
) -> list[dict]:
    """Retorna todas as partidas de uma competição/temporada.

    Cada partida contém: match_id, match_date, competition_stage (name),
    home_team, away_team, home_score, away_score, match_week, season, competition.

    Parâmetros verificados para treino: competition_id 43 (WC), 55 (Euro), 223 (Copa América).
    """
    name = f"matches_{competition_id}_{season_id}.json"
    if use_cache:
        cached = _load_cache(name)
        if cached is not None:
            return cached
    url = f"{_BASE_URL}/matches/{competition_id}/{season_id}.json"
    data = json.loads(_fetch_url(url))
    _save_cache(name, data)
    return data


def fetch_all_training_matches(*, use_cache: bool = True) -> list[dict]:
    """Retorna todas as partidas das competições de treino (§2.2), anotadas com o label.

    Cada partida recebe um campo extra ``_training_label`` com o nome da competição.
    Retorna uma lista plana ordenada por data.
    """
    all_matches: list[dict] = []
    for comp in TRAINING_COMPETITIONS:
        matches = fetch_matches(comp.competition_id, comp.season_id, use_cache=use_cache)
        for m in matches:
            all_matches.append({**m, "_training_label": comp.label})
    all_matches.sort(key=lambda m: m.get("match_date", ""))
    return all_matches


def fetch_events(match_id: int, *, use_cache: bool = True) -> list[dict]:
    """Retorna os eventos (nível de ação) de uma partida: chutes, passes, etc.

    ⚠️  Para uso EXCLUSIVO na camada Tier 2 (SPEC.md §3.3) — chutes e xG opcionais.
        NÃO chamar no pipeline Tier 1 (o modelo de produção não usa esses dados).

    Cada evento contém type, timestamp, team, player, location, shot (quando type=Shot),
    incluindo shot.statsbomb_xg quando disponível.
    """
    name = f"events_{match_id}.json"
    if use_cache:
        cached = _load_cache(name)
        if cached is not None:
            return cached
    url = f"{_BASE_URL}/events/{match_id}.json"
    data = json.loads(_fetch_url(url))
    _save_cache(name, data)
    return data


# ---------------------------------------------------------------------------
# CLI de verificação e download
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from collections import Counter

    parser = argparse.ArgumentParser(
        description="Baixa dados de treino do StatsBomb Open Data para cache local."
    )
    parser.add_argument(
        "--download", action="store_true", help="Baixa matches de todas as competições de treino."
    )
    parser.add_argument(
        "--force", action="store_true", help="Ignora cache existente e refaz download."
    )
    args = parser.parse_args()

    use_cache = not args.force

    print("=== Competições de treino (SPEC.md §2.2) ===")
    for comp in TRAINING_COMPETITIONS:
        try:
            matches = fetch_matches(comp.competition_id, comp.season_id, use_cache=use_cache)
            stages = Counter(m.get("competition_stage", {}).get("name", "?") for m in matches)
            stage_str = "  ".join(f"{s}:{n}" for s, n in sorted(stages.items()))
            cached_marker = "✓cache" if _cache_path(f"matches_{comp.competition_id}_{comp.season_id}.json").exists() else "↓fetch"
            print(f"  [{cached_marker}] {comp.label:<28} {len(matches):>3} partidas  {stage_str}")
        except RuntimeError as e:
            print(f"  [ERRO] {comp.label}: {e}", file=sys.stderr)

    if args.download:
        print("\n=== Download completo via --download já feito acima ===")

    print(f"\nCache em: {_CACHE_DIR}")
