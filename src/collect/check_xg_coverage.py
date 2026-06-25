"""Verificação de cobertura de xG (expected goals) na API-Football para a Copa 2026.

Confirma duas coisas, conforme SPEC.md §2.6:
  1. Cobertura declarada da liga (endpoint `leagues`, league=1, season=2026).
  2. Presença real de `expected_goals` por partida (endpoint `fixtures/statistics`).

Suporta os dois provedores. Defina UMA das variáveis de ambiente:

  # Acesso direto (host v3.football.api-sports.io):
  export APISPORTS_KEY="sua_chave_aqui"

  # Acesso via RapidAPI:
  export RAPIDAPI_KEY="sua_chave_aqui"

Depois:
    python src/collect/check_xg_coverage.py

Sem dependências externas — apenas stdlib. Não consome muitas requisições:
faz 1 chamada de cobertura, 1 de fixtures e checa xG numa amostra pequena de jogos.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

# Seleção de provedor por variável de ambiente. APISPORTS_KEY tem precedência.
APISPORTS_HOST = "v3.football.api-sports.io"
RAPIDAPI_HOST = "api-football-v1.p.rapidapi.com"

WORLD_CUP_LEAGUE_ID = 1
SEASON = 2026
XG_SAMPLE_SIZE = 5  # quantas partidas finalizadas inspecionar para presença de xG


def _provider() -> tuple[str, str, dict]:
    """Retorna (base_url, host, headers) conforme a chave disponível."""
    if os.environ.get("APISPORTS_KEY"):
        return (
            f"https://{APISPORTS_HOST}",
            APISPORTS_HOST,
            {"x-apisports-key": os.environ["APISPORTS_KEY"]},
        )
    if os.environ.get("RAPIDAPI_KEY"):
        return (
            f"https://{RAPIDAPI_HOST}/v3",
            RAPIDAPI_HOST,
            {
                "x-rapidapi-key": os.environ["RAPIDAPI_KEY"],
                "x-rapidapi-host": RAPIDAPI_HOST,
            },
        )
    sys.exit("ERRO: defina APISPORTS_KEY (acesso direto) ou RAPIDAPI_KEY antes de rodar.")


def _get(path: str, params: dict) -> dict:
    base_url, _host, headers = _provider()
    url = f"{base_url}/{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        sys.exit(f"ERRO HTTP {e.code} em {path}: {e.read().decode()[:300]}")
    except urllib.error.URLError as e:
        sys.exit(f"ERRO de rede em {path}: {e.reason}")


def check_league_coverage() -> None:
    print(f"== Cobertura declarada (league={WORLD_CUP_LEAGUE_ID}, season={SEASON}) ==")
    data = _get("leagues", {"id": WORLD_CUP_LEAGUE_ID, "season": SEASON})
    items = data.get("response", [])
    if not items:
        print("  Nenhuma liga retornada — confira o id da liga e a temporada.")
        return
    seasons = items[0].get("seasons", [])
    season = next((s for s in seasons if s.get("year") == SEASON), None)
    if not season:
        print(f"  Temporada {SEASON} não encontrada na resposta.")
        return
    fixtures_cov = season.get("coverage", {}).get("fixtures", {})
    stats_flag = fixtures_cov.get("statistics_fixtures")
    print(f"  coverage.fixtures.statistics_fixtures = {stats_flag}")
    print("  (true = suportado, mas disponibilidade de xG varia por partida)\n")


def check_xg_presence() -> None:
    print(f"== Presença real de xG em até {XG_SAMPLE_SIZE} partidas finalizadas ==")
    fixtures = _get(
        "fixtures",
        {"league": WORLD_CUP_LEAGUE_ID, "season": SEASON, "status": "FT"},
    ).get("response", [])
    if not fixtures:
        print("  Nenhuma partida finalizada ainda (ou sem acesso à temporada no plano).")
        return

    found, checked = 0, 0
    for fx in fixtures[:XG_SAMPLE_SIZE]:
        fixture_id = fx["fixture"]["id"]
        teams = f'{fx["teams"]["home"]["name"]} x {fx["teams"]["away"]["name"]}'
        stats = _get("fixtures/statistics", {"fixture": fixture_id}).get("response", [])
        types = {s["type"] for team in stats for s in team.get("statistics", [])}
        has_xg = "expected_goals" in types
        checked += 1
        found += has_xg
        print(f'  [{"xG OK " if has_xg else "sem xG"}] fixture {fixture_id}: {teams}')

    print(
        f"\n  Resultado: xG presente em {found}/{checked} partidas amostradas."
        f'\n  => {"Usar xG como feature principal." if found == checked else "Ativar proxy de xG (SPEC.md §3.3) quando ausente."}'
    )


if __name__ == "__main__":
    check_league_coverage()
    check_xg_presence()
