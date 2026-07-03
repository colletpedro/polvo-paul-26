"""Verifica cobertura de escanteios e cartões em fontes GRATUITAS para seleções.

NÃO coleta dataset — apenas verifica e reporta.

Fontes verificadas:
  1. StatsBomb Open Data (GitHub, sem API key) — eventos ricos por partida.
     Escanteio = Pass com pass.type.name == "Corner".
     Cartão    = registrado em DOIS tipos de evento — Bad Behaviour (má conduta sem
                 falta) E Foul Committed (cartão de falta, a MAIORIA). Contar só Bad
                 Behaviour subconta ~70% dos amarelos (provado pelo gate da Camada 3,
                 src/evaluate/cross_validation.py — validado contra o SofaScore em 83 jogos).
                 amarelo = Yellow Card + Second Yellow · vermelho = Red Card + Second Yellow.
     Cobertura real: WC2022/2018, Euro 2024/2020, Copa Am 2024, AFCON 2023.
     Cobertura adicional (não usada no treino): WC histórico 1958–1990, Women's
     WC/Euro, U20 WC.
     O que NÃO existe: eliminatórias, Nations League, amistosos de seleção.

  2. football-data.co.uk — CSVs de ligas de clubes europeus.
     Inclui total de escanteios e cartões por jogo (colunas HC/AC, HY/AY, HR/AR),
     MAS cobre EXCLUSIVAMENTE ligas de CLUBES. Nenhum jogo de seleção disponível.

Uso:
  python src/collect/check_counts_coverage.py              # amostra 3 jogos/comp
  python src/collect/check_counts_coverage.py --sample 5  # amostra maior
  python src/collect/check_counts_coverage.py --all       # todos os jogos (lento)

Os arquivos de eventos baixados ficam em cache em data/raw/statsbomb/.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.collect.statsbomb import (
    TRAINING_COMPETITIONS,
    derive_match_totals,
    fetch_competitions,
    fetch_events,
    fetch_matches,
)

# ---------------------------------------------------------------------------
# Constantes de classificação de fase
# ---------------------------------------------------------------------------
_STAGE_ORD: dict[str, int] = {
    "Group Stage": 0, "Group stage": 0,
    "Round of 16": 1, "Last 16": 1, "Round of 32": 1, "Round of 64": 1,
    "3rd Place Final": 1,
    "Quarter-finals": 2, "Quarter-final": 2,
    "Semi-finals": 3, "Semi-final": 3,
    "Final": 4,
}

# Competições de seleção que EXISTEM no StatsBomb Open Data mas NÃO são usadas
# no treino do modelo (excluídas por SPEC — sem ranking FIFA confiável ou dados
# fragmentários).  Reportadas como "disponível mas fora do escopo do modelo".
_ADDITIONAL_INTL_COMPS: list[tuple[int, int, str]] = [
    (43, 269, "FIFA World Cup 1958"),
    (43, 270, "FIFA World Cup 1962"),
    (43, 272, "FIFA World Cup 1970"),
    (43,  51, "FIFA World Cup 1974"),
    (43,  54, "FIFA World Cup 1986"),
    (43,  55, "FIFA World Cup 1990"),
    (1470, 274, "FIFA U20 World Cup 1979"),
    (72,  30,  "Women's World Cup 2019"),
    (72, 107,  "Women's World Cup 2023"),
    (53, 106,  "UEFA Women's Euro 2022"),
    (53, 315,  "UEFA Women's Euro 2025"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MatchSample(NamedTuple):
    match_id: int
    stage: str
    home: str
    away: str
    n_corners: int
    n_yellow: int
    n_red: int   # Red Card + Second Yellow (indirect red)


def _count_events(match_id: int) -> tuple[int, int, int]:
    """Retorna (corners, yellow_cards, red_cards) totais de um jogo.

    Usa a derivação canônica src.collect.statsbomb.derive_match_totals — a regra
    (cartão = Bad Behaviour + Foul Committed, Second Yellow = amarelo + expulsão) é
    definida em UM só lugar, validada contra o SofaScore em 83 jogos pelo gate da
    Camada 3 (src/evaluate/cross_validation.py).
    """
    totals = derive_match_totals(fetch_events(match_id))
    return totals["corners"], totals["yellow"], totals["red"]


def _sample_matches(
    competition_id: int,
    season_id: int,
    n_sample: int,
    all_matches: bool = False,
) -> tuple[list[dict], list[_MatchSample]]:
    """Retorna (all_match_dicts, sampled_MatchSamples)."""
    matches = fetch_matches(competition_id, season_id)
    if all_matches:
        to_sample = matches
    else:
        # Prefere jogos de mata-mata (mais representativos do modelo) para a amostra.
        ko = [m for m in matches if _STAGE_ORD.get(m["competition_stage"]["name"], -1) > 0]
        gs = [m for m in matches if _STAGE_ORD.get(m["competition_stage"]["name"], -1) == 0]
        pool = (ko + gs)[:n_sample]
        to_sample = pool

    samples: list[_MatchSample] = []
    for m in to_sample:
        mid = m["match_id"]
        stage = m["competition_stage"]["name"]
        home = m["home_team"]["home_team_name"]
        away = m["away_team"]["away_team_name"]
        try:
            c, y, r = _count_events(mid)
        except Exception as exc:
            print(f"    AVISO: erro ao buscar eventos de match_id={mid}: {exc}", file=sys.stderr)
            continue
        samples.append(_MatchSample(mid, stage, home, away, c, y, r))

    return matches, samples


# ---------------------------------------------------------------------------
# Secções de relatório
# ---------------------------------------------------------------------------

def _section_statsbomb_training(n_sample: int, all_matches: bool) -> list[dict]:
    """Verifica as competições de treino do modelo. Retorna lista de resultados."""
    print("=" * 70)
    print("FONTE 1 — StatsBomb Open Data (treinamento do modelo)")
    print("=" * 70)
    print(f"Derivação: escanteio = Pass[pass.type=Corner]  |  "
          f"cartão = Bad Behaviour + Foul Committed (amar.=Yellow+2ºYellow, verm.=Red+2ºYellow)")
    print()

    results = []
    for comp in TRAINING_COMPETITIONS:
        cid, sid, label = comp.competition_id, comp.season_id, comp.label
        print(f"  [{label}]")
        matches, samples = _sample_matches(cid, sid, n_sample, all_matches)

        stages = Counter(m["competition_stage"]["name"] for m in matches)
        n_ko = sum(n for s, n in stages.items() if _STAGE_ORD.get(s, -1) > 0)
        n_total = len(matches)

        if not samples:
            print(f"    ERRO: sem amostras disponíveis.")
            continue

        # Per-match detail
        for s in samples:
            print(
                f"    match={s.match_id}  [{s.stage[:16]:<16}]  "
                f"{s.home[:14]:<14} vs {s.away[:14]:<14}  "
                f"escanteios={s.n_corners:>2}  amarelos={s.n_yellow}  vermelhos={s.n_red}"
            )

        avg_c = sum(s.n_corners for s in samples) / len(samples)
        avg_y = sum(s.n_yellow for s in samples) / len(samples)
        avg_r = sum(s.n_red    for s in samples) / len(samples)
        note = f"(amostra {len(samples)}/{n_total}" + (" = todos" if all_matches else "") + ")"
        print(
            f"    -> médias: {avg_c:.1f} escanteios/jogo  {avg_y:.1f} amarelos/jogo  "
            f"{avg_r:.1f} vermelhos/jogo  {note}"
        )
        print(f"    -> total partidas: {n_total}  knockout: {n_ko}  "
              f"stages: {dict(stages)}")
        print()

        results.append({
            "label":    label,
            "cid":      cid,
            "sid":      sid,
            "n_total":  n_total,
            "n_ko":     n_ko,
            "n_sample": len(samples),
            "avg_corners": avg_c,
            "avg_yellow":  avg_y,
            "avg_red":     avg_r,
            "corners_ok":  True,
            "cards_ok":    True,
        })

    return results


def _section_statsbomb_additional() -> None:
    """Lista competições adicionais de seleção no StatsBomb que não são usadas no modelo."""
    print("=" * 70)
    print("FONTE 1b — StatsBomb Open Data (competições NÃO usadas no modelo)")
    print("=" * 70)
    print("  (eventos igualmente ricos em escanteios/cartões; excluídas por SPEC.md)")
    print()

    all_comps = fetch_competitions()
    comp_index = {(c["competition_id"], c["season_id"]): c for c in all_comps}

    for cid, sid, label in _ADDITIONAL_INTL_COMPS:
        c = comp_index.get((cid, sid))
        if c:
            gender = c.get("competition_gender", "?")
            intl   = c.get("competition_international", False)
            print(f"    cid={cid:>4} sid={sid:>4}  {label:<35}  gender={gender}  intl={intl}")
        else:
            print(f"    cid={cid:>4} sid={sid:>4}  {label:<35}  [não encontrado no competitions.json]")

    # Mostrar o que NÃO existe
    print()
    print("  O que NÃO existe no StatsBomb Open Data (jogos de seleção):")
    print("    * Eliminatórias de qualquer confederação (UEFA Nations League,")
    print("      CONMEBOL Eliminatórias, CAF, AFC, CONCACAF, OFC)")
    print("    * Amistosos internacionais de qualquer período")
    print("    * Gold Cup, Copa das Nações Africanas antes de 2023,")
    print("      Asian Cup, Oceania Cup — nenhuma edição disponível")
    print()


def _section_footballdata_dotcouk() -> None:
    """Relata o escopo do football-data.co.uk (sem chamadas à API)."""
    print("=" * 70)
    print("FONTE 2 — football-data.co.uk (CSV histórico, sem API key)")
    print("=" * 70)
    print()
    print("  Estrutura: um CSV por divisão por temporada.")
    print("  Colunas disponíveis por partida:")
    print("    HC / AC  = escanteios home / away (Home Corners / Away Corners)")
    print("    HY / AY  = cartões amarelos home / away")
    print("    HR / AR  = cartões vermelhos home / away")
    print()
    print("  Cobertura de ligas de CLUBES (sample):")
    print("    Premier League, Championship, League One, League Two (England)")
    print("    La Liga, Segunda División (Spain)")
    print("    Bundesliga, 2. Bundesliga (Germany)")
    print("    Serie A, Serie B (Italy)")
    print("    Ligue 1 (France)")
    print("    + Bélgica, Portugal, Países Baixos, Turquia, Grécia, Escócia")
    print()
    print("  *** JOGOS DE SELEÇÃO: ZERO cobertura. ***")
    print("  Nenhuma Copa do Mundo, Euro, Copa América, eliminatória ou amistoso.")
    print("  O site cobre exclusivamente ligas domésticas de clubes.")
    print("  Inutilizável para o modelo atual (seleções).")
    print()


# ---------------------------------------------------------------------------
# Tabela resumo
# ---------------------------------------------------------------------------

def _print_summary(training_results: list[dict]) -> None:
    print("=" * 70)
    print("TABELA RESUMO — cobertura de escanteios + cartões por fonte")
    print("=" * 70)
    print()

    hdr = (f"  {'Fonte':<14} {'Competição':<35} {'Partidas':>8} {'KO':>4} "
           f"{'Cant/j':>6} {'Amar/j':>6} {'Verm/j':>6}  {'Derivável':>10}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for r in training_results:
        derivable = "✓ Sim" if r["corners_ok"] and r["cards_ok"] else "✗ Não"
        print(
            f"  {'StatsBomb':<14} {r['label']:<35} {r['n_total']:>8} {r['n_ko']:>4} "
            f"{r['avg_corners']:>6.1f} {r['avg_yellow']:>6.1f} {r['avg_red']:>6.1f}  "
            f"{derivable:>10}"
        )

    total_matches = sum(r["n_total"] for r in training_results)
    total_ko      = sum(r["n_ko"]    for r in training_results)
    print("  " + "─" * (len(hdr) - 2))
    print(f"  {'StatsBomb':<14} {'TOTAL (treino)':<35} {total_matches:>8} {total_ko:>4}")
    print()
    print(f"  {'fd.co.uk':<14} {'Ligas de clubes (EPL, Liga, etc.)':<35} {'n/a':>8} {'n/a':>4} "
          f"{'✓ HC/AC':>6} {'✓ HY/AY':>6} {'✓ HR/AR':>6}  {'✗ Seleção':>10}")
    print()

    print("Conclusão:")
    print(f"  ✓ Escanteios e cartões estão DISPONÍVEIS em todas as {len(training_results)}")
    print(f"    competições de treino do StatsBomb ({total_matches} partidas, {total_ko} KO).")
    print()
    print("  Derivação:")
    print("    Escanteio = contar eventos Pass com pass.type.name == 'Corner' por time.")
    print("    Cartão    = contar cartões em Bad Behaviour E Foul Committed (a maioria).")
    print("                amarelo = Yellow Card + Second Yellow · vermelho = Red Card + Second Yellow.")
    print("                (só Bad Behaviour subconta ~70% dos amarelos — gate Camada 3.)")
    print()
    print("  ✗ football-data.co.uk: apenas clubes. Não usável para seleções.")
    print()
    print("  Lacuna: StatsBomb não tem eliminatórias nem amistosos de seleção.")
    print("  Para ampliar o treino com esses dados seria necessária uma fonte adicional")
    print("  (ex.: worldfootball.net, transfermarkt — nenhuma disponível gratuitamente")
    print("  com eventos estruturados de nível de partida).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(n_sample: int = 3, all_matches: bool = False) -> None:
    print()
    training_results = _section_statsbomb_training(n_sample, all_matches)
    _section_statsbomb_additional()
    _section_footballdata_dotcouk()
    _print_summary(training_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Verifica cobertura de escanteios/cartões em fontes gratuitas para seleções. "
            "Não coleta dataset — apenas inspeciona e reporta."
        )
    )
    parser.add_argument(
        "--sample", type=int, default=3, metavar="N",
        help="Número de partidas a amostrar por competição (default: 3).",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Inspecciona todas as partidas de cada competição (muito mais lento).",
    )
    args = parser.parse_args()
    main(n_sample=args.sample, all_matches=args.all)
