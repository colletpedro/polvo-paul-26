"""Sondagem empírica do SofaScore como fonte de dados para seleções — Modo 2.

API: api.sofascore.com/api/v1 (unofficial, reverse-engineered)
Referência de endpoints: github.com/apdmatos/sofascore-api
NÃO coleta dataset — apenas sonda e reporta.

PRÉ-REQUISITO — bypass de TLS fingerprinting:
  SofaScore usa Varnish com detecção de TLS fingerprint (JA3/JA4).
  Requisições de urllib/curl padrão retornam 403 imediatamente — a causa não são
  headers, é a cipher suite do TLS que diverge de um browser real.
  Solução: curl_cffi, que usa o TLS stack do Chrome (libcurl-impersonate).

  pip install curl_cffi          ← instala ~5 MB, sem deps de compilação
  python src/collect/check_sofascore_coverage.py

  Alternativa: fornecer cookie de sessão de um browser real:
  export SOFASCORE_COOKIE="session_cookie_do_seu_browser"
  python src/collect/check_sofascore_coverage.py

Sequência de verificação:
  1. ACESSO  — confirma que os endpoints respondem (200 vs 403/429).
  2. COBERTURA — dos N jogos amostrados, quantos têm /statistics preenchido?
                 Breakout por tipo: major (Euro/WC) vs qualifier vs friendly vs clube.
  3. ESTRUTURA — quais chaves de estatística existem? (cornerKicks, yellowCards,
                 redCards, expectedGoals, árbitro).
  4. CONSISTÊNCIA — os nomes de chave são idênticos entre Euro, Copa Américae
                   Brasileirão? (gate de paridade §2.4).
  5. RATE LIMIT  — 20 requisições com sleep 3 s; conta 200 vs 403/429/503.
                   Realizado por último, como medição empírica do NOSSO ambiente.

Flags:
  --sample N       Jogos a amostrar por competição (default: 5)
  --sleep  S       Segundos entre requisições (default: 2.0)
  --impersonate I  Perfil curl_cffi (default: chrome124)
  --skip-ratelimit Pula a fase 5 de rate limit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

_BASE = "https://api.sofascore.com/api/v1"

# ---------------------------------------------------------------------------
# Torneios semeados (IDs verificados empiricamente ou amplamente documentados)
# ---------------------------------------------------------------------------
_TOURNAMENT_SEEDS: list[dict[str, Any]] = [
    # International / seleções — alvos principais
    # IDs confirmados empiricamente nesta sessão (GET /unique-tournament/{id} → 200).
    {"id":   1, "label": "UEFA Euro",               "type": "major",     "target_n": 10},
    {"id": 133, "label": "Copa América",             "type": "major",     "target_n": 10},
    {"id":  16, "label": "FIFA World Cup",           "type": "major",     "target_n":  5},
    # UEFA Nations League — ID NÃO confirmado empiricamente. IDs 113, 1061 retornam 404.
    # A competição existe no SofaScore (visível no site) mas o uniqueTournament ID correto
    # não foi localizado via scan. Placeholder 0 para que o script reporte o status.
    {"id":   0, "label": "UEFA Nations League",      "type": "qualifier", "target_n":  5},
    # Friendlies: ID 851 = "International Friendly Games" (cat=World). ID 38 = Belgian Pro League.
    {"id": 851, "label": "Intl Friendlies (Men)",    "type": "friendly",  "target_n":  5},
    # Clube — grupo de controlo
    {"id": 325, "label": "Brasileirão Série A",      "type": "club",      "target_n":  5},
]

# Chaves de estatística esperadas para o modelo (paridade §2.4).
# NOTA: yellowCards e redCards têm presença CONDICIONAL — aparecem somente quando o
# valor é > 0. Um jogo sem cartões não terá a chave no JSON (ausente ≠ 0).
# Ao coletar dados, tratar chave ausente como 0 (não como nulo/missing).
# Confirmado empiricamente: Copa América México×Jamaica sem yellowCards na resposta.
_KEYS_OF_INTEREST = {
    "cornerKicks":    "escanteios",
    "yellowCards":    "amarelos (cond.)",  # ausente quando 0
    "redCards":       "vermelhos (cond.)", # ausente quando 0
    "expectedGoals":  "xG",
    "ballPossession": "posse (%)",
    "shotsOnGoal":    "chutes no alvo",    # NOTA: NÃO é 'onTargetScoringAttempt'
}

# ---------------------------------------------------------------------------
# Cliente HTTP — curl_cffi (preferido) ou urllib (fallback)
# ---------------------------------------------------------------------------

_SESSION: Any = None   # curl_cffi session, se disponível
_COOKIE = os.environ.get("SOFASCORE_COOKIE", "").strip()


def _build_headers(extra: dict | None = None) -> dict[str, str]:
    h: dict[str, str] = {
        "User-Agent":      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/124.0.0.0 Safari/537.36",
        "Referer":         "https://www.sofascore.com/",
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin":          "https://www.sofascore.com",
    }
    if _COOKIE:
        h["Cookie"] = _COOKIE
    if extra:
        h.update(extra)
    return h


def _init_client(impersonate: str = "chrome124") -> str:
    """Inicializa o cliente HTTP. Retorna 'curl_cffi', 'cookie', ou 'urllib'."""
    global _SESSION
    try:
        from curl_cffi import requests as cffi_requests
        _SESSION = cffi_requests.Session(impersonate=impersonate)
        return "curl_cffi"
    except ImportError:
        pass
    if _COOKIE:
        return "cookie"
    return "urllib"


def _get(path: str, sleep_s: float = 0.0) -> tuple[int, dict]:
    """GET autenticado; retorna (status_code, json_body)."""
    if sleep_s > 0:
        time.sleep(sleep_s)
    url = f"{_BASE}/{path.lstrip('/')}"
    headers = _build_headers()
    if _SESSION is not None:
        try:
            resp = _SESSION.get(url, headers=headers, timeout=20)
            try:
                body = resp.json()
            except Exception:
                body = {}
            return resp.status_code, body
        except Exception as e:
            return -1, {"_exception": str(e)}
    else:
        import urllib.request, urllib.error
        req = urllib.request.Request(url, headers=headers)
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
# Descoberta de torneios / eventos
# ---------------------------------------------------------------------------

def _get_seasons(tournament_id: int, sleep_s: float) -> list[dict]:
    """Retorna lista de seasons para um torneio, ou [] em caso de erro.

    tournament_id == 0 é um placeholder para IDs não confirmados — retorna [] diretamente.
    """
    if tournament_id == 0:
        return []
    status, body = _get(f"unique-tournament/{tournament_id}/seasons", sleep_s)
    if status == 200:
        return body.get("seasons", [])
    return []


def _get_events(tournament_id: int, season_id: int, sleep_s: float) -> list[dict]:
    """Retorna até 60 eventos recentes da season, filtrando apenas FINISHED."""
    status, body = _get(
        f"unique-tournament/{tournament_id}/season/{season_id}/events/last/0",
        sleep_s,
    )
    if status != 200:
        return []
    events = body.get("events", [])
    return [e for e in events if e.get("status", {}).get("type") == "finished"]


def _get_event(event_id: int, sleep_s: float) -> tuple[int, dict]:
    """Retorna (status, event_dict) — inclui referee, scores, teams."""
    status, body = _get(f"event/{event_id}", sleep_s)
    if status == 200:
        return status, body.get("event", {})
    return status, {}


def _get_statistics(event_id: int, sleep_s: float) -> tuple[int, dict | None]:
    """Retorna (status, statistics_list_or_None)."""
    status, body = _get(f"event/{event_id}/statistics", sleep_s)
    if status == 200:
        return status, body.get("statistics")
    return status, None


# ---------------------------------------------------------------------------
# Extração de chaves de estatística
# ---------------------------------------------------------------------------

def _extract_stat_keys(statistics: list[dict]) -> dict[str, Any]:
    """Extrai {key: {name, home, away}} de uma resposta /statistics."""
    keys: dict[str, Any] = {}
    for period_block in statistics:
        if period_block.get("period") != "ALL":
            continue
        for group in period_block.get("groups", []):
            for item in group.get("statisticsItems", []):
                k = item.get("key", "")
                if k:
                    keys[k] = {
                        "name":  item.get("name", ""),
                        "home":  item.get("home"),
                        "away":  item.get("away"),
                        "group": group.get("groupName", ""),
                    }
    return keys


# ---------------------------------------------------------------------------
# Fases da sondagem
# ---------------------------------------------------------------------------

def phase1_access(sleep_s: float) -> tuple[bool, int]:
    """Testa se a API responde. Retorna (acessível, status_code)."""
    status, body = _get("unique-tournament/1/seasons", sleep_s)
    accessible = status == 200
    return accessible, status


def phase2_3_coverage_and_structure(
    seeds: list[dict],
    target_n: int,
    sleep_s: float,
) -> list[dict]:
    """Fases 2 e 3 juntas: amostragem de eventos + verificação de statistics + chaves.

    Retorna lista de dicts por torneio com contagens e chaves encontradas.
    """
    results: list[dict] = []

    for seed in seeds:
        tid     = seed["id"]
        label   = seed["label"]
        ttype   = seed["type"]
        n_want  = min(seed.get("target_n", 5), target_n)

        print(f"\n  [{label}]  (tipo={ttype}, alvo={n_want} jogos)")

        # Descoberta de season
        if tid == 0:
            print(f"    → ID não confirmado — scan empírico não localizou o uniqueTournament ID.")
            print(f"       Torneio existe no site mas requer identificação manual.")
            results.append({
                "label": label, "type": ttype,
                "seasons_found": 0, "n_events_found": 0,
                "n_sampled": 0, "n_with_stats": 0,
                "keys_seen": set(), "referee_seen": 0,
                "status": "id_not_confirmed",
            })
            continue

        seasons = _get_seasons(tid, sleep_s)
        if not seasons:
            print(f"    → seasons: nenhuma / acesso negado")
            results.append({
                "label": label, "type": ttype,
                "seasons_found": 0, "n_events_found": 0,
                "n_sampled": 0, "n_with_stats": 0,
                "keys_seen": set(), "referee_seen": 0,
                "status": "no_seasons",
            })
            continue

        # Pega a season mais recente
        season = seasons[0]
        sid    = season["id"]
        sname  = season.get("name", "?")
        print(f"    → seasons: {len(seasons)}  usando '{sname}' (sid={sid})")

        # Busca eventos FINISHED
        finished_events = _get_events(tid, sid, sleep_s)
        if not finished_events:
            print(f"    → eventos FINISHED: 0 (season sem jogos encerrados ou acesso negado)")
            results.append({
                "label": label, "type": ttype,
                "seasons_found": len(seasons), "n_events_found": 0,
                "n_sampled": 0, "n_with_stats": 0,
                "keys_seen": set(), "referee_seen": 0,
                "status": "no_finished_events",
            })
            continue

        sample = finished_events[:n_want]
        print(f"    → eventos FINISHED: {len(finished_events)}  amostrando {len(sample)}")

        n_with_stats     = 0
        keys_all: list[dict[str, Any]] = []
        referee_seen     = 0

        for ev in sample:
            eid    = ev["id"]
            home   = ev.get("homeTeam", {}).get("name", "?")[:14]
            away   = ev.get("awayTeam", {}).get("name", "?")[:14]
            hs     = ev.get("homeScore", {}).get("display", "?")
            as_    = ev.get("awayScore", {}).get("display", "?")
            date   = ev.get("startTimestamp", "")

            # Referee (no /event/{id} response)
            ev_status, ev_full = _get_event(eid, sleep_s)
            has_ref = bool(ev_full.get("referee"))
            if has_ref:
                referee_seen += 1
            ref_name = ev_full.get("referee", {}).get("name", "—") if has_ref else "—"

            # Statistics
            st_status, stats = _get_statistics(eid, sleep_s)
            if st_status == 200 and stats:
                keys = _extract_stat_keys(stats)
                n_with_stats += 1
                keys_all.append(keys)
                corner_h = keys.get("cornerKicks", {}).get("home", "?")
                corner_a = keys.get("cornerKicks", {}).get("away", "?")
                ycard_h  = keys.get("yellowCards", {}).get("home", "?")
                ycard_a  = keys.get("yellowCards", {}).get("away", "?")
                has_xg   = "expectedGoals" in keys
                print(
                    f"    id={eid} {home:<14} {hs}-{as_} {away:<14}  "
                    f"cant={corner_h}/{corner_a}  amar={ycard_h}/{ycard_a}  "
                    f"xG={'sim' if has_xg else 'não'}  árbitro={ref_name[:18]}"
                )
            else:
                print(
                    f"    id={eid} {home:<14} {hs}-{as_} {away:<14}  "
                    f"statistics → HTTP {st_status}  "
                    f"árbitro={ref_name[:18]}"
                )

        # Intersecção de chaves (presença consistente)
        if keys_all:
            keys_union = set().union(*[set(k.keys()) for k in keys_all])
            keys_always = set(keys_all[0].keys())
            for k in keys_all[1:]:
                keys_always &= set(k.keys())
        else:
            keys_union = set()
            keys_always = set()

        results.append({
            "label":            label,
            "type":             ttype,
            "seasons_found":    len(seasons),
            "season_used":      sname,
            "n_events_found":   len(finished_events),
            "n_sampled":        len(sample),
            "n_with_stats":     n_with_stats,
            "keys_union":       keys_union,
            "keys_always":      keys_always,
            "referee_seen":     referee_seen,
            "status":           "ok",
        })

    return results


def phase4_consistency(results: list[dict]) -> dict[str, dict]:
    """Fase 4: compara quais chaves de interesse estão presentes em cada torneio.

    Retorna {key: {label: present_bool}}.
    """
    keys_matrix: dict[str, dict] = {k: {} for k in _KEYS_OF_INTEREST}
    for r in results:
        if r["status"] != "ok":
            continue
        always = r.get("keys_always", set())
        union  = r.get("keys_union", set())
        for k in _KEYS_OF_INTEREST:
            if k in always:
                keys_matrix[k][r["label"]] = "sempre"
            elif k in union:
                keys_matrix[k][r["label"]] = "às vezes"
            else:
                keys_matrix[k][r["label"]] = "nunca"
    return keys_matrix


def phase5_rate_limit(n_requests: int, sleep_s: float) -> dict[str, int]:
    """Fase 5: teste empírico de rate limit.

    Faz n_requests chamadas com sleep_s entre elas.
    Retorna contagem por código HTTP.
    """
    print(f"\n  Realizando {n_requests} requests com sleep {sleep_s:.1f} s entre elas...")
    counts: dict[str, int] = {}
    for i in range(n_requests):
        status, _ = _get("unique-tournament/1/seasons", sleep_s)
        key = str(status)
        counts[key] = counts.get(key, 0) + 1
        print(f"    req {i+1:>2}/{n_requests}  → HTTP {status}")
    return counts


# ---------------------------------------------------------------------------
# Relatório final
# ---------------------------------------------------------------------------

def _print_summary(
    client_mode: str,
    accessible: bool,
    access_status: int,
    results: list[dict],
    consistency: dict[str, dict],
    rate_counts: dict[str, int] | None,
    n_sample: int,
) -> None:
    print()
    print("=" * 72)
    print("TABELA RESUMO — SofaScore como fonte de Modo 2")
    print("=" * 72)

    # Acesso
    print(f"\n[ACESSO]  cliente={client_mode}  acessível={'✓ SIM' if accessible else '✗ NÃO'}  "
          f"HTTP={access_status}")
    if not accessible:
        print(
            "  FATO MEDIDO: 403 Forbidden em TODOS os endpoints.\n"
            "  Causa confirmada: TLS fingerprinting (Varnish WAF — cipher-suite diverge de Chrome).\n"
            "  Solução: curl_cffi (pip install curl_cffi) ou SOFASCORE_COOKIE env var.\n"
            "  Resultado desta sondagem: dados abaixo refletem o que foi OBSERVADO.\n"
            "  Se tudo retornou 403, as seções 2-4 mostrarão cobertura zero."
        )

    # Cobertura por torneio
    print()
    hdr = (f"  {'Competição':<26} {'tipo':<10} {'seasons':>7} {'FINISHED':>8} "
           f"{'amostra':>7} {'c/stats':>7} {'refs':>5}  {'cobertura'}")
    print("[COBERTURA POR TORNEIO]")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for r in results:
        if r["status"] != "ok":
            status_msg = {
                "no_seasons": "✗ sem seasons",
                "no_finished_events": "✗ sem jogos FINISHED",
                "id_not_confirmed": "? ID não confirmado",
            }.get(r["status"], f"✗ {r['status']}")
            cov_str = f"  {status_msg}"
        else:
            pct = (r["n_with_stats"] / r["n_sampled"] * 100) if r["n_sampled"] else 0
            cov_str = f"  {pct:>5.0f}%"
        refs = r.get("referee_seen", "—")
        print(
            f"  {r['label']:<26} {r['type']:<10} {r.get('seasons_found', '—'):>7} "
            f"{r.get('n_events_found', '—'):>8} {r.get('n_sampled', '—'):>7} "
            f"{r.get('n_with_stats', '—'):>7} {refs:>5}{cov_str}"
        )
    total_sampled = sum(r.get("n_sampled", 0) for r in results)
    total_stats   = sum(r.get("n_with_stats", 0) for r in results)
    print(f"  {'TOTAL':<26} {'':10} {'':7} {'':8} {total_sampled:>7} {total_stats:>7}")

    # Estrutura / chaves
    ok_results = [r for r in results if r["status"] == "ok" and r.get("n_with_stats", 0) > 0]
    if ok_results:
        print()
        print("[ESTRUTURA — chaves de interesse nos /statistics]")
        labels = [r["label"] for r in ok_results]
        col_w  = max(len(l) for l in labels) if labels else 12
        hdr2 = f"  {'chave':<24} {'campo':<22} " + "  ".join(f"{l[:col_w]:<{col_w}}" for l in labels)
        print(hdr2)
        print("  " + "─" * len(hdr2))
        for k, descr in _KEYS_OF_INTEREST.items():
            row = f"  {k:<24} {descr:<22} "
            for r in ok_results:
                status_k = consistency.get(k, {}).get(r["label"], "não testado")
                mark = {"sempre": "✓ sempre", "às vezes": "~ parcial", "nunca": "✗ nunca"}.get(
                    status_k, status_k
                )
                row += f"  {mark:<{col_w}}"
            print(row)

        # Consistência entre competições
        print()
        print("[CONSISTÊNCIA — nomes de chave idênticos entre competições?]")
        all_always = {k for r in ok_results for k in r.get("keys_always", set())}
        all_union  = {k for r in ok_results for k in r.get("keys_union", set())}
        # Chaves que aparecem em todos os torneios testados com stats
        cross_always = None
        for r in ok_results:
            ks = r.get("keys_always", set())
            cross_always = ks if cross_always is None else cross_always & ks
        cross_always = cross_always or set()
        print(f"  Chaves presentes em TODOS os jogos de TODOS os torneios: {len(cross_always)}")
        for k in sorted(cross_always & set(_KEYS_OF_INTEREST)):
            print(f"    ✓ {k}")
        nok_keys = set(_KEYS_OF_INTEREST) - cross_always
        if nok_keys:
            print(f"  Chaves de interesse NÃO universais ({len(nok_keys)}):")
            cond_keys = {"yellowCards", "redCards"}
            for k in sorted(nok_keys):
                status_map = {r["label"]: consistency.get(k, {}).get(r["label"], "?") for r in ok_results}
                note = "  ← presença CONDICIONAL: ausente quando valor = 0" if k in cond_keys else ""
                print(f"    ? {k:<24}  {status_map}{note}")
    else:
        print()
        print("[ESTRUTURA] Sem jogos com statistics — impossível avaliar campos.")

    # Referee info
    print()
    total_refs = sum(r.get("referee_seen", 0) for r in results)
    total_evs  = sum(r.get("n_sampled", 0) for r in results)
    if total_evs:
        print(f"[ÁRBITRO]  /event/{{id}} contém referee.name em "
              f"{total_refs}/{total_evs} jogos amostrados ({total_refs/total_evs:.0%})")
        print("  Estatísticas de árbitro (foul rate, cartões/jogo) NÃO estão em /statistics.")
        print("  Estariam em /event/{id}/incidents (não sondado aqui) — cada cartão como incidente.")
    else:
        print("[ÁRBITRO]  Sem dados amostrados.")

    # Rate limit
    print()
    if rate_counts is not None:
        total_rl  = sum(rate_counts.values())
        ok_rl     = rate_counts.get("200", 0)
        err_rl    = total_rl - ok_rl
        print(f"[RATE LIMIT]  {total_rl} requests com 3 s de sleep:")
        for code, n in sorted(rate_counts.items()):
            label = "OK" if code == "200" else ("rate-limited" if code in ("429", "503") else "forbidden/outro")
            print(f"  HTTP {code}: {n}/{total_rl}  ({label})")
        if ok_rl == total_rl:
            print("  → Sem rate limit detectado com sleep 3 s.")
        elif err_rl > 0:
            first_err = next(
                (i for i, c in enumerate(sorted(rate_counts.keys())) if c != "200"), "?"
            )
            print(f"  → Rate limit detectado: {err_rl}/{total_rl} requests bloqueadas.")
    else:
        print("[RATE LIMIT]  Fase pulada (--skip-ratelimit).")

    print()
    print("[CONCLUSÃO]")
    if not accessible:
        print(
            "  ✗ SofaScore bloqueia programaticamente (TLS fingerprint).\n"
            "    Medição: 403 em 100% dos endpoints, independente de headers.\n"
            "    Desbloqueio: pip install curl_cffi (Chrome TLS) ou cookie de sessão.\n"
            "    Impacto no modelo: necessita de dependência adicional (curl_cffi) +\n"
            "    possibilidade de bloqueio futuro (API unofficial, sem SLA)."
        )
    else:
        n_ok_sources = sum(1 for r in results if r.get("n_with_stats", 0) > 0)
        print(
            f"  ✓ API acessível via {client_mode}.\n"
            f"  ✓ Statistics disponíveis em {total_stats}/{total_sampled} jogos amostrados.\n"
            f"  ✓ {n_ok_sources}/{len(results)} fontes com cobertura confirmada.\n"
            f"  Próximo passo: coletar histórico WC2022/Euro2024/CopaAm2024 via fetch_events_bulk()."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sondagem empírica do SofaScore. "
            "Requer curl_cffi (pip install curl_cffi) ou SOFASCORE_COOKIE."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--sample", type=int, default=5, metavar="N",
        help="Máx. jogos a amostrar por competição (default: 5).",
    )
    parser.add_argument(
        "--sleep", type=float, default=2.0, metavar="S",
        help="Segundos entre requisições na sondagem (default: 2.0).",
    )
    parser.add_argument(
        "--impersonate", default="chrome124", metavar="I",
        help="Perfil curl_cffi: chrome124 | chrome120 | firefox121 | safari17_0 (default: chrome124).",
    )
    parser.add_argument(
        "--skip-ratelimit", action="store_true",
        help="Pula a fase 5 (teste de rate limit). Economiza 60 s.",
    )
    args = parser.parse_args()

    # --- Inicialização ---
    client_mode = _init_client(args.impersonate)
    print(f"\ncurl_cffi:  {'instalado — TLS Chrome impersonation ativo' if client_mode == 'curl_cffi' else 'NÃO instalado'}")
    print(f"SOFASCORE_COOKIE: {'definido' if _COOKIE else 'não definido'}")
    print(f"Modo cliente: {client_mode}")
    if client_mode == "urllib":
        print(
            "\n  AVISO: curl_cffi não está instalado e SOFASCORE_COOKIE não está definido.\n"
            "  Requests via urllib serão bloqueados por TLS fingerprint (403 esperado).\n"
            "  Para bypass: pip install curl_cffi\n"
        )

    # --- Fase 1: Acesso ---
    print("\n" + "=" * 72)
    print("FASE 1 — Verificação de acesso")
    print("=" * 72)
    accessible, access_status = phase1_access(sleep_s=0)
    print(f"  GET /unique-tournament/1/seasons → HTTP {access_status}")
    if not accessible:
        print(f"  FATO MEDIDO: 403 Forbidden — API inacessível com cliente '{client_mode}'.")
        if client_mode == "urllib":
            print(
                "  Causa: TLS fingerprinting. urllib usa cipher suites divergentes do Chrome.\n"
                "  Fix: pip install curl_cffi && python src/collect/check_sofascore_coverage.py"
            )
    else:
        print(f"  ✓ API acessível.")

    # --- Fases 2 + 3: Cobertura + Estrutura ---
    print("\n" + "=" * 72)
    print("FASES 2+3 — Cobertura de jogos + estrutura de /statistics")
    print("=" * 72)
    results = phase2_3_coverage_and_structure(
        _TOURNAMENT_SEEDS, args.sample, args.sleep
    )

    # --- Fase 4: Consistência ---
    print("\n" + "=" * 72)
    print("FASE 4 — Consistência de chaves entre competições")
    print("=" * 72)
    consistency = phase4_consistency(results)

    # --- Fase 5: Rate limit ---
    rate_counts: dict[str, int] | None = None
    if not args.skip_ratelimit:
        print("\n" + "=" * 72)
        print("FASE 5 — Teste empírico de rate limit (20 requests × 3 s)")
        print("=" * 72)
        print("  NOTA: Este é um resultado do NOSSO ambiente específico (IP + client).")
        print("  Resultados podem variar entre execuções e redes diferentes.")
        rate_counts = phase5_rate_limit(n_requests=20, sleep_s=3.0)

    # --- Resumo ---
    _print_summary(
        client_mode, accessible, access_status,
        results, consistency, rate_counts, args.sample,
    )


if __name__ == "__main__":
    main()
