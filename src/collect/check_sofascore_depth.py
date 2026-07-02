"""Sondagem de PROFUNDIDADE HISTÓRICA do SofaScore — Modo 2, passo 0 (BLOQUEADOR #1).

SPEC_MARKETS.md §2.2 (⚠️ NÃO VERIFICADO) e §10 passo 0.

NÃO coleta dataset. Só mapeia o que existe e mede cobertura de /statistics em
temporadas ANTIGAS (< 2024), que é o que decide o volume de treino do Modo 2.

Reutiliza integralmente o padrão de acesso já validado em check_sofascore_coverage.py
(curl_cffi + impersonação de Chrome, bypass do TLS fingerprinting / Varnish).

Duas fases:
  FASE A — DESCOBERTA DE IDs
    Para cada torneio-alvo, localizar o uniqueTournament id (via id conhecido ou
    /search/all?q=...) e listar TODAS as seasons (ano + id) que o SofaScore expõe.
    Saída: tabela (torneio × anos disponíveis).

  FASE B — COBERTURA DE /statistics (só no que a Fase A encontrou)
    Para cada (torneio, temporada) — priorizando temporadas ANTERIORES a 2024 —,
    amostrar N jogos FINISHED e medir:
      % com /event/{id}/statistics preenchido;
      cornerKicks / yellowCards / redCards / expectedGoals presentes?
    Saída: tabela (cobertura por temporada), marcando
    FATO_MEDIDO vs ID_NAO_ENCONTRADO vs SEM_STATISTICS.

Uso:
  .venv/bin/python src/collect/check_sofascore_depth.py
  .venv/bin/python src/collect/check_sofascore_depth.py --sample 5 --depth-seasons 4

Flags:
  --sample N          Jogos a amostrar por temporada na Fase B (default: 5)
  --depth-seasons K   Máx. temporadas < 2024 a sondar por torneio (default: 4)
  --sleep S           Segundos entre requisições (default: 3.0 — rate limit §2.2)
  --impersonate I     Perfil curl_cffi (default: chrome124)
  --include-recent    Também sondar 1 temporada >= 2024 por torneio (comparação)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any

_BASE = "https://api.sofascore.com/api/v1"

# ---------------------------------------------------------------------------
# Torneios-alvo (§10 passo 0). known_id != None => já confirmado empiricamente.
# query => usado em /search/all quando known_id é None (ou para reconfirmar).
# name_must => palavras que o nome do torneio DEVE conter (case-insensitive).
# name_not  => palavras que o eliminam (variantes Women/U17/U20/U23/Qual quando
#             não são o alvo).
# ---------------------------------------------------------------------------
_TARGETS: list[dict[str, Any]] = [
    {"label": "FIFA World Cup",        "known_id": 16,  "query": "world cup",
     "name_must": ["world", "cup"], "name_not": ["women", "u17", "u20", "u23", "qual", "club", "beach", "futsal"]},
    {"label": "UEFA Euro",             "known_id": 1,   "query": "european championship",
     "name_must": ["europ"], "name_not": ["women", "u17", "u19", "u21", "qual", "league"]},
    {"label": "Copa América",          "known_id": 133, "query": "copa america",
     "name_must": ["copa", "america"], "name_not": ["women", "u20", "sub"]},
    {"label": "AFCON",                 "known_id": None, "query": "africa cup of nations",
     "name_must": ["africa", "cup", "nations"], "name_not": ["women", "u17", "u20", "u23", "qual", "championship"]},
    {"label": "AFC Asian Cup",         "known_id": None, "query": "afc asian cup",
     "name_must": ["asian", "cup"], "name_not": ["women", "u17", "u20", "u23", "qual", "beach", "futsal", "west", "east"]},
    {"label": "CONCACAF Gold Cup",     "known_id": None, "query": "concacaf gold cup",
     "name_must": ["gold", "cup"], "name_not": ["women", "u17", "u20", "qual"]},
    {"label": "UEFA Nations League",   "known_id": 10783, "query": "uefa nations league",
     "name_must": ["uefa", "nations", "league"], "name_not": ["women", "concacaf", "qual"]},
    # WCQ: /search/all by "world cup qualification" surfaces FIBA basketball / futsal /
    # rugby FIRST — os ids de FUTEBOL foram confirmados manualmente e fixados aqui
    # (senão o script casa basquete e reporta corner/card = 0 espúrio).
    {"label": "WCQ UEFA (Europe)",     "known_id": 11,  "query": "world cup qual uefa",
     "name_must": ["world cup qual"], "name_not": ["women", "u17", "u20", "fiba"]},
    {"label": "WCQ CONMEBOL (S.Am)",   "known_id": 295, "query": "world cup qual conmebol",
     "name_must": ["world cup qual"], "name_not": ["women", "u17", "u20", "fiba"]},
    {"label": "WCQ CAF (Africa)",      "known_id": 13,  "query": "world cup qual caf",
     "name_must": ["world cup qual"], "name_not": ["women", "u17", "u20", "fiba"]},
    {"label": "Intl Friendlies (Men)", "known_id": 851, "query": "international friendlies",
     "name_must": ["friendl"], "name_not": ["women", "club", "u17", "u20", "u21", "u23"]},
]

# Chaves-alvo do Modo 2. yellowCards/redCards têm presença CONDICIONAL (ausentes
# quando valor == 0). expectedGoals é condicional ao torneio. cornerKicks é o
# sinal-âncora: se cornerKicks está presente, o /statistics é utilizável.
_KEYS_TARGET = ["cornerKicks", "yellowCards", "redCards", "expectedGoals"]

# ---------------------------------------------------------------------------
# Cliente HTTP — curl_cffi (obrigatório para bypass de TLS fingerprint / Varnish)
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
    import urllib.request, urllib.error
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
# FASE A — descoberta de IDs e seasons
# ---------------------------------------------------------------------------

def _year_of(season_name: str) -> int | None:
    """Extrai o ano (o MAIOR ano se houver intervalo, ex. '2018/2019' -> 2019).

    Também aceita o formato '24/25' (Nations League etc.), mapeando 'YY' -> 20YY.
    """
    s = season_name or ""
    years = [int(y) for y in re.findall(r"(?:19|20)\d{2}", s)]
    if not years:
        # formato curto 'YY/YY' ou 'YY' — assume século 2000
        yy = re.findall(r"\b(\d{2})\b", s)
        years = [2000 + int(y) for y in yy]
    return max(years) if years else None


def _search_tournament(target: dict, sleep_s: float) -> tuple[int | None, str]:
    """Localiza o uniqueTournament id via /search/all. Retorna (id, nome_encontrado)."""
    q = target["query"].replace(" ", "%20")
    status, body = _get(f"search/all?q={q}", sleep_s)
    if status != 200:
        return None, f"search HTTP {status}"
    results = body.get("results", body) if isinstance(body, dict) else body
    if not isinstance(results, list):
        return None, "search sem results"

    must = [w.lower() for w in target.get("name_must", [])]
    ban = [w.lower() for w in target.get("name_not", [])]
    for item in results:
        if not isinstance(item, dict):
            continue
        if item.get("type") not in ("uniqueTournament", "tournament"):
            continue
        ent = item.get("entity", {}) or {}
        name = (ent.get("name") or "").lower()
        if not name or ent.get("id") is None:
            continue
        if any(b in name for b in ban):
            continue
        if all(m in name for m in must):
            return int(ent["id"]), ent.get("name", "")
    return None, "sem match em /search/all"


def _confirm_id(tid: int, sleep_s: float) -> tuple[bool, str]:
    """GET /unique-tournament/{id} para confirmar existência. Retorna (ok, nome)."""
    status, body = _get(f"unique-tournament/{tid}", sleep_s)
    if status == 200:
        ut = body.get("uniqueTournament", {}) or {}
        return True, ut.get("name", "?")
    return False, f"HTTP {status}"


def _get_seasons(tid: int, sleep_s: float) -> list[dict]:
    status, body = _get(f"unique-tournament/{tid}/seasons", sleep_s)
    return body.get("seasons", []) if status == 200 else []


def phase_a(targets: list[dict], sleep_s: float) -> list[dict]:
    """Descobre id + lista de seasons por torneio. Retorna lista enriquecida."""
    out: list[dict] = []
    for t in targets:
        label = t["label"]
        print(f"\n  [{label}]")
        tid = t.get("known_id")
        found_name = ""

        if tid is not None:
            ok, name = _confirm_id(tid, sleep_s)
            if ok:
                found_name = name
                print(f"    → id conhecido {tid} confirmado: '{name}'")
            else:
                print(f"    → id conhecido {tid} FALHOU ({name}); tentando /search/all...")
                tid = None

        if tid is None:
            tid, found_name = _search_tournament(t, sleep_s)
            if tid is not None:
                print(f"    → /search/all → id {tid}: '{found_name}'")
            else:
                print(f"    → ID_NAO_ENCONTRADO: {found_name}")

        if tid is None:
            out.append({"label": label, "id": None, "found_name": found_name,
                        "seasons": [], "status": "ID_NAO_ENCONTRADO"})
            continue

        seasons = _get_seasons(tid, sleep_s)
        enriched = []
        for s in seasons:
            enriched.append({
                "sid": s.get("id"),
                "name": s.get("name", "?"),
                "year": _year_of(s.get("year", "") or s.get("name", "")),
            })
        years = sorted({e["year"] for e in enriched if e["year"]}, reverse=True)
        print(f"    → {len(enriched)} seasons. Anos: "
              f"{', '.join(str(y) for y in years) if years else '(sem ano parseável)'}")
        out.append({"label": label, "id": tid, "found_name": found_name,
                    "seasons": enriched, "status": "OK" if enriched else "SEM_SEASONS"})
    return out


# ---------------------------------------------------------------------------
# FASE B — cobertura de /statistics em temporadas antigas
# ---------------------------------------------------------------------------

def _get_finished_events(tid: int, sid: int, sleep_s: float) -> list[dict]:
    status, body = _get(
        f"unique-tournament/{tid}/season/{sid}/events/last/0", sleep_s)
    if status != 200:
        return []
    evs = body.get("events", [])
    return [e for e in evs if e.get("status", {}).get("type") == "finished"]


def _stat_keys(event_id: int, sleep_s: float) -> tuple[int, set[str]]:
    """Retorna (http_status, set de chaves presentes no período ALL)."""
    status, body = _get(f"event/{event_id}/statistics", sleep_s)
    if status != 200:
        return status, set()
    stats = body.get("statistics")
    if not stats:
        return status, set()
    keys: set[str] = set()
    for period in stats:
        if period.get("period") != "ALL":
            continue
        for group in period.get("groups", []):
            for item in group.get("statisticsItems", []):
                k = item.get("key", "")
                if k:
                    keys.add(k)
    return status, keys


def _pick_seasons(seasons: list[dict], depth_k: int, include_recent: bool) -> list[dict]:
    """Seleciona temporadas a sondar: prioriza < 2024 (§10 passo 0).

    Pega as `depth_k` temporadas < 2024 mais RECENTES (as mais próximas de 2024
    são as mais relevantes p/ decidir a fronteira histórica), opcionalmente + 1
    temporada >= 2024 para comparação de referência.
    """
    dated = [s for s in seasons if s["year"]]
    pre = sorted([s for s in dated if s["year"] < 2024], key=lambda s: s["year"], reverse=True)
    picked = pre[:depth_k]
    if include_recent:
        recent = sorted([s for s in dated if s["year"] >= 2024],
                        key=lambda s: s["year"], reverse=True)
        if recent:
            picked = picked + [recent[0]]
    return picked


def phase_b(discovered: list[dict], sample_n: int, depth_k: int,
            include_recent: bool, sleep_s: float) -> list[dict]:
    """Mede cobertura de /statistics por (torneio, temporada). Retorna linhas."""
    rows: list[dict] = []
    for d in discovered:
        if d["status"] != "OK":
            rows.append({"label": d["label"], "season": "—", "year": None,
                         "verdict": "ID_NAO_ENCONTRADO" if d["status"] == "ID_NAO_ENCONTRADO"
                         else "SEM_SEASONS", "n": 0, "with_stats": 0, "key_hits": {}})
            continue

        picks = _pick_seasons(d["seasons"], depth_k, include_recent)
        if not picks:
            rows.append({"label": d["label"], "season": "—", "year": None,
                         "verdict": "SEM_TEMPORADA_ANTIGA", "n": 0, "with_stats": 0, "key_hits": {}})
            continue

        tid = d["id"]
        for s in picks:
            sid, sname, yr = s["sid"], s["name"], s["year"]
            print(f"\n  [{d['label']}] temporada '{sname}' (year={yr}, sid={sid})")
            events = _get_finished_events(tid, sid, sleep_s)
            if not events:
                print(f"    → 0 eventos FINISHED (season sem jogos ou acesso negado)")
                rows.append({"label": d["label"], "season": sname, "year": yr,
                             "verdict": "SEM_EVENTOS", "n": 0, "with_stats": 0, "key_hits": {}})
                continue

            sample = events[:sample_n]
            with_stats = 0
            key_hits = {k: 0 for k in _KEYS_TARGET}
            for ev in sample:
                eid = ev["id"]
                home = ev.get("homeTeam", {}).get("name", "?")[:13]
                away = ev.get("awayTeam", {}).get("name", "?")[:13]
                st, keys = _stat_keys(eid, sleep_s)
                if st == 200 and keys:
                    with_stats += 1
                    hits = []
                    for k in _KEYS_TARGET:
                        if k in keys:
                            key_hits[k] += 1
                            hits.append(k)
                    print(f"    id={eid} {home:<13} vs {away:<13} stats=OK "
                          f"[{', '.join(hits) if hits else 'nenhuma chave-alvo'}]")
                else:
                    print(f"    id={eid} {home:<13} vs {away:<13} stats → HTTP {st} (vazio)")

            n = len(sample)
            verdict = "FATO_MEDIDO" if with_stats > 0 else "SEM_STATISTICS"
            print(f"    → {with_stats}/{n} com /statistics  ({verdict})")
            rows.append({"label": d["label"], "season": sname, "year": yr,
                         "verdict": verdict, "n": n, "with_stats": with_stats,
                         "key_hits": key_hits})
    return rows


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------

def _print_table_a(discovered: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("TABELA A — Torneios × IDs × anos disponíveis  (FASE A)")
    print("=" * 78)
    print(f"  {'Torneio':<22} {'id':>7}  {'#seasons':>8}  anos disponíveis")
    print("  " + "─" * 74)
    for d in discovered:
        if d["id"] is None:
            print(f"  {d['label']:<22} {'—':>7}  {'—':>8}  ID_NAO_ENCONTRADO ({d['found_name']})")
            continue
        years = sorted({e["year"] for e in d["seasons"] if e["year"]}, reverse=True)
        pre = [y for y in years if y < 2024]
        yr_str = ", ".join(str(y) for y in years) if years else "(sem ano parseável)"
        if len(yr_str) > 44:
            yr_str = yr_str[:41] + "..."
        flag = "" if pre else "  ⚠ SÓ >=2024"
        print(f"  {d['label']:<22} {d['id']:>7}  {len(d['seasons']):>8}  {yr_str}{flag}")


def _print_table_b(rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("TABELA B — Cobertura de /statistics por temporada  (FASE B)")
    print("=" * 78)
    kh = "  ".join(f"{k[:6]:>6}" for k in _KEYS_TARGET)
    print(f"  {'Torneio':<22} {'temporada':<16} {'ano':>4} {'stats':>7}  {kh}  veredito")
    print("  " + "─" * 74)
    for r in rows:
        if r["n"] == 0:
            print(f"  {r['label']:<22} {str(r['season'])[:16]:<16} "
                  f"{str(r['year'] or '—'):>4} {'—':>7}  "
                  f"{'':<{len(kh)}}  {r['verdict']}")
            continue
        cov = f"{r['with_stats']}/{r['n']}"
        kcells = "  ".join(f"{r['key_hits'].get(k, 0):>6}" for k in _KEYS_TARGET)
        print(f"  {r['label']:<22} {str(r['season'])[:16]:<16} "
              f"{str(r['year'] or '—'):>4} {cov:>7}  {kcells}  {r['verdict']}")
    print("\n  Legenda: colunas de chave = nº de jogos (dos amostrados) com a chave presente.")
    print("           yellowCards/redCards ausentes quando valor == 0 (tratar ausência = 0).")
    print("           FATO_MEDIDO = >=1 jogo com /statistics · SEM_STATISTICS = 0 jogos ·")
    print("           ID_NAO_ENCONTRADO / SEM_TEMPORADA_ANTIGA / SEM_EVENTOS = não medível.")


def _print_conclusion(discovered: list[dict], rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("[CONCLUSÃO — profundidade histórica (BLOQUEADOR #1)]")
    print("=" * 78)
    measured = [r for r in rows if r["verdict"] == "FATO_MEDIDO"]
    pre_measured = [r for r in measured if r["year"] and r["year"] < 2024]
    total_pre = [r for r in rows if r["year"] and r["year"] < 2024 and r["n"] > 0]
    ids_ok = sum(1 for d in discovered if d["id"] is not None)
    print(f"  Torneios com id resolvido: {ids_ok}/{len(discovered)}")
    if total_pre:
        good = sum(1 for r in pre_measured if r["with_stats"] >= max(1, r["n"] // 2))
        print(f"  Temporadas < 2024 sondadas: {len(total_pre)}  |  "
              f"com /statistics (>=1 jogo): {len(pre_measured)}  |  "
              f"cobertura >=50%: {good}")
    else:
        print("  Nenhuma temporada < 2024 pôde ser medida.")
    print("\n  Interpretação (§10 passo 0):")
    print("   - Boa cobertura em temporadas < 2024  → SofaScore-only viável (paridade trivial).")
    print("   - Cobertura fraca/ausente pré-2024    → treino StatsBomb + inferência SofaScore")
    print("     (exige o gate da Camada 3, §10 passo 1).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Sondagem de profundidade histórica do SofaScore (Modo 2, passo 0).",
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--sample", type=int, default=5, metavar="N",
                   help="Jogos a amostrar por temporada na Fase B (default: 5).")
    p.add_argument("--depth-seasons", type=int, default=4, metavar="K",
                   help="Máx. temporadas < 2024 a sondar por torneio (default: 4).")
    p.add_argument("--sleep", type=float, default=3.0, metavar="S",
                   help="Segundos entre requisições (default: 3.0 — rate limit §2.2).")
    p.add_argument("--impersonate", default="chrome124", metavar="I",
                   help="Perfil curl_cffi (default: chrome124).")
    p.add_argument("--include-recent", action="store_true",
                   help="Também sondar 1 temporada >=2024 por torneio (referência).")
    args = p.parse_args()

    client = _init_client(args.impersonate)
    print(f"Cliente HTTP: {client}", end="")
    if client != "curl_cffi":
        print("  ⚠ curl_cffi ausente — TLS fingerprint bloqueará (403). Instale curl_cffi.")
        if client == "urllib" and not _COOKIE:
            sys.exit(1)
    else:
        print(f"  (impersonate={args.impersonate})")

    # Sanity de acesso
    st, _ = _get("unique-tournament/16/seasons", 0)
    print(f"Sanity /unique-tournament/16/seasons → HTTP {st}")
    if st != 200:
        print("  ✗ Acesso negado — abortando. (Confirme curl_cffi / cookie.)")
        sys.exit(1)

    print("\n" + "=" * 78)
    print("FASE A — Descoberta de IDs e seasons")
    print("=" * 78)
    discovered = phase_a(_TARGETS, args.sleep)

    print("\n" + "=" * 78)
    print("FASE B — Cobertura de /statistics (priorizando temporadas < 2024)")
    print("=" * 78)
    rows = phase_b(discovered, args.sample, args.depth_seasons,
                   args.include_recent, args.sleep)

    _print_table_a(discovered)
    _print_table_b(rows)
    _print_conclusion(discovered, rows)


if __name__ == "__main__":
    main()
