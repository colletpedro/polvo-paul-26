"""Manifest + hash do cache do coletor SofaScore — detecção de mudança/corrupção.

O cache `data/raw/sofascore/hist_*.jsonl` é o insumo REAL do treino do Modo 2, NÃO é
git-tracked (`.gitignore`: `data/raw/`), e depende de uma API não-oficial sem contrato
de estabilidade. Este manifest congela o estado conhecido-bom (recém-auditado 394/394
exato vs API) para DETECÇÃO de mudança/corrupção — não é recuperação (snapshot Parquet
fica adiado, §decisão). O manifest em si É git-tracked (fica em `data/reference/`, a
exceção do `.gitignore`).

Determinismo: o sha256 é do ARQUIVO como está no disco (bytes crus, sem normalizar). O
coletor faz append na ordem de coleta e o backfill reescreve preservando a ordem — se uma
re-coleta legítima mudar a ordem/conteúdo, gera-se um manifest NOVO conscientemente via
--generate. O manifest protege ESTE estado auditado, não um estado "canônico" abstrato.
Para diagnóstico fino, além do hash guarda-se a lista ordenada de event_ids por season —
assim --verify distingue "mudou o quê" (jogos +/−, ou só re-escrita) de um mero "mudou".

Uso:
  python src/collect/manifest.py --generate     # escreve data/reference/sofascore_manifest.json
  python src/collect/manifest.py --verify        # compara disco × manifest; exit != 0 se divergir

Flags:
  --dir DIR         Diretório do cache a varrer (default: data/raw/sofascore). Permite
                    apontar para uma cópia temporária (teste).
  --manifest PATH   Caminho do manifest (default: data/reference/sofascore_manifest.json).
  --annotation STR  Anotação livre gravada no manifest (default: nota de auditoria de hoje).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_DEFAULT_CACHE_DIR = _ROOT / "data" / "raw" / "sofascore"
_DEFAULT_MANIFEST = _ROOT / "data" / "reference" / "sofascore_manifest.json"
_GLOB = "hist_*.jsonl"


# ---------------------------------------------------------------------------
# Leitura de um arquivo hist_*.jsonl → entrada do manifest
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    """sha256 dos BYTES CRUS do arquivo (streaming, sem carregar tudo em memória)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_entry(path: Path) -> dict:
    """Entrada do manifest para um arquivo: sha256, nº de linhas (jogos), event_ids ordenados."""
    n_lines = 0
    ids: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_lines += 1
            try:
                ids.append(int(json.loads(line)["event_id"]))
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                pass  # linha malformada: conta em n_lines mas não em event_ids (delta acusa)
    return {"sha256": _sha256(path), "n_lines": n_lines, "event_ids": sorted(ids)}


def _scan(cache_dir: Path) -> dict[str, dict]:
    """Varre hist_*.jsonl (ordenado por nome) → {nome: entry}."""
    return {p.name: _file_entry(p) for p in sorted(cache_dir.glob(_GLOB))}


# ---------------------------------------------------------------------------
# Geração
# ---------------------------------------------------------------------------

def generate(cache_dir: Path, manifest_path: Path, annotation: str) -> dict:
    files = _scan(cache_dir)
    if not files:
        raise SystemExit(f"✗ Nenhum {_GLOB} encontrado em {cache_dir}")
    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "annotation": annotation,
        "cache_dir": str(cache_dir),
        "glob": _GLOB,
        "total_games": sum(e["n_lines"] for e in files.values()),
        "n_files": len(files),
        "files": files,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# Verificação
# ---------------------------------------------------------------------------

def verify(cache_dir: Path, manifest_path: Path) -> tuple[dict, list[str]]:
    """Compara disco × manifest. Retorna (manifest, lista de problemas). Vazia = OK."""
    if not manifest_path.exists():
        return {}, [f"MANIFEST AUSENTE: {manifest_path}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected: dict[str, dict] = manifest.get("files", {})
    current = _scan(cache_dir)
    problems: list[str] = []

    for name in expected:
        if name not in current:
            problems.append(f"ARQUIVO FALTANDO no disco: {name}")
    for name in current:
        if name not in expected:
            problems.append(f"ARQUIVO NOVO (fora do manifest): {name}")

    for name, exp in expected.items():
        cur = current.get(name)
        if cur is None:
            continue
        if cur["sha256"] == exp["sha256"]:
            continue
        problems.append(f"HASH DIVERGE: {name}")
        problems.append(f"    sha256 {exp['sha256'][:16]}… → {cur['sha256'][:16]}…")
        if cur["n_lines"] != exp["n_lines"]:
            problems.append(f"    n_lines {exp['n_lines']} → {cur['n_lines']}")
        se, sc = set(exp["event_ids"]), set(cur["event_ids"])
        added, removed = sorted(sc - se), sorted(se - sc)
        if added:
            problems.append(f"    event_ids ADICIONADOS ({len(added)}): "
                            f"{added[:20]}{' …' if len(added) > 20 else ''}")
        if removed:
            problems.append(f"    event_ids REMOVIDOS ({len(removed)}): "
                            f"{removed[:20]}{' …' if len(removed) > 20 else ''}")
        if not added and not removed:
            problems.append("    mesmos event_ids — só conteúdo/ordem de linha mudou "
                            "(re-escrita/backfill/corrupção de campo)")
    return manifest, problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_manifest_summary(manifest: dict, manifest_path: Path) -> None:
    print(f"  gerado_em: {manifest['generated_at']}")
    print(f"  anotação:  {manifest['annotation']}")
    print(f"  cache_dir: {manifest['cache_dir']}")
    print(f"\n  {'arquivo':<26} {'linhas':>6}  sha256")
    print("  " + "─" * 60)
    for name, e in manifest["files"].items():
        print(f"  {name:<26} {e['n_lines']:>6}  {e['sha256'][:24]}…")
    print("  " + "─" * 60)
    print(f"  {'TOTAL':<26} {manifest['total_games']:>6}  "
          f"({manifest['n_files']} arquivos)")
    print(f"\n  → manifest: {manifest_path}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Manifest+hash do cache SofaScore (detecção de mudança/corrupção).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--generate", action="store_true", help="Gera/atualiza o manifest.")
    p.add_argument("--verify", action="store_true",
                   help="Verifica o cache atual contra o manifest (exit != 0 se divergir).")
    p.add_argument("--dir", default=None, metavar="DIR",
                   help="Diretório do cache (default: data/raw/sofascore).")
    p.add_argument("--manifest", default=None, metavar="PATH",
                   help="Caminho do manifest (default: data/reference/sofascore_manifest.json).")
    p.add_argument("--annotation", default=None, metavar="STR",
                   help="Anotação livre gravada no manifest (--generate).")
    args = p.parse_args()

    cache_dir = Path(args.dir) if args.dir else _DEFAULT_CACHE_DIR
    manifest_path = Path(args.manifest) if args.manifest else _DEFAULT_MANIFEST

    if args.generate == args.verify:  # nenhum ou ambos
        p.print_help()
        print("\n  Escolha exatamente um: --generate OU --verify.")
        return 2

    if args.generate:
        today = datetime.now().date().isoformat()
        annotation = args.annotation or (
            f"auditado 394/394 exato vs API em {today}, via "
            "cross_validation --check-cache --full-stats"
        )
        manifest = generate(cache_dir, manifest_path, annotation)
        print("=== MANIFEST GERADO ===")
        _print_manifest_summary(manifest, manifest_path)
        return 0

    # --verify
    manifest, problems = verify(cache_dir, manifest_path)
    print("=== VERIFICAÇÃO DO CACHE vs MANIFEST ===")
    if manifest:
        print(f"  manifest: {manifest_path}  (gerado_em {manifest.get('generated_at','?')})")
        print(f"  esperado: {manifest.get('total_games','?')} jogos em "
              f"{manifest.get('n_files','?')} arquivos")
    if not problems:
        print("  ✓ OK — cache idêntico ao manifest (hash, linhas e event_ids batem).")
        return 0
    print(f"  ✗ {sum(1 for x in problems if not x.startswith('    '))} divergência(s):")
    for x in problems:
        print(f"  {x}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
