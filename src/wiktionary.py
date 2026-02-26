"""
Module d'accès au Wiktionnaire français via l'API Wikimedia.

Récupère pour un mot : définitions, synonymes, antonymes,
vocabulaire apparenté (dérivés, apparentés) et anagrammes.
"""
import logging
import re
from dataclasses import dataclass, field
import requests

logger = logging.getLogger(__name__)

WIKTIONARY_API = "https://fr.wiktionary.org/w/api.php"
WIKTIONARY_PAGE_URL = "https://fr.wiktionary.org/wiki"


@dataclass
class WiktionaryResult:
    """Résultat d'une recherche Wiktionnaire pour un mot (français)."""

    mot: str
    definitions: list[str] = field(default_factory=list)
    synonymes: list[str] = field(default_factory=list)
    antonymes: list[str] = field(default_factory=list)
    vocabulaire_apparente: list[str] = field(default_factory=list)  # dérivés + apparentés
    anagrammes: list[str] = field(default_factory=list)
    page_url: str = ""
    erreur: str | None = None


def _fetch_wikitext_with_title(mot_norm: str, titre: str) -> tuple[str | None, str | None]:
    """Effectue la requête API avec un titre donné."""
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "titles": titre,
        "format": "json",
    }
    headers = {
        "User-Agent": "DatasetStyle/1.0 (Streamlit app; Wiktionary lookup) requests/"
        + requests.__version__
    }
    try:
        resp = requests.get(WIKTIONARY_API, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return None, "Aucune page trouvée."
        page_id = next(iter(pages))
        page = pages[page_id]
        if int(page_id) == -1:
            return None, f"Le mot « {mot_norm} » n'existe pas dans le Wiktionnaire français."
        revs = page.get("revisions", [])
        if not revs:
            return None, "Page sans contenu."
        wikitext = revs[0].get("slots", {}).get("main", {}).get("*", "")
        return wikitext, None
    except requests.Timeout:
        return None, "Délai dépassé : le Wiktionnaire n'a pas répondu."
    except requests.RequestException as e:
        logger.exception("Erreur réseau Wiktionnaire: %s", e)
        return None, f"Erreur réseau : {e}"


def _fetch_wikitext(mot: str) -> tuple[str | None, str | None]:
    """
    Récupère le wikitext de la page du mot sur fr.wiktionary.org.
    Essaie d'abord le mot tel quel (triste, chien), puis avec première lettre en majuscule.
    """
    mot_norm = mot.strip()
    if not mot_norm:
        return None, "Aucun mot saisi."
    titres = [mot_norm]
    if len(mot_norm) > 1:
        alt = mot_norm[0].upper() + mot_norm[1:].lower()
        if alt != mot_norm:
            titres.append(alt)
    for titre in titres:
        wikitext, err = _fetch_wikitext_with_title(mot_norm, titre)
        if err is None:
            return wikitext, None
        if err and "n'existe pas" not in err:
            return wikitext, err
    return _fetch_wikitext_with_title(mot_norm, mot_norm)


def _extract_french_block(wikitext: str) -> str:
    """Isole le bloc français (== {{langue|fr}} == ...) du wikitext."""
    start = re.search(r"==\s*\{\{langue\|fr\}\}\s*==", wikitext)
    if not start:
        return ""
    begin = start.end()
    next_lang = re.search(r"\n==\s*\{\{langue\|", wikitext[begin:])
    end = begin + next_lang.start() if next_lang else len(wikitext)
    return wikitext[begin:end]


def _strip_wiki_inline(text: str) -> str:
    """Enlève les liens [[...]] et garde le libellé (ou le lien s'il n'y a pas de pipe)."""
    def repl(m: re.Match[str]) -> str:
        content = m.group(1)
        if "|" in content:
            return content.split("|", 1)[1].strip()
        return content
    text = re.sub(r"\[\[([^\]|]+\|[^\]]+)\]\]", repl, text)
    text = re.sub(r"\[\[([^\]]+)\]\]", repl, text)
    # Enlever les templates {{...}} (grossier : tout entre {{ et }})
    text = re.sub(r"\{\{[^}]+\}\}", "", text)
    text = re.sub(r"''+", "", text)
    return text.strip()


def _extract_list_items(block: str) -> list[str]:
    """Extrait les items de listes * [[mot]] ou * [[a]], [[b]] dans un bloc de wikitext."""
    items: list[str] = []
    for m in re.finditer(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]", block):
        word = m.group(1).strip()
        if word and word not in items:
            items.append(word)
    return items


def _extract_section_content(
    wikitext: str, section_pattern: str, level: int = 4
) -> str:
    """
    Trouve une section (ex. {{S|synonymes}}) et retourne son contenu jusqu'à la prochaine section de même niveau.
    level 3 = ===, level 4 = ====
    """
    pattern = re.escape(section_pattern)
    if level == 3:
        regex = re.compile(
            r"===\s*\{\{S\|" + pattern + r"\s*\}\}\s*===\s*\n(.*?)(?=\n===|\Z)",
            re.DOTALL,
        )
    else:
        regex = re.compile(
            r"====\s*\{\{S\|" + pattern + r"\s*\}\}\s*====\s*\n(.*?)(?=\n====|\n===|\Z)",
            re.DOTALL,
        )
    m = regex.search(wikitext)
    return m.group(1).strip() if m else ""


def _extract_definitions(fr_block: str) -> list[str]:
    """Extrait les définitions (lignes # ...) du bloc français, sans les exemples trop longs."""
    definitions: list[str] = []
    for line in fr_block.split("\n"):
        line = line.strip()
        if line.startswith("#"):
            # Enlever le # initial et les #* (sous-exemples)
            if line.startswith("#*"):
                continue
            def_line = line.lstrip("#").strip()
            def_line = _strip_wiki_inline(def_line)
            if len(def_line) > 5:
                definitions.append(def_line)
    return definitions


def _extract_anagrammes(fr_block: str) -> list[str]:
    """Extrait les anagrammes depuis {{anagrammes|lang=fr|mot1|mot2|...}} ou liste * [[mot]]."""
    # Template {{anagrammes|lang=fr|mot1|mot2}}
    m = re.search(r"\{\{anagrammes\|[^|]*\|[^|]*\|([^}]+)\}\}", fr_block)
    if m:
        part = m.group(1)
        return [p.strip() for p in re.split(r"\|", part) if p.strip()]
    # Section anagrammes avec liste * [[mot]]
    section = _extract_section_content(fr_block, r"anagrammes", level=3)
    if section and "{{voir anagrammes" not in section:
        return _extract_list_items(section)
    return []


def fetch_wiktionary(mot: str) -> WiktionaryResult:
    """
    Interroge le Wiktionnaire français et parse le résultat.

    Returns:
        WiktionaryResult avec définitions, synonymes, antonymes,
        vocabulaire apparenté (dérivés + apparentés) et anagrammes.
    """
    mot_norm = mot.strip()
    titre = (
        mot_norm[0].upper() + mot_norm[1:].lower()
        if len(mot_norm) > 1
        else mot_norm.upper()
    )
    result = WiktionaryResult(
        mot=mot_norm,
        page_url=f"{WIKTIONARY_PAGE_URL}/{titre}" if mot_norm else "",
    )

    wikitext, err = _fetch_wikitext(mot_norm)
    if err or not wikitext:
        result.erreur = err or "Contenu vide."
        return result

    fr_block = _extract_french_block(wikitext)
    if not fr_block:
        result.erreur = "Aucune section française trouvée pour ce mot."
        return result

    result.definitions = _extract_definitions(fr_block)

    for section_name, attr in [
        ("synonymes", "synonymes"),
        ("antonymes", "antonymes"),
        ("dérivés", "vocabulaire_apparente"),
        ("apparentés", "vocabulaire_apparente"),
        ("vocabulaire", "vocabulaire_apparente"),
    ]:
        content = _extract_section_content(fr_block, re.escape(section_name), level=4)
        if content:
            items = _extract_list_items(content)
            if attr == "vocabulaire_apparente":
                # Thésaurus : {{voir thésaurus|fr|...}} — on peut ignorer ou extraire le thème
                for item in items:
                    if item not in result.vocabulaire_apparente:
                        result.vocabulaire_apparente.append(item)
            else:
                setattr(result, attr, items)

    # Dédupliquer vocabulaire apparenté
    result.vocabulaire_apparente = list(dict.fromkeys(result.vocabulaire_apparente))

    result.anagrammes = _extract_anagrammes(fr_block)

    return result
