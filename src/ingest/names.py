"""Normalización de nombres de selecciones al canon del dataset martj42."""

# Mapea variantes (Transfermarkt, FIFA, eloratings, Wikipedia) -> martj42
ALIASES = {
    "USA": "United States",
    "Korea Republic": "South Korea",
    "Korea, South": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Congo DR": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "China PR": "China",
    "Cabo Verde": "Cape Verde",
    "Serbia and Montenegro": "Serbia",
}


def canon(name: str) -> str:
    if not isinstance(name, str):
        return name
    name = name.strip()
    return ALIASES.get(name, name)
