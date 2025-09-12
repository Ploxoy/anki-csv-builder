# config/signalword_groups.py
# Signalword groups per CEFR level.
# The structure is:
#   SIGNALWORD_GROUPS = {
#       "group_name": { "A1": [...], "A2": [...], ... },
#       ...
#   }
# Keep this file logic-free: plain data only.

SIGNALWORD_GROUPS = {
    "cause_effect": {
        "A1": ["omdat", "want", "dus"],
        "A2": ["daardoor", "daarom", "zodat"],
        "B1": ["doordat", "waardoor"],
        "B2": ["ten gevolge van", "als gevolg van", "te danken aan", "te wijten aan", "wegens", "vanwege", "aangezien"],
        "C1": ["derhalve", "vandaar dat", "dien ten gevolge", "op grond daarvan"],
        "C2": ["immers", "in zoverre", "zulks omdat", "dit impliceert dat", "uit dien hoofde"],
    },
    "contrast": {
        "A1": ["maar"],
        "A2": ["toch", "of", "ofwel"],
        "B1": ["echter"],
        "B2": ["niettemin", "enerzijds ... anderzijds", "daarentegen", "integendeel", "in tegenstelling tot"],
        "C1": ["desondanks", "nochtans", "hoezeer ook", "ondanks dat"],
        "C2": ["hoe paradoxaal ook", "zij het dat", "al ware het maar", "weliswaar ... maar"],
    },
    "condition_goal": {
        "A1": ["als", "om ... te"],
        "A2": ["wanneer", "tenzij"],
        "B1": ["indien", "mits"],
        "B2": ["opdat", "daartoe", "met als doel", "met behulp van", "door middel van"],
        "C1": ["gesteld dat", "ingeval", "voor zover", "op voorwaarde dat"],
        "C2": ["indien en voorzover", "in de veronderstelling dat", "teneinde", "met het oog op"],
    },
    "example_addition": {
        "A1": ["en", "ook"],
        "A2": ["bijvoorbeeld", "zoals"],
        "B1": ["verder", "bovendien"],
        "B2": ["eveneens", "zowel ... als", "daarnaast", "ten slotte", "onder andere", "ter illustratie", "ter verduidelijking"],
        "C1": ["neem nu", "stel dat", "dat wil zeggen", "met name"],
        "C2": ["te weten", "als zodanig", "zulks ter illustratie", "onder meer ... doch niet uitsluitend"],
    },
    "comparison": {
        "A1": ["zoals", "net als"],
        "A2": ["hetzelfde als"],
        "B1": ["evenals"],
        "B2": ["in vergelijking met", "vergeleken met"],
        "C1": ["analoge wijze", "op gelijke wijze", "evenzeer"],
        "C2": ["mutatis mutandis", "naar analogie van"],
    },
    "summary_conclusion": {
        "A1": ["dus"],
        "A2": ["daarom"],
        "B1": ["kortom"],
        "B2": ["uiteindelijk", "samenvattend", "concluderend", "hieruit volgt", "met andere woorden", "al met al"],
        "C1": ["alles overziend", "alles bijeen genomen", "resumerend"],
        "C2": ["derhalve concluderen wij dat", "dit leidt onvermijdelijk tot de slotsom dat"],
    },
}