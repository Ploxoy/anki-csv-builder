from core.signalwords import pick_allowed_for_level
pool = pick_allowed_for_level(CFG_SIGNALWORD_GROUPS, "B2", n=3, seed=42)
print(pool)