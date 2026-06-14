  # Doedutch Cards — UI Style Guide

A clean, minimal, **Anki‑inspired** interface optimized for clarity, consistency, and accessibility.

---

## 🎨 Color Palette

| Role | Color | Hex | Description |
|------|--------|------|-------------|
| Page background | Light gray | `#f8f8f8` | Neutral background similar to AnkiWeb |
| Primary text | Dark gray | `#333333` | Readable, high contrast text |
| Secondary text | Muted gray-blue | `#6c757d` | Used for captions and notes |
| Accent | Blue | `#3178c6` | Main brand color for buttons and links |
| Hover state | Deep blue | `#255a9b` | Hover/active interaction color |
| Success | Green | `#28a745` | Positive feedback messages |
| Warning | Amber | `#ffc107` | Soft alert or warning highlights |
| Error | Red | `#dc3545` | Critical or error states |
| Panels/cards | White | `#ffffff` | Card and container background |
| Borders/shadows | Light gray | `#e0e0e0` | Separators and soft shadows |

---

## 🔠 Typography

- **Primary interface font:** `'Inter'`, `'Segoe UI'`, sans-serif  
- **Headings font:** `'Nunito Sans'` or `'Poppins'` for a friendly academic tone  
- **Monospace (code, CSV):** `'JetBrains Mono'` or `'Consolas'`  
- Base text size: **16 px**; headings +20–30 % larger

---

## 🧩 Base CSS Example

```css
body {
    background-color: #f8f8f8;
    font-family: 'Inter', 'Segoe UI', sans-serif;
    color: #333333;
}
h1, h2, h3, h4 {
    font-family: 'Nunito Sans', 'Segoe UI', sans-serif;
    color: #3178c6;
}
a, .stButton>button {
    background-color: #3178c6 !important;
    color: white !important;
    border-radius: 6px;
    border: none;
    transition: background-color 0.2s ease;
}
a:hover, .stButton>button:hover {
    background-color: #255a9b !important;
}
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e0e0e0;
}
```

---

## 🧱 Guidelines for Consistency

- Keep layouts **centered** with generous whitespace.  
- Use consistent **border-radius (6–8 px)** for buttons and cards.  
- Avoid strong shadows; use `#e0e0e0` outlines for separation.  
- Limit color accents to buttons, links, and headings.  
- Prefer **light theme** — dark mode optional later.  
- Icons: use emoji or **Lucide icons** for clarity.  
- Ensure **responsive layout** (max width ≤ 1200 px).  
- Maintain visual hierarchy: H1 → 22 px, H2 → 18 px, body → 16 px.  
- Each page: **max 2 accent colors** (blue + green or blue + orange).  

---

## 🗂️ UI Conventions (Tabs + Local Notices)

- Keep top-level navigation tab-based: **Generate / Settings / Admin**.
- In `Generate`, preserve linear visual flow: **Input → Run → Review → Export**.
- Use **one primary CTA** per flow stage; avoid duplicated global action buttons.
- Display feedback locally per context (tab + section), not as global cross-tab banners.
- Recommended notice scopes:
  - Generate: `input`, `run`, `audio`, `export`
  - Settings: `toolbar`, `access`, `generation`, `audio`, `export`
  - Admin: `toolbar`, `invite`, `users`, `usage`
- Keep progress compact by default; expose batch/stage details via explicit toggle.
- For long provider calls, show explicit waiting status (`waiting provider...`) in progress details.
- Keep spacing on 8/12/16/24 scale and avoid oversized gaps between checkbox and label.
- Keep text-card reuse and audio-library reuse visually distinct:
  - Text cards: `reused saved cards`, `saved cards`
  - Audio: `reused audio clips`, `saved audio clips`
- In `Review`, card details may expose JSON diagnostics, but should also provide direct actions next to the card: audio preview/playback for `AudioWord` and `AudioSentence` when available.

---

**Prepared for:** Doedutch Project (Streamlit UI)  
**Design reference:** AnkiWeb aesthetic, adapted under fair use (no logos, code, or trademarks copied).
