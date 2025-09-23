# Ethical Capital — Brand Kit

Use this kit to create on-brand assets for Ethical Capital.

## Brand Overview
- Public name: Ethical Capital
- Legal/disclaimers: Invest Vegan LLC DBA Ethical Capital
- Voice: professional, principled, direct; emphasize “process,” “evidence,” and “ethics”. Do not use “Truth‑Seeking Analysis”.

## Logos
- Primary SVG: logos/ecic-logo.svg
- Alternate: logos/ecic-logo-alt.svg
- Favicon: logos/favicon.svg
- Usage: keep aspect ratio; avoid recolor; ensure clearspace ≥ 1× logo “E” height; min height 24 px (UI), 48–64 px (hero).

## Color & Gradient
Core palette (see styles/ecic-brand.scss and config/_brand.yml):
- Purple (Primary) `#581c87`
- Purple Light `#6b46c1`
- Teal (Accent) `#14b8a6`
- Teal Light (Success) `#2dd4bf`
- Amber (Warning) `#f59e0b`
- Red (Danger) `#ef4444`
- Light Surface `#f9fafb`, Background `#ffffff`
- Text Dark `#111827`, Text Medium `#4b5563`, Border Gray `#e5e7eb`

Brand gradient:
```
linear-gradient(135deg, #14b8a6 0%, #581c87 40%, #6b21a8 100%)
```

## Typography
- Headings: Outfit (Google Fonts)
- Body: Raleway (Google Fonts)
- Headings: bold for H1/H2; Title Case for major headings; line-height 1.35–1.6

## Components
- Buttons: Primary Purple `#581c87` (white text); Outline variant with Purple border/text.
- Cards: White/light backgrounds, 2 px border `#e2e8f0` or 4 px left accent `#581c87`.
- Charts: Strategy line Purple `#581c87`; Benchmark line Neutral Gray `#6b7280`.

## Data/Benchmarks
- Benchmark phrasing: “MSCI ACWI (represented by iShares MSCI ACWI ETF)”
- Income benchmark: “Bloomberg Aggregate Bond Index”
- Performance copy: “Net of fees”; “Time‑Weighted Return (TWR)”; “Representative account basis”; always include “Past performance does not guarantee future results.”

## Copy Patterns
- CTAs: short, benefit‑oriented. Avoid superlatives; prefer evidence‑based language.

## Imagery
- Prefer .webp, optimized. Provide descriptive alt text.

## Do / Don’t
- Do: Use Primary/Accent thoughtfully; maintain AA+ contrast; keep layouts clean.
- Don’t: Recolor logos; invent new brand colors; lead with performance without context.

## File Map
- logos/: SVG marks (primary, alt, favicon, social default)
- styles/: ecic-brand.scss (colors, fonts, components)
- config/: _brand.yml (brand tokens)

