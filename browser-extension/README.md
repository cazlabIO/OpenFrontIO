# OpenFront Troop Growth Preview

A small Chrome extension that adds one row to the in-game HUD:

`[ 91% ]  →  [ 98% ]`

The percentages compare the current growth rate with the fastest possible rate
for your current troop cap. The left box is the current rate; the right box is
the immediate efficiency after sending the percentage selected in OpenFront's
attack slider. It keeps your current territory, cities, and max troops
unchanged; land gained or lost after the attack will naturally change the real
rate.

- 90–100%: green
- 75–89%: yellow
- 50–74%: orange
- Below 50%: red

## Install

1. Open `chrome://extensions`.
2. Turn on **Developer mode**.
3. Click **Load unpacked**.
4. Select the `browser-extension` directory containing this file.
5. Reload OpenFront.

The extension uses no Chrome APIs. Its site access is limited to OpenFront and
local OpenFront development pages.
