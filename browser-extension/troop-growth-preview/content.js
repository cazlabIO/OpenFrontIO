(() => {
  "use strict";

  const HOST_ID = "openfront-growth-preview";
  const UPDATE_INTERVAL_MS = 100;

  if (document.getElementById(HOST_ID)) return;

  let host = null;
  let nowBox = null;
  let nowValue = null;
  let afterBox = null;
  let afterValue = null;
  let lastRender = "";
  let peakPlayer = null;
  let peakMaxTroops = null;
  let peakRate = 0;

  function createPreview(controlPanel) {
    host = document.createElement("div");
    host.id = HOST_ID;
    host.style.pointerEvents = "none";
    host.style.display = "none";

    const shadow = host.attachShadow({ mode: "open" });
    shadow.innerHTML = `
      <style>
        :host {
          display: block;
          color: #fff;
          font: 600 12px/1.25 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .preview {
          align-items: center;
          border-bottom: 1px solid rgb(75 85 99 / 0.8);
          display: flex;
          gap: 10px;
          justify-content: center;
          padding: 7px 8px;
          white-space: nowrap;
        }

        .metric {
          align-items: center;
          background: var(--metric-background);
          border: 1px solid var(--metric-color);
          border-radius: 8px;
          color: var(--metric-color);
          display: flex;
          justify-content: center;
          min-height: 28px;
          padding: 5px 10px;
          width: 108px;
        }

        .metric-value {
          font-size: 17px;
          font-variant-numeric: tabular-nums;
          font-weight: 800;
        }

        .arrow {
          color: rgb(156 163 175);
        }

        @media (max-width: 639px) {
          .preview {
            gap: 6px;
            padding: 5px 6px;
          }

          .metric {
            min-height: 26px;
            padding: 4px 8px;
            width: 100px;
          }
        }
      </style>
      <div
        class="preview"
        title="Growth efficiency compared with your peak possible troop growth. The after-push estimate holds territory and max troops constant."
      >
        <div
          class="metric"
          data-now-box
          title="Current growth efficiency"
          aria-label="Current growth efficiency"
        >
          <span class="metric-value" data-now></span>
        </div>
        <span class="arrow">→</span>
        <div class="metric" data-after-box>
          <span class="metric-value" data-after></span>
        </div>
      </div>
    `;

    nowBox = shadow.querySelector("[data-now-box]");
    nowValue = shadow.querySelector("[data-now]");
    afterBox = shadow.querySelector("[data-after-box]");
    afterValue = shadow.querySelector("[data-after]");
    lastRender = "";
    controlPanel.insertAdjacentElement("beforebegin", host);
  }

  function efficiencyPalette(percentage) {
    if (percentage >= 90) {
      return { color: "rgb(74 222 128)", background: "rgb(74 222 128 / 0.12)" };
    }
    if (percentage >= 75) {
      return { color: "rgb(250 204 21)", background: "rgb(250 204 21 / 0.12)" };
    }
    if (percentage >= 50) {
      return { color: "rgb(251 146 60)", background: "rgb(251 146 60 / 0.12)" };
    }
    return { color: "rgb(248 113 113)", background: "rgb(248 113 113 / 0.12)" };
  }

  function applyPalette(element, percentage) {
    const palette = efficiencyPalette(percentage);
    element.style.setProperty("--metric-color", palette.color);
    element.style.setProperty("--metric-background", palette.background);
  }

  function playerAtTroopCount(player, troops) {
    return new Proxy(player, {
      get(target, property) {
        if (property === "troops") return () => troops;

        const value = Reflect.get(target, property, target);
        return typeof value === "function" ? value.bind(target) : value;
      },
    });
  }

  function growthRateAt(config, player, troops) {
    return Number(config.troopIncreaseRate(playerAtTroopCount(player, troops)));
  }

  function findPeakGrowthRate(config, player, maxTroops) {
    let low = 0;
    let high = maxTroops;

    // The growth curve has one peak. Ternary search keeps this tied to the
    // game's live calculation instead of duplicating its constants here.
    for (let i = 0; i < 28; i++) {
      const first = low + (high - low) / 3;
      const second = high - (high - low) / 3;
      if (
        growthRateAt(config, player, first) <
        growthRateAt(config, player, second)
      ) {
        low = first;
      } else {
        high = second;
      }
    }

    return growthRateAt(config, player, (low + high) / 2);
  }

  function growthEfficiency(rate, maximumRate) {
    if (!Number.isFinite(rate) || maximumRate <= 0) return 0;
    return Math.round(Math.min(100, Math.max(0, (rate / maximumRate) * 100)));
  }

  function update() {
    const controlPanel = document.querySelector("control-panel");
    if (!controlPanel) {
      if (host) host.style.display = "none";
      return;
    }

    if (!host || host.nextElementSibling !== controlPanel) {
      host?.remove();
      createPreview(controlPanel);
    }

    try {
      const game = controlPanel.game;
      const player = game?.myPlayer?.();
      if (!player || !player.isAlive?.()) {
        host.style.display = "none";
        return;
      }

      const config = game.config();
      const troops = Number(player.troops());
      const maxTroops = Number(config.maxTroops(player));
      const rawRatio = Number(controlPanel.uiState?.attackRatio);
      const ratio = Number.isFinite(rawRatio)
        ? Math.min(1, Math.max(0.01, rawRatio))
        : 0.2;
      const selectedTroops = Math.min(troops, Math.floor(troops * ratio));
      const remainingTroops = troops - selectedTroops;

      const currentRate = Number(config.troopIncreaseRate(player));
      const projectedRate = growthRateAt(config, player, remainingTroops);

      if (peakPlayer !== player || peakMaxTroops !== maxTroops) {
        peakPlayer = player;
        peakMaxTroops = maxTroops;
        peakRate = findPeakGrowthRate(config, player, maxTroops);
      }

      const currentEfficiency = growthEfficiency(currentRate, peakRate);
      const projectedEfficiency = growthEfficiency(projectedRate, peakRate);
      const pushPercentage = Math.round(ratio * 100);

      const rendered = [
        currentEfficiency,
        projectedEfficiency,
        pushPercentage,
      ].join(":");

      if (rendered !== lastRender) {
        nowValue.textContent = `${currentEfficiency}%`;
        applyPalette(nowBox, currentEfficiency);
        afterValue.textContent = `${projectedEfficiency}%`;
        applyPalette(afterBox, projectedEfficiency);
        const projectedLabel = `Growth efficiency after ${pushPercentage}% push`;
        afterBox.title = projectedLabel;
        afterBox.setAttribute("aria-label", projectedLabel);
        lastRender = rendered;
      }

      host.style.display = "block";
    } catch {
      host.style.display = "none";
    }
  }

  update();
  setInterval(update, UPDATE_INTERVAL_MS);
})();
