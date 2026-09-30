// EgoRecall project page: sticky bar, reveal-on-scroll, videos, BibTeX copy, hover hints, charts, and the category
// cloud.
// Dataset charts read static/data/stats.json (written by the maintainers' stats script from the released
// annotations); result charts use the paper's numbers.

const COLORS = {
  ink: "#1c2230",
  muted: "#667085",
  grid: "#eef0f3",
  blue: "#1d5fb8",
  blueLight: "#9dbde8",
  orange: "#e8671a",
  teal: "#0f8a7a",
  green: "#22a55a",
  slateLight: "#c9d0da",
  pair: "#475467",
};
const AXIS_COLORS = { history: COLORS.blue, egocentric: COLORS.orange, allocentric: COLORS.teal };

// ---------- Sticky bar: visible once the title area's buttons scroll out of view ----------

function initTopbar() {
  const topbar = document.getElementById("topbar");
  const heroButtons = document.getElementById("hero-buttons");
  new IntersectionObserver(([entry]) => {
    topbar.classList.toggle("is-visible", !entry.isIntersecting && entry.boundingClientRect.top < 0);
  }).observe(heroButtons);
}

// ---------- Reveal blocks as they enter the viewport ----------

function initReveal() {
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-shown");
        observer.unobserve(entry.target);
      }
    }
  }, { rootMargin: "0px 0px -8% 0px" });
  document.querySelectorAll(".reveal").forEach((element) => observer.observe(element));
}

// ---------- Videos: the hero animation and the Aria clip ----------
// Each plays once while at least half of it is in view and stops on its last frame, which completes the figure.
// Scrolling away pauses it and coming back resumes it; a pause by the viewer, or the end, holds. With reduced motion,
// the videos wait for the viewer.

// Each file is fetched whole and played from memory, because seeking needs HTTP range requests, which simple local
// servers such as python -m http.server do not answer; the files are a few megabytes. If the fetch fails, the video
// streams from its <source> instead.
const wholeVideos = new Map();

function loadWhole(video) {
  if (!wholeVideos.has(video)) {
    const source = video.querySelector("source");
    const loading = fetch(source.src)
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.blob();
      })
      .then((blob) => {
        // A viewer may have started the video from its source with the browser's controls in the meantime
        if (video.readyState !== HTMLMediaElement.HAVE_NOTHING) return;
        source.remove();
        // Decode from memory at once, so seeking shows frames before the first play
        video.preload = "auto";
        video.src = URL.createObjectURL(blob);
      })
      .catch((error) => console.warn(`${source.src} could not be fetched whole, so it streams instead.`, error));
    wholeVideos.set(video, loading);
  }
  return wholeVideos.get(video);
}

// Browsers may refuse to play a video the viewer has not clicked, for example to save power, and a pause during
// loading cancels play(); in both cases the video stays paused.
function playVideo(video) {
  return loadWhole(video).then(() => video.play()).catch(() => {});
}

// The hero's timeline: play, pause, and replay; a scrubber that pauses the video where it is left, so a stage can be
// read; and numbered marks that play from where each of the teaser's panels starts
function initTimeline(timeline) {
  const video = document.getElementById(timeline.dataset.timeline);
  const button = timeline.querySelector(".play-button");
  const scrubber = timeline.querySelector("input[type=range]");

  const showState = () => {
    const state = video.ended ? "ended" : video.paused ? "paused" : "playing";
    button.dataset.state = state;
    button.setAttribute("aria-label", { ended: "Replay", paused: "Play", playing: "Pause" }[state]);
  };
  // While the video plays, move the scrubber on every display frame
  const follow = () => {
    scrubber.value = String(video.currentTime);
    if (!video.paused) requestAnimationFrame(follow);
  };
  video.addEventListener("play", () => {
    showState();
    requestAnimationFrame(follow);
  });
  ["pause", "ended", "seeked"].forEach((type) => video.addEventListener(type, showState));

  button.addEventListener("click", () => (video.paused ? playVideo(video) : video.pause()));
  scrubber.addEventListener("input", () => {
    video.pause();
    loadWhole(video).then(() => {
      video.currentTime = Number(scrubber.value);
    });
  });
  timeline.querySelectorAll(".marker").forEach((marker) => {
    const time = Number(marker.dataset.time);
    marker.style.left = `${(100 * time) / Number(scrubber.max)}%`;
    marker.addEventListener("click", () => {
      loadWhole(video).then(() => {
        video.currentTime = time;
        scrubber.value = String(time);
        playVideo(video);
      });
    });
  });
  showState();
}

function initVideos() {
  document.querySelectorAll("[data-timeline]").forEach(initTimeline);
  const videos = document.querySelectorAll("video[data-autoplay]");

  // Fetch each file once its video comes within a screen's height of the view
  const loader = new IntersectionObserver((entries) => {
    for (const { target: video, isIntersecting } of entries) {
      if (!isIntersecting) continue;
      loader.unobserve(video);
      loadWhole(video);
    }
  }, { rootMargin: "100% 0px" });
  videos.forEach((video) => loader.observe(video));

  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const inView = new Set();
  const held = new Set(); // paused by the viewer, or at the end
  const scrolledAway = new Set(); // paused by scrolling away, to resume on return
  const player = new IntersectionObserver((entries) => {
    for (const { target: video, isIntersecting } of entries) {
      if (isIntersecting) {
        inView.add(video);
        // The file may still be loading; play only if the video is still in view once it has arrived
        if (!held.has(video)) loadWhole(video).then(() => inView.has(video) && playVideo(video));
      } else {
        inView.delete(video);
        if (!video.paused) {
          scrolledAway.add(video);
          video.pause();
        }
      }
    }
  }, { threshold: 0.5 });
  videos.forEach((video) => {
    video.addEventListener("pause", () => {
      if (!scrolledAway.delete(video)) held.add(video);
    });
    video.addEventListener("play", () => held.delete(video));
    player.observe(video);
  });
}

// ---------- BibTeX copy button ----------

function initCopyButtons() {
  document.querySelectorAll("[data-copy]").forEach((button) => {
    button.addEventListener("click", async () => {
      const text = document.getElementById(button.dataset.copy).textContent;
      await navigator.clipboard.writeText(text);
      button.textContent = "Copied";
      setTimeout(() => { button.textContent = "Copy"; }, 1600);
    });
  });
}

// ---------- Hover hints: elements with data-tip show their text below them on hover or focus ----------

function initTooltips() {
  const tooltip = document.createElement("div");
  tooltip.className = "tooltip";
  tooltip.setAttribute("role", "tooltip");
  document.body.append(tooltip);

  const show = (element) => {
    tooltip.textContent = element.dataset.tip;
    tooltip.classList.add("is-visible");
    // Center below the element, kept inside the window
    const box = element.getBoundingClientRect();
    const left = box.left + box.width / 2 - tooltip.offsetWidth / 2;
    tooltip.style.left = `${Math.min(Math.max(8, left), window.innerWidth - tooltip.offsetWidth - 8)}px`;
    tooltip.style.top = `${box.bottom + 8}px`;
  };
  const hide = () => tooltip.classList.remove("is-visible");

  document.querySelectorAll("[data-tip]").forEach((element) => {
    element.tabIndex = 0;
    element.addEventListener("mouseenter", () => show(element));
    element.addEventListener("mouseleave", hide);
    element.addEventListener("focus", () => show(element));
    element.addEventListener("blur", hide);
  });
  window.addEventListener("scroll", hide, { passive: true });
}

// ---------- Charts ----------

// Write each bar's value just past its end.
const valueLabels = {
  id: "valueLabels",
  afterDatasetsDraw(chart, _args, options) {
    const { ctx } = chart;
    const horizontal = chart.options.indexAxis === "y";
    ctx.save();
    ctx.font = `600 11.5px Inter, system-ui, sans-serif`;
    ctx.fillStyle = COLORS.ink;
    chart.data.datasets.forEach((dataset, datasetIndex) => {
      chart.getDatasetMeta(datasetIndex).data.forEach((bar, index) => {
        const text = options.format(dataset.data[index]);
        if (horizontal) {
          ctx.textAlign = "left";
          ctx.textBaseline = "middle";
          ctx.fillText(text, bar.x + 6, bar.y);
        } else {
          ctx.textAlign = "center";
          ctx.textBaseline = "bottom";
          ctx.fillText(text, bar.x, bar.y - 4);
        }
      });
    });
    ctx.restore();
  },
};

function percent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function barChart(canvasId, { labels, datasets, horizontal = false, max, format, legend = false, suggestedMax }) {
  return new Chart(document.getElementById(canvasId), {
    type: "bar",
    data: { labels, datasets },
    plugins: [valueLabels],
    options: {
      indexAxis: horizontal ? "y" : "x",
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: horizontal ? { right: 44 } : { top: 18 } },
      plugins: {
        legend: { display: legend, position: "top", align: "start", labels: { boxWidth: 10, boxHeight: 10, padding: 14 } },
        tooltip: {
          backgroundColor: COLORS.ink,
          padding: 10,
          cornerRadius: 8,
          callbacks: { label: (item) => ` ${item.dataset.label ? item.dataset.label + ": " : ""}${format(item.raw)}` },
        },
        valueLabels: { format },
      },
      scales: {
        [horizontal ? "x" : "y"]: {
          beginAtZero: true,
          max,
          suggestedMax,
          grid: { color: COLORS.grid },
          border: { display: false },
          ticks: { callback: (value) => (format === percent ? `${Math.round(value * 100)}%` : value) },
        },
        [horizontal ? "y" : "x"]: {
          grid: { display: false },
          border: { color: COLORS.grid },
          ticks: { color: COLORS.ink },
        },
      },
    },
  });
}

const CHARTS = {
  // How queries combine the three reference axes; pairs of axes share one dark color.
  "chart-axes": (stats) => {
    const names = {
      egocentric: "Egocentric only",
      "history+egocentric": "History + egocentric",
      history: "History only",
      "history+allocentric": "History + allocentric",
      "egocentric+allocentric": "Egocentric + allocentric",
      allocentric: "Allocentric only",
      none: "No axis",
    };
    const entries = Object.entries(stats.axis_combinations);
    barChart("chart-axes", {
      horizontal: true,
      labels: entries.map(([key]) => names[key] ?? key),
      datasets: [{
        data: entries.map(([, share]) => share),
        backgroundColor: entries.map(([key]) => AXIS_COLORS[key] ?? (key === "none" ? COLORS.slateLight : COLORS.pair)),
        borderRadius: 4,
        barPercentage: 0.75,
      }],
      format: percent,
    });
  },

  // Finding 1: full F1 on a 0-100 scale, the best method highlighted.
  "chart-full-f1": () => {
    const methods = [
      ["EgoR-MTU3D", 14.1], ["Qwen 3.8-27B", 12.1], ["MTU3D fine-tuned", 10.1],
      ["Qwen3D-3B", 9.6], ["DirectMe-GT-PD", 8.7], ["DAAAM", 7.9],
    ];
    barChart("chart-full-f1", {
      horizontal: true,
      labels: methods.map(([name]) => name),
      datasets: [{
        label: "Full F1",
        data: methods.map(([, value]) => value),
        backgroundColor: methods.map((_, index) => (index === 0 ? COLORS.blue : COLORS.slateLight)),
        borderRadius: 4,
        barPercentage: 0.75,
      }],
      max: 100,
      format: (value) => value.toFixed(1),
    });
  },

  // Finding 2: full F1 when all targets are visible versus all remembered.
  "chart-memory": () => {
    barChart("chart-memory", {
      labels: ["EgoR-MTU3D", "Qwen 3.8-27B", "DirectMe-GT-PD"],
      datasets: [
        { label: "All targets visible", data: [16.8, 14.6, 11.0], backgroundColor: COLORS.green, borderRadius: 4 },
        { label: "All targets remembered", data: [11.8, 10.6, 7.7], backgroundColor: COLORS.blue, borderRadius: 4 },
      ],
      legend: true,
      suggestedMax: 20,
      format: (value) => value.toFixed(1),
    });
  },

  // Finding 4: selection F1 of fine-tuned MTU3D with each encoder and with all three.
  "chart-encoders": () => {
    barChart("chart-encoders", {
      labels: ["Fine-tuned", "+ history", "+ recency", "+ viewpoint", "All three"],
      datasets: [{
        label: "Selection F1",
        data: [15.9, 17.8, 18.0, 19.7, 22.2],
        backgroundColor: [COLORS.slateLight, COLORS.blueLight, COLORS.blueLight, COLORS.blueLight, COLORS.blue],
        borderRadius: 4,
        barPercentage: 0.7,
      }],
      suggestedMax: 25,
      format: (value) => value.toFixed(1),
    });
  },
};

// Size each category by the square root of its target count, then order the words alphabetically.
function renderCategoryCloud(stats) {
  const cloud = document.getElementById("category-cloud");
  const counts = stats.top_target_categories.map((entry) => entry.targets);
  const [low, high] = [Math.min(...counts), Math.max(...counts)];
  const entries = [...stats.top_target_categories].sort((a, b) => a.label.localeCompare(b.label));
  for (const { label, targets } of entries) {
    const word = document.createElement("span");
    const scale = Math.sqrt((targets - low) / (high - low));
    word.textContent = label;
    word.style.fontSize = `${(12.5 + 17.5 * scale).toFixed(1)}px`;
    word.title = `${label}: ${targets.toLocaleString("en-US")} targets`;
    cloud.append(word);
  }
}

// Draw each chart when it first scrolls into view, so its animation is seen.
function initCharts(stats) {
  if (!window.Chart) {
    console.warn("Chart.js did not load; charts are skipped.");
    return;
  }
  Chart.defaults.font.family = "Inter, system-ui, sans-serif";
  Chart.defaults.color = COLORS.muted;

  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      observer.unobserve(entry.target);
      CHARTS[entry.target.id](stats);
    }
  }, { rootMargin: "0px 0px -10% 0px" });
  for (const id of Object.keys(CHARTS)) {
    if (id !== "chart-axes" || stats) observer.observe(document.getElementById(id));
  }
}

async function loadStats() {
  try {
    const response = await fetch("static/data/stats.json");
    return await response.json();
  } catch (error) {
    console.warn("Dataset statistics could not be loaded; serve the page over HTTP to see them.", error);
    return null;
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  initTopbar();
  initReveal();
  initVideos();
  initCopyButtons();
  initTooltips();
  const stats = await loadStats();
  if (stats) renderCategoryCloud(stats);
  initCharts(stats);
});
