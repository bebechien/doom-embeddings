#!/usr/bin/env python3
"""
Generates a self-contained, interactive 3D HTML Code Space Explorer (`index.html`)
from `doom_code_space.json` (AST chunks + EmbeddingGemma 768D embeddings & 3D projections).
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON = os.path.join(BASE_DIR, "doom_code_space.json")
OUTPUT_HTML = os.path.join(BASE_DIR, "index.html")
OUTPUT_HTML_ALIAS = os.path.join(BASE_DIR, "doom_code_explorer.html")

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>DOOM (1993) AST Code Space Explorer — EmbeddingGemma 3D Visualization</title>
<style>
  :root {
    --bg-dark: #090b10;
    --bg-panel: rgba(14, 17, 26, 0.92);
    --bg-card: rgba(22, 27, 40, 0.85);
    --bg-hover: rgba(38, 46, 68, 0.85);
    --border: rgba(255, 255, 255, 0.11);
    --border-accent: rgba(249, 115, 22, 0.55);
    --text-main: #f1f5f9;
    --text-muted: #94a3b8;
    --text-dim: #64748b;
    --doom-red: #ef4444;
    --doom-orange: #f97316;
    --doom-gold: #f59e0b;
    --cyan: #06b6d4;
    --purple: #c084fc;
    --green: #10b981;
    --font-mono: 'JetBrains Mono', 'Fira Code', 'SFMono-Regular', Consolas, monospace;
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }

  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  body {
    background: var(--bg-dark);
    color: var(--text-main);
    font-family: var(--font-sans);
    overflow: hidden;
    height: 100vh;
    width: 100vw;
    user-select: none;
  }

  #canvas-container {
    position: absolute;
    inset: 0;
    z-index: 1;
    background:
      radial-gradient(circle at 50% 45%, rgba(30, 20, 45, 0.45) 0%, rgba(9, 11, 16, 1) 75%);
  }

  canvas {
    display: block;
    width: 100%;
    height: 100%;
    cursor: grab;
  }
  canvas:active {
    cursor: grabbing;
  }

  /* Top Header Bar */
  .topbar {
    position: absolute;
    top: 12px;
    left: 348px;
    right: 438px;
    z-index: 10;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 10px 16px;
    background: var(--bg-panel);
    backdrop-filter: blur(14px);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    pointer-events: auto;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 11px;
  }

  .brand-badge {
    background: linear-gradient(135deg, #dc2626, #f97316);
    color: #fff;
    font-family: var(--font-mono);
    font-weight: 800;
    font-size: 13px;
    letter-spacing: 1px;
    padding: 5px 9px;
    border-radius: 6px;
    box-shadow: 0 0 15px rgba(239, 68, 68, 0.45);
  }

  .brand-title h1 {
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.2px;
    color: #f8fafc;
  }

  .brand-title p {
    font-size: 11px;
    color: var(--text-muted);
    font-family: var(--font-mono);
  }

  .top-controls {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }

  .ctrl-group {
    display: flex;
    align-items: center;
    gap: 5px;
    background: rgba(255, 255, 255, 0.04);
    padding: 4px 8px;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.07);
  }

  .ctrl-group label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--text-muted);
    font-weight: 600;
  }

  select, button.btn {
    background: rgba(15, 23, 42, 0.9);
    color: var(--text-main);
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 11.5px;
    font-family: var(--font-sans);
    cursor: pointer;
    transition: all 0.15s ease;
  }

  select:hover, button.btn:hover {
    border-color: var(--doom-orange);
    background: rgba(30, 41, 59, 0.95);
  }

  button.btn.active {
    background: rgba(249, 115, 22, 0.22);
    border-color: var(--doom-orange);
    color: #fdba74;
  }

  /* Left Sidebar */
  .sidebar-left {
    position: absolute;
    top: 12px;
    bottom: 12px;
    left: 12px;
    width: 324px;
    z-index: 10;
    background: var(--bg-panel);
    backdrop-filter: blur(14px);
    border: 1px solid var(--border);
    border-radius: 12px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.55);
  }

  /* Right Sidebar */
  .sidebar-right {
    position: absolute;
    top: 12px;
    bottom: 12px;
    right: 12px;
    width: 414px;
    z-index: 10;
    background: var(--bg-panel);
    backdrop-filter: blur(14px);
    border: 1px solid var(--border);
    border-radius: 12px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.55);
  }

  .panel-header {
    padding: 13px 15px 10px;
    border-bottom: 1px solid var(--border);
  }

  .panel-header h2 {
    font-size: 12.5px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .search-box {
    margin-top: 9px;
    position: relative;
  }

  .search-box input {
    width: 100%;
    padding: 8px 28px 8px 11px;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.14);
    background: rgba(9, 11, 16, 0.8);
    color: var(--text-main);
    font-size: 12.5px;
    font-family: var(--font-mono);
    outline: none;
  }

  .search-box input:focus {
    border-color: var(--doom-orange);
    box-shadow: 0 0 0 2px rgba(249, 115, 22, 0.2);
  }

  .search-clear {
    position: absolute;
    right: 8px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 13px;
    color: var(--text-muted);
    cursor: pointer;
    display: none;
  }

  .filter-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
    margin-top: 8px;
  }

  .stats-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 6px;
    padding: 9px 14px;
    background: rgba(0, 0, 0, 0.25);
    border-bottom: 1px solid var(--border);
  }

  .stat-cell {
    text-align: center;
  }

  .stat-val {
    font-family: var(--font-mono);
    font-size: 13px;
    font-weight: 700;
    color: #f8fafc;
  }

  .stat-lbl {
    font-size: 9.5px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }

  .scroll-area {
    flex: 1;
    overflow-y: auto;
    padding: 10px 12px;
  }

  .scroll-area::-webkit-scrollbar {
    width: 6px;
  }
  .scroll-area::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.16);
    border-radius: 3px;
  }

  .section-title {
    font-size: 10.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    color: var(--text-muted);
    margin: 8px 0 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 8px;
    border-radius: 7px;
    cursor: pointer;
    transition: background 0.12s;
    margin-bottom: 3px;
    border: 1px solid transparent;
  }

  .legend-item:hover {
    background: var(--bg-hover);
  }

  .legend-item.selected {
    background: rgba(249, 115, 22, 0.14);
    border-color: rgba(249, 115, 22, 0.45);
  }

  .legend-item.dimmed {
    opacity: 0.38;
  }

  .swatch {
    width: 11px;
    height: 11px;
    border-radius: 50%;
    flex-shrink: 0;
    box-shadow: 0 0 6px currentColor;
  }

  .legend-info {
    flex: 1;
    min-width: 0;
  }

  .legend-name {
    font-size: 11.5px;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .legend-sub {
    font-size: 10px;
    color: var(--text-dim);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-family: var(--font-mono);
  }

  .legend-count {
    font-family: var(--font-mono);
    font-size: 10.5px;
    color: var(--text-muted);
    background: rgba(255, 255, 255, 0.06);
    padding: 2px 6px;
    border-radius: 10px;
  }

  .chunk-list-item {
    padding: 7px 9px;
    border-radius: 7px;
    background: var(--bg-card);
    border: 1px solid rgba(255, 255, 255, 0.05);
    margin-bottom: 5px;
    cursor: pointer;
    transition: all 0.12s;
  }

  .chunk-list-item:hover {
    background: var(--bg-hover);
    border-color: rgba(255, 255, 255, 0.18);
  }

  .chunk-list-item.active {
    border-color: var(--doom-orange);
    background: rgba(249, 115, 22, 0.14);
  }

  .chunk-item-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
  }

  .chunk-sym {
    font-family: var(--font-mono);
    font-size: 12px;
    font-weight: 600;
    color: #f8fafc;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .chunk-badge {
    font-size: 9.5px;
    font-family: var(--font-mono);
    padding: 1px 5px;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.08);
    color: var(--text-muted);
    flex-shrink: 0;
  }

  .chunk-meta {
    font-size: 10.5px;
    color: var(--text-dim);
    margin-top: 2px;
    font-family: var(--font-mono);
  }

  /* Inspector Details */
  .inspector-content {
    flex: 1;
    overflow-y: auto;
    padding: 14px;
    user-select: text;
  }

  .inspector-content::-webkit-scrollbar {
    width: 6px;
  }
  .inspector-content::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.16);
    border-radius: 3px;
  }

  .insp-title {
    font-family: var(--font-mono);
    font-size: 16px;
    font-weight: 700;
    color: #fff;
    word-break: break-all;
  }

  .insp-filepath {
    font-family: var(--font-mono);
    font-size: 11.5px;
    color: var(--doom-orange);
    margin-top: 4px;
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }

  .tag-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 10px;
  }

  .pill {
    font-size: 10.5px;
    font-family: var(--font-mono);
    padding: 3px 8px;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #e2e8f0;
  }

  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 6px;
    margin: 12px 0;
  }

  .metric-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 7px 6px;
    text-align: center;
  }

  .metric-box .v {
    font-family: var(--font-mono);
    font-size: 13.5px;
    font-weight: 700;
    color: #f8fafc;
  }

  .metric-box .k {
    font-size: 9.5px;
    color: var(--text-dim);
    text-transform: uppercase;
    margin-top: 2px;
  }

  .code-block {
    background: #06080c;
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 8px;
    padding: 11px 12px;
    font-family: var(--font-mono);
    font-size: 11.5px;
    line-height: 1.5;
    color: #e2e8f0;
    overflow-x: auto;
    white-space: pre;
    max-height: 360px;
    overflow-y: auto;
    margin-top: 6px;
  }

  .neighbor-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 6px 8px;
    border-radius: 6px;
    background: var(--bg-card);
    border: 1px solid rgba(255, 255, 255, 0.06);
    margin-bottom: 4px;
    cursor: pointer;
    transition: all 0.12s;
  }

  .neighbor-row:hover {
    background: var(--bg-hover);
    border-color: var(--purple);
  }

  .sim-bar-wrap {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-shrink: 0;
  }

  .sim-score {
    font-family: var(--font-mono);
    font-size: 10.5px;
    color: #e9d5ff;
  }

  .chip-Wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: 5px;
  }

  .call-chip {
    font-family: var(--font-mono);
    font-size: 11px;
    padding: 3px 7px;
    border-radius: 5px;
    background: rgba(6, 182, 212, 0.13);
    border: 1px solid rgba(6, 182, 212, 0.35);
    color: #67e8f9;
    cursor: pointer;
    transition: all 0.12s;
  }

  .call-chip:hover {
    background: rgba(6, 182, 212, 0.28);
  }

  .call-chip.caller {
    background: rgba(245, 158, 11, 0.13);
    border-color: rgba(245, 158, 11, 0.35);
    color: #fcd34d;
  }
  .call-chip.caller:hover {
    background: rgba(245, 158, 11, 0.28);
  }

  /* Hover Tooltip */
  #tooltip {
    position: absolute;
    pointer-events: none;
    z-index: 30;
    background: rgba(10, 13, 22, 0.95);
    border: 1px solid var(--doom-orange);
    border-radius: 8px;
    padding: 8px 11px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.7);
    display: none;
    max-width: 320px;
  }

  #tooltip .tt-name {
    font-family: var(--font-mono);
    font-size: 12.5px;
    font-weight: 700;
    color: #fff;
  }

  #tooltip .tt-sub {
    font-family: var(--font-mono);
    font-size: 10.5px;
    color: var(--doom-orange);
    margin-top: 2px;
  }

  #tooltip .tt-meta {
    font-size: 10.5px;
    color: var(--text-muted);
    margin-top: 3px;
  }

  /* Bottom HUD Legend for Edges */
  .bottom-hud {
    position: absolute;
    bottom: 14px;
    left: 348px;
    right: 438px;
    z-index: 10;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 14px;
    background: var(--bg-panel);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 10px;
    font-size: 11px;
    color: var(--text-muted);
    pointer-events: auto;
  }

  .edge-key {
    display: flex;
    align-items: center;
    gap: 14px;
    font-family: var(--font-mono);
    font-size: 10.5px;
  }

  .edge-dot {
    display: inline-block;
    width: 14px;
    height: 2px;
    vertical-align: middle;
    margin-right: 5px;
  }

  /* Syntax highlighting helpers */
  .tok-kw { color: #f472b6; font-weight: 600; }
  .tok-type { color: #38bdf8; }
  .tok-num { color: #fbbf24; }
  .tok-str { color: #a3e635; }
  .tok-com { color: #64748b; font-style: italic; }
  .tok-fn { color: #c084fc; }

  @media (max-width: 1280px) {
    .topbar, .bottom-hud {
      left: 300px;
      right: 360px;
    }
    .sidebar-left { width: 280px; }
    .sidebar-right { width: 340px; }
  }
</style>
</head>
<body>

<div id="canvas-container">
  <canvas id="space-canvas"></canvas>
</div>

<!-- Top Navigation & 3D Controls -->
<div class="topbar">
  <div class="brand">
    <div class="brand-badge">DOOM // AST</div>
    <div class="brand-title">
      <h1>id Software DOOM Code Space (3D)</h1>
      <p id="model-subtitle">EmbeddingGemma-300M (768D) &bull; Tree-sitter C AST</p>
    </div>
  </div>

  <div class="top-controls">
    <div class="ctrl-group">
      <label>3D Projection</label>
      <select id="proj-select">
        <option value="umap" selected>3D UMAP (Manifold)</option>
        <option value="tsne">3D t-SNE (Cosine)</option>
        <option value="pca">3D PCA (Linear)</option>
      </select>
    </div>

    <div class="ctrl-group">
      <label>Color By</label>
      <select id="color-select">
        <option value="cluster" selected>EmbeddingGemma Clusters (K=12)</option>
        <option value="subsystem">DOOM Engine Subsystem</option>
        <option value="category">AST Node Category</option>
        <option value="hdbscan">HDBSCAN Density Clusters</option>
        <option value="complexity">Cyclomatic Complexity Heatmap</option>
      </select>
    </div>

    <div class="ctrl-group">
      <label>Point Size</label>
      <select id="size-select">
        <option value="ast" selected>AST Node Count</option>
        <option value="complexity">Cyclomatic Complexity</option>
        <option value="lines">Line Span</option>
        <option value="uniform">Uniform</option>
      </select>
    </div>

    <button class="btn active" id="btn-labels" title="Toggle 3D Cluster Centroid Labels">Labels</button>
    <button class="btn active" id="btn-hulls" title="Toggle 3D Cluster Aura Spheres">Hulls</button>
    <button class="btn" id="btn-all-edges" title="Show All 1,613 AST Call Graph Edges">Call Web</button>
    <button class="btn active" id="btn-spin" title="Toggle Gentle 3D Auto-Orbit">Auto-Orbit</button>
    <button class="btn" id="btn-reset" title="Reset 3D Camera View">Reset View</button>
  </div>
</div>

<!-- Left Sidebar: Search, Filters, Cluster Legend & Matching AST Chunks -->
<div class="sidebar-left">
  <div class="panel-header">
    <h2>
      <span>AST Explorer &amp; Clusters</span>
      <span id="visible-count" style="color: var(--doom-orange); font-family: var(--font-mono);"></span>
    </h2>
    <div class="search-box">
      <input type="text" id="search-input" placeholder="Search symbol, C code, file (e.g. P_Mobj, BSP)..." />
      <span class="search-clear" id="search-clear">&times;</span>
    </div>
    <div class="filter-row">
      <select id="filter-category">
        <option value="ALL">All AST Categories</option>
      </select>
      <select id="filter-subsystem">
        <option value="ALL">All Subsystems</option>
      </select>
    </div>
  </div>

  <div class="stats-strip">
    <div class="stat-cell">
      <div class="stat-val" id="st-chunks">0</div>
      <div class="stat-lbl">AST Chunks</div>
    </div>
    <div class="stat-cell">
      <div class="stat-val" id="st-files">0</div>
      <div class="stat-lbl">C/H Files</div>
    </div>
    <div class="stat-cell">
      <div class="stat-val" id="st-clusters">12</div>
      <div class="stat-lbl">Clusters</div>
    </div>
    <div class="stat-cell">
      <div class="stat-val" id="st-edges">0</div>
      <div class="stat-lbl">Call Edges</div>
    </div>
  </div>

  <div class="scroll-area">
    <div class="section-title">
      <span id="legend-title">Clusters (Click to Filter)</span>
      <span id="clear-cluster-btn" style="cursor:pointer; color:var(--doom-orange); display:none;">Show All</span>
    </div>
    <div id="legend-list"></div>

    <div class="section-title" style="margin-top: 14px;">
      <span>AST Chunks</span>
      <span id="list-limit-note" style="font-weight:400; color:var(--text-dim);"></span>
    </div>
    <div id="chunk-list"></div>
  </div>
</div>

<!-- Right Sidebar: Selected AST Chunk Inspector -->
<div class="sidebar-right">
  <div class="panel-header">
    <h2>
      <span>AST Node &amp; Embedding Inspector</span>
      <span id="insp-id" style="font-family: var(--font-mono); color: var(--text-dim);"></span>
    </h2>
  </div>
  <div class="inspector-content" id="inspector-body"></div>
</div>

<!-- Bottom HUD -->
<div class="bottom-hud">
  <div>
    <strong>3D Controls:</strong> Left-Drag Rotate &bull; Right/Shift-Drag Pan &bull; Scroll Zoom &bull; Click Node to Inspect
  </div>
  <div class="edge-key">
    <span><i class="edge-dot" style="background:#06b6d4;"></i>AST Callee (Calls)</span>
    <span><i class="edge-dot" style="background:#f59e0b;"></i>AST Caller (Called By)</span>
    <span><i class="edge-dot" style="background:#c084fc;"></i>768D EmbeddingGemma Neighbor</span>
  </div>
</div>

<!-- Floating Tooltip -->
<div id="tooltip">
  <div class="tt-name" id="tt-name"></div>
  <div class="tt-sub" id="tt-sub"></div>
  <div class="tt-meta" id="tt-meta"></div>
</div>

<script>
const DATA = __DOOM_DATA_JSON__;

// Distinct vibrant palette for clusters & subsystems
const PALETTE = [
  "#f97316", // 0: DOOM Orange
  "#06b6d4", // 1: Cyber Cyan
  "#10b981", // 2: Plasma Green
  "#c084fc", // 3: BFG Purple
  "#f43f5e", // 4: Baron Crimson
  "#eab308", // 5: Mega Gold
  "#3b82f6", // 6: Soul Blue
  "#ec4899", // 7: Demon Magenta
  "#14b8a6", // 8: Nukage Teal
  "#a855f7", // 9: Hell Violet
  "#84cc16", // 10: Acid Lime
  "#fb7185", // 11: Cacodemon Rose
  "#38bdf8",
  "#fbbf24",
  "#4ade80",
  "#e879f9"
];

const chunks = DATA.chunks;
const clusters = DATA.clusters;
const meta = DATA.metadata;

// Populate top stats
document.getElementById("st-chunks").textContent = meta.total_chunks;
document.getElementById("st-files").textContent = meta.total_files;
document.getElementById("st-clusters").textContent = meta.n_clusters;
document.getElementById("st-edges").textContent = meta.total_call_edges;

// Build lookup tables for subsystems & categories
const subsystems = Array.from(new Set(chunks.map(c => c.subsystem))).sort();
const categories = Array.from(new Set(chunks.map(c => c.category))).sort();
const subsystemIdx = Object.fromEntries(subsystems.map((s, i) => [s, i]));
const categoryIdx = Object.fromEntries(categories.map((c, i) => [c, i]));

const catSelect = document.getElementById("filter-category");
categories.forEach(cat => {
  const opt = document.createElement("option");
  opt.value = cat;
  opt.textContent = cat;
  catSelect.appendChild(opt);
});

const subSelect = document.getElementById("filter-subsystem");
subsystems.forEach(sub => {
  const opt = document.createElement("option");
  opt.value = sub;
  opt.textContent = sub;
  subSelect.appendChild(opt);
});

// Interactive State
let currentProj = "umap";
let colorMode = "cluster";
let sizeMode = "ast";
let showLabels = true;
let showHulls = true;
let showAllEdges = false;
let autoOrbit = true;

let searchQuery = "";
let filterCategory = "ALL";
let filterSubsystem = "ALL";
let activeGroupFilter = null; // cluster ID, subsystem name, etc.
let selectedChunkId = null;
let hoveredChunkId = null;

// Current animated 3D positions for each chunk
const posCurr = chunks.map(c => ({ x: c.x, y: c.y, z: c.z }));
const posTarget = chunks.map(c => ({ x: c.x, y: c.y, z: c.z }));

// Camera state (Spherical orbit + target offset)
const cam = {
  yaw: 0.55,
  pitch: 0.28,
  distance: 255,
  targetX: 0,
  targetY: 0,
  targetZ: 0,
  goalX: 0,
  goalY: 0,
  goalZ: 0,
  goalDist: 255
};

function getChunkColor(ch) {
  if (colorMode === "cluster") {
    return PALETTE[ch.cluster % PALETTE.length];
  } else if (colorMode === "subsystem") {
    return PALETTE[subsystemIdx[ch.subsystem] % PALETTE.length];
  } else if (colorMode === "category") {
    return PALETTE[categoryIdx[ch.category] % PALETTE.length];
  } else if (colorMode === "hdbscan") {
    if (ch.hdbscan_cluster < 0) return "#475569";
    return PALETTE[ch.hdbscan_cluster % PALETTE.length];
  } else if (colorMode === "complexity") {
    const cc = ch.cyclomatic_complexity;
    if (cc <= 2) return "#06b6d4";
    if (cc <= 6) return "#10b981";
    if (cc <= 14) return "#eab308";
    if (cc <= 25) return "#f97316";
    return "#ef4444";
  }
  return "#f97316";
}

function getChunkRadius(ch) {
  if (sizeMode === "uniform") return 3.8;
  if (sizeMode === "complexity") {
    return Math.min(10.5, 2.6 + Math.sqrt(ch.cyclomatic_complexity) * 1.15);
  }
  if (sizeMode === "lines") {
    return Math.min(10.5, 2.5 + Math.log2(ch.line_count + 1) * 0.95);
  }
  // Default: AST node count
  return Math.min(10.5, 2.3 + Math.log2(ch.ast_node_count + 1) * 0.78);
}

function getGroupKey(ch) {
  if (colorMode === "cluster") return ch.cluster;
  if (colorMode === "subsystem") return ch.subsystem;
  if (colorMode === "category") return ch.category;
  if (colorMode === "hdbscan") return ch.hdbscan_cluster;
  return "all";
}

function isChunkVisible(ch) {
  if (filterCategory !== "ALL" && ch.category !== filterCategory) return false;
  if (filterSubsystem !== "ALL" && ch.subsystem !== filterSubsystem) return false;
  if (activeGroupFilter !== null && getGroupKey(ch) !== activeGroupFilter) return false;
  if (searchQuery) {
    const q = searchQuery.toLowerCase();
    const matchName = ch.name.toLowerCase().includes(q);
    const matchFile = ch.file.toLowerCase().includes(q);
    const matchDoc = (ch.doc_comment || "").toLowerCase().includes(q);
    const matchCode = (ch.code || "").toLowerCase().includes(q);
    if (!matchName && !matchFile && !matchDoc && !matchCode) return false;
  }
  return true;
}

const visibleFlags = new Array(chunks.length).fill(true);

function updateFiltersAndUI() {
  let visCount = 0;
  const matchedChunks = [];
  for (let i = 0; i < chunks.length; i++) {
    const vis = isChunkVisible(chunks[i]);
    visibleFlags[i] = vis;
    if (vis) {
      visCount++;
      matchedChunks.push(chunks[i]);
    }
  }
  document.getElementById("visible-count").textContent = `${visCount} / ${chunks.length}`;
  renderLegend();
  renderChunkList(matchedChunks);
}

function renderLegend() {
  const container = document.getElementById("legend-list");
  container.innerHTML = "";
  const clearBtn = document.getElementById("clear-cluster-btn");
  clearBtn.style.display = activeGroupFilter !== null ? "inline" : "none";

  let items = [];
  if (colorMode === "cluster") {
    items = clusters.map(cl => ({
      key: cl.id,
      color: PALETTE[cl.id % PALETTE.length],
      name: cl.short_name,
      sub: `${cl.dominant_subsystem.split("(")[0].trim()} • ${cl.exemplars.slice(0, 2).join(", ")}`,
      count: cl.count,
      centroid: currentProj === "umap" ? cl.centroid_umap : (currentProj === "tsne" ? cl.centroid_tsne : cl.centroid_pca)
    }));
  } else if (colorMode === "subsystem") {
    items = subsystems.map((sub, idx) => {
      const cnt = chunks.filter(c => c.subsystem === sub).length;
      return {
        key: sub,
        color: PALETTE[idx % PALETTE.length],
        name: sub,
        sub: "DOOM Engine Subsystem",
        count: cnt
      };
    });
  } else if (colorMode === "category") {
    items = categories.map((cat, idx) => {
      const cnt = chunks.filter(c => c.category === cat).length;
      return {
        key: cat,
        color: PALETTE[idx % PALETTE.length],
        name: cat,
        sub: "Tree-sitter C AST Node Class",
        count: cnt
      };
    });
  } else if (colorMode === "hdbscan") {
    const hdbIds = Array.from(new Set(chunks.map(c => c.hdbscan_cluster))).sort((a, b) => a - b);
    items = hdbIds.map(hid => {
      const cnt = chunks.filter(c => c.hdbscan_cluster === hid).length;
      return {
        key: hid,
        color: hid < 0 ? "#475569" : PALETTE[hid % PALETTE.length],
        name: hid < 0 ? "Unclustered / Noise Outliers" : `HDBSCAN Density Cluster #${hid}`,
        sub: `${cnt} AST nodes`,
        count: cnt
      };
    });
  } else {
    items = [
      { key: "all", color: "#06b6d4", name: "Low Complexity (1 - 2)", sub: "Simple leaf functions & structs", count: chunks.filter(c => c.cyclomatic_complexity <= 2).length },
      { key: "all", color: "#10b981", name: "Moderate (3 - 6)", sub: "Standard gameplay & helper routines", count: chunks.filter(c => c.cyclomatic_complexity >= 3 && c.cyclomatic_complexity <= 6).length },
      { key: "all", color: "#eab308", name: "Elevated (7 - 14)", sub: "Branching physics & rendering logic", count: chunks.filter(c => c.cyclomatic_complexity >= 7 && c.cyclomatic_complexity <= 14).length },
      { key: "all", color: "#f97316", name: "High (15 - 25)", sub: "Complex state machines & responders", count: chunks.filter(c => c.cyclomatic_complexity >= 15 && c.cyclomatic_complexity <= 25).length },
      { key: "all", color: "#ef4444", name: "Extreme (26+)", sub: "Core dispatchers & collision solvers", count: chunks.filter(c => c.cyclomatic_complexity >= 26).length }
    ];
  }

  items.forEach(item => {
    const el = document.createElement("div");
    const isSel = activeGroupFilter === item.key;
    const isDim = activeGroupFilter !== null && !isSel;
    el.className = `legend-item ${isSel ? "selected" : ""} ${isDim ? "dimmed" : ""}`;
    el.innerHTML = `
      <span class="swatch" style="background:${item.color}; color:${item.color};"></span>
      <div class="legend-info">
        <div class="legend-name">${escapeHtml(item.name)}</div>
        <div class="legend-sub">${escapeHtml(item.sub)}</div>
      </div>
      <span class="legend-count">${item.count}</span>
    `;
    el.addEventListener("click", () => {
      if (colorMode === "complexity") return;
      if (activeGroupFilter === item.key) {
        activeGroupFilter = null;
        cam.goalX = 0; cam.goalY = 0; cam.goalZ = 0; cam.goalDist = 255;
      } else {
        activeGroupFilter = item.key;
        if (item.centroid) {
          cam.goalX = item.centroid[0];
          cam.goalY = item.centroid[1];
          cam.goalZ = item.centroid[2];
          cam.goalDist = 145;
        }
      }
      updateFiltersAndUI();
    });
    container.appendChild(el);
  });
}

function renderChunkList(matched) {
  const container = document.getElementById("chunk-list");
  container.innerHTML = "";
  const limit = 80;
  document.getElementById("list-limit-note").textContent =
    matched.length > limit ? `Showing top ${limit} of ${matched.length}` : `${matched.length} shown`;

  // Sort by AST node count descending or exact search match
  const sorted = [...matched].sort((a, b) => {
    if (searchQuery) {
      const qa = a.name.toLowerCase().startsWith(searchQuery.toLowerCase()) ? 1 : 0;
      const qb = b.name.toLowerCase().startsWith(searchQuery.toLowerCase()) ? 1 : 0;
      if (qa !== qb) return qb - qa;
    }
    return b.ast_node_count - a.ast_node_count;
  });

  sorted.slice(0, limit).forEach(ch => {
    const div = document.createElement("div");
    div.className = `chunk-list-item ${selectedChunkId === ch.id ? "active" : ""}`;
    const col = getChunkColor(ch);
    div.innerHTML = `
      <div class="chunk-item-top">
        <span class="chunk-sym" style="color:${col};">${escapeHtml(ch.name)}</span>
        <span class="chunk-badge">${escapeHtml(ch.category)}</span>
      </div>
      <div class="chunk-meta">${escapeHtml(ch.file)}:${ch.start_line}-${ch.end_line} &bull; CC:${ch.cyclomatic_complexity}</div>
    `;
    div.addEventListener("click", () => selectChunk(ch.id, true));
    container.appendChild(div);
  });
}

function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function highlightCCode(code) {
  let html = escapeHtml(code);
  // Comments
  html = html.replace(/(\/\/[^\n]*|\/\*[\s\S]*?\*\/)/g, '<span class="tok-com">$1</span>');
  // Keywords
  html = html.replace(/\b(if|else|for|while|do|switch|case|default|break|continue|return|goto|sizeof|typedef|struct|enum|union|static|extern|const|unsigned|signed|volatile|register)\b/g, '<span class="tok-kw">$1</span>');
  // DOOM & C Types
  html = html.replace(/\b(void|int|char|short|long|float|double|boolean|byte|fixed_t|angle_t|mobj_t|player_t|sector_t|line_t|side_t|subsector_t|seg_t|visplane_t|vissprite_t|thinker_t|state_t|mobjinfo_t|pspdef_t|event_t|ticcmd_t|patch_t)\b/g, '<span class="tok-type">$1</span>');
  // Numbers & Hex
  html = html.replace(/\b(0x[0-9a-fA-F]+|\d+)\b/g, '<span class="tok-num">$1</span>');
  return html;
}

function selectChunk(id, flyCamera = true) {
  selectedChunkId = id;
  const ch = chunks[id];
  if (!ch) return;

  if (flyCamera) {
    const p = posTarget[id];
    cam.goalX = p.x;
    cam.goalY = p.y;
    cam.goalZ = p.z;
    cam.goalDist = Math.min(cam.distance, 125);
    autoOrbit = false;
    document.getElementById("btn-spin").classList.remove("active");
  }

  document.getElementById("insp-id").textContent = `#${ch.id} • ${ch.ast_type}`;
  const col = getChunkColor(ch);

  const neighborsHtml = (ch.neighbors || []).map(nb => {
    const nch = chunks[nb.id];
    const ncol = getChunkColor(nch);
    const pct = Math.max(10, Math.round(nb.sim * 100));
    return `
      <div class="neighbor-row" onclick="selectChunk(${nch.id}, true)">
        <div style="min-width:0; flex:1;">
          <div style="font-family:var(--font-mono); font-size:11.5px; font-weight:600; color:${ncol}; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
            ${escapeHtml(nch.name)}
          </div>
          <div style="font-size:10px; color:var(--text-dim); font-family:var(--font-mono);">
            ${escapeHtml(nch.file)}:${nch.start_line} &bull; ${escapeHtml(nch.category)}
          </div>
        </div>
        <div class="sim-bar-wrap">
          <span class="sim-score">${nb.sim.toFixed(3)}</span>
        </div>
      </div>
    `;
  }).join("");

  const calleesHtml = (ch.callee_ids || []).map(cid => {
    const cch = chunks[cid];
    return `<span class="call-chip" onclick="selectChunk(${cch.id}, true)" title="${escapeHtml(cch.file)}">${escapeHtml(cch.name)}()</span>`;
  }).join("");

  const callersHtml = (ch.caller_ids || []).map(cid => {
    const cch = chunks[cid];
    return `<span class="call-chip caller" onclick="selectChunk(${cch.id}, true)" title="${escapeHtml(cch.file)}">${escapeHtml(cch.name)}()</span>`;
  }).join("");

  const docHtml = ch.doc_comment
    ? `<div class="code-block" style="color:#94a3b8; max-height:110px; margin-bottom:10px;">${escapeHtml(ch.doc_comment)}</div>`
    : "";

  const body = document.getElementById("inspector-body");
  body.innerHTML = `
    <div class="insp-title" style="color:${col};">${escapeHtml(ch.name)}</div>
    <div class="insp-filepath">
      <span>${escapeHtml(ch.file)} : lines ${ch.start_line}&ndash;${ch.end_line}</span>
    </div>

    <div class="tag-row">
      <span class="pill" style="border-color:${col};">${escapeHtml(ch.cluster_name)}</span>
      <span class="pill">${escapeHtml(ch.subsystem)}</span>
      <span class="pill">${escapeHtml(ch.category)}</span>
    </div>

    <div class="metrics-grid">
      <div class="metric-box">
        <div class="v">${ch.ast_node_count}</div>
        <div class="k">AST Nodes</div>
      </div>
      <div class="metric-box">
        <div class="v">${ch.ast_depth}</div>
        <div class="k">AST Depth</div>
      </div>
      <div class="metric-box">
        <div class="v">${ch.cyclomatic_complexity}</div>
        <div class="k">Complexity</div>
      </div>
      <div class="metric-box">
        <div class="v">${ch.line_count}</div>
        <div class="k">Lines</div>
      </div>
    </div>

    ${ch.file_description ? `<div style="font-size:11.5px; color:var(--text-muted); margin-bottom:10px; line-height:1.4;"><strong>File Purpose:</strong> ${escapeHtml(ch.file_description)}</div>` : ""}

    <div class="section-title">
      <span>Top-6 Nearest Neighbors (768D EmbeddingGemma Cosine)</span>
    </div>
    <div>${neighborsHtml}</div>

    ${calleesHtml ? `
      <div class="section-title" style="margin-top:12px;">
        <span>AST Outgoing Calls (${ch.callee_ids.length})</span>
      </div>
      <div class="chip-Wrap">${calleesHtml}</div>
    ` : ""}

    ${callersHtml ? `
      <div class="section-title" style="margin-top:12px;">
        <span>AST Incoming Callers (${ch.caller_ids.length})</span>
      </div>
      <div class="chip-Wrap">${callersHtml}</div>
    ` : ""}

    <div class="section-title" style="margin-top:14px;">
      <span>AST Extracted C Source Code</span>
      <span style="font-family:var(--font-mono); font-size:10px;">${escapeHtml(ch.ast_type)}</span>
    </div>
    ${docHtml}
    <div class="code-block">${highlightCCode(ch.code)}</div>
  `;

  // Highlight active item in left list
  document.querySelectorAll(".chunk-list-item").forEach(el => el.classList.remove("active"));
}

// Set target coordinates when projection changes
function setProjection(proj) {
  currentProj = proj;
  for (let i = 0; i < chunks.length; i++) {
    const ch = chunks[i];
    if (proj === "umap") {
      posTarget[i].x = ch.x;
      posTarget[i].y = ch.y;
      posTarget[i].z = ch.z;
    } else if (proj === "tsne") {
      posTarget[i].x = ch.tsne_x;
      posTarget[i].y = ch.tsne_y;
      posTarget[i].z = ch.tsne_z;
    } else {
      posTarget[i].x = ch.pca_x;
      posTarget[i].y = ch.pca_y;
      posTarget[i].z = ch.pca_z;
    }
  }
  if (selectedChunkId !== null) {
    cam.goalX = posTarget[selectedChunkId].x;
    cam.goalY = posTarget[selectedChunkId].y;
    cam.goalZ = posTarget[selectedChunkId].z;
  }
  renderLegend();
}

// 3D Canvas Engine with Perspective Projection, Depth Sorting, Orbit/Pan/Zoom, and Raypicking
const canvas = document.getElementById("space-canvas");
const ctx = canvas.getContext("2d");
let width = window.innerWidth;
let height = window.innerHeight;
let dpr = window.devicePixelRatio || 1;

function resizeCanvas() {
  width = window.innerWidth;
  height = window.innerHeight;
  dpr = window.devicePixelRatio || 1;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener("resize", resizeCanvas);
resizeCanvas();

// Project a 3D world point (wx, wy, wz) into screen coordinates (sx, sy, depth, scale)
const FOV = 520;
function project3D(wx, wy, wz) {
  const dx = wx - cam.targetX;
  const dy = wy - cam.targetY;
  const dz = wz - cam.targetZ;

  const cosY = Math.cos(cam.yaw);
  const sinY = Math.sin(cam.yaw);
  const cosP = Math.cos(cam.pitch);
  const sinP = Math.sin(cam.pitch);

  // Rotate around Y axis (yaw)
  const rx = dx * cosY - dz * sinY;
  const rz1 = dx * sinY + dz * cosY;

  // Rotate around X axis (pitch)
  const ry = dy * cosP - rz1 * sinP;
  const rz = dy * sinP + rz1 * cosP + cam.distance;

  if (rz <= 10) return null;
  const scale = FOV / rz;
  // Offset center slightly left of right-sidebar for balanced framing
  const centerX = width * 0.475;
  const centerY = height * 0.50;
  return {
    sx: centerX + rx * scale,
    sy: centerY - ry * scale,
    rz: rz,
    scale: scale
  };
}

// Mouse interaction for 3D Orbit, Pan, Zoom & Click
let isDragging = false;
let dragMode = "orbit";
let dragStartX = 0;
let dragStartY = 0;
let totalDragDist = 0;

canvas.addEventListener("contextmenu", e => e.preventDefault());

canvas.addEventListener("mousedown", e => {
  isDragging = true;
  dragStartX = e.clientX;
  dragStartY = e.clientY;
  totalDragDist = 0;
  if (e.button === 2 || e.shiftKey) {
    dragMode = "pan";
  } else {
    dragMode = "orbit";
  }
});

window.addEventListener("mousemove", e => {
  const mx = e.clientX;
  const my = e.clientY;

  if (isDragging) {
    const dx = mx - dragStartX;
    const dy = my - dragStartY;
    totalDragDist += Math.hypot(dx, dy);
    dragStartX = mx;
    dragStartY = my;

    if (dragMode === "orbit") {
      cam.yaw += dx * 0.0065;
      cam.pitch = Math.max(-1.45, Math.min(1.45, cam.pitch + dy * 0.0065));
      autoOrbit = false;
      document.getElementById("btn-spin").classList.remove("active");
    } else {
      const panSpeed = cam.distance * 0.0016;
      const cosY = Math.cos(cam.yaw);
      const sinY = Math.sin(cam.yaw);
      cam.goalX -= dx * cosY * panSpeed;
      cam.goalZ -= dx * sinY * panSpeed;
      cam.goalY += dy * panSpeed;
    }
    return;
  }

  // Hit-test projected nodes for hover tooltip
  if (mx < 340 || mx > width - 426) {
    hoveredChunkId = null;
    document.getElementById("tooltip").style.display = "none";
    return;
  }

  let bestId = null;
  let bestDist = 13;
  for (let i = 0; i < screenNodes.length; i++) {
    const sn = screenNodes[i];
    if (!sn || !visibleFlags[sn.id]) continue;
    const d = Math.hypot(mx - sn.sx, my - sn.sy);
    if (d < sn.r + 5 && d < bestDist) {
      bestDist = d;
      bestId = sn.id;
    }
  }

  hoveredChunkId = bestId;
  const tt = document.getElementById("tooltip");
  if (bestId !== null) {
    const ch = chunks[bestId];
    document.getElementById("tt-name").textContent = ch.name;
    document.getElementById("tt-sub").textContent = `${ch.file}:${ch.start_line} • ${ch.category}`;
    document.getElementById("tt-meta").textContent = `${ch.cluster_name} | AST Nodes: ${ch.ast_node_count} | CC: ${ch.cyclomatic_complexity}`;
    tt.style.display = "block";
    tt.style.left = `${Math.min(width - 340, mx + 14)}px`;
    tt.style.top = `${Math.max(20, my - 10)}px`;
  } else {
    tt.style.display = "none";
  }
});

window.addEventListener("mouseup", e => {
  if (isDragging && totalDragDist < 5 && e.target === canvas) {
    if (hoveredChunkId !== null) {
      selectChunk(hoveredChunkId, false);
    }
  }
  isDragging = false;
});

canvas.addEventListener("wheel", e => {
  e.preventDefault();
  const factor = e.deltaY > 0 ? 1.1 : 0.91;
  cam.goalDist = Math.max(45, Math.min(620, cam.goalDist * factor));
}, { passive: false });

// Keep projected nodes for hit testing
let screenNodes = new Array(chunks.length);

function draw3DAxesGrid() {
  const axes = [
    { x1: -110, y1: 0, z1: 0, x2: 110, y2: 0, z2: 0, col: "rgba(239, 68, 68, 0.22)", lbl: "+X" },
    { x1: 0, y1: -110, z1: 0, x2: 0, y2: 110, z2: 0, col: "rgba(16, 185, 129, 0.22)", lbl: "+Y" },
    { x1: 0, y1: 0, z1: -110, x2: 0, y2: 0, z2: 110, col: "rgba(59, 130, 246, 0.22)", lbl: "+Z" }
  ];
  ctx.lineWidth = 1;
  axes.forEach(ax => {
    const p1 = project3D(ax.x1, ax.y1, ax.z1);
    const p2 = project3D(ax.x2, ax.y2, ax.z2);
    if (p1 && p2) {
      ctx.strokeStyle = ax.col;
      ctx.beginPath();
      ctx.moveTo(p1.sx, p1.sy);
      ctx.lineTo(p2.sx, p2.sy);
      ctx.stroke();
    }
  });
}

function renderLoop() {
  // Smooth interpolation of 3D point positions & camera
  for (let i = 0; i < chunks.length; i++) {
    posCurr[i].x += (posTarget[i].x - posCurr[i].x) * 0.12;
    posCurr[i].y += (posTarget[i].y - posCurr[i].y) * 0.12;
    posCurr[i].z += (posTarget[i].z - posCurr[i].z) * 0.12;
  }

  cam.targetX += (cam.goalX - cam.targetX) * 0.1;
  cam.targetY += (cam.goalY - cam.targetY) * 0.1;
  cam.targetZ += (cam.goalZ - cam.targetZ) * 0.1;
  cam.distance += (cam.goalDist - cam.distance) * 0.12;

  if (autoOrbit && !isDragging) {
    cam.yaw += 0.0022;
  }

  ctx.clearRect(0, 0, width, height);
  draw3DAxesGrid();

  // Project all nodes
  const drawOrder = [];
  for (let i = 0; i < chunks.length; i++) {
    const p = posCurr[i];
    const pr = project3D(p.x, p.y, p.z);
    if (!pr) {
      screenNodes[i] = null;
      continue;
    }
    const baseR = getChunkRadius(chunks[i]);
    const r = Math.max(1.8, baseR * pr.scale * 0.45);
    const nodeObj = {
      id: i,
      sx: pr.sx,
      sy: pr.sy,
      rz: pr.rz,
      scale: pr.scale,
      r: r
    };
    screenNodes[i] = nodeObj;
    drawOrder.push(nodeObj);
  }

  // Sort back-to-front by depth (rz descending)
  drawOrder.sort((a, b) => b.rz - a.rz);

  // 1. Optional: Draw cluster translucent aura spheres
  if (showHulls && colorMode === "cluster") {
    clusters.forEach(cl => {
      if (activeGroupFilter !== null && activeGroupFilter !== cl.id) return;
      const cpos = currentProj === "umap" ? cl.centroid_umap : (currentProj === "tsne" ? cl.centroid_tsne : cl.centroid_pca);
      const pr = project3D(cpos[0], cpos[1], cpos[2]);
      if (!pr) return;
      const hullR = Math.max(24, 42 * pr.scale * 0.55);
      const col = PALETTE[cl.id % PALETTE.length];
      const grad = ctx.createRadialGradient(pr.sx, pr.sy, 2, pr.sx, pr.sy, hullR);
      grad.addColorStop(0, col + "24");
      grad.addColorStop(0.65, col + "0d");
      grad.addColorStop(1, col + "00");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(pr.sx, pr.sy, hullR, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  // 2. Optional: Draw global call graph web
  if (showAllEdges) {
    ctx.strokeStyle = "rgba(148, 163, 184, 0.11)";
    ctx.lineWidth = 0.7;
    ctx.beginPath();
    for (let i = 0; i < chunks.length; i++) {
      if (!visibleFlags[i] || !screenNodes[i]) continue;
      const snA = screenNodes[i];
      const callees = chunks[i].callee_ids;
      for (let j = 0; j < callees.length; j++) {
        const cid = callees[j];
        if (!visibleFlags[cid] || !screenNodes[cid]) continue;
        const snB = screenNodes[cid];
        ctx.moveTo(snA.sx, snA.sy);
        ctx.lineTo(snB.sx, snB.sy);
      }
    }
    ctx.stroke();
  }

  // Build highlight sets for the active/hovered chunk
  const activeId = hoveredChunkId !== null ? hoveredChunkId : selectedChunkId;
  const calleeSet = new Set();
  const callerSet = new Set();
  const neighborSet = new Set();

  if (activeId !== null && chunks[activeId]) {
    const ach = chunks[activeId];
    (ach.callee_ids || []).forEach(id => calleeSet.add(id));
    (ach.caller_ids || []).forEach(id => callerSet.add(id));
    (ach.neighbors || []).forEach(nb => neighborSet.add(nb.id));
  }

  // 3. Draw 3D synaptic curves for selected/hovered chunk (Callees, Callers, 768D Neighbors)
  if (activeId !== null && screenNodes[activeId]) {
    const origin = screenNodes[activeId];

    const drawArc = (targetId, strokeCol, dashed = false) => {
      const dest = screenNodes[targetId];
      if (!dest) return;
      ctx.save();
      ctx.strokeStyle = strokeCol;
      ctx.lineWidth = 1.75;
      if (dashed) ctx.setLineDash([4, 4]);
      ctx.beginPath();
      const mx = (origin.sx + dest.sx) * 0.5;
      const my = (origin.sy + dest.sy) * 0.5 - 18;
      ctx.moveTo(origin.sx, origin.sy);
      ctx.quadraticCurveTo(mx, my, dest.sx, dest.sy);
      ctx.stroke();
      ctx.restore();
    };

    neighborSet.forEach(nid => drawArc(nid, "rgba(192, 132, 252, 0.72)", true));
    callerSet.forEach(cid => drawArc(cid, "rgba(245, 158, 11, 0.85)", false));
    calleeSet.forEach(cid => drawArc(cid, "rgba(6, 182, 212, 0.9)", false));
  }

  // 4. Draw 3D AST Chunk Nodes
  for (let i = 0; i < drawOrder.length; i++) {
    const sn = drawOrder[i];
    const id = sn.id;
    const ch = chunks[id];
    const vis = visibleFlags[id];

    const isSel = id === selectedChunkId;
    const isHov = id === hoveredChunkId;
    const isLinked = calleeSet.has(id) || callerSet.has(id) || neighborSet.has(id);

    if (!vis && !isSel && !isLinked) {
      ctx.fillStyle = "rgba(71, 85, 105, 0.12)";
      ctx.beginPath();
      ctx.arc(sn.sx, sn.sy, Math.max(1.2, sn.r * 0.6), 0, Math.PI * 2);
      ctx.fill();
      continue;
    }

    const col = getChunkColor(ch);
    let alpha = 0.86;
    if (activeId !== null && !isSel && !isHov && !isLinked && searchQuery === "") {
      alpha = 0.32;
    }

    ctx.save();
    ctx.globalAlpha = alpha;

    // Outer glow for selected, hovered, or linked nodes
    if (isSel || isHov || isLinked) {
      ctx.shadowColor = col;
      ctx.shadowBlur = isSel || isHov ? 18 : 10;
    }

    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.arc(sn.sx, sn.sy, isSel || isHov ? sn.r * 1.45 : sn.r, 0, Math.PI * 2);
    ctx.fill();

    // Specular highlight dot or ring
    if (isSel || isHov) {
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.stroke();
    } else if (ch.category === "Struct / Typedef" || ch.category === "Enum") {
      ctx.strokeStyle = "rgba(255,255,255,0.55)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    ctx.restore();

    // Draw symbol label next to selected/hovered/linked nodes
    if (isSel || isHov || (isLinked && sn.scale > 1.35)) {
      ctx.save();
      ctx.font = `${isSel || isHov ? "700 11.5px" : "500 10px"} 'JetBrains Mono', monospace`;
      ctx.fillStyle = isSel || isHov ? "#ffffff" : "#e2e8f0";
      ctx.shadowColor = "rgba(0,0,0,0.9)";
      ctx.shadowBlur = 4;
      ctx.fillText(ch.name, sn.sx + sn.r + 5, sn.sy + 3);
      ctx.restore();
    }
  }

  // 5. Optional: Draw 3D Cluster Centroid Badges
  if (showLabels && colorMode === "cluster") {
    clusters.forEach(cl => {
      if (activeGroupFilter !== null && activeGroupFilter !== cl.id) return;
      const cpos = currentProj === "umap" ? cl.centroid_umap : (currentProj === "tsne" ? cl.centroid_tsne : cl.centroid_pca);
      const pr = project3D(cpos[0], cpos[1], cpos[2]);
      if (!pr) return;

      const txt = cl.short_name;
      ctx.save();
      ctx.font = "600 10.5px 'JetBrains Mono', monospace";
      const tw = ctx.measureText(txt).width;
      const bx = pr.sx - tw * 0.5 - 6;
      const by = pr.sy - 10;
      const bw = tw + 12;
      const bh = 19;

      ctx.fillStyle = "rgba(9, 11, 16, 0.82)";
      ctx.strokeStyle = PALETTE[cl.id % PALETTE.length];
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.roundRect(bx, by, bw, bh, 5);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "#f8fafc";
      ctx.fillText(txt, bx + 6, by + 13);
      ctx.restore();
    });
  }

  requestAnimationFrame(renderLoop);
}

// Wire up UI Controls
document.getElementById("proj-select").addEventListener("change", e => {
  setProjection(e.target.value);
});

document.getElementById("color-select").addEventListener("change", e => {
  colorMode = e.target.value;
  activeGroupFilter = null;
  updateFiltersAndUI();
  if (selectedChunkId !== null) selectChunk(selectedChunkId, false);
});

document.getElementById("size-select").addEventListener("change", e => {
  sizeMode = e.target.value;
});

document.getElementById("btn-labels").addEventListener("click", e => {
  showLabels = !showLabels;
  e.target.classList.toggle("active", showLabels);
});

document.getElementById("btn-hulls").addEventListener("click", e => {
  showHulls = !showHulls;
  e.target.classList.toggle("active", showHulls);
});

document.getElementById("btn-all-edges").addEventListener("click", e => {
  showAllEdges = !showAllEdges;
  e.target.classList.toggle("active", showAllEdges);
});

document.getElementById("btn-spin").addEventListener("click", e => {
  autoOrbit = !autoOrbit;
  e.target.classList.toggle("active", autoOrbit);
});

document.getElementById("btn-reset").addEventListener("click", () => {
  activeGroupFilter = null;
  cam.goalX = 0;
  cam.goalY = 0;
  cam.goalZ = 0;
  cam.goalDist = 255;
  cam.pitch = 0.28;
  updateFiltersAndUI();
});

document.getElementById("clear-cluster-btn").addEventListener("click", () => {
  activeGroupFilter = null;
  cam.goalX = 0; cam.goalY = 0; cam.goalZ = 0; cam.goalDist = 255;
  updateFiltersAndUI();
});

const searchInput = document.getElementById("search-input");
const searchClear = document.getElementById("search-clear");

searchInput.addEventListener("input", e => {
  searchQuery = e.target.value.trim();
  searchClear.style.display = searchQuery ? "block" : "none";
  updateFiltersAndUI();
});

searchClear.addEventListener("click", () => {
  searchInput.value = "";
  searchQuery = "";
  searchClear.style.display = "none";
  updateFiltersAndUI();
});

catSelect.addEventListener("change", e => {
  filterCategory = e.target.value;
  updateFiltersAndUI();
});

subSelect.addEventListener("change", e => {
  filterSubsystem = e.target.value;
  updateFiltersAndUI();
});

// Initialize UI and select an iconic DOOM function by default (e.g., D_DoomMain or R_RenderPlayerView or P_XYMovement)
updateFiltersAndUI();
const defaultChunk = chunks.find(c => c.name === "P_XYMovement") || chunks.find(c => c.name === "D_DoomMain") || chunks[0];
if (defaultChunk) {
  selectChunk(defaultChunk.id, false);
}
requestAnimationFrame(renderLoop);
</script>
</body>
</html>
"""


def main():
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"{INPUT_JSON} not found. Run process_doom_ast.py first.")

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Polish cluster short_name if generic English words like "The" appeared in TF-IDF
    extra_stops = {"the", "for", "and", "from", "with", "that", "this", "not", "are", "only"}
    cluster_Map = {}
    for cl in data.get("clusters", []):
        kws = [w for w in cl.get("keywords", []) if w.lower() not in extra_stops]
        if kws:
            kw_title = ", ".join(w.capitalize() for w in kws[:3])
            short_sub = cl["dominant_subsystem"].split("(")[0].strip()
            cl["short_name"] = f"C{cl['id']:02d}: {kw_title}"
            cl["name"] = f"C{cl['id']:02d}: {kw_title} ({short_sub})"
        cluster_Map[cl["id"]] = cl["short_name"]

    for ch in data.get("chunks", []):
        if ch.get("cluster") in cluster_Map:
            ch["cluster_name"] = cluster_Map[ch["cluster"]]

    with open(INPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Compact JSON for fast HTML load
    compact_json = json.dumps(data, separators=(",", ":"))
    html_out = HTML_TEMPLATE.replace("__DOOM_DATA_JSON__", compact_json)

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_out)

    with open(OUTPUT_HTML_ALIAS, "w", encoding="utf-8") as f:
        f.write(html_out)

    size_mb = os.path.getsize(OUTPUT_HTML) / (1024 * 1024)
    print(f"Generated {OUTPUT_HTML} and {OUTPUT_HTML_ALIAS} ({size_mb:.2f} MB).")
    for cl in data.get("clusters", []):
        print(f"  {cl['name']} — {cl['count']} chunks")


if __name__ == "__main__":
    main()
