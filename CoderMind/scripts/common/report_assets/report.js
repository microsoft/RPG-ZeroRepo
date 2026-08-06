/* CoderMind Run Report — dashboard-v1
   Renders the 5-view report from window.CMIND_REPORT (a real snapshot).
   Principles baked in:
     - Encoder and Decoder are shown separately, never merged into one percent.
     - RPG dependency-mapping coverage is NOT presented as "implementation".
     - Absent data renders as an honest empty/not-run state, not a fake value. */
(function () {
  "use strict";
  var R = window.CMIND_REPORT || {};
  var RPG_DOCUMENT = window.CMIND_RPG_HTML || "";
  var HISTORY = window.CMIND_HISTORY_INDEX || { roots: [], retention: {}, summary: {} };
  var historyFilter = "all";
  var historySearch = "";
  var historyCoverageExpanded = false;
  var collapsedHistoryRoots = {};

  /* ---------- helpers ---------- */
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  function esc(v) {
    return String(v == null ? "" : v).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function num(v) { return v == null ? "—" : (typeof v === "number" ? v.toLocaleString() : esc(v)); }
  function shortSha(v) { return v ? String(v).slice(0, 8) : "—"; }
  function fmtDur(s) {
    if (s == null || isNaN(s)) return "—";
    if (s < 1) return Math.round(s * 1000) + " ms";
    if (s < 60) return s.toFixed(s < 10 ? 2 : 1) + " s";
    var m = Math.floor(s / 60), r = Math.round(s % 60);
    return m + "m " + String(r).padStart(2, "0") + "s";
  }
  function delta(v) { return v == null ? "—" : v > 0 ? "+" + v : String(v); }
  function statusKind(s) {
    s = String(s == null ? "" : s).toLowerCase();
    if (["success", "completed", "ok", "passed", "available", "true"].indexOf(s) >= 0) return "ok";
    if (["running", "in_progress"].indexOf(s) >= 0) return "running";
    if (["failed", "error", "false", "timed_out", "cancelled", "interrupted"].indexOf(s) >= 0) return "fail";
    if (["completed_with_warnings", "warning", "warn", "partial", "invalid", "unreadable", "degraded"].indexOf(s) >= 0) return "warn";
    return "pending";
  }
  function pill(status, labelOverride) {
    var k = statusKind(status);
    var label = labelOverride || (status == null ? "n/a" : String(status));
    return '<span class="pill ' + k + '">' + esc(label) + "</span>";
  }
  function qBadge(q) { return q ? '<span class="q ' + esc(q) + '">' + esc(q) + "</span>" : ""; }
  function stageDisplayStatus(stage) {
    return stage.quality !== "measured" && stage.kind === "ok" ? "artifact present" : stage.status;
  }
  function stageEvidence(stage) {
    return stage.kind === "pending" ? "no evidence" : stage.quality;
  }

  /* ---------- stage catalog (presentation, not data) ---------- */
  var CATALOG = {
    encoder: {
      label: "Encoder", tagline: "Code → RPG", icon: "EN",
      phases: [
        { id: "encode", label: "Encode", stages: [["parse_rpg", "Parse RPG", "PR"], ["dep_graph", "Dependency Graph", "DG"], ["save_rpg", "Save RPG", "SV"], ["visualize", "Visualization", "VZ"]] },
        { id: "update", label: "Update RPG", loop: true, stages: [["load_rpg", "Load", "LD"], ["process_diff", "Process Diff", "PD"], ["align_paths", "Align Paths", "AP"], ["advance_git", "Advance Git", "AG"], ["save_rpg", "Save RPG", "SV"]] }
      ]
    },
    decoder: {
      label: "Decoder", tagline: "Requirements → Code", icon: "DE",
      phases: [
        { id: "feature", label: "Feature", stages: [["feature_spec", "Feature Spec", "FS"], ["feature_build", "Feature Build", "FB"], ["feature_refactor", "Feature Refactor", "FR"]] },
        { id: "plan", label: "Architecture & Plan", stages: [["build_skeleton", "Skeleton", "SK"], ["build_data_flow", "Data Flow", "DF"], ["design_base_classes", "Base Classes", "BC"], ["design_interfaces", "Interfaces", "IF"], ["plan_tasks", "Tasks", "TK"]] },
        { id: "impl", label: "Implementation", stages: [["code_gen", "Code Gen", "CG"], ["rpg_edit", "RPG Edit", "RE"]] }
      ]
    }
  };

  var pipeById = {};
  (R.pipeline || []).forEach(function (p) { pipeById[p.id] = p; });
  var latestStageByName = {};
  (R.runs || []).forEach(function (run) {
    (run.stages || []).forEach(function (st) {
      if (st.name && !latestStageByName[st.name]) latestStageByName[st.name] = { stage: st, run_id: run.run_id, command: run.command };
    });
  });
  var activeMode = (R.workspace && R.workspace.mode) || "unknown";
  // Current Changes is the default workspace; Full Graph is opt-in per session.
  var rpgMode = "change";
  var rpgChangeFilter = "all";
  var rpgChangeSearch = "";
  var rpgChangeExpanded = false;
  var rpgChangeListOpen = false;
  var rpgContextMode = "context";
  var rpgFullStyle = "normal";
  var rpgFocusedNode = null;

  function resolveStage(pipelineKey, stageId) {
    var run = latestStageByName[stageId];
    var pipe = (pipelineKey === activeMode) ? pipeById[stageId] : null;
    var status = "not_started", quality = "inferred", duration = null, runId = null, attempt = null, error = null;
    if (run) {
      status = run.stage.status === "success" ? "completed" : (run.stage.status || "completed");
      quality = "measured"; duration = run.stage.duration_s; runId = run.run_id;
      attempt = run.stage.attempt; error = run.stage.error;
    } else if (pipe) {
      status = pipe.status; quality = pipe.quality || "inferred"; runId = pipe.run_id;
      duration = pipe.duration_s; attempt = pipe.attempt; error = pipe.error;
    }
    return { id: stageId, status: status, kind: statusKind(status), quality: quality, duration_s: duration, run_id: runId, attempt: attempt, error: error };
  }
  function pipelineProgress(pipelineKey) {
    var cat = CATALOG[pipelineKey], catalogTotal = 0, seen = {};
    cat.phases.forEach(function (ph) {
      ph.stages.forEach(function (s) {
        var key = ph.id + ":" + s[0]; if (seen[key]) return; seen[key] = 1; catalogTotal++;
      });
    });
    if (pipelineKey === activeMode && (R.pipeline || []).length) {
      var snapshotSteps = R.pipeline || [];
      var completed = snapshotSteps.filter(function (step) { return statusKind(step.status) === "ok"; }).length;
      return {
        done: completed,
        total: snapshotSteps.length,
        percent: snapshotSteps.length ? Math.round(completed / snapshotSteps.length * 100) : 0,
        source: "snapshot",
        catalogTotal: catalogTotal
      };
    }
    return { done: 0, total: catalogTotal, percent: 0, source: "catalog", catalogTotal: catalogTotal };
  }

  /* ---------- drawer ---------- */
  function openDrawer(kicker, title, bodyHtml) {
    $("#drawerKicker").textContent = kicker;
    $("#drawerTitle").textContent = title;
    $("#drawerBody").innerHTML = bodyHtml;
    $("#drawer").classList.add("open");
    $("#drawer").setAttribute("aria-hidden", "false");
    $("#drawerScrim").classList.add("open");
  }
  function closeDrawer() {
    $("#drawer").classList.remove("open");
    $("#drawer").setAttribute("aria-hidden", "true");
    $("#drawerScrim").classList.remove("open");
  }

  function kvTable(rows) {
    return '<table class="kv"><tbody>' + rows.map(function (r) {
      return "<tr><td>" + esc(r[0]) + "</td><td>" + (r[1] == null ? "—" : r[1]) + "</td></tr>";
    }).join("") + "</tbody></table>";
  }

  function stageDrawer(pipelineKey, stageId, label) {
    var r = resolveStage(pipelineKey, stageId);
    var run = latestStageByName[stageId];
    var checks = [];
    if (run) {
      var rr = (R.runs || []).filter(function (x) { return x.run_id === run.run_id; })[0] || {};
      checks = (rr.verification || []).filter(function (c) { return c.name === stageId || c.name === label; });
    } else {
      checks = ((R.verification && R.verification.checks) || []).filter(function (c) {
        return c.name === stageId || c.name === label ||
          (stageId === "code_gen" && ["code generation tasks", "code_gen final test"].indexOf(c.name) >= 0);
      });
    }
    var body = '<div class="drawer-section">' + kvTable([
      ["status", pill(r.status, stageDisplayStatus(r))],
      ["pipeline", CATALOG[pipelineKey].label],
      ["duration", fmtDur(r.duration_s)],
      ["evidence", qBadge(stageEvidence(r)) || "—"],
      ["attempt", r.attempt != null ? r.attempt : "—"],
      ["run", r.run_id ? '<span class="mono">' + esc(r.run_id) + "</span>" : "—"],
      ["error", r.error ? esc(typeof r.error === "object" ? (r.error.message || JSON.stringify(r.error)) : r.error) : "—"]
    ]) + "</div>";
    if (checks.length) {
      body += '<div class="drawer-section"><h4>Verification</h4>' + kvTable(checks.map(function (c) {
        return [c.name, pill(c.status) + (c.detail ? ' <span class="muted">' + esc(typeof c.detail === "object" ? JSON.stringify(c.detail) : c.detail) + "</span>" : "")];
      })) + "</div>";
    }
    if (r.kind === "pending") {
      body += '<div class="note"><span class="i">ℹ</span><span>This stage has no recorded execution in the current snapshot. It is part of the ' + esc(CATALOG[pipelineKey].label) + ' pipeline but has not run in this workspace.</span></div>';
    }
    openDrawer(CATALOG[pipelineKey].label + " · stage", label, body);
  }

  function runDrawer(runId) {
    var run = (R.runs || []).filter(function (x) { return x.run_id === runId; })[0];
    if (!run) return;
    var m = run.metrics || {};
    var head = kvTable([
      ["command", '<span class="mono">' + esc(run.command) + "</span>"],
      ["status", pill(run.display_status || run.status)],
      ["trigger", run.trigger || "—"],
      ["started", '<span class="mono">' + esc(run.started_at || "—") + "</span>"],
      ["duration", fmtDur(run.duration_s)],
      ["run id", '<span class="mono">' + esc(run.run_id) + "</span>"]
    ]);
    var timeline = (run.stages || []).length
      ? '<div class="timeline">' + run.stages.map(function (s) {
          return '<div class="tl-item ' + statusKind(s.status) + '"><div class="t">' + esc(s.name) + "</div><div class=\"d\">" + pill(s.status) + " · " + fmtDur(s.duration_s) + (s.attempt > 1 ? " · attempt " + s.attempt : "") + "</div></div>";
        }).join("") + "</div>"
      : '<div class="empty"><p>No internal stages recorded for this run.</p></div>';
    var metricsRows = [];
    [["node_count", "RPG nodes"], ["edge_count", "RPG edges"], ["nodes_delta", "RPG nodes Δ"], ["edges_delta", "RPG edges Δ"], ["dep_nodes", "Dep nodes"], ["dep_edges", "Dep edges"], ["functional_areas", "Functional areas"]].forEach(function (p) {
      if (m[p[0]] != null) metricsRows.push([p[1], p[0].indexOf("delta") >= 0 ? delta(m[p[0]]) : num(m[p[0]])]);
    });
    var checks = run.verification || [];
    var next = run.next_actions || [];
    var tel = run.telemetry || {};
    var body = '<div class="drawer-section">' + head + "</div>";
    body += '<div class="drawer-section"><h4>Stage timeline</h4>' + timeline + "</div>";
    if (metricsRows.length) body += '<div class="drawer-section"><h4>Graph metrics</h4>' + kvTable(metricsRows) + "</div>";
    if (checks.length) body += '<div class="drawer-section"><h4>Verification</h4>' + kvTable(checks.map(function (c) { return [c.name, pill(c.status) + " " + qBadge(c.quality)]; })) + "</div>";
    if (m.previous_commit || m.new_commit) body += '<div class="drawer-section"><h4>Git range</h4>' + kvTable([["from", '<span class="mono">' + shortSha(m.previous_commit) + "</span>"], ["to", '<span class="mono">' + shortSha(m.new_commit) + "</span>"]]) + "</div>";
    if (tel.llm || tel.mcp) {
      var llm = tel.llm || {}, tk = llm.tokens || {};
      body += '<div class="drawer-section"><h4>Telemetry</h4>' + kvTable([
        ["llm calls", llm.calls != null ? llm.calls : "—"],
        ["tokens", tk.total != null ? num(tk.total) : "—"],
        ["mcp calls", tel.mcp && tel.mcp.calls != null ? tel.mcp.calls : "—"]
      ]) + "</div>";
    }
    if (next.length) body += '<div class="drawer-section"><h4>Next actions</h4>' + kvTable(next.map(function (a) { return [a.label || "action", (a.command ? '<span class="mono">' + esc(a.command) + "</span> " : "") + (a.detail ? '<span class="muted">' + esc(a.detail) + "</span>" : "") + " " + qBadge(a.quality)]; })) + "</div>";
    openDrawer("Run detail", run.command, body);
  }

  /* ---------- top bar ---------- */
  function renderTop() {
    var ws = R.workspace || {}, cur = R.current_state || {}, git = ws.git || {};
    var latestAutomation = (R.automation && R.automation.latest) || {};
    var displayedStatus = (R.runs || []).length ? cur.status : (latestAutomation.status || cur.status);
    $("#brandRepo").textContent = ws.name || (R.rpg && R.rpg.repo_name) || "CoderMind";
    $("#brandMeta").textContent = "run report · " + (ws.tool_version ? "v" + ws.tool_version : "");
    var modeEl = $("#modePill");
    modeEl.textContent = activeMode;
    modeEl.className = "mode-pill " + (activeMode === "encoder" || activeMode === "decoder" ? activeMode : "");
    modeEl.title = "Detected workspace mode, derived from available snapshot artifacts";
    var kind = statusKind(displayedStatus);
    $("#statusDot").className = "status-dot " + kind;
    $("#topStatus").textContent = "LATEST " + String(displayedStatus || "unknown").toUpperCase();
    $("#topCommit").textContent = shortSha(git.commit) + (git.dirty ? " ✱" : "");
  }

  /* ---------- Overview ---------- */
  function pipeMiniCard(key) {
    var cat = CATALOG[key], p = pipelineProgress(key);
    var isActive = key === activeMode;
    var countLabel = p.source === "snapshot" ? p.done + "/" + p.total : "not run";
    var phases = cat.phases.map(function (ph) {
      var dots = ph.stages.map(function (s) {
        var k = resolveStage(key, s[0]).kind;
        return '<span class="pdot ' + k + '" title="' + esc(s[1]) + '"></span>';
      }).join("");
      return '<button type="button" class="pmini-phase" data-pipeline-jump="' + key + "/" + ph.id
        + '" aria-label="Open ' + esc(cat.label + " " + ph.label) + ' pipeline details"><span class="pmini-plabel">'
        + esc(ph.label) + (ph.loop ? ' <span class="loop">↻</span>' : "")
        + '</span><span class="pmini-dots">' + dots + '</span><span class="pmini-arrow" aria-hidden="true">›</span></button>';
    }).join("");
    return '<div class="pmini ' + key + '">'
      + '<button type="button" class="pmini-top pmini-jump" data-pipeline-jump="' + key
      + '" aria-label="Open detailed ' + esc(cat.label) + ' pipeline"><span class="pipe-tag">' + cat.icon
      + '</span><span class="pmini-name"><strong>' + cat.label + '</strong><span class="sub mono">' + esc(cat.tagline)
      + '</span></span><span class="count mono">' + countLabel + '</span><span class="pmini-arrow" aria-hidden="true">›</span></button>'
      + '<div class="bar ' + (key === "decoder" ? "dec" : "") + (p.percent === 100 ? " ok" : "") + '"><i style="width:' + p.percent + '%"></i></div>'
      + '<div class="pmini-phases">' + phases + "</div>"
      + (!isActive && p.done === 0 ? '<div class="pmini-empty mono">not run in this workspace</div>' : "")
      + "</div>";
  }

  function renderOverview() {
    var ws = R.workspace || {}, cur = R.current_state || {}, git = ws.git || {}, rpg = R.rpg || {};
    var fg = rpg.feature_graph || {}, dg = rpg.dependency_graph || {}, mp = rpg.mapping || {};
    var runs = R.runs || [];
    var health = R.source_health || [];
    var warnings = health.filter(function (h) { return ["partial", "invalid", "unreadable"].indexOf(h.status) >= 0; });
    var nextActions = (R.verification && R.verification.next_actions) || [];
    var automation = R.automation || {}, latestAutomation = automation.latest || {};
    var mcpActivity = automation.mcp || {}, hookActivity = automation.hooks || {};

    var ctx = '<div class="context">' + [
      ["Workspace folder", esc(ws.name || "—")],
      ["Branch", esc(git.branch || "—")],
      ["Source commit", shortSha(git.commit) + (git.dirty ? ' <span class="pill warn">dirty</span>' : "")],
      ["Tool", "v" + esc(ws.tool_version || "—")],
      ["Generated", esc((R.generated_at || "").replace("T", " ").replace("Z", " UTC"))]
    ].map(function (c) { return '<div class="cell"><div class="k">' + c[0] + '</div><div class="v">' + c[1] + "</div></div>"; }).join("") + "</div>";

    var latest = runs[0];
    var current;
    if (!latest && latestAutomation.status) {
      var automationDetail = latestAutomation.type === "mcp"
        ? num(mcpActivity.sessions) + " sessions · " + num(mcpActivity.calls) + " calls"
        : num(hookActivity.invocations) + " hook invocations · " + num(hookActivity.updates) + " RPG updates";
      current = '<div class="surface"><div class="surface-head"><h3>Latest recorded activity</h3><span class="hint mono">on-demand automation</span></div><div class="surface-body">'
        + kvTable([
          ["activity", '<span class="mono">' + esc(latestAutomation.label || latestAutomation.type) + "</span>"],
          ["status", pill(latestAutomation.status)],
          ["summary", automationDetail],
          ["started", '<span class="mono">' + esc(latestAutomation.started_at || "—") + "</span> · " + historyDuration(latestAutomation.duration_ms)]
        ]) + "</div></div>";
    } else {
      var latestMetadata = (latest && latest.metadata) || {};
      var hookTrigger = latestMetadata.hook_type || latestAutomation.type === "hook"
        ? "Git hook / " + humanizeHistoryName(latestMetadata.hook_type || latestAutomation.hook_type || "hook")
          + ((latestMetadata.hook_sha || latestAutomation.git_sha) ? " @ " + shortSha(latestMetadata.hook_sha || latestAutomation.git_sha) : "")
        : (latest && latest.trigger ? humanizeHistoryName(latest.trigger) : "—");
      current = '<div class="surface"><div class="surface-head"><h3>Latest recorded run</h3><span class="hint mono">detected ' + esc(activeMode) + " workspace</span></div><div class=\"surface-body\">"
        + kvTable([
          ["command", latest ? '<span class="mono">' + esc(cur.command || latest.command) + "</span>" : "—"],
          ["run status", pill(cur.status)],
          ["trigger", hookTrigger],
          ["running stage", cur.current_stage ? esc(cur.current_stage) : '<span class="muted">none recorded</span>'],
          ["started", latest ? '<span class="mono">' + esc(latest.started_at || "—") + "</span> · " + fmtDur(latest.duration_s) : "—"]
        ]) + "</div></div>";
    }

    var pipelines = '<div class="surface"><div class="surface-head"><h3>Pipeline readiness</h3><span class="hint">active snapshot steps; inferred means artifact present</span></div><div class="surface-body grid cols-2">'
      + pipeMiniCard("encoder") + pipeMiniCard("decoder") + "</div></div>";

    var rpgTiles = '<div class="surface"><div class="surface-head"><h3>RPG state</h3><span class="hint mono">' + esc(rpg.repo_name || "") + '</span></div><div class="surface-body tiles cols-3" style="grid-template-columns:repeat(3,1fr)">' + [
      ["rpg", num(fg.nodes), "RPG tree nodes"],
      ["rpg", num(fg.semantic_edges), "Non-hierarchy RPG edges"],
      ["rpg", num(fg.functional_areas), "Functional areas"],
      ["", num(dg.nodes) + " <small>/ " + num(dg.edges) + "</small>", "Dependency graph nodes / edges"],
      ["", num(mp.mapped_dep_nodes) + " <small>/ " + num(mp.total_dep_nodes) + "</small>", "Dependency nodes mapped to RPG"],
      ["", num(mp.unmapped_dep_nodes), "Unmapped dep nodes"]
    ].map(function (t) { return '<div class="tile ' + t[0] + '"><div class="n">' + t[1] + '</div><div class="l">' + t[2] + "</div></div>"; }).join("") + "</div></div>";

    var attention = '<div class="surface"><div class="surface-head"><h3>Attention &amp; next steps</h3></div><div class="surface-body">';
    var automationNotes = [];
    if (mcpActivity.calls) automationNotes.push(
      '<div class="note"><span class="i">ℹ</span><span><b>MCP is on demand</b> · '
      + num(mcpActivity.sessions) + " server sessions, " + num(mcpActivity.calls)
      + " tool calls. MCP calls do not complete Encoder pipeline stages.</span></div>"
    );
    if (hookActivity.invocations) automationNotes.push(
      '<div class="note"><span class="i">ℹ</span><span><b>Git hooks</b> · '
      + num(hookActivity.post_commit) + " post-commit, " + num(hookActivity.post_merge)
      + " post-merge, " + num(hookActivity.updates) + " asynchronous RPG updates.</span></div>"
    );
    if (hookActivity.attribution_mismatches) automationNotes.push(
      '<div class="note warn"><span class="i">⚠</span><span><b>Hook sequencing</b> · '
      + num(hookActivity.attribution_mismatches)
      + " asynchronous update(s) targeted a later HEAD than the commit that triggered them.</span></div>"
    );
    if (!warnings.length && !nextActions.length && !automationNotes.length) {
      attention += '<div class="empty"><p>No data-quality warnings. Pipeline is idle with all recorded checks passing.</p></div>';
    } else {
      if (nextActions.length) attention += kvTable(nextActions.map(function (a) { return [a.label || "next", (a.command ? '<span class="mono">' + esc(a.command) + "</span> " : "") + (a.detail ? '<span class="muted">' + esc(a.detail) + "</span>" : "")]; }));
      if (warnings.length) attention += '<div style="margin-top:10px">' + warnings.map(function (w) { return '<div class="note warn"><span class="i">⚠</span><span><b>' + esc(w.source) + "</b> · " + esc(w.status) + "</span></div>"; }).join("") + "</div>";
      if (automationNotes.length) attention += '<div style="margin-top:10px" class="grid">' + automationNotes.join("") + "</div>";
    }
    attention += "</div></div>";

    var recent = '<div class="surface"><div class="surface-head"><h3>Recent runs</h3><span class="hint">click to inspect</span></div><div class="rows">'
      + (runs.length ? runs.slice(0, 6).map(function (run) {
          return '<div class="row runs-row" data-run="' + esc(run.run_id) + '"><span class="cmd">' + esc(run.command) + '</span><span class="when mono">' + esc(run.started_at || "—") + "</span>" + pill(run.display_status || run.status) + '<span class="dur">' + fmtDur(run.duration_s) + '</span><span class="mono faint" title="RPG node-count change versus the previous RPG version">RPG Δ ' + delta((run.metrics || {}).nodes_delta) + "</span><span class=\"chev\">›</span></div>";
        }).join("") : '<div class="empty"><p>No run history recorded.</p></div>')
      + "</div></div>";

    $("#view-overview").innerHTML =
      '<div class="page-head"><div><div class="eyebrow">CoderMind</div><h2>Overview</h2><p>What the latest snapshot records — pipeline readiness, RPG state, and data requiring attention.</p></div></div>'
      + ctx
      + '<div class="grid cols-2" style="margin-bottom:16px"><div class="grid" style="gap:16px">' + current + attention + "</div>" + pipelines + "</div>"
      + '<div style="margin-bottom:16px">' + rpgTiles + "</div>"
      + recent;
  }

  /* ---------- Pipeline ---------- */
  function stageCard(key, stage) {
    var r = resolveStage(key, stage[0]);
    var meta = r.kind === "ok" && r.duration_s != null ? fmtDur(r.duration_s)
      : r.kind === "ok" && r.quality !== "measured" ? "artifact evidence"
      : r.kind === "pending" ? "not run" : String(r.status);
    return '<button class="stage-card ' + (r.kind === "ok" ? "done" : r.kind) + '" data-stage="' + key + ":" + stage[0] + '" data-label="' + esc(stage[1]) + '">'
      + '<div class="st-top"><span class="stage-ico">' + stage[2] + "</span>" + pill(r.status, stageDisplayStatus(r)) + "</div>"
      + "<strong>" + esc(stage[1]) + "</strong>"
      + '<div class="meta">' + esc(meta) + " · " + esc(stageEvidence(r)) + "</div></button>";
  }
  function pipeSection(key) {
    var cat = CATALOG[key], p = pipelineProgress(key);
    var phases = cat.phases.map(function (ph) {
      return '<section class="phase" id="pipeline-' + key + "-" + ph.id + '" tabindex="-1"><div class="phase-title">'
        + esc(ph.label) + (ph.loop ? ' <span class="loop">↻ loop</span>' : "") + "</div><div class=\"stages\">"
        + ph.stages.map(function (s) { return stageCard(key, s); }).join("") + "</div></section>";
    }).join("");
    var empty = (key !== activeMode && p.done === 0)
      ? '<div class="note" style="margin:0 17px 16px"><span class="i">ℹ</span><span>No ' + esc(cat.label) + ' run recorded in this <span class="mono">' + esc(activeMode) + '</span>-mode workspace. Stages below show the pipeline shape only.</span></div>'
      : "";
    var count = p.source === "snapshot"
      ? p.done + "/" + p.total + " snapshot steps · " + p.percent + "%"
      : "not run · " + p.catalogTotal + " catalog stages";
    return '<section class="pipe ' + key + '" id="pipeline-' + key + '" tabindex="-1"><div class="pipe-head"><span class="pipe-tag">'
      + cat.icon + '</span><div><h3>' + cat.label + '</h3><div class="sub mono">' + esc(cat.tagline)
      + '</div></div><span class="count mono">' + count + "</span></div>" + empty + phases + "</section>";
  }
  function renderPipeline() {
    $("#view-pipeline").innerHTML =
      '<div class="page-head"><div><div class="eyebrow">Pipeline</div><h2>Pipeline readiness and stage evidence</h2><p>The active pipeline summary comes from snapshot steps. <b>Measured</b> means a recorded stage run, <b>inferred</b> means artifact-only evidence, and <b>no evidence</b> means neither was found.</p></div></div>'
      + pipeSection("encoder") + pipeSection("decoder");
  }

  /* ---------- RPG ---------- */
  function deltaStr(v) { return v == null ? "0" : v > 0 ? "+" + v : String(v); }
  function deltaCls(v) { return v > 0 ? "pos" : v < 0 ? "neg" : "zero"; }
  // Runs that touched the RPG (encoder builds / incremental updates).
  function rpgRuns() {
    return (R.runs || []).filter(function (r) {
      var m = r.metrics || {}, ch = r.changes || {};
      return r.command === "update_rpg" || r.command === "encode" || (ch && ch.graph_deltas) || m.nodes_delta != null;
    });
  }
  function rpgChangeRows() {
    var diff = R.rpg_latest_change || {};
    var rows = [];
    [["feature", diff.feature_nodes || {}], ["dependency", diff.dependency_nodes || {}]].forEach(function (entry) {
      ["added", "removed", "modified"].forEach(function (kind) {
        (entry[1][kind] || []).forEach(function (node) {
          rows.push({
            scope: entry[0],
            kind: kind,
            node_id: String(node.node_id || ""),
            name: node.name || node.node_id || "Unnamed node",
            node_type: node.node_type || "unknown",
            path: node.path,
            parent_id: node.parent_id,
            previous_parent_id: node.previous_parent_id,
            changed_fields: node.changed_fields || []
          });
        });
      });
    });
    return rows;
  }
  function rpgChangeCounts(rows) {
    var counts = { all: rows.length, added: 0, removed: 0, modified: 0 };
    var scope = {
      added: { feature: 0, dependency: 0 },
      removed: { feature: 0, dependency: 0 },
      modified: { feature: 0, dependency: 0 }
    };
    rows.forEach(function (row) {
      counts[row.kind]++;
      scope[row.kind][row.scope]++;
    });
    return { total: counts, scope: scope };
  }
  function changeFilterHtml(rows) {
    var counts = rpgChangeCounts(rows);
    var config = [
      ["all", "All changes", "Δ"],
      ["added", "Added", "+"],
      ["removed", "Removed", "−"],
      ["modified", "Modified", "~"]
    ];
    return '<div class="change-filters">' + config.map(function (item) {
      var kind = item[0], scopes = counts.scope[kind];
      var detail = kind === "all"
        ? "Feature + Dependency"
        : (scopes.feature + " Feature · " + scopes.dependency + " Dependency");
      return '<button class="change-filter ' + kind + (rpgChangeFilter === kind ? " active" : "") + '" data-rpg-filter="' + kind + '">'
        + '<span class="change-symbol">' + item[2] + '</span><span class="change-filter-copy"><strong>' + item[1]
        + '</strong><small>' + detail + '</small></span><b>' + counts.total[kind] + '</b></button>';
    }).join("") + "</div>";
  }
  function normalizeNodePath(value) {
    if (Array.isArray(value)) return value.join(" / ");
    return value == null ? "" : String(value);
  }
  function filteredChangeRows(rows) {
    var query = rpgChangeSearch.trim().toLowerCase();
    return rows.filter(function (row) {
      if (rpgChangeFilter !== "all" && row.kind !== rpgChangeFilter) return false;
      if (!query) return true;
      return (row.name + " " + row.node_id + " " + normalizeNodePath(row.path)).toLowerCase().indexOf(query) >= 0;
    });
  }
  function changeNodeListHtml(rows) {
    var filtered = filteredChangeRows(rows);
    var visible = rpgChangeExpanded ? filtered : filtered.slice(0, 8);
    var list = visible.length ? visible.map(function (row) {
      var selected = rpgFocusedNode && rpgFocusedNode.scope === row.scope && rpgFocusedNode.node_id === row.node_id;
      return '<button class="change-node-row' + (selected ? " selected" : "") + '" data-rpg-node="' + esc(row.node_id)
        + '" data-rpg-scope="' + row.scope + '" data-rpg-kind="' + row.kind + '">'
        + '<span class="change-node-mark ' + row.kind + '">' + ({added: "+", removed: "−", modified: "~"}[row.kind]) + '</span>'
        + '<span class="change-node-main"><strong>' + esc(row.name) + '</strong><small>' + esc(normalizeNodePath(row.path) || row.node_id) + '</small></span>'
        + '<span class="scope-badge">' + (row.scope === "feature" ? "Feature" : "Dependency") + '</span><span class="chev">›</span></button>';
    }).join("") : '<div class="change-list-empty">No nodes match this filter.</div>';
    var toggle = filtered.length > 8
      ? '<button class="btn ghost sm change-list-toggle" data-rpg-expand="' + (rpgChangeExpanded ? "collapse" : "expand") + '">'
        + (rpgChangeExpanded ? "Collapse" : "Show all " + filtered.length) + '</button>'
      : "";
    return '<div class="change-list-head"><div><strong>Changed nodes</strong><span>' + filtered.length + ' visible</span></div>'
      + '<label class="change-search"><span>⌕</span><input id="rpgChangeSearch" value="' + esc(rpgChangeSearch) + '" placeholder="Search changed nodes"></label>'
      + '<button class="btn ghost sm" data-rpg-list-close aria-label="Close changed node list">✕</button></div>'
      + '<div class="change-node-list">' + list + '</div>' + toggle;
  }
  function refreshChangeListPanel() {
    var panel = $("#rpgChangeList");
    if (!rpgChangeListOpen) {
      if (panel) panel.remove();
      return;
    }
    if (!panel) {
      panel = document.createElement("section");
      panel.id = "rpgChangeList";
      panel.className = "change-list-panel";
      var workbar = $(".current-changes .graph-workbar");
      if (workbar) workbar.before(panel);
    }
    panel.innerHTML = changeNodeListHtml(rpgChangeRows());
  }
  function operationContextHtml(run) {
    var diff = R.rpg_latest_change || {};
    return '<div class="change-context"><div class="change-context-main"><span class="status-mark">●</span><div><strong>'
      + esc((diff.operation || (run && run.command) || "RPG update").replace(/--json/g, "").trim())
      + '</strong><small>' + esc(diff.committed_at || (run && run.started_at) || "") + '</small></div></div>'
      + '<div class="change-context-meta"><span>' + pill((run && (run.display_status || run.status)) || "available") + '</span>'
      + '<span class="mono">' + esc(diff.parent_short || "empty") + ' → ' + esc(diff.short_commit || "current") + '</span>'
      + '<span>' + qBadge(diff.quality || "measured") + '</span></div></div>';
  }
  function graphMessage() {
    var diff = R.rpg_latest_change || {};
    return {
      type: "cmind:rpg-highlight",
      mode: rpgMode === "change" ? "changes" : "full",
      filter: rpgChangeFilter,
      contextMode: rpgContextMode,
      emphasize: rpgMode === "change" || rpgFullStyle === "emphasize",
      focus: rpgFocusedNode,
      feature: diff.feature_nodes || {},
      dependency: diff.dependency_nodes || {},
      semanticEdges: diff.semantic_edges || {},
      dependencyEdges: diff.dependency_edges || {},
      hierarchyEdges: diff.feature_hierarchy_edges || {},
      mappingEdges: diff.mapping_edges || {}
    };
  }
  function postRpgGraphState() {
    var frame = $("#rpgFrame");
    if (frame && frame.contentWindow) frame.contentWindow.postMessage(graphMessage(), "*");
  }
  function rpgVersionDrawer(commit) {
    var v = (R.rpg_history || []).filter(function (x) { return x.commit === commit; })[0];
    if (!v) return;
    var cmd = "cmind script rpg_version.py --commit " + v.short_commit;
    var body = '<div class="drawer-section">' + kvTable([
      ["operation", esc(v.operation)],
      ["meta commit", '<span class="mono">' + esc(v.short_commit) + "</span>"],
      ["source commit", v.source_short ? '<span class="mono">' + esc(v.source_short) + "</span>" : "—"],
      ["committed", '<span class="mono">' + esc(v.committed_at || "—") + "</span>"],
      ["RPG nodes", v.node_count == null ? '<span class="muted">not joined to a run</span>' : num(v.node_count) + (v.nodes_delta != null ? ' <span class="' + deltaCls(v.nodes_delta) + '">' + deltaStr(v.nodes_delta) + "</span>" : "")],
      ["message", '<span class="mono faint">' + esc(v.message) + "</span>"]
    ]) + "</div>";
    body += '<div class="drawer-section"><h4>View this version (on-demand)</h4>'
      + '<p class="muted" style="font-size:12px;margin:0 0 10px">The full graph for a past version is <b>not embedded</b> — read it from the meta-git only when needed:</p>'
      + '<div class="cmd-box mono">' + esc(cmd) + " --diff</div>"
      + '<div class="cmd-box mono">' + esc(cmd) + " --output rpg-" + esc(v.short_commit) + ".json</div>"
      + '<p class="muted" style="font-size:11px;margin:10px 0 0"><b>--diff</b> lists exact added / removed / modified nodes and edge changes vs the previous RPG version; <b>--output</b> extracts that version\'s full <span class="mono">rpg.json</span>.</p></div>';
    openDrawer("RPG version", v.operation + " · " + v.short_commit, body);
  }
  function openRpgHistoryDrawer() {
    var hist = R.rpg_history || [];
    var rows = hist.length ? hist.map(function (v) {
      var delta = v.nodes_delta == null ? "metadata" : deltaStr(v.nodes_delta) + " nodes / " + deltaStr(v.edges_delta) + " edges";
      return '<button class="history-version-row" data-rpg-commit="' + esc(v.commit) + '"><span><strong>' + esc(v.operation)
        + '</strong><small>' + esc(String(v.committed_at || "").replace("T", " ").replace("Z", "")) + '</small></span>'
        + '<span class="mono">' + esc(v.short_commit) + '</span><span class="faint">' + esc(delta) + '</span><span>›</span></button>';
    }).join("") : '<div class="empty"><p>No RPG history recorded.</p></div>';
    openDrawer("RPG history", hist.length + " versions", '<div class="history-version-list">' + rows + '</div>');
  }
  function renderRpg() {
    var hasRpgData = !!(R.graph && R.graph.feature_root);
    if (!hasRpgData) {
      $("#view-rpg").innerHTML = '<div class="rpg-hostbar"><span class="mono muted">RPG visualization</span></div>'
        + '<div class="rpg-frame"><div class="empty"><div class="big">RPG not generated</div>'
        + '<p>No <span class="mono">rpg.json</span> is available in this workspace snapshot. Continue the pipeline until RPG data is produced; graph, mapping, and history controls will then appear here.</p></div></div>';
      return;
    }
    var historyCount = (R.rpg_history || []).length;
    var head = '<div class="rpg-hostbar"><span class="mono muted">Interactive RPG visualization</span>'
      + '<div class="rpg-actions"><button class="btn sm" id="rpgHistoryBtn">History (' + historyCount + ')</button>'
      + '<button class="btn sm" id="rpgFull">⛶ Fullscreen</button>'
      + '<a class="btn sm" href="rpg.html" target="_blank" rel="noopener">Open in tab ↗</a></div></div>';
    var frame = '<div class="rpg-frame" id="rpgFrameWrap"><iframe class="ready" id="rpgFrame" title="RPG graph"></iframe>'
      + '<button class="rpg-fs-exit" id="rpgFsExit" hidden>✕ Exit fullscreen</button></div>';
    $("#view-rpg").innerHTML = head + frame;

    var frameElement = $("#rpgFrame");
    frameElement.addEventListener("load", function () {
      syncRpgTheme();
    });
    if (RPG_DOCUMENT) frameElement.srcdoc = RPG_DOCUMENT;
    else frameElement.src = "rpg.html";

    var wrap = $("#rpgFrameWrap");
    function fsRequest(el) {
      (el.requestFullscreen || el.webkitRequestFullscreen || el.msRequestFullscreen).call(el);
    }
    function fsExit() { (document.exitFullscreen || document.webkitExitFullscreen || function () { }).call(document); }
    $("#rpgFull").onclick = function () {
      var fsEl = document.fullscreenElement || document.webkitFullscreenElement;
      if (fsEl === wrap) fsExit(); else fsRequest(wrap);
    };
    $("#rpgFsExit").onclick = fsExit;
  }

  /* ---------- Run History ---------- */
  function historyDuration(ms) { return ms == null ? "—" : fmtDur(Number(ms) / 1000); }
  function historyTime(value) {
    if (!value) return "—";
    var date = new Date(value);
    return isNaN(date.getTime()) ? String(value) : date.toISOString().replace("T", " ").replace(".000Z", "Z");
  }
  function historyShortTime(value) {
    if (!value) return "—";
    var date = new Date(value);
    return isNaN(date.getTime()) ? String(value) : date.toISOString().slice(0, 19).replace("T", " ") + "Z";
  }
  function humanizeHistoryName(value) {
    return String(value || "Activity").replace(/[_-]+/g, " ").replace(/\b\w/g, function (c) { return c.toUpperCase(); })
      .replace(/\bRpg\b/g, "RPG").replace(/\bMcp\b/g, "MCP").replace(/\bLlm\b/g, "LLM");
  }
  function historyDisplayName(node, child) {
    if (child) return humanizeHistoryName(node.name || node.logical_key);
    var key = String(node.logical_key || "");
    var prefix = key.indexOf("encoder-") === 0 ? "Encoder / " : key.indexOf("decoder-") === 0 ? "Decoder / " : "";
    var name = node.name || key.replace(/^(encoder|decoder)-/, "");
    if (!child && node.kind === "mcp.session") {
      var context = (node.details || {}).client_context;
      return "MCP session" + (context ? " / " + humanizeHistoryName(context) : "")
        + " · " + num((node.metrics || {}).calls || (node.children || []).length) + " calls";
    }
    if (!child && node.kind === "hook.workflow") {
      var details = node.details || {};
      return "Git hook / " + humanizeHistoryName(details.hook_type || name)
        + (details.git_sha ? " @ " + shortSha(details.git_sha) : "");
    }
    if (node.kind === "hook.operation") return "Encoder / Hooks / " + humanizeHistoryName(name);
    if (node.kind === "tool.mcp") return "Encoder / MCP / " + humanizeHistoryName(name);
    if (node.kind === "codegen.batch") return "Decoder / Code Gen / " + humanizeHistoryName(name);
    return prefix + humanizeHistoryName(name);
  }
  function historyTooltip(node) {
    return [historyDisplayName(node, false), "Status: " + (node.status || "unknown"),
      "Started: " + historyTime(node.started_at), "Finished: " + historyTime(node.finished_at),
      "Duration: " + historyDuration(node.duration_ms), "Trigger: " + (node.trigger || "unknown"),
      "Evidence: " + (node.quality || "unknown")].join("\n");
  }
  function historyMatches(root) {
    var key = String(root.logical_key || "").toLowerCase(), kind = String(root.kind || "");
    var failed = statusKind(root.status) === "fail" || (root.children || []).some(function (child) { return statusKind(child.status) === "fail"; });
    var filterMatch = historyFilter === "all"
      || (historyFilter === "encoder" && key.indexOf("encoder-") === 0)
      || (historyFilter === "decoder" && key.indexOf("decoder-") === 0)
      || (historyFilter === "hooks" && (kind === "hook.operation" || kind === "hook.workflow"))
      || (historyFilter === "mcp" && (kind === "tool.mcp" || kind === "mcp.session"))
      || (historyFilter === "failures" && failed);
    if (!filterMatch) return false;
    var query = historySearch.trim().toLowerCase();
    if (!query) return true;
    var values = [root.name, root.logical_key, root.status, root.trigger, root.source];
    (root.children || []).forEach(function (child) { values.push(child.name, child.logical_key, child.status); });
    return values.some(function (value) {
      return String(value || "").toLowerCase().replace(/[_-]+/g, " ").indexOf(query.replace(/[_-]+/g, " ")) >= 0;
    });
  }
  function historyNodeMatchesQuery(node) {
    var query = historySearch.trim().toLowerCase().replace(/[_-]+/g, " ");
    if (!query) return true;
    return [node.name, node.logical_key, node.status, node.trigger, node.source].some(function (value) {
      return String(value || "").toLowerCase().replace(/[_-]+/g, " ").indexOf(query) >= 0;
    });
  }
  function historyRow(node, root, child) {
    var hasChildren = !child && (root.children || []).length > 0;
    var collapsed = !!collapsedHistoryRoots[root.span_id];
    return '<div class="history-row ' + (child ? "child " : "root ") + statusKind(node.status)
      + '" data-tip="' + esc(historyTooltip(node)) + '">'
      + (hasChildren
        ? '<span class="history-tree-toggle" role="button" tabindex="0" data-history-toggle="' + esc(root.span_id)
          + '" aria-label="' + (collapsed ? "Expand" : "Collapse") + ' ' + esc(historyDisplayName(root, false))
          + '" aria-expanded="' + (!collapsed) + '">' + (collapsed ? "▸" : "▾") + '</span>'
        : '<span class="history-tree-mark" aria-hidden="true">' + (child ? "└" : "•") + '</span>')
      + '<span class="history-open" role="button" tabindex="0" data-history-root="' + esc(root.span_id)
      + '" data-history-span="' + esc(node.span_id) + '" data-history-detail="' + esc(root.detail_path || "")
      + '" aria-label="Open details for ' + esc(historyDisplayName(node, child)) + '"><span class="history-main"><strong>'
      + esc(historyDisplayName(node, child)) + '</strong><small>'
      + esc(node.kind || "activity") + " · " + esc(node.quality || "unknown") + '</small></span>'
      + '<span class="history-start mono">' + esc(historyShortTime(node.started_at)) + '</span>'
      + pill(node.status || "unknown") + '<span class="history-duration mono">' + esc(historyDuration(node.duration_ms))
      + '</span><span class="chev">›</span></span></div>';
  }
  function filteredHistoryRoots() { return (HISTORY.roots || []).filter(historyMatches); }
  function historyTreeHtml() {
    var roots = filteredHistoryRoots();
    if (!roots.length) return '<div class="empty"><div class="big">No matching history</div><p>Adjust the filters or search query.</p></div>';
    return roots.map(function (root) {
      var rootMatches = historyNodeMatchesQuery(root);
      var childrenToShow = (root.children || []).filter(function (child) {
        return !historySearch.trim() || rootMatches || historyNodeMatchesQuery(child);
      });
      var children = childrenToShow.map(function (child) { return historyRow(child, root, true); }).join("");
      return '<section class="history-group">' + historyRow(root, root, false)
        + (children ? '<div class="history-children' + (collapsedHistoryRoots[root.span_id] ? " collapsed" : "") + '">' + children + '</div>' : "") + '</section>';
    }).join("");
  }
  function historyFind(node, spanId) {
    if (!node) return null;
    if (String(node.span_id) === String(spanId)) return node;
    for (var i = 0; i < (node.children || []).length; i++) {
      var match = historyFind(node.children[i], spanId); if (match) return match;
    }
    return null;
  }
  function loadHistoryDetail(rootId, detailPath) {
    window.CMIND_HISTORY_DETAILS = window.CMIND_HISTORY_DETAILS || {};
    if (window.CMIND_HISTORY_DETAILS[rootId]) return Promise.resolve(window.CMIND_HISTORY_DETAILS[rootId]);
    if (!detailPath) return Promise.resolve((HISTORY.roots || []).filter(function (root) { return root.span_id === rootId; })[0]);
    return new Promise(function (resolve, reject) {
      var script = document.createElement("script"); script.src = detailPath;
      script.onload = function () { resolve(window.CMIND_HISTORY_DETAILS[rootId]); script.remove(); };
      script.onerror = function () { reject(new Error("Unable to load " + detailPath)); script.remove(); };
      document.head.appendChild(script);
    });
  }
  function historyTimeline(node) {
    var children = node.children || [];
    if (!children.length) return '<div class="empty"><p>No child activity recorded.</p></div>';
    return '<div class="timeline">' + children.map(function (child) {
      return '<div class="tl-item ' + statusKind(child.status) + '"><div class="t">' + esc(humanizeHistoryName(child.name))
        + '</div><div class="d">' + pill(child.status) + ' · ' + esc(historyDuration(child.duration_ms))
        + ' · ' + qBadge(child.quality) + '</div>' + historyTimeline(child) + '</div>';
    }).join("") + '</div>';
  }
  function openHistoryDrawer(root, target) {
    var body = '<div class="drawer-section">' + kvTable([["status", pill(target.status)],
      ["started", '<span class="mono">' + esc(historyTime(target.started_at)) + '</span>'],
      ["finished", '<span class="mono">' + esc(historyTime(target.finished_at)) + '</span>'],
      ["duration", historyDuration(target.duration_ms)], ["trigger", esc(target.trigger || "—")],
      ["attempt", target.attempt == null ? "—" : target.attempt], ["evidence", qBadge(target.quality) || "—"],
      ["logical key", '<span class="mono">' + esc(target.logical_key || "—") + '</span>'],
      ["trace", '<span class="mono">' + esc(target.trace_id || root.trace_id || "—") + '</span>']]) + '</div>';
    if (target.error) body += '<div class="note warn"><span class="i">⚠</span><span>' + esc(typeof target.error === "object" ? (target.error.message || JSON.stringify(target.error)) : target.error) + '</span></div>';
    var details = target.details || {};
    var metrics = target.metrics || {};
    if (metrics.prev_ref || metrics.previous_commit || metrics.new_commit) {
      body += '<div class="drawer-section"><h4>Processed Git range</h4>' + kvTable([
        ["from", '<span class="mono">' + shortSha(metrics.prev_ref || metrics.previous_commit) + '</span>'],
        ["to", '<span class="mono">' + shortSha(metrics.new_commit) + '</span>']
      ]) + '</div>';
    }
    var detailLabels = {
      artifact_key: "artifact", artifact_origin: "artifact origin", change_type: "change",
      size_bytes: "size bytes", content_sha256: "SHA-256", provider: "provider", model: "model",
      purpose: "purpose", tool: "MCP tool", mode: "mode", call_id: "call id",
      server_session_id: "server session", client_context: "client context",
      batch_id: "batch id", task_id: "task id",
      task_type: "task type", attempts_used: "attempts used", file_path: "file",
      result_type: "result", script: "script", exit_code: "exit code", git_sha: "git SHA",
      hook_type: "hook"
    };
    var detailRows = Object.keys(detailLabels).filter(function (key) { return details[key] != null; })
      .map(function (key) { return [detailLabels[key], '<span class="mono">' + esc(details[key]) + '</span>']; });
    if (detailRows.length) body += '<div class="drawer-section"><h4>Evidence</h4>' + kvTable(detailRows) + '</div>';
    body += '<div class="drawer-section"><h4>Child activity</h4>' + historyTimeline(target) + '</div>';
    openDrawer("Run History", historyDisplayName(target, target !== root), body);
  }
  function renderRuns() {
    var hasHistory = (HISTORY.roots || []).length > 0;
    if (!hasHistory) {
      var runs = R.runs || [];
      var fallback = runs.length ? runs.map(function (run) {
        return '<div class="row runs-row" data-run="' + esc(run.run_id) + '"><span class="cmd">' + esc(run.command) + '</span><span class="when mono">' + esc(run.started_at || "—") + "</span>" + pill(run.display_status || run.status) + '<span class="dur">' + fmtDur(run.duration_s) + '</span><span class="mono faint">' + (run.stages || []).length + ' stages</span><span class="chev">›</span></div>';
      }).join("") : '<div class="empty"><div class="big">No history</div><p>No CoderMind activity has been recorded in this workspace.</p></div>';
      $("#view-runs").innerHTML = '<div class="page-head"><div><div class="eyebrow">History</div><h2>Run History</h2><p>Historical execution data from this workspace.</p></div></div><div class="surface"><div class="rows">' + fallback + '</div></div>';
      return;
    }
    var health = R.source_health || [];
    var visibleHealth = health.filter(function (h) {
      return historyCoverageExpanded || h.expectation === "required" || ["partial", "invalid", "unreadable"].indexOf(h.status) >= 0;
    });
    var healthGrid = '<div class="health">' + visibleHealth.map(function (h) {
      var statusLabel = h.status === "missing" ? "not found" : h.status === "not_applicable" ? "not applicable" : h.status;
      var recordHint = h.records != null ? " · " + h.records + " records" : "";
      return '<div class="h ' + esc(h.status) + '" title="' + esc((h.expectation || "optional") + " · " + statusLabel + recordHint) + '"><span class="dot"></span><span class="mono">' + esc(h.source) + '</span><span>' + esc(h.expectation || "optional") + " · " + esc(statusLabel) + "</span></div>";
    }).join("") + "</div>";
    var retention = HISTORY.retention || {}, usedMb = ((Number(retention.bytes_used) || 0) / 1048576).toFixed(1);
    var maxMb = ((Number(retention.max_bytes) || 0) / 1048576).toFixed(0);
    var controls = [["all", "All"], ["encoder", "Encoder"], ["decoder", "Decoder"], ["hooks", "Hooks"], ["mcp", "MCP"], ["failures", "Failures"]]
      .map(function (item) { return '<button class="' + (historyFilter === item[0] ? "active" : "") + '" data-history-filter="' + item[0] + '">' + item[1] + '</button>'; }).join("");
    $("#view-runs").innerHTML = '<div class="page-head history-head"><div><div class="eyebrow">History</div><h2>Run History</h2><p>All recorded CoderMind activity in this workspace. Pipeline shows only the latest state.</p></div></div>'
      + '<div class="history-toolbar"><div class="seg history-filters">' + controls + '</div><label class="history-search"><span>⌕</span><input id="historySearch" value="' + esc(historySearch) + '" placeholder="Search runs and stages"></label></div>'
      + '<div class="note history-retention"><span class="i">ℹ</span><span>History is retained automatically for <b>' + num(retention.days || 90) + ' days</b> or <b>' + esc(maxMb) + ' MB</b>. Current activity storage: <b>' + esc(usedMb) + ' MB</b>.</span></div>'
      + '<div class="surface history-surface"><div class="surface-head"><h3>Execution tree</h3><span class="hint" id="historyCount">' + num(filteredHistoryRoots().length) + ' roots · newest first</span></div><div id="historyTree">' + historyTreeHtml() + '</div></div>'
      + '<details class="surface history-diagnostics"><summary><span>Data coverage</span><small>Why some history details may be unavailable</small></summary><div class="surface-body">' + healthGrid
      + (health.length > visibleHealth.length ? '<button class="btn sm" data-history-coverage="all">Show all ' + health.length + ' sources</button>' : '')
      + '<div class="note" style="margin-top:12px"><span class="i">ℹ</span><span>Missing optional telemetry means the report cannot show that detail; it does not prove the activity failed or never occurred.</span></div></div></details>';
  }

  /* ---------- routing ---------- */
  var VIEWS = ["overview", "pipeline", "rpg", "runs"];
  var pipelineHighlightTimer = null;
  function currentView() {
    var h = (location.hash || "").replace(/^#/, "").split("/")[0];
    return VIEWS.indexOf(h) >= 0 ? h : "overview";
  }
  function pipelineTarget() {
    var parts = (location.hash || "").replace(/^#/, "").split("/");
    if (parts[0] !== "pipeline" || ["encoder", "decoder"].indexOf(parts[1]) < 0) return null;
    var cat = CATALOG[parts[1]];
    var phase = parts[2] && cat.phases.some(function (item) { return item.id === parts[2]; }) ? parts[2] : null;
    return document.getElementById("pipeline-" + parts[1] + (phase ? "-" + phase : ""));
  }
  function revealPipelineTarget() {
    var target = pipelineTarget();
    if (!target) { window.scrollTo(0, 0); return; }
    $$(".pipeline-jump-highlight").forEach(function (el) { el.classList.remove("pipeline-jump-highlight"); });
    target.classList.add("pipeline-jump-highlight");
    var reducedMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    target.scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: target.classList.contains("phase") ? "center" : "start" });
    try { target.focus({ preventScroll: true }); } catch (e) { target.focus(); }
    if (pipelineHighlightTimer) clearTimeout(pipelineHighlightTimer);
    pipelineHighlightTimer = setTimeout(function () { target.classList.remove("pipeline-jump-highlight"); }, 2400);
  }
  function route() {
    var v = currentView();
    $$(".view").forEach(function (el) { el.classList.toggle("active", el.id === "view-" + v); });
    $$("#nav button").forEach(function (b) { b.classList.toggle("active", b.dataset.view === v); });
    if (v === "overview") renderOverview();
    else if (v === "pipeline") renderPipeline();
    else if (v === "rpg") renderRpg();
    else if (v === "runs") renderRuns();
    if (v === "pipeline" && pipelineTarget()) requestAnimationFrame(revealPipelineTarget);
    else window.scrollTo(0, 0);
  }

  /* ---------- events ---------- */
  $("#nav").addEventListener("click", function (e) {
    var b = e.target.closest("button[data-view]"); if (!b) return;
    location.hash = b.dataset.view;
  });
  document.addEventListener("click", function (e) {
    var historyToggleEl = e.target.closest("[data-history-toggle]");
    if (historyToggleEl) {
      e.preventDefault(); e.stopPropagation();
      var historyRootId = historyToggleEl.getAttribute("data-history-toggle");
      collapsedHistoryRoots[historyRootId] = !collapsedHistoryRoots[historyRootId];
      var tree = $("#historyTree"); if (tree) tree.innerHTML = historyTreeHtml();
      return;
    }
    var historyFilterEl = e.target.closest("[data-history-filter]");
    if (historyFilterEl) { historyFilter = historyFilterEl.getAttribute("data-history-filter"); renderRuns(); return; }
    if (e.target.closest("[data-history-coverage]")) { historyCoverageExpanded = true; renderRuns(); var coverage = $(".history-diagnostics"); if (coverage) coverage.open = true; return; }
    var historyRowEl = e.target.closest("[data-history-root]");
    if (historyRowEl) {
      var rootId = historyRowEl.getAttribute("data-history-root"), targetId = historyRowEl.getAttribute("data-history-span");
      loadHistoryDetail(rootId, historyRowEl.getAttribute("data-history-detail")).then(function (root) { openHistoryDrawer(root, historyFind(root, targetId) || root); });
      return;
    }
    var pipelineJump = e.target.closest("[data-pipeline-jump]");
    if (pipelineJump) { location.hash = "pipeline/" + pipelineJump.getAttribute("data-pipeline-jump"); return; }
    var rmodeEl = e.target.closest("[data-rmode]");
    if (rmodeEl) { rpgMode = rmodeEl.getAttribute("data-rmode"); renderRpg(); return; }
    var filterEl = e.target.closest("[data-rpg-filter]");
    if (filterEl) {
      rpgChangeFilter = filterEl.getAttribute("data-rpg-filter");
      rpgChangeExpanded = false;
      rpgChangeListOpen = true;
      rpgFocusedNode = null;
      $$("[data-rpg-filter]").forEach(function (button) { button.classList.toggle("active", button === filterEl); });
      refreshChangeListPanel();
      postRpgGraphState();
      return;
    }
    if (e.target.closest("[data-rpg-list-close]")) {
      rpgChangeListOpen = false;
      rpgChangeSearch = "";
      refreshChangeListPanel();
      postRpgGraphState();
      return;
    }
    var expandEl = e.target.closest("[data-rpg-expand]");
    if (expandEl) {
      rpgChangeExpanded = expandEl.getAttribute("data-rpg-expand") === "expand";
      $("#rpgChangeList").innerHTML = changeNodeListHtml(rpgChangeRows());
      return;
    }
    var nodeEl = e.target.closest("[data-rpg-node]");
    if (nodeEl) {
      rpgFocusedNode = {
        node_id: nodeEl.getAttribute("data-rpg-node"),
        scope: nodeEl.getAttribute("data-rpg-scope"),
        kind: nodeEl.getAttribute("data-rpg-kind")
      };
      $$(".change-node-row").forEach(function (row) { row.classList.toggle("selected", row === nodeEl); });
      postRpgGraphState();
      return;
    }
    var contextEl = e.target.closest("[data-rpg-context]");
    if (contextEl) {
      rpgContextMode = contextEl.getAttribute("data-rpg-context");
      $$("[data-rpg-context]").forEach(function (button) { button.classList.toggle("active", button === contextEl); });
      postRpgGraphState();
      return;
    }
    var styleEl = e.target.closest("[data-rpg-full-style]");
    if (styleEl) {
      rpgFullStyle = styleEl.getAttribute("data-rpg-full-style");
      $$("[data-rpg-full-style]").forEach(function (button) { button.classList.toggle("active", button === styleEl); });
      postRpgGraphState();
      return;
    }
    if (e.target.closest("#rpgHistoryBtn")) { openRpgHistoryDrawer(); return; }
    var rverEl = e.target.closest("[data-rpg-commit]");
    if (rverEl) { rpgVersionDrawer(rverEl.getAttribute("data-rpg-commit")); return; }
    var runEl = e.target.closest("[data-run]");
    if (runEl) { runDrawer(runEl.getAttribute("data-run")); return; }
    var stEl = e.target.closest("[data-stage]");
    if (stEl) { var parts = stEl.getAttribute("data-stage").split(":"); stageDrawer(parts[0], parts[1], stEl.getAttribute("data-label")); return; }
  });
  document.addEventListener("input", function (e) {
    if (e.target.id === "historySearch") {
      historySearch = e.target.value;
      var historyTree = $("#historyTree"); if (historyTree) historyTree.innerHTML = historyTreeHtml();
      var historyCount = $("#historyCount"); if (historyCount) historyCount.textContent = filteredHistoryRoots().length + " roots · newest first";
      return;
    }
    if (e.target.id !== "rpgChangeSearch") return;
    rpgChangeSearch = e.target.value;
    rpgChangeExpanded = false;
    var panel = $("#rpgChangeList");
    if (!panel) return;
    panel.innerHTML = changeNodeListHtml(rpgChangeRows());
    var input = $("#rpgChangeSearch");
    if (input) {
      input.focus();
      input.setSelectionRange(input.value.length, input.value.length);
    }
  });
  $("#drawerClose").addEventListener("click", closeDrawer);
  $("#drawerScrim").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") closeDrawer(); });
  document.addEventListener("keydown", function (e) {
    if ((e.key === "Enter" || e.key === " ") && e.target.closest("[data-history-toggle]")) {
      e.preventDefault(); e.target.click();
    }
    else if ((e.key === "Enter" || e.key === " ") && e.target.closest("[data-history-root]")) {
      e.preventDefault(); e.target.closest("[data-history-root]").click();
    }
  });
  window.addEventListener("hashchange", route);
  function onFsChange() {
    var wrap = $("#rpgFrameWrap"), exit = $("#rpgFsExit");
    var fsEl = document.fullscreenElement || document.webkitFullscreenElement;
    if (exit) exit.hidden = fsEl !== wrap;
  }
  document.addEventListener("fullscreenchange", onFsChange);
  document.addEventListener("webkitfullscreenchange", onFsChange);

  /* ---------- theme ---------- */
  var themeBtn = $("#themeToggle");
  function currentTheme() {
    return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
  }
  function syncRpgTheme() {
    var frame = $("#rpgFrame");
    if (frame && frame.contentWindow) frame.contentWindow.postMessage({ type: "cmind:theme", theme: currentTheme() }, "*");
  }
  function setReportTheme(theme, persist) {
    var value = theme === "light" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", value);
    if (persist) { try { localStorage.setItem("cmind-report-theme", value); } catch (e) { } }
    applyThemeIcon();
    syncRpgTheme();
  }
  function applyThemeIcon() {
    var t = currentTheme();
    if (themeBtn) themeBtn.textContent = t === "light" ? "☀" : "☾";
  }
  if (themeBtn) themeBtn.addEventListener("click", function () {
    setReportTheme(currentTheme() === "light" ? "dark" : "light", true);
  });
  window.addEventListener("message", function (event) {
    var frame = $("#rpgFrame");
    if (!frame || event.source !== frame.contentWindow || !event.data || event.data.type !== "cmind:theme-change") return;
    setReportTheme(event.data.theme, true);
  });
  applyThemeIcon();

  /* ---------- boot ---------- */
  renderTop();
  if (!location.hash) location.hash = "overview";
  route();
})();
