import json
import argparse
from pathlib import Path

from common.paths import DATA_FLOW_FILE, DATA_FLOW_VIZ_FILE

# HTML Template
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RPG Kit Data Flow Architecture</title>
    <!-- ECharts -->
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <!-- Mermaid -->
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js"></script>
    <!-- SVG Pan Zoom -->
    <script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f8f9fa; color: #333; height: 100vh; display: flex; flex-direction: column; }
        header { background: #2c3e50; color: white; padding: 1rem 2rem; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        h2 { margin: 0; font-size: 1.2rem; }
        .controls { display: flex; gap: 10px; }
        button { background: #34495e; color: white; border: 1px solid #46607a; padding: 8px 16px; border-radius: 4px; cursor: pointer; transition: all 0.2s; font-size: 14px; }
        button:hover { background: #46607a; }
        button.active { background: #3498db; border-color: #3498db; font-weight: bold; }
        
        #main-container { flex: 1; position: relative; overflow: hidden; display: flex; justify-content: center; align-items: center; background: white; margin: 20px; border-radius: 8px; box-shadow: 0 0 15px rgba(0,0,0,0.05); }
        
        /* View Containers */
        .view-pane { width: 100%; height: 100%; position: absolute; top: 0; left: 0; visibility: hidden; opacity: 0; transition: opacity 0.3s; padding: 0; box-sizing: border-box; overflow: hidden; }
        .view-pane.active { visibility: visible; opacity: 1; }
        
        /* Mermaid Specific */
        #mermaid-view, #uml-view { display: flex; justify-content: center; align-items: center; width: 100%; height: 100%; background: #fdfdfd; }
        #mermaid-graph, #uml-graph { width: 100%; height: 100%; display: flex; justify-content: center; align-items: center; }
        /* Force SVG to fill container so pan-zoom works well */
        #mermaid-graph svg, #uml-graph svg { width: 100% !important; height: 100% !important; max-width: none !important; }
        
        /* Details Panel */
        #details-panel {
            position: absolute; bottom: 20px; right: 20px; width: 300px;
            background: white; border-radius: 8px; box-shadow: 0 5px 20px rgba(0,0,0,0.15);
            padding: 15px; border-left: 5px solid #3498db;
            display: none; z-index: 100;
        }
        #details-panel h3 { margin-top: 0; color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 8px; }
        #details-panel p { font-size: 14px; line-height: 1.5; color: #555; }
        .tag { display: inline-block; background: #eef2f7; padding: 2px 6px; border-radius: 4px; border: 1px solid #dce4ec; font-family: monospace; font-size: 12px; color: #3498db; margin-bottom: 5px; }

    </style>
</head>
<body>

<header>
    <h2>Data Flow: Architecture View</h2>
    <div class="controls">
        <button onclick="switchView('mermaid')" id="btn-mermaid" class="active">Flowchart (Structural)</button>
        <button onclick="switchView('uml')" id="btn-uml">UML Sequence</button>
        <button onclick="switchView('chord')" id="btn-chord">Chord (Relationships)</button>
    </div>
</header>
<div id="main-container">
    <!-- Mermaid Container -->
    <div id="mermaid-view" class="view-pane active">
        <div class="mermaid" id="mermaid-graph">
            <!-- Mermaid content will be injected here -->
        </div>
    </div>

    <!-- UML Container -->
    <div id="uml-view" class="view-pane">
        <div class="mermaid" id="uml-graph"></div>
    </div>

    <!-- ECharts Container -->
    <div id="chord-view" class="view-pane"></div>
</div>

<div id="details-panel">
    <h3 id="panel-title">Details</h3>
    <div id="panel-content"></div>
</div>

<script>
    // ---------------- Data Loading ----------------
    // INJECTED_DATA_START
    const rawData = __JSON_DATA_PLACEHOLDER__;
    // INJECTED_DATA_END

    // Color palette
    const colors = ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272', '#fc8452', '#9a60b4'];
    const nodeColors = {};
    if (rawData.subtree_order) {
        rawData.subtree_order.forEach((node, i) => nodeColors[node] = colors[i % colors.length]);
    } else {
        // Fallback if subtree_order is missing, collect unique nodes
        const nodes = new Set();
        rawData.data_flow.forEach(f => { nodes.add(f.source); nodes.add(f.target); });
        Array.from(nodes).forEach((node, i) => nodeColors[node] = colors[i % colors.length]);
        rawData.subtree_order = Array.from(nodes);
    }

    // ---------------- View Switching Logic ----------------
    let myChart = null;

    function switchView(viewName) {
        document.querySelectorAll('.view-pane').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.controls button').forEach(el => el.classList.remove('active'));
        
        document.getElementById(`btn-${viewName}`).classList.add('active');
        document.getElementById(`${viewName}-view`).classList.add('active');

        if (viewName === 'chord') {
            if (!myChart) initChordChart();
            else myChart.resize();
        } else if (viewName === 'uml') {
            initUML();
        }
    }

    // ---------------- UML Sequence Logic ----------------
    function initUML() {
        if (document.getElementById('uml-graph').getAttribute('data-processed')) return;
        
        const nodes = rawData.subtree_order;
        // Construct Mermaid Sequence Diagram
        let graphDef = 'sequenceDiagram\\n';
        graphDef += '    autonumber\\n';
        
        // Participants
        nodes.forEach(node => {
            const label = node.replace(/"/g, "'");
            graphDef += `    participant ${label}\\n`;
        });

        // Messages
        rawData.data_flow.forEach(flow => {
            const src = flow.source.replace(/"/g, "'");
            const tgt = flow.target.replace(/"/g, "'");
            // Wrap text for clearer display if too long? 
            // For now simple display
            const label = flow.data_id ? flow.data_id.replace(/"/g, "'") : "Data";
            graphDef += `    ${src}->>${tgt}: ${label}\\n`;
        });

        const element = document.getElementById('uml-graph');
        element.textContent = graphDef;
        element.removeAttribute('data-processed');
        
        mermaid.run({
            nodes: [element],
            suppressErrors: false
        }).then(() => {
            const svg = element.querySelector('svg');
            if (svg) {
                // Initialize SVG Pan Zoom
                svgPanZoom(svg, {
                    zoomEnabled: true,
                    controlIconsEnabled: true,
                    fit: true,
                    center: true,
                    minZoom: 0.1,
                    maxZoom: 10,
                    zoomScaleSensitivity: 0.4
                });
            }
        }).catch(err => console.error(err));
    }

    // ---------------- Mermaid Logic ----------------
    function initMermaid() {
        const nodes = rawData.subtree_order;
        // Construct Mermaid Graph Definition
        let graphDef = 'graph LR\\n';
        
        // Define Styles
        // Increased font size and stroke width for better visibility
        graphDef += `    classDef default fill:#f9f9f9,stroke:#333,stroke-width:3px,rx:8,ry:8,font-size:20px,font-weight:bold,padding:15px;\\n`;
        
        // Nodes
        nodes.forEach((node, idx) => {
            const safeId = "node_" + idx;
            // Mermaid safe label: escape quotes if necessary
            const label = node.replace(/"/g, "'");
            graphDef += `    ${safeId}["${label}"]\\n`;
            
            const color = nodeColors[node] || '#ccc';
            // Dark text color for contrast
            graphDef += `    style ${safeId} fill:${color},color:#222,stroke:#333,stroke-width:2px\\n`;
        });

        // Edges
        rawData.data_flow.forEach((flow, idx) => {
            const srcIdx = nodes.indexOf(flow.source);
            const tgtIdx = nodes.indexOf(flow.target);
            
            // If explicit order not found, fallback to name-based ID (less safe but works for fallback)
            // But here we assume input data controls nodes.
            if (srcIdx === -1 || tgtIdx === -1) {
                console.warn(`Node not found in subtree_order: ${flow.source} or ${flow.target}`);
                return; 
            }
            
            const srcId = "node_" + srcIdx;
            const tgtId = "node_" + tgtIdx;
            const label = flow.data_id ? ` -- "${flow.data_id}" --> ` : ` --> `;
            
            graphDef += `    ${srcId}${label}${tgtId}\\n`;
        });

        const element = document.getElementById('mermaid-graph');
        element.textContent = graphDef;
        element.removeAttribute('data-processed'); 
        
        mermaid.initialize({ startOnLoad: false, securityLevel: 'loose' });
        
        mermaid.run({
            nodes: [element],
            suppressErrors: false
        }).then(() => {
            const svg = element.querySelector('svg');
            if (svg) {
                // Initialize SVG Pan Zoom
                var panZoom = svgPanZoom(svg, {
                    zoomEnabled: true,
                    controlIconsEnabled: true,
                    fit: false, // Changed to false so it doesn't shrink heavily
                    center: true, // Keep it centered
                    minZoom: 0.1,
                    maxZoom: 10,
                    zoomScaleSensitivity: 0.4 // smoother zoom
                });
                
                // Ensure it starts at a good readable size (100% or slightly less if huge)
                panZoom.zoom(1.0);
                panZoom.center();
            }
        }).catch(err => console.error(err));
    }

    // ---------------- ECharts Chord Logic ----------------
    function initChordChart() {
        const dom = document.getElementById("chord-view");
        myChart = echarts.init(dom);
        
        const nodes = rawData.subtree_order.map(name => ({ name: name }));
        const links = rawData.data_flow.map(flow => ({
            source: flow.source,
            target: flow.target,
            value: 1,
            info: flow // Store full info for tooltip
        }));

        const option = {
            tooltip: {
                trigger: 'item',
                formatter: function (params) {
                    if (params.dataType === 'edge') {
                        const info = params.data.info;
                        return `<b>${info.source} → ${info.target}</b><br/>
                                Data: ${info.data_type}<br/>
                                <i>${info.transformation ? info.transformation.substring(0, 50) + "..." : ""}</i>`;
                    } else {
                        return params.name;
                    }
                }
            },
            series: [{
                type: 'graph',
                layout: 'circular',
                circular: { rotateLabel: true },
                data: nodes,
                links: links,
                roam: true,
                label: { position: 'right', formatter: '{b}' },
                lineStyle: { color: 'source', curveness: 0.3 },
                itemStyle: {
                    color: (params) => nodeColors[params.name]
                },
                emphasis: { focus: 'adjacency', lineStyle: { width: 4 } }
            }]
        };
        myChart.setOption(option);
        
        myChart.on('click', function(params) {
            if (params.dataType === 'edge') {
                showDetails(params.data.info);
            }
        });
        
        window.addEventListener('resize', myChart.resize);
    }

    // ---------------- Shared Details Logic ----------------
    function showDetails(flowData) {
        const panel = document.getElementById('details-panel');
        const content = document.getElementById('panel-content');
        const title = document.getElementById('panel-title');
        
        panel.style.display = 'block';
        title.innerHTML = `${flowData.source} <span style="font-size:0.8em"> -></span> ${flowData.target}`;
        
        content.innerHTML = `
            <div class="tag">${flowData.data_id || 'N/A'}</div>
            <div class="tag" style="background:#fff3cd; color:#856404; border-color:#ffeeba">${flowData.data_type || 'Unknown'}</div>
            <p>${flowData.transformation || 'No description available.'}</p>
        `;
    }
    
    // Initialize
    initMermaid();

</script>

</body>
</html>
"""

def generate_visualization(json_path, output_path):
    print(f"Reading data from: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found {json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {json_path}")
        return

    # Process data to JSON string
    json_str = json.dumps(data, indent=2)
    
    # Inject into HTML
    html_content = HTML_TEMPLATE.replace('__JSON_DATA_PLACEHOLDER__', json_str)
    
    print(f"Writing visualization to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Data Flow Visualization")
    parser.add_argument("--input", "-i", type=Path, default=DATA_FLOW_FILE, help="Input data flow JSON file")
    parser.add_argument("--output", "-o", type=Path, default=DATA_FLOW_VIZ_FILE, help="Output HTML file")
    
    args = parser.parse_args()
    
    generate_visualization(args.input, args.output)
