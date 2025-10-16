let profilingData = null;
let expandedNodes = new Set();
let minTimeFilter = 0;
let currentDirectory = '';
let selectedFilePath = null;
let hardwareSpecs = null;
let detectedHardware = null;
let selectedHardwareKey = null;

// Initialize
window.addEventListener('load', () => {
    setupFileLoader();
    loadHardwareSpecs();

    // Ensure expand all checkbox is unchecked by default
    document.getElementById('expandAll').checked = false;

    // Auto-load if data is already available
    fetch('/data.json')
        .then(r => {
            if (r.ok) return r.json();
            throw new Error('No initial data');
        })
        .then(data => {
            profilingData = data;
            showContent();
            renderAll();
        })
        .catch(() => {
            // No initial data, show file loader
            console.log('No initial data, waiting for file selection');
        })
        .finally(() => {
            // Load directory browser regardless
            loadDirectory(null);
        });
});

function setupFileLoader() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');

    // Click to browse
    dropZone.addEventListener('click', () => fileInput.click());

    // File input change
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) loadFromFile(file);
    });

    // Drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && file.name.endsWith('.json')) {
            loadFromFile(file);
        } else {
            alert('Please drop a valid JSON file');
        }
    });
}

function loadFromFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const data = JSON.parse(e.target.result);
            profilingData = data;
            showContent();
            renderAll();
        } catch (err) {
            alert('Error parsing JSON: ' + err.message);
        }
    };
    reader.readAsText(file);
}

function loadDirectory(path) {
    // Show loading state
    const browser = document.getElementById('directoryBrowser');
    browser.classList.add('loading');

    const pathDisplay = document.getElementById('currentPath');
    const dirList = document.getElementById('dirList');
    const fileList = document.getElementById('fileList');

    // Clear previous content
    dirList.innerHTML = '';
    fileList.innerHTML = '';

    currentDirectory = path;

    const url = (path === null || path === '') ? '/list_dirs' : `/list_dirs?path=${encodeURIComponent(path)}`;
    const fileUrl = (path === null || path === '') ? '/list_files' : `/list_files?path=${encodeURIComponent(path)}`;

    // Load both directories and files in parallel
    Promise.all([
        fetch(url).then(r => r.json()),
        fetch(fileUrl).then(r => r.json())
    ])
    .then(([dirs, files]) => {
        console.log('Loaded directories:', dirs);
        console.log('Loaded files:', files);

        // Update path display
        if (path) {
            pathDisplay.textContent = path;
        } else {
            pathDisplay.textContent = '(current directory)';
        }

        // Render directories
        if (dirs.length === 0) {
            dirList.innerHTML = '<li style="padding: 10px 15px; color: #999; font-size: 13px;">No subdirectories</li>';
        } else {
            dirs.forEach(dir => {
                const li = document.createElement('li');
                li.className = 'dir-item';
                li.innerHTML = `
                    <span class="dir-icon">📁</span>
                    <div class="file-info">
                        <div class="dir-name">${dir.name}</div>
                    </div>
                `;
                li.onclick = () => loadDirectory(dir.path);
                dirList.appendChild(li);
            });
        }

        // Render files
        if (files.length === 0) {
            fileList.innerHTML = '<li style="padding: 10px 15px; color: #999; font-size: 13px;">No JSON files in this directory</li>';
        } else {
            files.forEach(file => {
                const li = document.createElement('li');
                li.className = 'file-item';
                const date = new Date(file.mtime * 1000).toLocaleString();
                const size = (file.size / 1024).toFixed(1);

                li.innerHTML = `
                    <span class="file-icon">📄</span>
                    <div class="file-info">
                        <div class="file-name">${file.name}</div>
                        <div class="file-meta">${size} KB • ${date}</div>
                    </div>
                `;
                li.onclick = () => selectFile(file.path, li);
                fileList.appendChild(li);
            });
        }
    })
    .catch(err => {
        console.error('Error loading directory:', err);
        pathDisplay.textContent = 'Error loading directory';
        dirList.innerHTML = '<li style="padding: 15px; color: #f44; font-size: 13px;">Error loading directories</li>';
        fileList.innerHTML = '<li style="padding: 15px; color: #f44; font-size: 13px;">Error loading files</li>';
    })
    .finally(() => {
        // Hide loading state
        browser.classList.remove('loading');
    });
}

function selectFile(path, element) {
    // Remove previous selection
    document.querySelectorAll('.file-item').forEach(item => {
        item.classList.remove('selected');
    });

    // Mark as selected
    element.classList.add('selected');
    selectedFilePath = path;
    document.getElementById('loadButton').disabled = false;
}

function loadSelectedFile() {
    if (!selectedFilePath) {
        alert('Please select a file');
        return;
    }

    fetch(`/load_file?path=${encodeURIComponent(selectedFilePath)}`)
        .then(r => r.json())
        .then(data => {
            profilingData = data;
            showContent();
            renderAll();
        })
        .catch(err => alert('Error loading file: ' + err.message));
}

function showContent() {
    const fileLoader = document.getElementById('fileLoader');
    fileLoader.classList.add('compact');
    fileLoader.querySelector('h3').textContent = '📂 Load Another File';
    document.getElementById('contentWrapper').classList.add('visible');
}

function switchTab(tab) {
    document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

    event.target.classList.add('active');
    document.getElementById(tab + '-tab').classList.add('active');

    // Re-render flame graph when switching to it (to get correct width)
    if (tab === 'flame') {
        setTimeout(() => renderFlameGraph(), 10);
    }

    // Re-render tensor flow when switching to it
    if (tab === 'tensor') {
        setTimeout(() => renderTensorFlow(), 10);
    }

    // Re-render bottleneck analysis when switching to it
    if (tab === 'bottleneck') {
        setTimeout(() => renderBottleneckAnalysis(), 10);
    }
}

function renderAll() {
    renderMetadata();
    renderStats();
    renderTree();
    renderTable();
    renderFlameGraph();
    renderTensorFlow();
    renderBottleneckAnalysis();
}

function renderMetadata() {
    const meta = document.getElementById('metadata');
    const formatBadge = profilingData.format
        ? `<span class="format-badge format-${profilingData.format}">${profilingData.format.replace('_', ' ')}</span>`
        : '';
    meta.innerHTML = `
        <strong>Timestamp:</strong> ${new Date(profilingData.timestamp).toLocaleString()}
        ${formatBadge}
    `;
}

function renderStats() {
    const stats = document.getElementById('stats');
    const tree = profilingData.profiling_tree;

    let totalTime = 0;
    let totalOps = 0;
    let maxTime = 0;
    let maxOp = '';
    let totalDataAccess = 0;
    let maxDataAccess = 0;
    let maxDataOp = '';

    function traverse(nodes) {
        nodes.forEach(node => {
            totalOps++;
            totalTime += node.avg_latency_ms;
            if (node.avg_latency_ms > maxTime) {
                maxTime = node.avg_latency_ms;
                maxOp = node.name;
            }
            if (node.total_data_access_mb) {
                totalDataAccess += node.total_data_access_mb;
                if (node.total_data_access_mb > maxDataAccess) {
                    maxDataAccess = node.total_data_access_mb;
                    maxDataOp = node.name;
                }
            }
            if (node.children) traverse(node.children);
        });
    }

    traverse(tree);

    const statCards = `
        <div class="stat-card">
            <div class="label">Total Time</div>
            <div class="value">${tree[0].avg_latency_ms.toFixed(2)}<span class="unit">ms</span></div>
        </div>
        <div class="stat-card">
            <div class="label">Operations</div>
            <div class="value">${totalOps}</div>
        </div>
        <div class="stat-card">
            <div class="label">Slowest Operation</div>
            <div class="value">${maxOp}</div>
            <div style="font-size: 12px; color: #999; margin-top: 4px;">${maxTime.toFixed(2)} ms</div>
        </div>
        ${totalDataAccess > 0 ? `
        <div class="stat-card">
            <div class="label">Total Data Access</div>
            <div class="value">${totalDataAccess.toFixed(1)}<span class="unit">MB</span></div>
        </div>
        ` : ''}
        ${maxDataAccess > 0 ? `
        <div class="stat-card">
            <div class="label">Largest Data Op</div>
            <div class="value">${maxDataOp}</div>
            <div style="font-size: 12px; color: #999; margin-top: 4px;">${maxDataAccess.toFixed(1)} MB</div>
        </div>
        ` : ''}
    `;

    stats.innerHTML = statCards;
}

function renderTree() {
    const container = document.getElementById('treeView');
    container.innerHTML = '<ul class="tree-node">' +
        profilingData.profiling_tree.map(node => renderNode(node)).join('') +
        '</ul>';
}

function renderNode(node) {
    if (node.avg_latency_ms < minTimeFilter) return '';

    const hasChildren = node.children && node.children.length > 0;
    const nodeId = node.full_path;
    const isExpanded = expandedNodes.has(nodeId);

    // Use percent_of_total if available, otherwise fall back to 100 for root
    const percent = node.percent_of_total !== undefined ? node.percent_of_total.toFixed(1) : '100.0';

    // Build tensor info tooltip
    let tensorTooltip = '';
    if (node.typical_input_tensors || node.typical_output_tensors || node.total_data_access_mb) {
        tensorTooltip = buildTensorTooltip(node);
    }

    return `
        <li class="tree-node">
            <div class="node-content ${tensorTooltip ? 'has-tensor-info' : ''}"
                 onclick="toggleNode('${nodeId}')"
                 style="cursor: ${hasChildren ? 'pointer' : 'default'}"
                 ${tensorTooltip ? `title="${escapeHtml(tensorTooltip)}"` : ''}>
                <span class="toggle ${hasChildren ? '' : 'empty'}">
                    ${hasChildren ? (isExpanded ? '▼' : '▶') : ''}
                </span>
                <span class="node-name">${node.name}</span>
                <span class="node-time">${node.avg_latency_ms.toFixed(2)} ms</span>
                ${node.total_data_access_mb ? `<span class="node-data-access">${node.total_data_access_mb.toFixed(1)} MB</span>` : ''}
                <span class="node-percent">${percent}%</span>
                <div class="node-bar">
                    <div class="node-bar-fill" style="width: ${Math.min(parseFloat(percent), 100)}%"></div>
                </div>
                <span class="node-samples">${node.samples} sample${node.samples > 1 ? 's' : ''}</span>
            </div>
            ${hasChildren ? `
                <ul class="children ${isExpanded ? '' : 'hidden'}" id="children-${nodeId}">
                    ${node.children.map(child => renderNode(child)).join('')}
                </ul>
            ` : ''}
        </li>
    `;
}

function buildTensorTooltip(node) {
    let tooltip = [];

    if (node.typical_input_tensors && node.typical_input_tensors.length > 0) {
        tooltip.push(`Input Tensors (${node.typical_input_tensors.length}):`);
        node.typical_input_tensors.forEach((t, idx) => {
            const shape = t.shape.join('x');
            tooltip.push(`  ${idx + 1}: [${shape}] ${t.size_mb.toFixed(2)} MB (${t.device}, ${t.dtype})`);
        });
    }

    if (node.typical_output_tensors && node.typical_output_tensors.length > 0) {
        tooltip.push(`Output Tensors (${node.typical_output_tensors.length}):`);
        node.typical_output_tensors.forEach((t, idx) => {
            const shape = t.shape.join('x');
            tooltip.push(`  ${idx + 1}: [${shape}] ${t.size_mb.toFixed(2)} MB (${t.device}, ${t.dtype})`);
        });
    }

    if (node.total_data_access_mb) {
        tooltip.push(`Total Data Access: ${node.total_data_access_mb.toFixed(2)} MB`);
    }

    return tooltip.join('\n');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function toggleNode(nodeId) {
    if (expandedNodes.has(nodeId)) {
        expandedNodes.delete(nodeId);
    } else {
        expandedNodes.add(nodeId);
    }
    renderTree();
}

function toggleExpandAll() {
    const expandAll = document.getElementById('expandAll').checked;

    function collectAllNodes(nodes) {
        let ids = [];
        nodes.forEach(node => {
            ids.push(node.full_path);
            if (node.children) {
                ids = ids.concat(collectAllNodes(node.children));
            }
        });
        return ids;
    }

    if (expandAll) {
        expandedNodes = new Set(collectAllNodes(profilingData.profiling_tree));
    } else {
        expandedNodes.clear();
    }
    renderTree();
}

function applyFilters() {
    minTimeFilter = parseFloat(document.getElementById('minTime').value) || 0;
    renderTree();
    renderTable();
}

function renderTable() {
    const container = document.getElementById('tableView');

    let rows = [];
    function traverse(nodes, depth = 0) {
        nodes.forEach(node => {
            if (node.avg_latency_ms >= minTimeFilter) {
                rows.push({...node, depth});
                if (node.children) traverse(node.children, depth + 1);
            }
        });
    }
    traverse(profilingData.profiling_tree);

    // Sort by time descending
    rows.sort((a, b) => b.avg_latency_ms - a.avg_latency_ms);

    container.innerHTML = `
        <table>
            <thead>
                <tr>
                    <th>Operation</th>
                    <th>Avg Time (ms)</th>
                    <th>Data Access (MB)</th>
                    <th>Input Tensors</th>
                    <th>Output Tensors</th>
                    <th>Min (ms)</th>
                    <th>Max (ms)</th>
                    <th>Std Dev (ms)</th>
                    <th>Samples</th>
                    <th>% of Total</th>
                </tr>
            </thead>
            <tbody>
                ${rows.map(row => {
                    const inputInfo = row.input_tensor_summary
                        ? `${row.input_tensor_summary.num_tensors} (${row.input_tensor_summary.total_size_mb.toFixed(1)} MB)`
                        : '-';
                    const outputInfo = row.output_tensor_summary
                        ? `${row.output_tensor_summary.num_tensors} (${row.output_tensor_summary.total_size_mb.toFixed(1)} MB)`
                        : '-';
                    const dataAccess = row.total_data_access_mb ? row.total_data_access_mb.toFixed(2) : '-';

                    return `
                        <tr class="${row.total_data_access_mb > 100 ? 'high-memory-op' : ''}">
                            <td style="padding-left: ${row.depth * 20 + 12}px">${row.name}</td>
                            <td><strong>${row.avg_latency_ms.toFixed(3)}</strong></td>
                            <td>${dataAccess}</td>
                            <td>${inputInfo}</td>
                            <td>${outputInfo}</td>
                            <td>${row.min_ms.toFixed(3)}</td>
                            <td>${row.max_ms.toFixed(3)}</td>
                            <td>${row.std_dev_ms.toFixed(3)}</td>
                            <td>${row.samples}</td>
                            <td>${row.percent_of_total ? row.percent_of_total.toFixed(1) + '%' : '-'}</td>
                        </tr>
                    `;
                }).join('')}
            </tbody>
        </table>
    `;
}

function renderFlameGraph() {
    const container = document.getElementById('flameView');

    if (!profilingData || !profilingData.profiling_tree) {
        console.error('No profiling data available for flame graph');
        container.innerHTML = '<div style="padding: 20px; color: #999;">No profiling data loaded</div>';
        return;
    }

    // Get container width, use a minimum if container is hidden
    const containerWidth = container.clientWidth || container.parentElement.clientWidth || 1200;
    const width = Math.max(containerWidth - 60, 600);
    const barHeight = 20;
    const barGap = 1;

    console.log('Rendering flame graph, container width:', containerWidth, 'svg width:', width);
    console.log('Profiling tree:', profilingData.profiling_tree);

    // Calculate total height needed
    function getMaxDepth(node, depth = 0) {
        if (!node.children || node.children.length === 0) return depth;
        return Math.max(...node.children.map(c => getMaxDepth(c, depth + 1)));
    }

    const maxDepth = Math.max(...profilingData.profiling_tree.map(n => getMaxDepth(n))) + 1;
    const height = maxDepth * (barHeight + barGap) + 40;

    console.log('Max depth:', maxDepth, 'Height:', height);

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', width);
    svg.setAttribute('height', height);
    svg.style.background = '#fff';

    // Calculate total time for root width calculation
    const totalTime = profilingData.profiling_tree.reduce((sum, n) => sum + n.avg_latency_ms, 0);

    function renderFlameNode(node, x, nodeWidth, depth) {
        if (nodeWidth < 0.5) return;

        const y = depth * (barHeight + barGap);

        // Material Design color palette
        const materialColors = [
            '#EF5350', // Red 400
            '#EC407A', // Pink 400
            '#AB47BC', // Purple 400
            '#7E57C2', // Deep Purple 400
            '#5C6BC0', // Indigo 400
            '#42A5F5', // Blue 400
            '#29B6F6', // Light Blue 400
            '#26C6DA', // Cyan 400
            '#26A69A', // Teal 400
            '#66BB6A', // Green 400
            '#9CCC65', // Light Green 400
            '#D4E157', // Lime 400
            '#FFEE58', // Yellow 400
            '#FFCA28', // Amber 400
            '#FFA726', // Orange 400
            '#FF7043', // Deep Orange 400
        ];

        // Create rect
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', x);
        rect.setAttribute('y', y);
        rect.setAttribute('width', nodeWidth);
        rect.setAttribute('height', barHeight);
        rect.setAttribute('class', 'flame-rect');

        // Pick color based on depth, cycling through material palette
        const color = materialColors[depth % materialColors.length];
        rect.setAttribute('fill', color);

        // Add tooltip
        const percent = node.percent_of_total ? node.percent_of_total.toFixed(1) : '?';
        const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        title.textContent = `${node.name}\n${node.avg_latency_ms.toFixed(2)} ms (${percent}%)\n${node.samples} sample${node.samples > 1 ? 's' : ''}`;
        rect.appendChild(title);

        svg.appendChild(rect);

        // Add text if wide enough
        if (nodeWidth > 40) {
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', x + 4);
            text.setAttribute('y', y + barHeight / 2 + 4);
            text.setAttribute('class', 'flame-text');
            text.setAttribute('fill', '#fff');
            text.setAttribute('font-size', '11px');
            text.setAttribute('font-family', 'sans-serif');

            // Truncate text to fit
            const maxChars = Math.floor(nodeWidth / 7);
            const displayText = node.name.length > maxChars
                ? node.name.substring(0, maxChars - 1) + '…'
                : node.name;
            text.textContent = displayText;
            svg.appendChild(text);
        }

        // Render children
        if (node.children && node.children.length > 0) {
            let childX = x;
            const nodeTime = node.avg_latency_ms;

            node.children.forEach(child => {
                // Calculate child width proportional to parent width
                const childProportion = child.avg_latency_ms / nodeTime;
                const childWidth = nodeWidth * childProportion;

                renderFlameNode(child, childX, childWidth, depth + 1);
                childX += childWidth;
            });
        }
    }

    // Render all root nodes
    let currentX = 0;
    profilingData.profiling_tree.forEach(node => {
        const nodeWidth = (node.avg_latency_ms / totalTime) * width;
        console.log(`Rendering root node: ${node.name}, width: ${nodeWidth}, x: ${currentX}`);
        renderFlameNode(node, currentX, nodeWidth, 0);
        currentX += nodeWidth;
    });

    console.log('Total SVG children:', svg.childNodes.length);

    container.innerHTML = '';
    container.appendChild(svg);

    console.log('Flame graph rendered successfully');
}

function renderTensorFlow() {
    const container = document.getElementById('tensorView');

    if (!profilingData || !profilingData.profiling_tree) {
        container.innerHTML = '<div class="no-data">No profiling data loaded</div>';
        return;
    }

    // Build tensor dependency graph
    const tensorGraph = buildTensorDependencyGraph(profilingData.profiling_tree);

    if (!tensorGraph.operations.length) {
        container.innerHTML = `
            <div class="no-data">
                <p>No tensor information available in this profile.</p>
                <p style="margin-top: 10px; color: #999; font-size: 13px;">
                    Tensor tracking is only available when profiling with tensor metadata enabled.
                </p>
            </div>
        `;
        return;
    }

    // Render tensor flow summary
    const html = `
        <div class="tensor-summary">
            <div class="tensor-stat-card">
                <div class="tensor-stat-label">Total Operations</div>
                <div class="tensor-stat-value">${tensorGraph.operations.length}</div>
            </div>
            <div class="tensor-stat-card">
                <div class="tensor-stat-label">Unique Tensors</div>
                <div class="tensor-stat-value">${tensorGraph.uniqueTensors.size}</div>
            </div>
            <div class="tensor-stat-card">
                <div class="tensor-stat-label">Total Data Flow</div>
                <div class="tensor-stat-value">${tensorGraph.totalDataFlow.toFixed(1)} MB</div>
            </div>
            <div class="tensor-stat-card">
                <div class="tensor-stat-label">Max Tensor Size</div>
                <div class="tensor-stat-value">${tensorGraph.maxTensorSize.toFixed(1)} MB</div>
            </div>
        </div>

        <div class="tensor-operations-list">
            <h4>Operations with Tensor Data</h4>
            <table class="tensor-table">
                <thead>
                    <tr>
                        <th>Operation</th>
                        <th>Input</th>
                        <th>Output</th>
                        <th>Data Transfer</th>
                        <th>Time (ms)</th>
                    </tr>
                </thead>
                <tbody>
                    ${tensorGraph.operations.map(op => `
                        <tr class="${op.dataTransfer > 100 ? 'high-data-transfer' : ''}">
                            <td><strong>${op.name}</strong></td>
                            <td>
                                ${op.inputTensors.map(t =>
                                    `<div class="tensor-badge">${t.shape.join('×')} (${t.size_mb.toFixed(1)} MB)</div>`
                                ).join('')}
                            </td>
                            <td>
                                ${op.outputTensors.map(t =>
                                    `<div class="tensor-badge">${t.shape.join('×')} (${t.size_mb.toFixed(1)} MB)</div>`
                                ).join('')}
                            </td>
                            <td><strong>${op.dataTransfer.toFixed(1)} MB</strong></td>
                            <td>${op.time.toFixed(2)} ms</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>

        ${tensorGraph.warnings.length > 0 ? `
        <div class="tensor-warnings">
            <h4>⚠️ Performance Warnings</h4>
            <ul>
                ${tensorGraph.warnings.map(w => `<li>${w}</li>`).join('')}
            </ul>
        </div>
        ` : ''}
    `;

    container.innerHTML = html;
}

function buildTensorDependencyGraph(tree) {
    const operations = [];
    const uniqueTensors = new Set();
    let totalDataFlow = 0;
    let maxTensorSize = 0;
    const warnings = [];

    function traverse(nodes) {
        nodes.forEach(node => {
            if (node.typical_input_tensors || node.typical_output_tensors) {
                const inputTensors = node.typical_input_tensors || [];
                const outputTensors = node.typical_output_tensors || [];

                // Track unique tensors
                inputTensors.forEach(t => uniqueTensors.add(t.id));
                outputTensors.forEach(t => uniqueTensors.add(t.id));

                // Calculate data transfer
                const inputSize = inputTensors.reduce((sum, t) => sum + t.size_mb, 0);
                const outputSize = outputTensors.reduce((sum, t) => sum + t.size_mb, 0);
                const dataTransfer = inputSize + outputSize;

                totalDataFlow += dataTransfer;

                // Track max tensor size
                [...inputTensors, ...outputTensors].forEach(t => {
                    if (t.size_mb > maxTensorSize) {
                        maxTensorSize = t.size_mb;
                    }
                });

                // Generate warnings
                if (dataTransfer > 1000) {
                    warnings.push(`${node.name}: Very large data transfer (${dataTransfer.toFixed(0)} MB)`);
                }

                if (node.avg_latency_ms > 50 && dataTransfer > 100) {
                    warnings.push(`${node.name}: High latency with large data transfer may indicate memory bottleneck`);
                }

                operations.push({
                    name: node.name,
                    inputTensors,
                    outputTensors,
                    dataTransfer,
                    time: node.avg_latency_ms,
                });
            }

            if (node.children) traverse(node.children);
        });
    }

    traverse(tree);

    // Sort operations by data transfer (descending)
    operations.sort((a, b) => b.dataTransfer - a.dataTransfer);

    return {
        operations,
        uniqueTensors,
        totalDataFlow,
        maxTensorSize,
        warnings,
    };
}

// ============================================================================
// Bottleneck Analysis (Roofline Model)
// ============================================================================

function loadHardwareSpecs() {
    // Load hardware specs from JSON
    fetch('/hardware_specs.json')
        .then(r => r.json())
        .then(specs => {
            hardwareSpecs = specs;
            console.log('Hardware specs loaded:', Object.keys(specs));
        })
        .catch(err => console.error('Failed to load hardware specs:', err));

    // Detect current hardware
    fetch('/detect_hardware')
        .then(r => r.json())
        .then(hw => {
            detectedHardware = hw;
            console.log('Detected hardware:', hw);

            // Auto-select detected hardware
            if (hw.detected && hw.vendor && hw.chip) {
                if (hw.vendor === 'apple') {
                    selectedHardwareKey = `apple_silicon.${hw.chip}`;
                } else if (hw.vendor === 'nvidia') {
                    selectedHardwareKey = `nvidia.${hw.chip}`;
                }
                console.log('Auto-selected hardware:', selectedHardwareKey);
            }
        })
        .catch(err => console.error('Failed to detect hardware:', err));
}

function getHardwareSpec(customKey = null) {
    // Use custom key if provided, otherwise use selected or detected hardware
    const key = customKey || selectedHardwareKey;

    if (!hardwareSpecs) {
        console.warn('Hardware specs not loaded, using fallback');
        return getFallbackSpec();
    }

    // Parse key like "apple_silicon.M4_Pro" or "nvidia.RTX_4090"
    if (key && key.includes('.')) {
        const [vendor, chip] = key.split('.');
        if (hardwareSpecs[vendor] && hardwareSpecs[vendor][chip]) {
            const spec = { ...hardwareSpecs[vendor][chip] };
            // Calculate ridge point
            spec.ridge_point = (spec.peak_tflops_fp16 * 1e12) / (spec.peak_bandwidth_gbs * 1e9);
            return spec;
        }
    }

    // Fallback to M1 if nothing specified
    return getFallbackSpec();
}

function getFallbackSpec() {
    return {
        name: 'Generic Apple Silicon',
        peak_tflops_fp16: 10.0,
        peak_bandwidth_gbs: 200,
        ridge_point: 50.0,
    };
}

function getAllHardwareOptions() {
    if (!hardwareSpecs) return [];

    const options = [];

    // Add Apple Silicon options
    if (hardwareSpecs.apple_silicon) {
        Object.entries(hardwareSpecs.apple_silicon).forEach(([key, spec]) => {
            options.push({
                key: `apple_silicon.${key}`,
                label: spec.name,
                vendor: 'Apple Silicon',
            });
        });
    }

    // Add NVIDIA options
    if (hardwareSpecs.nvidia) {
        Object.entries(hardwareSpecs.nvidia).forEach(([key, spec]) => {
            options.push({
                key: `nvidia.${key}`,
                label: spec.name,
                vendor: 'NVIDIA',
            });
        });
    }

    return options;
}

function analyzeBottleneck(operation, hardwareSpec) {
    if (!operation.duration_ms || !operation.flops || !operation.data_bytes) {
        return null;
    }

    const duration_sec = operation.duration_ms / 1000;
    const ridge_point = hardwareSpec.ridge_point;
    const arithmetic_intensity = operation.arithmetic_intensity ||
                                  (operation.flops / Math.max(operation.data_bytes, 1));

    // Calculate actual performance
    const actual_tflops = (operation.flops / 1e12) / duration_sec;
    const actual_bandwidth_gbs = (operation.data_bytes / 1e9) / duration_sec;

    // Calculate utilization
    const compute_utilization = actual_tflops / hardwareSpec.peak_tflops_fp16;
    const bandwidth_utilization = actual_bandwidth_gbs / hardwareSpec.peak_bandwidth_gbs;

    // Determine bottleneck spectrum position (-2.0 to +2.0)
    const log_ai = Math.log10(Math.max(arithmetic_intensity, 0.01));
    const log_ridge = Math.log10(ridge_point);
    const bottleneck_spectrum = Math.max(-2, Math.min(2, (log_ai - log_ridge)));

    // Primary bottleneck
    const primary_bottleneck = arithmetic_intensity < ridge_point ? 'bandwidth' : 'compute';
    const bottleneck_severity = Math.abs(bottleneck_spectrum);

    return {
        primary_bottleneck,
        bottleneck_spectrum,
        bottleneck_severity,
        heat_color: getBottleneckHeatColor(bottleneck_spectrum),
        arithmetic_intensity,
        ridge_point,
        compute_utilization: Math.min(1, compute_utilization),
        bandwidth_utilization: Math.min(1, bandwidth_utilization),
        limiting_factor: primary_bottleneck,
        efficiency: primary_bottleneck === 'bandwidth' ? bandwidth_utilization : compute_utilization,
        actual_tflops,
        actual_bandwidth_gbs,
    };
}

function getBottleneckHeatColor(spectrum) {
    // Convert bottleneck spectrum position to heat color
    // -2.0 → Dark Red (very bandwidth-bound)
    // -1.0 → Red (bandwidth-bound)
    //  0.0 → Purple (balanced)
    // +1.0 → Blue (compute-bound)
    // +2.0 → Dark Blue (very compute-bound)

    const t = (spectrum + 2) / 4; // Normalize to 0-1

    let r, g, b;
    if (t < 0.5) {
        // Bandwidth side: Red → Purple
        const local_t = t * 2;
        r = 255;
        g = Math.floor(50 * local_t);
        b = Math.floor(128 * local_t);
    } else {
        // Compute side: Purple → Blue
        const local_t = (t - 0.5) * 2;
        r = Math.floor(128 * (1 - local_t));
        g = Math.floor(50 * (1 - local_t));
        b = Math.floor(128 + 127 * local_t);
    }

    return `rgb(${r}, ${g}, ${b})`;
}

function renderBottleneckAnalysis() {
    const container = document.getElementById('bottleneckView');
    if (!container) return;

    if (!profilingData.operation_log || profilingData.operation_log.length === 0) {
        container.innerHTML = `
            <div class="bottleneck-warning">
                <h3>⚠️ No Operation Data Available</h3>
                <p>Bottleneck analysis requires operation timing and metrics data.</p>
                <p>This data is available in profiles generated with hook-based tracking enabled.</p>
            </div>
        `;
        return;
    }

    const hardwareSpec = getHardwareSpec();
    const operations = profilingData.operation_log;

    // Analyze all operations
    const analysisResults = operations.map(op => ({
        operation: op,
        analysis: analyzeBottleneck(op, hardwareSpec)
    }));

    const analyses = analysisResults.filter(item => item.analysis !== null);
    const skippedOps = analysisResults.filter(item => item.analysis === null);

    // Log skipped operations for debugging
    if (skippedOps.length > 0) {
        console.warn(`Skipped ${skippedOps.length} operations without complete metrics:`,
            skippedOps.slice(0, 5).map(item => item.operation.name));
    }

    if (analyses.length === 0) {
        container.innerHTML = `
            <div class="bottleneck-warning">
                <h3>⚠️ Insufficient Metrics Data</h3>
                <p>Operations are missing required fields (duration_ms, flops, data_bytes).</p>
                <p>${operations.length} operations found, but none have complete bottleneck metrics.</p>
            </div>
        `;
        return;
    }

    // Build hardware selector dropdown
    const hardwareOptions = getAllHardwareOptions();
    let selectorHTML = '';
    if (hardwareOptions.length > 0) {
        selectorHTML = `
            <div class="hardware-selector">
                <label for="hardwareSelect">Hardware:</label>
                <select id="hardwareSelect" onchange="onHardwareChange()">
                    ${hardwareOptions.map(opt => `
                        <option value="${opt.key}" ${selectedHardwareKey === opt.key ? 'selected' : ''}>
                            ${opt.label} ${detectedHardware && detectedHardware.chip && opt.key.endsWith(detectedHardware.chip) ? '(detected)' : ''}
                        </option>
                    `).join('')}
                </select>
            </div>
        `;
    }

    // Render sections
    container.innerHTML = `
        <div class="bottleneck-header">
            <div class="bottleneck-header-row">
                <h3>Bottleneck Analysis (Roofline Model)</h3>
                ${selectorHTML}
            </div>
            <div class="hardware-info">
                <strong>Hardware:</strong> ${hardwareSpec.name} |
                <strong>Peak Compute:</strong> ${hardwareSpec.peak_tflops_fp16.toFixed(1)} TFLOPs (FP16) |
                <strong>Peak Bandwidth:</strong> ${hardwareSpec.peak_bandwidth_gbs.toFixed(0)} GB/s |
                <strong>Ridge Point:</strong> ${hardwareSpec.ridge_point.toFixed(1)} FLOPs/Byte
            </div>
        </div>

        <div class="bottleneck-legend">
            <div class="legend-title">Bottleneck Heat Scale:</div>
            <div class="heat-scale">
                <div class="heat-bar"></div>
                <div class="heat-labels">
                    <span>Bandwidth-Bound</span>
                    <span>Balanced</span>
                    <span>Compute-Bound</span>
                </div>
            </div>
        </div>

        <div class="bottleneck-summary" id="bottleneckSummary"></div>
        <div class="bottleneck-timeline" id="bottleneckTimeline"></div>
        <div class="bottleneck-table" id="bottleneckTable"></div>
    `;

    // Store analyses globally for filter access
    window.currentBottleneckAnalyses = analyses;

    renderBottleneckSummary(analyses, hardwareSpec);
    renderBottleneckTimeline(analyses);
    renderBottleneckTable(analyses);
}

function renderBottleneckSummary(analyses, hardwareSpec) {
    const container = document.getElementById('bottleneckSummary');
    if (!container) return;

    // Calculate aggregate statistics
    let totalBandwidthBound = 0;
    let totalComputeBound = 0;
    let totalTime = 0;

    analyses.forEach(({operation, analysis}) => {
        if (analysis.primary_bottleneck === 'bandwidth') {
            totalBandwidthBound += operation.duration_ms;
        } else {
            totalComputeBound += operation.duration_ms;
        }
        totalTime += operation.duration_ms;
    });

    const bandwidthPct = (totalBandwidthBound / totalTime) * 100;
    const computePct = (totalComputeBound / totalTime) * 100;

    // Average utilization
    const avgComputeUtil = analyses.reduce((sum, {analysis}) =>
        sum + analysis.compute_utilization, 0) / analyses.length;
    const avgBandwidthUtil = analyses.reduce((sum, {analysis}) =>
        sum + analysis.bandwidth_utilization, 0) / analyses.length;

    container.innerHTML = `
        <h4>Overall Performance Summary</h4>
        <div class="summary-stats">
            <div class="stat-card">
                <div class="label">Bandwidth-Bound Time</div>
                <div class="value" style="color: #ff3333;">${bandwidthPct.toFixed(1)}<span class="unit">%</span></div>
                <div class="subtext">${totalBandwidthBound.toFixed(2)} ms</div>
            </div>
            <div class="stat-card">
                <div class="label">Compute-Bound Time</div>
                <div class="value" style="color: #3333ff;">${computePct.toFixed(1)}<span class="unit">%</span></div>
                <div class="subtext">${totalComputeBound.toFixed(2)} ms</div>
            </div>
            <div class="stat-card">
                <div class="label">Avg Compute Utilization</div>
                <div class="value">${(avgComputeUtil * 100).toFixed(1)}<span class="unit">%</span></div>
                <div class="subtext">of ${hardwareSpec.peak_tflops_fp16} TFLOPs</div>
            </div>
            <div class="stat-card">
                <div class="label">Avg Bandwidth Utilization</div>
                <div class="value">${(avgBandwidthUtil * 100).toFixed(1)}<span class="unit">%</span></div>
                <div class="subtext">of ${hardwareSpec.peak_bandwidth_gbs} GB/s</div>
            </div>
        </div>
    `;
}

function renderBottleneckTimeline(analyses) {
    const container = document.getElementById('bottleneckTimeline');
    if (!container) return;

    // Get unique operation types for filter
    const operationTypes = [...new Set(analyses.map(a => a.operation.module_type))].sort();

    // Build filter dropdown
    const filterOptions = [
        '<option value="all">All Operations</option>',
        ...operationTypes.map(type => `<option value="${type}">${type}</option>`)
    ].join('');

    container.innerHTML = `
        <div class="timeline-header">
            <h4>Operation Timeline (colored by bottleneck type)</h4>
            <div class="timeline-controls">
                <label for="timelineFilter">Filter:</label>
                <select id="timelineFilter" onchange="onTimelineFilterChange()">
                    ${filterOptions}
                </select>
                <span id="timelineFilterStats" class="filter-stats"></span>
            </div>
        </div>
        <div class="timeline-container" id="timelineChart"></div>
    `;

    // Render timeline with current filter
    renderTimelineBars(analyses, 'all');
}

function renderTimelineBars(analyses, filterType) {
    const chart = document.getElementById('timelineChart');
    if (!chart) return;

    // Filter operations by type
    const filteredAnalyses = filterType === 'all'
        ? analyses
        : analyses.filter(a => a.operation.module_type === filterType);

    if (filteredAnalyses.length === 0) {
        chart.innerHTML = '<div class="no-data">No operations of this type</div>';
        updateTimelineStats(0, 0);
        return;
    }

    const startTime = new Date(analyses[0].operation.timestamp);

    // Calculate total time span (use all operations for consistent scale)
    let maxTime = 0;
    analyses.forEach(({operation}) => {
        const relativeTime = (new Date(operation.timestamp) - startTime);
        maxTime = Math.max(maxTime, relativeTime + operation.duration_ms);
    });

    // Calculate filtered time for stats
    const filteredTotalTime = filteredAnalyses.reduce((sum, {operation}) => sum + operation.duration_ms, 0);

    // Render timeline bars
    let html = '<div class="timeline-bars">';
    filteredAnalyses.forEach(({operation, analysis}) => {
        const relativeTime = (new Date(operation.timestamp) - startTime);
        const leftPct = (relativeTime / maxTime) * 100;
        const widthPct = (operation.duration_ms / maxTime) * 100;

        html += `
            <div class="timeline-bar"
                 style="left: ${leftPct}%; width: ${widthPct}%; background-color: ${analysis.heat_color};"
                 title="${operation.name}\nType: ${operation.module_type}\nDuration: ${operation.duration_ms.toFixed(3)} ms\nBottleneck: ${analysis.primary_bottleneck}\nAI: ${analysis.arithmetic_intensity.toFixed(2)} FLOPs/Byte">
            </div>
        `;
    });
    html += '</div>';

    chart.innerHTML = html;
    updateTimelineStats(filteredAnalyses.length, filteredTotalTime);
}

function updateTimelineStats(count, totalTime) {
    const statsElement = document.getElementById('timelineFilterStats');
    if (statsElement) {
        if (count > 0) {
            statsElement.textContent = `(${count} ops, ${totalTime.toFixed(2)} ms total)`;
        } else {
            statsElement.textContent = '';
        }
    }
}

function onTimelineFilterChange() {
    const filterSelect = document.getElementById('timelineFilter');
    if (!filterSelect) return;

    const filterType = filterSelect.value;

    // Get analyses from the current render
    // We need to store analyses globally or re-fetch from profilingData
    if (window.currentBottleneckAnalyses) {
        renderTimelineBars(window.currentBottleneckAnalyses, filterType);
    }
}

function renderBottleneckTable(analyses) {
    const container = document.getElementById('bottleneckTable');
    if (!container) return;

    // Aggregate by operation type
    const typeStats = {};

    analyses.forEach(({operation, analysis}) => {
        const type = operation.module_type;
        if (!typeStats[type]) {
            typeStats[type] = {
                count: 0,
                totalTime: 0,
                bandwidthBound: 0,
                computeBound: 0,
                totalFlops: 0,
                totalBytes: 0,
            };
        }

        typeStats[type].count++;
        typeStats[type].totalTime += operation.duration_ms;
        if (analysis.primary_bottleneck === 'bandwidth') {
            typeStats[type].bandwidthBound++;
        } else {
            typeStats[type].computeBound++;
        }
        typeStats[type].totalFlops += operation.flops;
        typeStats[type].totalBytes += operation.data_bytes;
    });

    // Convert to array and sort by total time
    const rows = Object.entries(typeStats)
        .map(([type, stats]) => ({
            type,
            ...stats,
            avgTime: stats.totalTime / stats.count,
            primaryBottleneck: stats.bandwidthBound > stats.computeBound ? 'bandwidth' : 'compute',
            bandwidthPct: (stats.bandwidthBound / stats.count) * 100,
        }))
        .sort((a, b) => b.totalTime - a.totalTime);

    // Get total operations from the profiling data
    const totalOps = profilingData.operation_log ? profilingData.operation_log.length : 0;
    const analyzedOps = analyses.length;
    const coveragePct = totalOps > 0 ? (analyzedOps / totalOps * 100).toFixed(1) : 0;

    let html = `
        <h4>Operation Type Summary</h4>
        <div style="margin-bottom: 10px; font-size: 13px; color: #666;">
            Showing ${analyzedOps} of ${totalOps} operations (${coveragePct}% coverage).
            ${totalOps > analyzedOps ? `<strong>${totalOps - analyzedOps} operations</strong> missing metrics (duration/FLOPs/bytes). ` : ''}
            ${totalOps > analyzedOps ? 'These may include custom kernels or non-module operations like attention.' : ''}
        </div>
        <table class="bottleneck-summary-table">
            <thead>
                <tr>
                    <th>Operation Type</th>
                    <th>Count</th>
                    <th>Avg Time (ms)</th>
                    <th>Total Time (ms)</th>
                    <th>Primary Bottleneck</th>
                    <th>Bandwidth-Bound %</th>
                </tr>
            </thead>
            <tbody>
    `;

    rows.forEach(row => {
        const bottleneckColor = row.primaryBottleneck === 'bandwidth' ? '#ff6666' : '#6666ff';
        html += `
            <tr>
                <td><strong>${row.type}</strong></td>
                <td>${row.count}</td>
                <td>${row.avgTime.toFixed(3)}</td>
                <td>${row.totalTime.toFixed(2)}</td>
                <td style="color: ${bottleneckColor}; font-weight: bold;">${row.primaryBottleneck}</td>
                <td>${row.bandwidthPct.toFixed(1)}%</td>
            </tr>
        `;
    });

    html += `
            </tbody>
        </table>
    `;

    container.innerHTML = html;
}

function onHardwareChange() {
    const selector = document.getElementById("hardwareSelect");
    if (selector) {
        selectedHardwareKey = selector.value;
        console.log("Hardware changed to:", selectedHardwareKey);
        // Re-render bottleneck analysis with new hardware
        renderBottleneckAnalysis();
    }
}
