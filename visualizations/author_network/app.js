// ============================================================================
// Twitter Authors Map - WebGL Renderer
// Renders matched authors using regl for WebGL
// ============================================================================

// Data
let nodes = [];
let summary = {};
const ASSET_VERSION = '1';
const versionedAsset = path => `${path}?v=${ASSET_VERSION}`;

// Descriptive labels for the supplied network communities
// (data/community_labels.json). Communities outside the labelled set keep their
// bare numeric ID and are described collectively as long-tail communities.
let communityLabels = {};        // { "4": {label, display_label, short_description, ...} }
let labelledCommunityOrder = []; // labelled community IDs, largest first
let communitySizes = {};         // exact matched-author count for every community

// Author -> selected Example post metadata. Lazily loaded after the main point
// cloud is on screen and kept in ordinary JS memory, never in GPU buffers.
let examplePosts = null;
let examplePostsState = 'idle';  // idle | loading | ready | failed

// Per-emotion mean/SD across all matched authors, used only to pick each
// author's two most distinctive emotions for display. Computed at load time
// from values already present in nodes.json; no stored value is modified.
let emotionStats = null;

// Currently selected author (click, or search hit) and hovered author
let selectedNode = null;
let hoverNode = null;

const EMOTION_KEYS = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love',
    'optimism', 'pessimism', 'sadness', 'surprise', 'trust'];

// Field name mapping (compact JSON uses short names)
const F = {
    id: 'id', x: 'x', y: 'y',
    community: 'c',
    fullDegree: 'fd', matchedDegree: 'md',
    dominantTopic: 'dt', dominantProb: 'dp',
    positive: 'sp', neutral: 'sn', negative: 'sg',
    topic: i => 't' + i,
    anger: 'ea', anticipation: 'eb', disgust: 'ec', fear: 'ed',
    joy: 'ee', love: 'ef', optimism: 'eg', pessimism: 'eh',
    sadness: 'ei', surprise: 'ej', trust: 'ek'
};

// View state
let viewX = 0, viewY = 0, viewScale = 1;
let minX, maxX, minY, maxY;
let viewMode = '2d';
let nodeIndexById = new Map();

// The 3D coordinates are a separate, lazy binary payload in viewer-node order.
// No analytical value is recomputed in the browser and the accepted 2D
// coordinates remain the default view.
let positions3D = null;
let visible3D = null;
let layout3DMeta = null;
let layout3DState = 'idle'; // idle | loading | ready | failed
let layout3DPromise = null;

// Start close to the accepted 2D-facing orientation so major branches are
// immediately legible, while retaining enough obliqueness to show depth.
const CAMERA3D_HOME_YAW = 0.08;
const CAMERA3D_HOME_PITCH = 0.08;
const camera3D = {
    yaw: CAMERA3D_HOME_YAW, pitch: CAMERA3D_HOME_PITCH, distance: 3.2,
    homeDistance: 3.2,
    target: [0, 0, 0],
    desiredYaw: CAMERA3D_HOME_YAW, desiredPitch: CAMERA3D_HOME_PITCH, desiredDistance: 3.2,
    desiredTarget: [0, 0, 0],
    eye: [0, 0, 3.2],
    view: new Float32Array(16),
    projection: new Float32Array(16),
    viewProjection: new Float32Array(16),
};
let lastFrameTime = 0;
let pickDirty = true;
let pickFramebuffer = null;
let pickReadback = new Uint8Array(4);
let hoverFramePending = false;
let pendingHoverEvent = null;

// Rendering
let regl, drawPoints2D, drawPoints3D, drawPick3D, drawSelected3D;
let positions, colors, sizes, pickIds;
let positionBuffer2D, positionBuffer3D, visibilityBuffer3D, colorBuffer, sizeBuffer, pickIdBuffer;
let selectedPosition3DBuffer;

// Current color mode
let colorMode = 'community';

// Color palettes
const COMMUNITY_COLORS = [
    [0.12, 0.47, 0.71], [1.00, 0.50, 0.05], [0.17, 0.63, 0.17], [0.84, 0.15, 0.16],
    [0.58, 0.40, 0.74], [0.55, 0.34, 0.29], [0.89, 0.47, 0.76], [0.50, 0.50, 0.50],
    [0.74, 0.74, 0.13], [0.09, 0.75, 0.81], [0.68, 0.78, 0.91], [1.00, 0.73, 0.47],
    [0.60, 0.87, 0.54], [1.00, 0.60, 0.59], [0.77, 0.69, 0.84], [0.77, 0.61, 0.58],
    [0.97, 0.71, 0.82], [0.78, 0.78, 0.78], [0.86, 0.86, 0.55], [0.62, 0.85, 0.90],
    [0.22, 0.23, 0.47], [0.39, 0.47, 0.22], [0.55, 0.43, 0.19], [0.52, 0.24, 0.22]
];

let TOPIC_NAMES = [];

const TOPIC_PALETTE = [
    [0.89, 0.10, 0.11], [0.22, 0.49, 0.72], [0.30, 0.69, 0.29], [0.60, 0.31, 0.64],
    [1.00, 0.50, 0.00], [1.00, 1.00, 0.20], [0.65, 0.34, 0.16], [0.97, 0.51, 0.75],
    [0.60, 0.60, 0.60], [0.40, 0.76, 0.65], [0.99, 0.55, 0.38], [0.55, 0.63, 0.80]
];

// Sentiment colours are expressed as
// exact 8-bit RGB fractions so WebGL points and CSS legends render identically.
const SENTIMENT_COLORS = {
    positive: [44/255, 162/255, 95/255],   // #2ca25f
    neutral: [227/255, 181/255, 5/255],   // #e3b505
    negative: [215/255, 48/255, 39/255]   // #d73027
};

const EMOTION_COLORS = {
    anger: [0/255, 114/255, 178/255],
    anticipation: [230/255, 159/255, 0/255],
    disgust: [0/255, 158/255, 115/255],
    fear: [204/255, 121/255, 167/255],
    joy: [86/255, 180/255, 233/255],
    love: [213/255, 94/255, 0/255],
    optimism: [240/255, 228/255, 66/255],
    pessimism: [51/255, 34/255, 136/255],
    sadness: [136/255, 204/255, 238/255],
    surprise: [170/255, 68/255, 153/255],
    trust: [68/255, 170/255, 153/255]
};

const SCALE_LOW = [0.85, 0.85, 0.83];
const EMOTION_SCALE_LOW = [0.86, 0.86, 0.90];
const GREY = [0.35, 0.35, 0.42];

// ============================================================================
// Community label helpers
// ============================================================================

// Presentation-only short label for the labelled labelled set; numeric IDs remain
// stored separately in each node and in the community key.
function communityLabel(c) {
    if (c === null || c === undefined || c < 0) return 'No community';
    const rec = communityLabels[String(c)];
    return rec ? rec.display_label : `Community ${c}`;
}

function communityIsLabelled(c) {
    return c !== null && c !== undefined && communityLabels[String(c)] !== undefined;
}

function communityDescription(c) {
    const rec = communityLabels[String(c)];
    if (rec) return rec.short_description;
    return 'No descriptive label has been supplied for this community.';
}

function communitySize(c) {
    if (c === null || c === undefined || c < 0) return null;
    const labelled = communityLabels[String(c)];
    return labelled?.n_authors ?? communitySizes[String(c)] ?? null;
}

function communityConfidence(c) {
    const rec = communityLabels[String(c)];
    if (!rec) return '';
    return rec.review_status === 'provisional'
        ? rec.confidence + ' (provisional)'
        : rec.confidence;
}

function topTopicScores(node, howMany) {
    const scores = [];
    for (let i = 0; i < TOPIC_NAMES.length; i++) {
        const value = node[F.topic(i)];
        if (value !== null && value !== undefined) {
            scores.push({ index: i, name: TOPIC_NAMES[i], value });
        }
    }
    scores.sort((a, b) => b.value - a.value);
    return scores.slice(0, howMany || 3);
}

// Two most distinctive emotions for an author: largest |z| against the
// matched-author mean and SD for that emotion.
function computeEmotionStats() {
    const stats = {};
    for (const k of EMOTION_KEYS) {
        const field = F[k];
        let n = 0, sum = 0, sumsq = 0;
        for (let i = 0; i < nodes.length; i++) {
            const v = nodes[i][field];
            if (v === null || v === undefined) continue;
            n++; sum += v; sumsq += v * v;
        }
        const mean = n ? sum / n : 0;
        const varr = n ? Math.max(sumsq / n - mean * mean, 0) : 0;
        stats[k] = { mean: mean, sd: Math.sqrt(varr) || 1e-9 };
    }
    return stats;
}

function distinctiveEmotions(node, howMany) {
    if (!emotionStats) return [];
    const scored = [];
    for (const k of EMOTION_KEYS) {
        const v = node[F[k]];
        if (v === null || v === undefined) continue;
        const s = emotionStats[k];
        scored.push({ name: k, value: v, z: (v - s.mean) / s.sd });
    }
    scored.sort((a, b) => Math.abs(b.z) - Math.abs(a.z));
    return scored.slice(0, howMany || 2);
}

function titleCase(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

function fmtSigned(z) { return (z >= 0 ? '+' : '−') + Math.abs(z).toFixed(2); }

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, ch => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[ch]);
}

// Diverging color interpolation: val from -1 to +1
function divergingColor(val, lowColor, midColor, highColor) {
    if (val <= 0) {
        // Interpolate from low to mid
        const t = val + 1; // 0 to 1
        return [
            lowColor[0] + t * (midColor[0] - lowColor[0]),
            lowColor[1] + t * (midColor[1] - lowColor[1]),
            lowColor[2] + t * (midColor[2] - lowColor[2])
        ];
    } else {
        // Interpolate from mid to high
        const t = val; // 0 to 1
        return [
            midColor[0] + t * (highColor[0] - midColor[0]),
            midColor[1] + t * (highColor[1] - midColor[1]),
            midColor[2] + t * (highColor[2] - midColor[2])
        ];
    }
}

// ============================================================================
// Minimal 3D camera mathematics (column-major WebGL matrices)
// ============================================================================

function clamp(value, low, high) {
    return Math.max(low, Math.min(high, value));
}

function lerp(a, b, t) {
    return a + (b - a) * t;
}

function normalise3(v) {
    const length = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / length, v[1] / length, v[2] / length];
}

function cross3(a, b) {
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ];
}

function perspectiveMatrix(out, fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2);
    out.fill(0);
    out[0] = f / aspect;
    out[5] = f;
    out[10] = (far + near) / (near - far);
    out[11] = -1;
    out[14] = (2 * far * near) / (near - far);
    return out;
}

function lookAtMatrix(out, eye, target, up) {
    const z = normalise3([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]);
    const x = normalise3(cross3(up, z));
    const y = cross3(z, x);

    out[0] = x[0]; out[1] = y[0]; out[2] = z[0]; out[3] = 0;
    out[4] = x[1]; out[5] = y[1]; out[6] = z[1]; out[7] = 0;
    out[8] = x[2]; out[9] = y[2]; out[10] = z[2]; out[11] = 0;
    out[12] = -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]);
    out[13] = -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]);
    out[14] = -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]);
    out[15] = 1;
    return out;
}

function multiplyMatrix4(out, a, b) {
    const result = new Float32Array(16);
    for (let column = 0; column < 4; column++) {
        for (let row = 0; row < 4; row++) {
            result[column * 4 + row] =
                a[row] * b[column * 4] +
                a[4 + row] * b[column * 4 + 1] +
                a[8 + row] * b[column * 4 + 2] +
                a[12 + row] * b[column * 4 + 3];
        }
    }
    out.set(result);
    return out;
}

function updateCameraMatrices(canvas, deltaSeconds) {
    const damping = 1 - Math.exp(-Math.min(deltaSeconds, 0.05) * 13);
    const previous = [camera3D.yaw, camera3D.pitch, camera3D.distance, ...camera3D.target];
    camera3D.yaw = lerp(camera3D.yaw, camera3D.desiredYaw, damping);
    camera3D.pitch = lerp(camera3D.pitch, camera3D.desiredPitch, damping);
    camera3D.distance = lerp(camera3D.distance, camera3D.desiredDistance, damping);
    for (let i = 0; i < 3; i++) {
        camera3D.target[i] = lerp(camera3D.target[i], camera3D.desiredTarget[i], damping);
    }

    const cp = Math.cos(camera3D.pitch);
    camera3D.eye[0] = camera3D.target[0] + camera3D.distance * cp * Math.sin(camera3D.yaw);
    camera3D.eye[1] = camera3D.target[1] + camera3D.distance * Math.sin(camera3D.pitch);
    camera3D.eye[2] = camera3D.target[2] + camera3D.distance * cp * Math.cos(camera3D.yaw);

    lookAtMatrix(camera3D.view, camera3D.eye, camera3D.target, [0, 1, 0]);
    perspectiveMatrix(camera3D.projection, Math.PI / 4, canvas.width / canvas.height, 0.01, 30);
    multiplyMatrix4(camera3D.viewProjection, camera3D.projection, camera3D.view);

    const current = [camera3D.yaw, camera3D.pitch, camera3D.distance, ...camera3D.target];
    if (current.some((value, i) => Math.abs(value - previous[i]) > 1e-7)) pickDirty = true;
}

function cameraBasis3D() {
    const forward = normalise3([
        camera3D.target[0] - camera3D.eye[0],
        camera3D.target[1] - camera3D.eye[1],
        camera3D.target[2] - camera3D.eye[2]
    ]);
    const right = normalise3(cross3(forward, [0, 1, 0]));
    const up = normalise3(cross3(right, forward));
    return { right, up };
}

// ============================================================================
// Initialization
// ============================================================================

async function init() {
    try {
        const [nodesResp, summaryResp, labelsResp] = await Promise.all([
            fetch('data/nodes.json'),
            fetch('data/build_summary.json'),
            fetch(versionedAsset('data/community_labels.json')).catch(() => null)
        ]);

        nodes = await nodesResp.json();
        summary = await summaryResp.json();
        if (!nodes.length) throw new Error('The viewer node file is empty');
        const count = Object.keys(nodes[0]).filter(key => /^t[0-9]+$/.test(key)).length;
        TOPIC_NAMES = Array.from({length: count}, (_, i) => summary.topic_labels?.[i] || `Topic ${i}`);
        const selector = document.getElementById('topic-select');
        selector.replaceChildren(...TOPIC_NAMES.map((label, i) => {
            const option = document.createElement('option');
            option.value = i;
            option.textContent = label;
            return option;
        }));

        if (labelsResp && labelsResp.ok) {
            const payload = await labelsResp.json();
            communityLabels = payload.communities || {};
            labelledCommunityOrder = (payload.order || Object.keys(communityLabels).map(Number));
            console.log(`Loaded ${Object.keys(communityLabels).length} community labels`);
        } else {
            console.warn('community_labels.json unavailable; falling back to numeric community IDs');
        }

        console.log(`Loaded ${nodes.length} nodes`);
        nodeIndexById = new Map(nodes.map((node, index) => [node[F.id], index]));

        // Distributional stats for the "distinctive emotions" display only.
        emotionStats = computeEmotionStats();

        // Update stats
        document.getElementById('stat-authors').textContent = nodes.length.toLocaleString();
        document.getElementById('stat-edges').textContent = summary.matched_edges?.toLocaleString() || '-';

        // Calculate bounds and exact matched-author community sizes.
        minX = Infinity; maxX = -Infinity;
        minY = Infinity; maxY = -Infinity;
        communitySizes = {};
        for (const node of nodes) {
            if (node.x < minX) minX = node.x;
            if (node.x > maxX) maxX = node.x;
            if (node.y < minY) minY = node.y;
            if (node.y > maxY) maxY = node.y;
            const c = node[F.community];
            if (c !== null && c !== undefined && c >= 0) {
                communitySizes[String(c)] = (communitySizes[String(c)] || 0) + 1;
            }
        }

        // Center view
        viewX = (minX + maxX) / 2;
        viewY = (minY + maxY) / 2;
        const rangeX = maxX - minX;
        const rangeY = maxY - minY;
        viewScale = 0.9 / Math.max(rangeX, rangeY) * 2;

        // Initialize WebGL
        initWebGL();

        // Build initial colors
        updateColors();

        // Hide loading
        document.getElementById('loading').classList.add('hidden');

        // Start render loop
        requestAnimationFrame(render);

        // Event listeners
        setupEventListeners();

        // Example-post lookup: fetched lazily after first paint so it never
        // delays the map. It remains separate from all GPU buffers.
        loadExamplePosts();

    } catch (err) {
        console.error('Failed to load data:', err);
        document.getElementById('loading-text').innerHTML =
            `<span style="color:#ff6b6b;">Error loading data</span><br>
            <span style="font-size:11px;color:#888;">${err.message}</span>`;
    }
}

function initWebGL() {
    const canvas = document.getElementById('graph-canvas');
    canvas.width = canvas.clientWidth * window.devicePixelRatio;
    canvas.height = canvas.clientHeight * window.devicePixelRatio;

    regl = createREGL({
        canvas: canvas,
        attributes: { antialias: true, alpha: false }
    });

    // Accepted 2D position buffer.
    positions = new Float32Array(nodes.length * 2);
    for (let i = 0; i < nodes.length; i++) {
        positions[i * 2] = nodes[i].x;
        positions[i * 2 + 1] = nodes[i].y;
    }

    // Prepare size buffer (log-scaled degree)
    sizes = new Float32Array(nodes.length);
    for (let i = 0; i < nodes.length; i++) {
        const deg = nodes[i][F.fullDegree] || 1;
        sizes[i] = Math.max(2, Math.log1p(deg) * 1.5);
    }

    // Shared colour and 3D-picking buffers.
    colors = new Float32Array(nodes.length * 3);
    pickIds = new Float32Array(nodes.length);
    for (let i = 0; i < nodes.length; i++) pickIds[i] = i + 1;

    positionBuffer2D = regl.buffer(positions);
    positionBuffer3D = regl.buffer({ data: new Float32Array(0), usage: 'static' });
    visibilityBuffer3D = regl.buffer({ data: new Float32Array(nodes.length), usage: 'static' });
    colorBuffer = regl.buffer({ data: colors, usage: 'dynamic' });
    sizeBuffer = regl.buffer(sizes);
    pickIdBuffer = regl.buffer(pickIds);
    selectedPosition3DBuffer = regl.buffer(new Float32Array([0, 0, 0]));

    // Original 2D point drawing command. Its shader and visual scaling are
    // deliberately retained so the approved default view is unchanged.
    drawPoints2D = regl({
        vert: `
            precision highp float;
            attribute vec2 position;
            attribute vec3 color;
            attribute float size;
            uniform vec2 viewOffset;
            uniform float viewScale;
            uniform vec2 resolution;
            varying vec3 vColor;

            void main() {
                vec2 pos = (position - viewOffset) * viewScale;
                pos.x *= resolution.y / resolution.x;
                gl_Position = vec4(pos, 0, 1);
                gl_PointSize = size * viewScale * 50.0;
                vColor = color;
            }
        `,
        frag: `
            precision highp float;
            varying vec3 vColor;

            void main() {
                vec2 cxy = 2.0 * gl_PointCoord - 1.0;
                float r = dot(cxy, cxy);
                if (r > 1.0) discard;
                float alpha = 1.0 - smoothstep(0.5, 1.0, r);
                gl_FragColor = vec4(vColor, alpha);
            }
        `,
        attributes: {
            position: positionBuffer2D,
            color: colorBuffer,
            size: sizeBuffer
        },
        uniforms: {
            viewOffset: regl.prop('viewOffset'),
            viewScale: regl.prop('viewScale'),
            resolution: regl.prop('resolution')
        },
        count: nodes.length,
        primitive: 'points',
        blend: {
            enable: true,
            func: { srcRGB: 'src alpha', srcAlpha: 1, dstRGB: 'one minus src alpha', dstAlpha: 1 }
        },
        depth: { enable: false }
    });

    // Perspective point cloud for the structure-preserving giant-component 3D layout.
    drawPoints3D = regl({
        vert: `
            precision highp float;
            attribute vec3 position;
            attribute float visibility;
            attribute vec3 color;
            attribute float size;
            uniform mat4 view;
            uniform mat4 projection;
            uniform float pointScale;
            varying vec3 vColor;
            varying float vDepth;

            void main() {
                if (visibility < 0.5) {
                    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                    gl_PointSize = 0.0;
                    vColor = color;
                    vDepth = 1.0;
                    return;
                }
                vec4 cameraPosition = view * vec4(position, 1.0);
                float depth = max(0.08, -cameraPosition.z);
                gl_Position = projection * cameraPosition;
                gl_PointSize = clamp(size * pointScale / depth, 1.0, 14.0);
                vColor = color;
                vDepth = depth;
            }
        `,
        frag: `
            precision highp float;
            uniform float fogNear;
            uniform float fogFar;
            varying vec3 vColor;
            varying float vDepth;

            void main() {
                vec2 cxy = 2.0 * gl_PointCoord - 1.0;
                float r = dot(cxy, cxy);
                if (r > 1.0) discard;
                float edge = 1.0 - smoothstep(0.52, 1.0, r);
                if (edge < 0.025) discard;
                float fog = 1.0 - smoothstep(fogNear, fogFar, vDepth);
                float alpha = edge * mix(0.06, 0.56, fog);
                gl_FragColor = vec4(vColor, alpha);
            }
        `,
        attributes: {
            position: positionBuffer3D,
            visibility: visibilityBuffer3D,
            color: colorBuffer,
            size: sizeBuffer
        },
        uniforms: {
            view: regl.prop('view'),
            projection: regl.prop('projection'),
            pointScale: regl.prop('pointScale'),
            fogNear: regl.prop('fogNear'),
            fogFar: regl.prop('fogFar')
        },
        count: nodes.length,
        primitive: 'points',
        blend: {
            enable: true,
            func: { srcRGB: 'src alpha', srcAlpha: 1, dstRGB: 'one minus src alpha', dstAlpha: 1 }
        },
        depth: { enable: true, mask: true, func: 'less' }
    });

    // Off-screen ID pass: each point is encoded as a 24-bit index. This makes
    // 3D hover/click selection independent of draw order and avoids scanning
    // all projected points on every pointer movement.
    drawPick3D = regl({
        vert: `
            precision highp float;
            attribute vec3 position;
            attribute float visibility;
            attribute float size;
            attribute float pickId;
            uniform mat4 view;
            uniform mat4 projection;
            uniform float pointScale;
            uniform float minPickSize;
            varying float vPickId;

            void main() {
                if (visibility < 0.5) {
                    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                    gl_PointSize = 0.0;
                    vPickId = 0.0;
                    return;
                }
                vec4 cameraPosition = view * vec4(position, 1.0);
                float depth = max(0.08, -cameraPosition.z);
                gl_Position = projection * cameraPosition;
                gl_PointSize = clamp(max(minPickSize, size * pointScale / depth), minPickSize, 26.0);
                vPickId = pickId;
            }
        `,
        frag: `
            precision highp float;
            varying float vPickId;

            void main() {
                vec2 cxy = 2.0 * gl_PointCoord - 1.0;
                if (dot(cxy, cxy) > 1.0) discard;
                float ident = floor(vPickId + 0.5);
                float red = mod(ident, 256.0);
                float green = mod(floor(ident / 256.0), 256.0);
                float blue = mod(floor(ident / 65536.0), 256.0);
                gl_FragColor = vec4(red, green, blue, 255.0) / 255.0;
            }
        `,
        attributes: {
            position: positionBuffer3D,
            visibility: visibilityBuffer3D,
            size: sizeBuffer,
            pickId: pickIdBuffer
        },
        uniforms: {
            view: regl.prop('view'),
            projection: regl.prop('projection'),
            pointScale: regl.prop('pointScale'),
            minPickSize: regl.prop('minPickSize')
        },
        count: nodes.length,
        primitive: 'points',
        blend: { enable: false },
        dither: false,
        depth: { enable: true, mask: true, func: 'less' }
    });

    drawSelected3D = regl({
        vert: `
            precision highp float;
            attribute vec3 position;
            uniform mat4 view;
            uniform mat4 projection;
            uniform float pointSize;
            void main() {
                vec4 cameraPosition = view * vec4(position, 1.0);
                gl_Position = projection * cameraPosition;
                gl_Position.z -= 0.0005 * gl_Position.w;
                gl_PointSize = pointSize;
            }
        `,
        frag: `
            precision highp float;
            void main() {
                vec2 cxy = 2.0 * gl_PointCoord - 1.0;
                float r = dot(cxy, cxy);
                if (r > 1.0 || r < 0.46) discard;
                gl_FragColor = vec4(1.0, 1.0, 1.0, 0.94);
            }
        `,
        attributes: { position: selectedPosition3DBuffer },
        uniforms: {
            view: regl.prop('view'),
            projection: regl.prop('projection'),
            pointSize: regl.prop('pointSize')
        },
        count: 1,
        primitive: 'points',
        blend: {
            enable: true,
            func: { srcRGB: 'src alpha', srcAlpha: 1, dstRGB: 'one minus src alpha', dstAlpha: 1 }
        },
        depth: { enable: false }
    });
}

// ============================================================================
// Rendering
// ============================================================================

function resizeCanvasIfNeeded(canvas) {
    const dpr = window.devicePixelRatio;
    const width = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const height = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width === width && canvas.height === height) return false;
    canvas.width = width;
    canvas.height = height;
    if (pickFramebuffer) pickFramebuffer.resize(width, height);
    pickDirty = true;
    return true;
}

function render(now) {
    const canvas = document.getElementById('graph-canvas');
    const dpr = window.devicePixelRatio;
    resizeCanvasIfNeeded(canvas);
    regl.clear({ color: [0.07, 0.07, 0.10, 1], depth: 1 });

    if (viewMode === '3d' && positions3D) {
        const deltaSeconds = lastFrameTime ? (now - lastFrameTime) / 1000 : 1 / 60;
        updateCameraMatrices(canvas, deltaSeconds);
        const fogNear = Math.max(0.15, camera3D.distance - 1.15);
        const fogFar = camera3D.distance + 1.7;
        drawPoints3D({
            view: camera3D.view,
            projection: camera3D.projection,
            pointScale: 1.65 * dpr,
            fogNear,
            fogFar
        });
        if (selectedNode && nodeVisibleIn3D(selectedNode)) {
            drawSelected3D({
                view: camera3D.view,
                projection: camera3D.projection,
                pointSize: 15 * dpr
            });
        }
    } else {
        drawPoints2D({
            viewOffset: [viewX, viewY],
            viewScale: viewScale,
            resolution: [canvas.width, canvas.height]
        });
    }

    lastFrameTime = now;

    requestAnimationFrame(render);
}

// ============================================================================
// Color computation
// ============================================================================

function updateColors() {
    const mode = colorMode;
    let validCount = 0;

    for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i];
        let color = GREY;

        if (mode === 'community') {
            const comm = node[F.community];
            if (comm !== null && comm !== undefined && comm >= 0) {
                color = COMMUNITY_COLORS[comm % COMMUNITY_COLORS.length];
                validCount++;
            }
        }
        else if (mode === 'dominant_topic') {
            const topicIdx = node[F.dominantTopic];
            if (topicIdx !== null && topicIdx !== undefined && topicIdx >= 0 && topicIdx < TOPIC_NAMES.length) {
                color = TOPIC_PALETTE[topicIdx % TOPIC_PALETTE.length];
                validCount++;
            }
        }
        else if (mode === 'sentiment') {
            const sentType = document.getElementById('sentiment-select').value;

            if (sentType === 'net') {
                const pos = node[F.positive];
                const neg = node[F.negative];
                if (pos !== null && pos !== undefined && neg !== null && neg !== undefined) {
                    // Continuous diverging scale: red (-1) -> neutral yellow (0) -> green (+1)
                    const val = Math.max(-1, Math.min(1, (pos - neg) * 1.5)); // Scale up for visibility
                    color = divergingColor(val,
                        SENTIMENT_COLORS.negative,
                        SENTIMENT_COLORS.neutral,
                        SENTIMENT_COLORS.positive
                    );
                    validCount++;
                }
            } else {
                // Map dropdown value to compact field name
                const sentField = sentType === 'positive' ? F.positive : sentType === 'negative' ? F.negative : F.neutral;
                const val = node[sentField];
                if (val !== null && val !== undefined) {
                    // Continuous scale for individual dimensions
                    const t = Math.min(1, Math.max(0, val));
                    const target = SENTIMENT_COLORS[sentType];
                    color = SCALE_LOW.map((channel, j) => channel + t * (target[j] - channel));
                    validCount++;
                }
            }
        }
        else if (mode === 'topic_membership') {
            const topicIdx = document.getElementById('topic-select').value;
            const val = node[F.topic(topicIdx)];
            if (val !== null && val !== undefined) {
                // Light grey (low) to saturated teal (high)
                const t = Math.min(1, Math.max(0, val * 2.5));
                color = [
                    0.85 - t * 0.65,  // 0.85 -> 0.20
                    0.85 - t * 0.25,  // 0.85 -> 0.60
                    0.88 - t * 0.13   // 0.88 -> 0.75
                ];
                validCount++;
            }
        }
        else if (mode === 'emotion') {
            const emotion = document.getElementById('emotion-select').value;
            // Map emotion name to compact field
            const emotionMap = {
                anger: F.anger, anticipation: F.anticipation, disgust: F.disgust, fear: F.fear,
                joy: F.joy, love: F.love, optimism: F.optimism, pessimism: F.pessimism,
                sadness: F.sadness, surprise: F.surprise, trust: F.trust
            };
            const val = node[emotionMap[emotion]];
            if (val !== null && val !== undefined) {
                // Sequential scale from pale neutral to the matching figure colour.
                const t = Math.min(1, Math.max(0, val * 1.8)); // Scale for visibility
                const target = EMOTION_COLORS[emotion];
                color = EMOTION_SCALE_LOW.map((channel, j) => channel + t * (target[j] - channel));
                validCount++;
            }
        }

        colors[i * 3] = color[0];
        colors[i * 3 + 1] = color[1];
        colors[i * 3 + 2] = color[2];
    }

    if (colorBuffer) colorBuffer.subdata(colors);
    updateLegend(mode);
}

function updateLegend(mode) {
    const container = document.getElementById('legend');
    container.innerHTML = '';

    // Helper to create RGB string
    const rgb = (c) => `rgb(${Math.round(c[0]*255)},${Math.round(c[1]*255)},${Math.round(c[2]*255)})`;

    if (mode === 'community') {
        // Discrete swatches: the labelled communities, largest first, then
        // a single entry for the unlabelled long tail.
        const comms = [...new Set(nodes.map(n => n[F.community]))]
            .filter(c => c !== null && c >= 0);

        let labelled = labelledCommunityOrder.filter(c => communityIsLabelled(c));
        if (labelled.length === 0) {
            labelled = comms.slice().sort((a, b) => a - b).slice(0, 12);
        }
        const nTail = comms.length - labelled.length;

        const items = labelled.map(c => ({
            label: communityLabel(c),
            color: COMMUNITY_COLORS[c % COMMUNITY_COLORS.length]
        }));
        if (nTail > 0) {
            items.push({
                label: `${nTail.toLocaleString()} long-tail communities`,
                color: null
            });
        }
        items.push({ label: 'No data', color: GREY });

        for (const item of items) {
            const div = document.createElement('div');
            div.className = 'legend-item';
            const swatch = item.color
                ? `<span class="legend-swatch" style="background:${rgb(item.color)}"></span>`
                : `<span class="legend-swatch legend-swatch-multi"></span>`;
            div.innerHTML = `${swatch}<span class="legend-label">${item.label}</span>`;
            container.appendChild(div);
        }
    }
    else if (mode === 'dominant_topic') {
        // Discrete swatches for topics
        for (let i = 0; i < TOPIC_NAMES.length; i++) {
            const div = document.createElement('div');
            div.className = 'legend-item';
            div.innerHTML = `
                <span class="legend-swatch" style="background:${rgb(TOPIC_PALETTE[i % TOPIC_PALETTE.length])}"></span>
                <span class="legend-label">${escapeHtml(TOPIC_NAMES[i])}</span>
            `;
            container.appendChild(div);
        }
        const noData = document.createElement('div');
        noData.className = 'legend-item';
        noData.innerHTML = `<span class="legend-swatch" style="background:${rgb(GREY)}"></span><span class="legend-label">No data</span>`;
        container.appendChild(noData);
    }
    else if (mode === 'sentiment') {
        // Gradient bar for sentiment
        const sentType = document.getElementById('sentiment-select').value;
        const gradientDiv = document.createElement('div');
        gradientDiv.className = 'legend-gradient-wrap';

        if (sentType === 'net') {
            gradientDiv.innerHTML = `
                <div class="legend-gradient" style="background: linear-gradient(to right, ${rgb(SENTIMENT_COLORS.negative)}, ${rgb(SENTIMENT_COLORS.neutral)}, ${rgb(SENTIMENT_COLORS.positive)})"></div>
                <div class="legend-gradient-labels"><span>Negative</span><span>Neutral</span><span>Positive</span></div>
            `;
        } else {
            gradientDiv.innerHTML = `
                <div class="legend-gradient" style="background: linear-gradient(to right, ${rgb(SCALE_LOW)}, ${rgb(SENTIMENT_COLORS[sentType])})"></div>
                <div class="legend-gradient-labels"><span>Low</span><span>High</span></div>
            `;
        }
        container.appendChild(gradientDiv);

        const noData = document.createElement('div');
        noData.className = 'legend-item';
        noData.style.marginTop = '8px';
        noData.innerHTML = `<span class="legend-swatch" style="background:${rgb(GREY)}"></span><span class="legend-label">No data</span>`;
        container.appendChild(noData);
    }
    else if (mode === 'topic_membership') {
        // Gradient bar for topic membership
        const gradientDiv = document.createElement('div');
        gradientDiv.className = 'legend-gradient-wrap';
        gradientDiv.innerHTML = `
            <div class="legend-gradient" style="background: linear-gradient(to right, ${rgb([0.85,0.85,0.88])}, ${rgb([0.20,0.60,0.75])})"></div>
            <div class="legend-gradient-labels"><span>Low</span><span>High</span></div>
        `;
        container.appendChild(gradientDiv);

        const noData = document.createElement('div');
        noData.className = 'legend-item';
        noData.style.marginTop = '8px';
        noData.innerHTML = `<span class="legend-swatch" style="background:${rgb(GREY)}"></span><span class="legend-label">No data</span>`;
        container.appendChild(noData);
    }
    else if (mode === 'emotion') {
        // Sequential gradient ending at the selected emotion's figure colour.
        const emotion = document.getElementById('emotion-select').value;
        const gradientDiv = document.createElement('div');
        gradientDiv.className = 'legend-gradient-wrap';
        gradientDiv.innerHTML = `
            <div class="legend-gradient" style="background: linear-gradient(to right, ${rgb(EMOTION_SCALE_LOW)}, ${rgb(EMOTION_COLORS[emotion])})"></div>
            <div class="legend-gradient-labels"><span>Low</span><span>High</span></div>
        `;
        container.appendChild(gradientDiv);

        const noData = document.createElement('div');
        noData.className = 'legend-item';
        noData.style.marginTop = '8px';
        noData.innerHTML = `<span class="legend-swatch" style="background:${rgb(GREY)}"></span><span class="legend-label">No data</span>`;
        container.appendChild(noData);
    }
}

// ============================================================================
// Projection switching and lazy 3D data
// ============================================================================

async function load3DLayout() {
    if (layout3DState === 'ready') return;
    if (layout3DPromise) return layout3DPromise;
    layout3DState = 'loading';
    updateProjectionUi();

    layout3DPromise = Promise.all([
        fetch(versionedAsset('data/layout3d.bin')),
        fetch(versionedAsset('data/layout3d_indices.bin')),
        fetch(versionedAsset('data/layout3d_meta.json'))
    ]).then(async ([binaryResponse, indexResponse, metadataResponse]) => {
        if (!binaryResponse.ok) throw new Error(`3D coordinates: HTTP ${binaryResponse.status}`);
        if (!indexResponse.ok) throw new Error(`3D render indices: HTTP ${indexResponse.status}`);
        if (!metadataResponse.ok) throw new Error(`3D metadata: HTTP ${metadataResponse.status}`);
        const [binary, indexBinary, metadata] = await Promise.all([
            binaryResponse.arrayBuffer(), indexResponse.arrayBuffer(), metadataResponse.json()
        ]);
        const expectedBytes = nodes.length * 3 * 4;
        const expectedIndexBytes = metadata.rendered_node_count * 4;
        if (metadata.node_count !== nodes.length) {
            throw new Error(`3D metadata has ${metadata.node_count} nodes; expected ${nodes.length}`);
        }
        if (binary.byteLength !== expectedBytes) {
            throw new Error(`3D coordinate payload is ${binary.byteLength} bytes; expected ${expectedBytes}`);
        }
        if (indexBinary.byteLength !== expectedIndexBytes) {
            throw new Error(`3D index payload is ${indexBinary.byteLength} bytes; expected ${expectedIndexBytes}`);
        }
        const coordinates = new Float32Array(binary);
        for (let i = 0; i < coordinates.length; i++) {
            if (!Number.isFinite(coordinates[i])) throw new Error(`Non-finite 3D coordinate at offset ${i}`);
        }
        const renderIndices = new Uint32Array(indexBinary);
        const visibility = new Float32Array(nodes.length);
        for (let i = 0; i < renderIndices.length; i++) {
            const index = renderIndices[i];
            if (index >= nodes.length) throw new Error(`3D render index ${index} is out of bounds`);
            if (visibility[index] > 0.5) throw new Error(`Duplicate 3D render index ${index}`);
            visibility[index] = 1;
        }

        positions3D = coordinates;
        visible3D = visibility;
        layout3DMeta = metadata;
        const fitRadius = Number(metadata.normalised_bounds?.radius_p995)
            || Number(metadata.normalised_bounds?.radius_p99)
            || Number(metadata.normalised_bounds?.max_radius)
            || 1.0;
        camera3D.homeDistance = clamp(fitRadius / Math.sin(Math.PI / 8) * 1.22, 2.8, 5.5);
        positionBuffer3D(positions3D);
        visibilityBuffer3D(visible3D);
        pickFramebuffer = regl.framebuffer({
            color: regl.texture({
                width: Math.max(1, document.getElementById('graph-canvas').width),
                height: Math.max(1, document.getElementById('graph-canvas').height),
                format: 'rgba',
                type: 'uint8',
                min: 'nearest',
                mag: 'nearest'
            }),
            depth: true
        });
        layout3DState = 'ready';
        pickDirty = true;
        updateSelected3DPosition();
        console.log(`Loaded ${metadata.algorithm} (${metadata.rendered_node_count.toLocaleString()} rendered nodes, seed ${metadata.seed})`);
    }).catch(error => {
        layout3DState = 'failed';
        layout3DPromise = null;
        console.error('Failed to load 3D layout:', error);
        throw error;
    }).finally(updateProjectionUi);

    return layout3DPromise;
}

async function setViewMode(mode) {
    if (mode !== '2d' && mode !== '3d') return;
    if (mode === '3d' && layout3DState !== 'ready') {
        try {
            await load3DLayout();
        } catch (error) {
            const hint = document.getElementById('interaction-hint');
            if (hint) hint.textContent = `3D view unavailable: ${error.message}`;
            return;
        }
    }
    if (viewMode === mode) return;
    viewMode = mode;
    document.getElementById('tooltip').style.display = 'none';
    if (mode === '3d') {
        reset3DView(true);
        updateSelected3DPosition();
    }
    pickDirty = true;
    updateProjectionUi();
}

function updateProjectionUi() {
    const button2D = document.getElementById('view-2d');
    const button3D = document.getElementById('view-3d');
    const hint = document.getElementById('interaction-hint');
    if (button2D) {
        button2D.classList.toggle('active', viewMode === '2d');
        button2D.setAttribute('aria-pressed', String(viewMode === '2d'));
    }
    if (button3D) {
        button3D.classList.toggle('active', viewMode === '3d');
        button3D.classList.toggle('loading', layout3DState === 'loading');
        button3D.setAttribute('aria-pressed', String(viewMode === '3d'));
        button3D.disabled = nodes.length === 0 || layout3DState === 'loading';
    }
    if (hint) {
        hint.textContent = viewMode === '3d'
            ? 'Drag to rotate · Shift/right-drag to pan · Wheel to zoom'
            : 'Drag to pan · Wheel to zoom';
    }
}

function reset3DView(immediate) {
    camera3D.desiredYaw = CAMERA3D_HOME_YAW;
    camera3D.desiredPitch = CAMERA3D_HOME_PITCH;
    camera3D.desiredDistance = camera3D.homeDistance;
    camera3D.desiredTarget = [0, 0, 0];
    if (immediate) {
        camera3D.yaw = camera3D.desiredYaw;
        camera3D.pitch = camera3D.desiredPitch;
        camera3D.distance = camera3D.desiredDistance;
        camera3D.target = camera3D.desiredTarget.slice();
    }
    pickDirty = true;
}

function updateSelected3DPosition() {
    if (!positions3D || !selectedNode || !selectedPosition3DBuffer) return;
    const index = nodeIndexById.get(selectedNode[F.id]);
    if (index === undefined || !visible3D || visible3D[index] < 0.5) return;
    selectedPosition3DBuffer(new Float32Array([
        positions3D[index * 3], positions3D[index * 3 + 1], positions3D[index * 3 + 2]
    ]));
}

function nodeVisibleIn3D(node) {
    if (!node || !visible3D) return false;
    const index = nodeIndexById.get(node[F.id]);
    return index !== undefined && visible3D[index] >= 0.5;
}

function focus3DNode(node) {
    if (!positions3D) return;
    const index = nodeIndexById.get(node[F.id]);
    if (index === undefined || !visible3D || visible3D[index] < 0.5) return;
    camera3D.desiredTarget = [
        positions3D[index * 3], positions3D[index * 3 + 1], positions3D[index * 3 + 2]
    ];
    camera3D.desiredDistance = Math.min(camera3D.desiredDistance, 1.15);
    pickDirty = true;
}

// ============================================================================
// Interaction
// ============================================================================

function setupEventListeners() {
    const canvas = document.getElementById('graph-canvas');

    // Gentle wheel zoom in 2D; perspective dolly in 3D.
    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        if (viewMode === '3d') {
            camera3D.desiredDistance = clamp(
                camera3D.desiredDistance * Math.exp(e.deltaY * 0.0011), 0.12, 8
            );
            pickDirty = true;
        } else {
            const factor = e.deltaY > 0 ? 0.97 : 1.03;
            viewScale *= factor;
            viewScale = Math.max(0.001, Math.min(100, viewScale));
        }
    }, { passive: false });

    let dragging = false;
    let dragMode = 'pan';
    let lastX = 0, lastY = 0;
    let downX = 0, downY = 0;
    let downButton = 0;

    canvas.addEventListener('mousedown', (e) => {
        if (e.button !== 0 && e.button !== 2) return;
        if (viewMode === '3d') e.preventDefault();
        dragging = true;
        downButton = e.button;
        dragMode = viewMode === '3d' && e.button === 0 && !e.shiftKey && !e.ctrlKey && !e.metaKey
            ? 'rotate' : 'pan';
        lastX = e.clientX;
        lastY = e.clientY;
        downX = e.clientX;
        downY = e.clientY;
        canvas.style.cursor = 'grabbing';
    });

    canvas.addEventListener('mousemove', (e) => {
        if (dragging) {
            const dx = e.clientX - lastX;
            const dy = e.clientY - lastY;
            if (viewMode === '3d') {
                if (dragMode === 'rotate') {
                    camera3D.desiredYaw -= dx * 0.0062;
                    camera3D.desiredPitch = clamp(camera3D.desiredPitch + dy * 0.0052, -1.38, 1.38);
                } else {
                    const basis = cameraBasis3D();
                    const amount = camera3D.desiredDistance * 0.00155;
                    for (let i = 0; i < 3; i++) {
                        camera3D.desiredTarget[i] -= basis.right[i] * dx * amount;
                        camera3D.desiredTarget[i] += basis.up[i] * dy * amount;
                    }
                }
                pickDirty = true;
            } else {
                const scale = 2 / (canvas.clientHeight * viewScale);
                viewX -= dx * scale;
                viewY += dy * scale;
            }
            lastX = e.clientX;
            lastY = e.clientY;
        } else {
            queueTooltip(e);
        }
    });

    window.addEventListener('mouseup', (e) => {
        if (!dragging) return;
        dragging = false;
        canvas.style.cursor = 'default';

        // A click (not a drag) on a point opens the selected-author panel
        const moved = Math.abs(e.clientX - downX) + Math.abs(e.clientY - downY);
        if (moved < 4 && downButton === 0) {
            const node = pickNodeAt(e);
            if (node) showAuthorPanel(node);
        }
    });

    canvas.addEventListener('mouseleave', () => {
        document.getElementById('tooltip').style.display = 'none';
    });
    canvas.addEventListener('contextmenu', e => {
        if (viewMode === '3d') e.preventDefault();
    });

    // Touch: one finger rotates (or taps a point); two fingers pan and dolly.
    let touchState = null;
    let touchDown = null;
    const touchSnapshot = touches => {
        const points = Array.from(touches).map(t => ({ x: t.clientX, y: t.clientY }));
        const centre = points.reduce((sum, p) => ({ x: sum.x + p.x, y: sum.y + p.y }), { x: 0, y: 0 });
        centre.x /= points.length;
        centre.y /= points.length;
        const distance = points.length > 1 ? Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y) : 0;
        return { points, centre, distance };
    };
    canvas.addEventListener('touchstart', e => {
        if (e.touches.length === 0) return;
        e.preventDefault();
        touchState = touchSnapshot(e.touches);
        if (e.touches.length === 1) {
            touchDown = { x: touchState.centre.x, y: touchState.centre.y, moved: 0 };
        }
        document.getElementById('tooltip').style.display = 'none';
    }, { passive: false });
    canvas.addEventListener('touchmove', e => {
        if (!touchState || e.touches.length === 0) return;
        e.preventDefault();
        const next = touchSnapshot(e.touches);
        const dx = next.centre.x - touchState.centre.x;
        const dy = next.centre.y - touchState.centre.y;
        if (viewMode === '2d') {
            const scale = 2 / (canvas.clientHeight * viewScale);
            viewX -= dx * scale;
            viewY += dy * scale;
            if (next.points.length > 1 && touchState.points.length > 1 && touchState.distance > 0) {
                viewScale = clamp(viewScale * next.distance / touchState.distance, 0.001, 100);
                touchDown = null;
            } else if (touchDown) {
                touchDown.moved += Math.abs(dx) + Math.abs(dy);
            }
        } else if (next.points.length === 1 && touchState.points.length === 1) {
            camera3D.desiredYaw -= dx * 0.0062;
            camera3D.desiredPitch = clamp(camera3D.desiredPitch + dy * 0.0052, -1.38, 1.38);
            if (touchDown) touchDown.moved += Math.abs(dx) + Math.abs(dy);
        } else if (next.points.length > 1 && touchState.points.length > 1) {
            const basis = cameraBasis3D();
            const amount = camera3D.desiredDistance * 0.0013;
            for (let i = 0; i < 3; i++) {
                camera3D.desiredTarget[i] -= basis.right[i] * dx * amount;
                camera3D.desiredTarget[i] += basis.up[i] * dy * amount;
            }
            if (next.distance > 0 && touchState.distance > 0) {
                camera3D.desiredDistance = clamp(
                    camera3D.desiredDistance * touchState.distance / next.distance, 0.12, 8
                );
            }
            touchDown = null;
        }
        touchState = next;
        pickDirty = true;
    }, { passive: false });
    canvas.addEventListener('touchend', e => {
        e.preventDefault();
        if (e.touches.length) {
            touchState = touchSnapshot(e.touches);
            return;
        }
        if (touchDown && touchDown.moved < 6) {
            const node = pickNodeAt({ clientX: touchDown.x, clientY: touchDown.y });
            if (node) showAuthorPanel(node);
        }
        touchState = null;
        touchDown = null;
    }, { passive: false });

    // Color mode radio buttons
    const radioItems = document.querySelectorAll('.radio-item');
    radioItems.forEach(item => {
        item.addEventListener('click', () => {
            // Update active state
            radioItems.forEach(r => r.classList.remove('active'));
            item.classList.add('active');

            // Get mode from data attribute
            const mode = item.dataset.mode;
            if (mode) {
                colorMode = mode;

                // Show/hide sub-options
                document.getElementById('sentiment-options').classList.toggle('visible', mode === 'sentiment');
                document.getElementById('topic-options').classList.toggle('visible', mode === 'topic_membership');
                document.getElementById('emotion-options').classList.toggle('visible', mode === 'emotion');

                updateColors();
            }
        });
    });

    // Sub-selects
    document.getElementById('sentiment-select').addEventListener('change', updateColors);
    document.getElementById('topic-select').addEventListener('change', updateColors);
    document.getElementById('emotion-select').addEventListener('change', updateColors);

    // Resize
    window.addEventListener('resize', () => {
        const canvas = document.getElementById('graph-canvas');
        resizeCanvasIfNeeded(canvas);
    });

    // Enter key in search
    document.getElementById('search-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') searchAuthor();
    });

    // Escape closes the selected-author panel
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeAuthorPanel();
    });

    updateProjectionUi();
}

function queueTooltip(e) {
    pendingHoverEvent = { clientX: e.clientX, clientY: e.clientY };
    if (hoverFramePending) return;
    hoverFramePending = true;
    requestAnimationFrame(() => {
        hoverFramePending = false;
        if (pendingHoverEvent) showTooltip(pendingHoverEvent);
        pendingHoverEvent = null;
    });
}

// Nearest node in the accepted 2D layout. Kept separate from the GPU ID pass
// so the original view's selection behaviour remains unchanged.
function pickNodeAt2D(e) {
    const canvas = document.getElementById('graph-canvas');
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Convert to data coordinates
    const aspect = canvas.clientWidth / canvas.clientHeight;
    const dataX = viewX + (x / canvas.clientWidth * 2 - 1) / viewScale * aspect;
    const dataY = viewY - (y / canvas.clientHeight * 2 - 1) / viewScale;

    // Find nearest node
    let nearest = null;
    let minDist = Infinity;
    const threshold = 0.08 / viewScale;

    for (const node of nodes) {
        const dx = node.x - dataX;
        const dy = node.y - dataY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < minDist && dist < threshold) {
            minDist = dist;
            nearest = node;
        }
    }
    return nearest;
}

function renderPickBuffer3D() {
    if (!pickFramebuffer || !positions3D) return;
    const canvas = document.getElementById('graph-canvas');
    pickFramebuffer.use(() => {
        regl.clear({ color: [0, 0, 0, 0], depth: 1 });
        drawPick3D({
            view: camera3D.view,
            projection: camera3D.projection,
            pointScale: 3.15 * window.devicePixelRatio,
            minPickSize: 7 * window.devicePixelRatio
        });
    });
    pickDirty = false;
}

function pickNodeAt3D(e) {
    if (!pickFramebuffer || !positions3D) return null;
    const canvas = document.getElementById('graph-canvas');
    const rect = canvas.getBoundingClientRect();
    if (e.clientX < rect.left || e.clientX >= rect.right || e.clientY < rect.top || e.clientY >= rect.bottom) {
        return null;
    }
    if (pickDirty) renderPickBuffer3D();
    const dpr = window.devicePixelRatio;
    const x = clamp(Math.floor((e.clientX - rect.left) * dpr), 0, canvas.width - 1);
    const y = clamp(canvas.height - 1 - Math.floor((e.clientY - rect.top) * dpr), 0, canvas.height - 1);
    regl.read({
        framebuffer: pickFramebuffer,
        x, y, width: 1, height: 1,
        data: pickReadback
    });
    const encoded = pickReadback[0] + pickReadback[1] * 256 + pickReadback[2] * 65536;
    return encoded > 0 && encoded <= nodes.length ? nodes[encoded - 1] : null;
}

function pickNodeAt(e) {
    return viewMode === '3d' ? pickNodeAt3D(e) : pickNodeAt2D(e);
}

function showTooltip(e) {
    const nearest = pickNodeAt(e);

    const tooltip = document.getElementById('tooltip');

    if (nearest) {
        hoverNode = nearest;
        let html = `<div class="tt-author">${nearest[F.id]}</div>`;
        html += `<div class="tt-row"><span class="tt-label">Community</span><span class="tt-value">${nearest[F.community]}</span></div>`;
        html += `<div class="tt-row"><span class="tt-label">Full degree</span><span class="tt-value">${nearest[F.fullDegree]?.toFixed(0) || '-'}</span></div>`;
        html += `<div class="tt-row"><span class="tt-label">Matched degree</span><span class="tt-value">${nearest[F.matchedDegree]?.toFixed(0) || '-'}</span></div>`;

        const domTopic = nearest[F.dominantTopic];
        if (domTopic !== null && domTopic !== undefined && domTopic >= 0) {
            const topicName = TOPIC_NAMES[domTopic] || `Topic ${domTopic}`;
            html += `<div class="tt-row"><span class="tt-label">Topic</span><span class="tt-value">${escapeHtml(topicName)}</span></div>`;
            html += `<div class="tt-row"><span class="tt-label">Topic prob</span><span class="tt-value">${(nearest[F.dominantProb] * 100).toFixed(0)}%</span></div>`;
        }

        const pos = nearest[F.positive];
        if (pos !== null && pos !== undefined) {
            html += `<div class="tt-row"><span class="tt-label">Sentiment</span><span class="tt-value">+${pos.toFixed(2)} / ${nearest[F.neutral].toFixed(2)} / -${nearest[F.negative].toFixed(2)}</span></div>`;
        }

        tooltip.innerHTML = html;
        tooltip.style.display = 'block';

        // Position tooltip
        let left = e.clientX + 12;
        let top = e.clientY + 12;

        // Keep within viewport
        const tooltipRect = tooltip.getBoundingClientRect();
        if (left + tooltipRect.width > window.innerWidth - 10) {
            left = e.clientX - tooltipRect.width - 12;
        }
        if (top + tooltipRect.height > window.innerHeight - 10) {
            top = e.clientY - tooltipRect.height - 12;
        }

        tooltip.style.left = left + 'px';
        tooltip.style.top = top + 'px';
    } else {
        hoverNode = null;
        tooltip.style.display = 'none';
    }
}

// ============================================================================
// Selected-author panel
// ============================================================================

// The dataset stores numeric author IDs only; no display name or @handle was
// retained in the matched author table, so the ID is what we can show.
function authorDisplay(node) {
    return node[F.id];
}

function loadExamplePosts() {
    if (examplePostsState !== 'idle') return;
    examplePostsState = 'loading';
    fetch(versionedAsset('data/representative_posts.json'))
        .then(r => (r.ok ? r.json() : Promise.reject(new Error('HTTP ' + r.status))))
        .then(j => {
            examplePosts = j.posts || j;
            examplePostsState = 'ready';
            console.log(`Loaded ${Object.keys(examplePosts).length} Example posts`);
            if (selectedNode) showAuthorPanel(selectedNode);
        })
        .catch(err => {
            examplePostsState = 'failed';
            console.warn('representative_posts.json unavailable:', err.message);
            if (selectedNode) showAuthorPanel(selectedNode);
        });
}

function formatScore(value) {
    return value === null || value === undefined ? '—' : Number(value).toFixed(3);
}

function examplePostHtml(node) {
    const title = `<div class='ap-example-title'>Example post</div>`;
    if (examplePostsState === 'loading' || examplePostsState === 'idle') {
        return title + `<div class='ap-empty'>Loading…</div>`;
    }
    if (examplePostsState === 'failed' || !examplePosts) {
        return title + `<div class='ap-empty'>Example post unavailable.</div>`;
    }
    const record = examplePosts[node[F.id]];
    if (!record) {
        return title + `<div class='ap-empty'>No sufficiently informative example post available.</div>`;
    }
    return title + `<blockquote class='ap-excerpt'>${escapeHtml(record[3])}</blockquote>`;
}

function showAuthorPanel(node) {
    selectedNode = node;
    updateSelected3DPosition();
    const panel = document.getElementById('author-panel');
    const body = document.getElementById('author-panel-body');
    if (!panel || !body) return;
    const community = node[F.community];
    const communityN = communitySize(community);
    const dominantTopic = node[F.dominantTopic];
    const dominantTopicName = dominantTopic !== null && dominantTopic !== undefined && dominantTopic >= 0
        ? (TOPIC_NAMES[dominantTopic] || `Topic ${dominantTopic}`)
        : '—';

    const row = (label, value) => `<div class='ap-row'><span class='ap-label'>${label}</span><span class='ap-value'>${value}</span></div>`;
    const score = (label, value) => `<div class='ap-score'><span>${label}</span><span>${formatScore(value)}</span></div>`;
    const emotions = [
        ['Anger', node[F.anger]],
        ['Anticipation', node[F.anticipation]],
        ['Disgust', node[F.disgust]],
        ['Fear', node[F.fear]],
        ['Joy', node[F.joy]],
        ['Love', node[F.love]],
        ['Optimism', node[F.optimism]],
        ['Pessimism', node[F.pessimism]],
        ['Sadness', node[F.sadness]],
        ['Surprise', node[F.surprise]],
        ['Trust', node[F.trust]],
    ];

    let html = `<div class='ap-author'>${escapeHtml(node[F.id])}</div>`;
    html += `<div class='ap-section ap-summary'>`;
    html += row('Community', escapeHtml(communityLabel(community)));
    if (communityN !== null) html += row('Community size', communityN.toLocaleString());
    html += row('Dominant topic', escapeHtml(dominantTopicName));
    html += row('Topic weight', formatScore(node[F.dominantProb]));
    html += row('Full-network degree', (node[F.fullDegree] ?? 0).toFixed(0));
    html += `</div>`;

    html += `<div class='ap-section'>`;
    html += `<div class='ap-section-title'>Sentiment scores <span>(raw)</span></div>`;
    html += `<div class='ap-score-grid ap-sentiment-grid'>`;
    html += score('Positive', node[F.positive]);
    html += score('Neutral', node[F.neutral]);
    html += score('Negative', node[F.negative]);
    html += `</div></div>`;

    html += `<div class='ap-section'>`;
    html += `<div class='ap-section-title'>Emotion scores <span>(raw)</span></div>`;
    html += `<div class='ap-score-grid'>`;
    html += emotions.map(([label, value]) => score(label, value)).join('');
    html += `</div></div>`;

    html += `<div class='ap-section'>${examplePostHtml(node)}</div>`;

    body.innerHTML = html;
    panel.classList.remove('hidden');
}
function closeAuthorPanel() {
    selectedNode = null;
    const panel = document.getElementById('author-panel');
    if (panel) panel.classList.add('hidden');
}

// ============================================================================
// Controls
// ============================================================================

function searchAuthor() {
    const query = document.getElementById('search-input').value.trim();
    if (!query) return;

    const index = nodeIndexById.get(query);
    const node = index === undefined ? null : nodes[index];
    const status = document.getElementById('search-status');
    if (node) {
        if (viewMode === '3d') {
            focus3DNode(node);
        } else {
            viewX = node.x;
            viewY = node.y;
            viewScale = 0.3;
        }

        if (status) {
            status.className = 'search-status found';
            status.innerHTML = `Found <strong>${escapeHtml(node[F.id])}</strong><br>`
                + escapeHtml(communityLabel(node[F.community]))
                + (viewMode === '3d' && !nodeVisibleIn3D(node)
                    ? '<br>Outside the 3D giant component'
                    : '');
        }
        showAuthorPanel(node);
    } else {
        if (status) {
            status.className = 'search-status missing';
            status.textContent = `Author ${query} not found among the ${nodes.length.toLocaleString()} matched authors`;
        }
        closeAuthorPanel();
    }
}
function resetView() {
    if (viewMode === '3d') {
        reset3DView(false);
        return;
    }
    viewX = (minX + maxX) / 2;
    viewY = (minY + maxY) / 2;
    const rangeX = maxX - minX;
    const rangeY = maxY - minY;
    viewScale = 0.9 / Math.max(rangeX, rangeY) * 2;
}

function fitToScreen() {
    resetView();
}

function zoomIn() {
    if (viewMode === '3d') {
        camera3D.desiredDistance = Math.max(0.12, camera3D.desiredDistance * 0.8);
        pickDirty = true;
        return;
    }
    viewScale *= 1.25;
    viewScale = Math.min(100, viewScale);
}

function zoomOut() {
    if (viewMode === '3d') {
        camera3D.desiredDistance = Math.min(8, camera3D.desiredDistance * 1.25);
        pickDirty = true;
        return;
    }
    viewScale *= 0.8;
    viewScale = Math.max(0.001, viewScale);
}

// Start
document.addEventListener('DOMContentLoaded', init);
