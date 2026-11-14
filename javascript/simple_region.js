(() => {
    const COLORS = [
        "#ef4444", // red
        "#3b82f6", // blue
        "#10b981", // emerald
        "#f59e0b", // amber
        "#a855f7", // purple
        "#ec4899", // pink
        "#06b6d4", // cyan
        "#84cc16", // lime
    ];

    const MAX_CANVAS_SIZE = 384;
    const HANDLE_RADIUS = 7;

    function getColor(idx) {
        return COLORS[idx % COLORS.length];
    }

    function hexToRgba(hex, alpha) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    function clamp01(v) {
        return Math.max(0, Math.min(1, v));
    }

    function setupTab(tab) {
        const root = document.getElementById(`sr_${tab}_root`);
        if (!root || root.dataset.initialized === "1") return;
        root.dataset.initialized = "1";

        const canvas = root.querySelector(".sr-canvas");
        const ctx = canvas.getContext("2d");
        const sizeLabel = root.querySelector(".sr-size-label");
        const listEl = root.querySelector(".sr-list");
        const splitCountInput = root.querySelector(".sr-split-count");
        const btnAdd = root.querySelector(".sr-btn-add");
        const btnSplitH = root.querySelector(".sr-btn-split-h");
        const btnSplitV = root.querySelector(".sr-btn-split-v");
        const btnSync = root.querySelector(".sr-btn-sync");
        const btnClear = root.querySelector(".sr-btn-clear");
        const bgFileInput = root.querySelector(".sr-bg-file");
        const btnBgI2i = root.querySelector(".sr-btn-bg-i2i");
        const btnBgClear = root.querySelector(".sr-btn-bg-clear");

        const prefix = tab === "t2i" ? "txt2img" : "img2img";
        const widthInput = () =>
            document.querySelector(`#${prefix}_width input[type='number']`);
        const heightInput = () =>
            document.querySelector(`#${prefix}_height input[type='number']`);

        let regions = [];
        let selectedIdx = -1;
        let dragState = null; // { mode: 'draw'|'move'|'resize', idx, corner, startX, startY, orig }
        let bgImage = null;
        let bgObjectUrl = null;

        function syncToGradio() {
            const box = document.querySelector(
                `#sr_${tab}_regions_json textarea, #sr_${tab}_regions_json input`
            );
            if (!box) return;
            const json = JSON.stringify(regions);
            if (box.value !== json) {
                box.value = json;
                box.dispatchEvent(new Event("input", { bubbles: true }));
                box.dispatchEvent(new Event("change", { bubbles: true }));
            }
        }

        function updateCanvasSize() {
            const wEl = widthInput();
            const hEl = heightInput();
            const w = wEl ? Math.max(64, parseInt(wEl.value, 10) || 512) : 512;
            const h = hEl ? Math.max(64, parseInt(hEl.value, 10) || 512) : 512;

            const scale = MAX_CANVAS_SIZE / Math.max(w, h);
            const cw = Math.max(120, Math.round(w * scale));
            const ch = Math.max(120, Math.round(h * scale));

            if (canvas.width !== cw || canvas.height !== ch) {
                canvas.width = cw;
                canvas.height = ch;
            }
            sizeLabel.textContent = `${w} × ${h}`;
            drawCanvas();
        }

        function drawCanvas() {
            const cw = canvas.width;
            const ch = canvas.height;

            ctx.clearRect(0, 0, cw, ch);

            if (bgImage) {
                ctx.save();
                ctx.globalAlpha = 0.45;
                ctx.drawImage(bgImage, 0, 0, cw, ch);
                ctx.restore();
            }

            // Subtle grid
            ctx.strokeStyle = "rgba(128, 128, 128, 0.18)";
            ctx.lineWidth = 1;
            for (let i = 1; i < 4; i++) {
                const gx = (cw * i) / 4;
                const gy = (ch * i) / 4;
                ctx.beginPath();
                ctx.moveTo(gx, 0);
                ctx.lineTo(gx, ch);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(0, gy);
                ctx.lineTo(cw, gy);
                ctx.stroke();
            }

            regions.forEach((reg, idx) => {
                const color = getColor(idx);
                const x = reg.x1 * cw;
                const y = reg.y1 * ch;
                const w = (reg.x2 - reg.x1) * cw;
                const h = (reg.y2 - reg.y1) * ch;
                const isSel = idx === selectedIdx;

                ctx.fillStyle = hexToRgba(color, isSel ? 0.34 : 0.2);
                ctx.fillRect(x, y, w, h);

                ctx.strokeStyle = color;
                ctx.lineWidth = isSel ? 2.5 : 1.5;
                ctx.strokeRect(x, y, w, h);

                // Badge
                ctx.fillStyle = color;
                ctx.fillRect(x, y, 24, 20);
                ctx.fillStyle = "#ffffff";
                ctx.font = "bold 12px sans-serif";
                ctx.textBaseline = "middle";
                ctx.fillText(`#${idx + 1}`, x + 4, y + 10);

                // Resize handles for selected region
                if (isSel) {
                    const corners = [
                        [x, y],
                        [x + w, y],
                        [x, y + h],
                        [x + w, y + h],
                    ];
                    ctx.fillStyle = "#ffffff";
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 2;
                    for (const [cx, cy] of corners) {
                        ctx.beginPath();
                        ctx.arc(cx, cy, 4.5, 0, Math.PI * 2);
                        ctx.fill();
                        ctx.stroke();
                    }
                }
            });

            // Preview rectangle while drawing
            if (dragState && dragState.mode === "draw" && dragState.preview) {
                const p = dragState.preview;
                const color = getColor(regions.length);
                const x = p.x1 * cw;
                const y = p.y1 * ch;
                const w = (p.x2 - p.x1) * cw;
                const h = (p.y2 - p.y1) * ch;
                ctx.fillStyle = hexToRgba(color, 0.25);
                ctx.fillRect(x, y, w, h);
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 3]);
                ctx.strokeRect(x, y, w, h);
                ctx.setLineDash([]);
            }
        }

        function renderList() {
            listEl.innerHTML = "";
            if (regions.length === 0) {
                const empty = document.createElement("div");
                empty.className = "sr-empty";
                empty.textContent =
                    "範囲がありません。左のキャンバスをドラッグするか、上の分割ボタンを押してください。";
                listEl.appendChild(empty);
                return;
            }

            regions.forEach((reg, idx) => {
                const color = getColor(idx);
                const card = document.createElement("div");
                card.className = `sr-card ${idx === selectedIdx ? "is-selected" : ""}`;
                card.style.borderLeftColor = color;

                card.addEventListener("click", () => {
                    if (selectedIdx !== idx) {
                        selectedIdx = idx;
                        drawCanvas();
                        listEl.querySelectorAll(".sr-card").forEach((c, i) => {
                            c.classList.toggle("is-selected", i === selectedIdx);
                        });
                    }
                });

                const header = document.createElement("div");
                header.className = "sr-card-header";

                const badge = document.createElement("span");
                badge.className = "sr-badge";
                badge.style.backgroundColor = color;
                badge.textContent = `#${idx + 1}`;

                const coords = document.createElement("span");
                coords.className = "sr-coords";
                coords.textContent = `X:${Math.round(reg.x1 * 100)}-${Math.round(
                    reg.x2 * 100
                )}%  Y:${Math.round(reg.y1 * 100)}-${Math.round(reg.y2 * 100)}%`;

                const delBtn = document.createElement("button");
                delBtn.type = "button";
                delBtn.className = "sr-btn-del";
                delBtn.textContent = "×";
                delBtn.title = "この範囲を削除";
                delBtn.addEventListener("click", (e) => {
                    e.stopPropagation();
                    regions.splice(idx, 1);
                    if (selectedIdx >= regions.length) {
                        selectedIdx = regions.length - 1;
                    }
                    syncToGradio();
                    drawCanvas();
                    renderList();
                });

                header.appendChild(badge);
                header.appendChild(coords);
                header.appendChild(delBtn);

                const textarea = document.createElement("textarea");
                textarea.className = "sr-prompt";
                textarea.rows = 2;
                textarea.placeholder = `領域 #${idx + 1} に追加するプロンプト...`;
                textarea.value = reg.prompt;
                textarea.addEventListener("input", () => {
                    regions[idx].prompt = textarea.value;
                    syncToGradio();
                });

                const weightRow = document.createElement("div");
                weightRow.className = "sr-weight-row";

                const weightLabel = document.createElement("span");
                weightLabel.className = "sr-weight-label";
                weightLabel.textContent = `重み: ${Number(reg.weight).toFixed(2)}`;

                const weightSlider = document.createElement("input");
                weightSlider.type = "range";
                weightSlider.className = "sr-weight-slider";
                weightSlider.min = "0.1";
                weightSlider.max = "2.0";
                weightSlider.step = "0.05";
                weightSlider.value = String(reg.weight);
                weightSlider.addEventListener("input", () => {
                    const val = parseFloat(weightSlider.value) || 1.0;
                    regions[idx].weight = val;
                    weightLabel.textContent = `重み: ${val.toFixed(2)}`;
                    syncToGradio();
                });

                weightRow.appendChild(weightLabel);
                weightRow.appendChild(weightSlider);

                card.appendChild(header);
                card.appendChild(textarea);
                card.appendChild(weightRow);
                listEl.appendChild(card);
            });
        }

        function getCanvasCoords(e) {
            const rect = canvas.getBoundingClientRect();
            const px = ((e.clientX - rect.left) / rect.width) * canvas.width;
            const py = ((e.clientY - rect.top) / rect.height) * canvas.height;
            return {
                px,
                py,
                nx: clamp01(px / canvas.width),
                ny: clamp01(py / canvas.height),
            };
        }

        function hitTestHandle(px, py, reg) {
            const cw = canvas.width;
            const ch = canvas.height;
            const corners = {
                nw: [reg.x1 * cw, reg.y1 * ch],
                ne: [reg.x2 * cw, reg.y1 * ch],
                sw: [reg.x1 * cw, reg.y2 * ch],
                se: [reg.x2 * cw, reg.y2 * ch],
            };
            for (const [name, [cx, cy]] of Object.entries(corners)) {
                if (Math.hypot(px - cx, py - cy) <= HANDLE_RADIUS + 2) {
                    return name;
                }
            }
            return null;
        }

        function hitTestBox(nx, ny) {
            for (let i = regions.length - 1; i >= 0; i--) {
                const r = regions[i];
                if (nx >= r.x1 && nx <= r.x2 && ny >= r.y1 && ny <= r.y2) {
                    return i;
                }
            }
            return -1;
        }

        canvas.addEventListener("mousedown", (e) => {
            if (e.button !== 0) return;
            const { px, py, nx, ny } = getCanvasCoords(e);

            // Check resize handle on currently selected region first
            if (selectedIdx >= 0 && selectedIdx < regions.length) {
                const corner = hitTestHandle(px, py, regions[selectedIdx]);
                if (corner) {
                    dragState = {
                        mode: "resize",
                        idx: selectedIdx,
                        corner,
                        orig: { ...regions[selectedIdx] },
                    };
                    return;
                }
            }

            // Check if clicking inside an existing region
            const hitIdx = hitTestBox(nx, ny);
            if (hitIdx >= 0) {
                selectedIdx = hitIdx;
                dragState = {
                    mode: "move",
                    idx: hitIdx,
                    startX: nx,
                    startY: ny,
                    orig: { ...regions[hitIdx] },
                };
                drawCanvas();
                renderList();
                return;
            }

            // Otherwise draw a new region
            dragState = {
                mode: "draw",
                startX: nx,
                startY: ny,
                preview: { x1: nx, y1: ny, x2: nx, y2: ny },
            };
        });

        window.addEventListener("mousemove", (e) => {
            if (!dragState) return;
            const { nx, ny } = getCanvasCoords(e);

            if (dragState.mode === "draw") {
                dragState.preview = {
                    x1: Math.min(dragState.startX, nx),
                    y1: Math.min(dragState.startY, ny),
                    x2: Math.max(dragState.startX, nx),
                    y2: Math.max(dragState.startY, ny),
                };
                drawCanvas();
            } else if (dragState.mode === "move") {
                const r = regions[dragState.idx];
                const o = dragState.orig;
                const w = o.x2 - o.x1;
                const h = o.y2 - o.y1;
                let dx = nx - dragState.startX;
                let dy = ny - dragState.startY;
                dx = Math.max(-o.x1, Math.min(1 - o.x2, dx));
                dy = Math.max(-o.y1, Math.min(1 - o.y2, dy));
                r.x1 = o.x1 + dx;
                r.x2 = o.x1 + dx + w;
                r.y1 = o.y1 + dy;
                r.y2 = o.y1 + dy + h;
                drawCanvas();
            } else if (dragState.mode === "resize") {
                const r = regions[dragState.idx];
                const minGap = 0.04;
                if (dragState.corner.includes("w")) {
                    r.x1 = Math.min(nx, r.x2 - minGap);
                }
                if (dragState.corner.includes("e")) {
                    r.x2 = Math.max(nx, r.x1 + minGap);
                }
                if (dragState.corner.includes("n")) {
                    r.y1 = Math.min(ny, r.y2 - minGap);
                }
                if (dragState.corner.includes("s")) {
                    r.y2 = Math.max(ny, r.y1 + minGap);
                }
                drawCanvas();
            }
        });

        window.addEventListener("mouseup", () => {
            if (!dragState) return;

            if (dragState.mode === "draw" && dragState.preview) {
                const p = dragState.preview;
                if (p.x2 - p.x1 >= 0.04 && p.y2 - p.y1 >= 0.04) {
                    regions.push({
                        x1: Number(p.x1.toFixed(3)),
                        y1: Number(p.y1.toFixed(3)),
                        x2: Number(p.x2.toFixed(3)),
                        y2: Number(p.y2.toFixed(3)),
                        prompt: "",
                        weight: 1.0,
                    });
                    selectedIdx = regions.length - 1;
                }
            } else if (dragState.mode === "move" || dragState.mode === "resize") {
                const r = regions[dragState.idx];
                r.x1 = Number(r.x1.toFixed(3));
                r.y1 = Number(r.y1.toFixed(3));
                r.x2 = Number(r.x2.toFixed(3));
                r.y2 = Number(r.y2.toFixed(3));
            }

            dragState = null;
            syncToGradio();
            drawCanvas();
            renderList();
        });

        function applySplit(horizontal) {
            const n = Math.max(2, Math.min(8, parseInt(splitCountInput.value, 10) || 2));
            const next = [];
            for (let i = 0; i < n; i++) {
                const prev = regions[i];
                const a = Number((i / n).toFixed(3));
                const b = Number(((i + 1) / n).toFixed(3));
                next.push({
                    x1: horizontal ? a : 0,
                    y1: horizontal ? 0 : a,
                    x2: horizontal ? b : 1,
                    y2: horizontal ? 1 : b,
                    prompt: prev ? prev.prompt : "",
                    weight: prev ? prev.weight : 1.0,
                });
            }
            regions = next;
            selectedIdx = 0;
            syncToGradio();
            drawCanvas();
            renderList();
        }

        if (btnAdd) {
            btnAdd.addEventListener("click", () => {
                const offset = (regions.length % 5) * 0.05;
                const x1 = Number(Math.min(0.55, 0.2 + offset).toFixed(3));
                const y1 = Number(Math.min(0.55, 0.2 + offset).toFixed(3));
                const x2 = Number(Math.min(0.95, x1 + 0.4).toFixed(3));
                const y2 = Number(Math.min(0.95, y1 + 0.4).toFixed(3));
                regions.push({
                    x1,
                    y1,
                    x2,
                    y2,
                    prompt: "",
                    weight: 1.0,
                });
                selectedIdx = regions.length - 1;
                syncToGradio();
                drawCanvas();
                renderList();
                listEl.scrollTop = listEl.scrollHeight;
            });
        }

        btnSplitH.addEventListener("click", () => {
            updateCanvasSize();
            applySplit(true);
        });
        btnSplitV.addEventListener("click", () => {
            updateCanvasSize();
            applySplit(false);
        });
        if (btnSync) {
            btnSync.addEventListener("click", () => {
                attachSizeListeners();
                updateCanvasSize();
            });
        }
        btnClear.addEventListener("click", () => {
            regions = [];
            selectedIdx = -1;
            syncToGradio();
            drawCanvas();
            renderList();
        });

        function clearBackground() {
            if (bgObjectUrl) {
                URL.revokeObjectURL(bgObjectUrl);
                bgObjectUrl = null;
            }
            bgImage = null;
            if (bgFileInput) bgFileInput.value = "";
            drawCanvas();
        }

        function setBackgroundFromSrc(src, objectUrl = null) {
            if (!src) return;
            const img = new Image();
            img.onload = () => {
                if (bgObjectUrl && bgObjectUrl !== objectUrl) {
                    URL.revokeObjectURL(bgObjectUrl);
                }
                bgObjectUrl = objectUrl;
                bgImage = img;
                drawCanvas();
            };
            img.src = src;
        }

        function loadBackgroundFile(file) {
            if (!file || !file.type.startsWith("image/")) return;
            const url = URL.createObjectURL(file);
            setBackgroundFromSrc(url, url);
        }

        if (bgFileInput) {
            bgFileInput.addEventListener("change", (e) => {
                const file = e.target.files && e.target.files[0];
                loadBackgroundFile(file);
            });
        }

        if (btnBgI2i) {
            btnBgI2i.addEventListener("click", () => {
                const i2iImg = document.querySelector("#img2img_image img");
                if (i2iImg && i2iImg.src) {
                    setBackgroundFromSrc(i2iImg.src);
                }
            });
        }

        if (btnBgClear) {
            btnBgClear.addEventListener("click", clearBackground);
        }

        canvas.addEventListener("dragover", (e) => {
            e.preventDefault();
        });
        canvas.addEventListener("drop", (e) => {
            e.preventDefault();
            const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
            loadBackgroundFile(file);
        });

        // Watch width/height changes (both number and range inputs)
        const attachSizeListeners = () => {
            document
                .querySelectorAll(`#${prefix}_width input, #${prefix}_height input`)
                .forEach((el) => {
                    if (!el.dataset.srWatched) {
                        el.dataset.srWatched = "1";
                        el.addEventListener("input", updateCanvasSize);
                        el.addEventListener("change", updateCanvasSize);
                    }
                });
        };

        const accordion = document.getElementById(`sr_${tab}_accordion`);
        if (accordion && !accordion.dataset.srSyncWatched) {
            accordion.dataset.srSyncWatched = "1";
            accordion.addEventListener("click", () => {
                setTimeout(() => {
                    attachSizeListeners();
                    updateCanvasSize();
                }, 50);
            });
        }

        attachSizeListeners();
        updateCanvasSize();
        renderList();
    }

    onUiLoaded(() => {
        setupTab("t2i");
        setupTab("i2i");
    });
})();
