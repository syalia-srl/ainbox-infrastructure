/* SYALIA UI — generic drag-to-resize for side panels. SOURCE OF TRUTH (copied
 * into each app's static, kept in sync). Mirrors magpie's chat-resize pattern.
 *
 * The panel must be a flex item with an explicit width (flex: none; width: …);
 * the middle pane should be flex: 1 / min-width: 0 so it absorbs the change.
 *
 *   makeResizable({ handle, panel, side: 'left'|'right', min, max, storageKey })
 */
function makeResizable({ handle, panel, side = "left", min = 160, max = 720, storageKey }) {
  if (!handle || !panel) return;
  if (storageKey) {
    const saved = parseInt(localStorage.getItem(storageKey) || "", 10);
    if (saved && saved >= min) panel.style.width = Math.min(saved, max) + "px";
  }
  let resizing = false;
  const move = (e) => {
    if (!resizing) return;
    const rect = panel.getBoundingClientRect();
    const w = side === "left" ? e.clientX - rect.left : rect.right - e.clientX;
    panel.style.width = Math.min(max, Math.max(min, w)) + "px";
  };
  const up = () => {
    if (!resizing) return;
    resizing = false;
    handle.classList.remove("resizing");
    document.body.style.userSelect = "";
    document.body.style.cursor = "";
    document.removeEventListener("mousemove", move);
    document.removeEventListener("mouseup", up);
    if (storageKey) {
      localStorage.setItem(storageKey, String(parseInt(panel.style.width, 10) || ""));
    }
  };
  handle.addEventListener("mousedown", (e) => {
    e.preventDefault();
    resizing = true;
    handle.classList.add("resizing");
    document.body.style.userSelect = "none";
    document.body.style.cursor = "col-resize";
    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", up);
  });
}

if (typeof window !== "undefined") window.makeResizable = makeResizable;
