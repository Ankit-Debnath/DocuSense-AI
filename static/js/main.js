// PDF Analyzer GenAI - Frontend Script
// Handles Markdown report rendering, Q&A conversation formatting, math notation, page navigation, and UI loading states

/**
 * Safely escapes HTML special characters into text entities.
 */
function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

/**
 * Retrieves the active CSRF token from the page meta tag if available.
 */
function getCSRFToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute("content") : "";
}

/**
 * Transforms raw Markdown into sanitized, safe HTML.
 * Stashes LaTeX mathematical notation to prevent parser mangling, parses GFM,
 * enhances page citation tags into interactive pills, and sanitizes via DOMPurify.
 * If DOMPurify is unavailable, fails closed by returning escaped plain text.
 */
function formatMarkdownToSafeHtml(rawMarkdown) {
    const text = (rawMarkdown || "").trim();
    if (!text) return "";

    // Fail-closed security guard:
    // If DOMPurify is not available, never attempt regex HTML sanitization.
    // Display content as safely escaped plain text instead of injecting raw HTML.
    if (typeof DOMPurify === "undefined" || typeof DOMPurify.sanitize !== "function") {
        console.warn("DOMPurify sanitizer is unavailable. Failing closed to escaped plain text for security.");
        return `<pre class="plain-text-fallback">${escapeHtml(text)}</pre>`;
    }

    // A. Mathematical Expression Stashing
    const mathBlocks = [];
    let processed = text;

    // 1. Display math $$ ... $$
    processed = processed.replace(/\$\$([\s\S]+?)\$\$/g, (match, math) => {
        const id = `%%MATH_BLOCK_${mathBlocks.length}%%`;
        mathBlocks.push({ id, math: `$$${math}$$` });
        return id;
    });

    // 2. Display math \[ ... \]
    processed = processed.replace(/\\\[([\s\S]+?)\\\]/g, (match, math) => {
        const id = `%%MATH_BLOCK_${mathBlocks.length}%%`;
        mathBlocks.push({ id, math: `\\[${math}\\]` });
        return id;
    });

    // 3. Inline math $ ... $ (ignoring pure currency like $10 or $25.50)
    processed = processed.replace(/(^|[^\$])\$([^\s\$][^\$\n]*?[^\s\$]|[^\s\$])\$(?!\$)/g, (match, prefix, math) => {
        if (/^\s*\d+(?:\.\d+)?\s*$/.test(math)) {
            return match;
        }
        const id = `%%MATH_INLINE_${mathBlocks.length}%%`;
        mathBlocks.push({ id, math: `$${math}$` });
        return `${prefix}${id}`;
    });

    // 4. Inline math \( ... \)
    processed = processed.replace(/\\\(([\s\S]+?)\\\)/g, (match, math) => {
        const id = `%%MATH_INLINE_${mathBlocks.length}%%`;
        mathBlocks.push({ id, math: `\\(${math}\\)` });
        return id;
    });

    // B. Convert Markdown to HTML using Marked.js
    let rawHtml = "";
    if (typeof marked !== "undefined" && typeof marked.parse === "function") {
        marked.setOptions({
            gfm: true,
            breaks: true,
            pedantic: false
        });
        rawHtml = marked.parse(processed);
    } else {
        rawHtml = escapeHtml(processed);
    }

    // C. Restore Protected Mathematical Formulas
    for (const item of mathBlocks) {
        rawHtml = rawHtml.replace(item.id, item.math);
    }

    // D. Enhance Page Citations [Page X] or [Pages X–Y] into interactive jump pills
    rawHtml = rawHtml.replace(/\[Page\s+(\d+)\]/gi, '<button type="button" class="page-citation-pill" data-page="$1" title="Jump to Page $1">Page $1</button>');
    rawHtml = rawHtml.replace(/\[Pages\s+(\d+)[–-](\d+)\]/gi, '<button type="button" class="page-citation-pill" data-page="$1" title="Jump to Pages $1–$2">Pages $1–$2</button>');

    // E. Sanitize with DOMPurify to guarantee XSS prevention
    const cleanHtml = DOMPurify.sanitize(rawHtml, {
        ALLOWED_TAGS: [
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'p', 'br', 'hr',
            'ul', 'ol', 'li',
            'strong', 'b', 'em', 'i', 's', 'del', 'mark',
            'blockquote', 'pre', 'code',
            'table', 'thead', 'tbody', 'tr', 'th', 'td',
            'a', 'span', 'div', 'button'
        ],
        ALLOWED_ATTR: ['href', 'target', 'rel', 'class', 'id', 'title', 'type', 'data-page'],
        FORBID_TAGS: ['script', 'style', 'iframe', 'frame', 'object', 'embed', 'form', 'input'],
        FORBID_ATTR: ['onerror', 'onload', 'onclick', 'onmouseover', 'onfocus', 'onblur', 'style']
    });

    // Wrap tables in responsive scroll wrapper
    return cleanHtml.replace(/<table>/gi, '<div class="table-wrapper"><table>').replace(/<\/table>/gi, '</table></div>');
}

/**
 * Applies KaTeX mathematical formatting to an element.
 */
function applyMathRendering(element) {
    if (typeof renderMathInElement === "function") {
        try {
            renderMathInElement(element, {
                delimiters: [
                    { left: "$$", right: "$$", display: true },
                    { left: "$", right: "$", display: false },
                    { left: "\\[", right: "\\]", display: true },
                    { left: "\\(", right: "\\)", display: false }
                ],
                throwOnError: false
            });
        } catch (err) {
            console.warn("KaTeX render error:", err);
        }
    } else {
        wrapFallbackMath(element);
    }
}

/**
 * Attaches interactive click-to-jump handlers to all page citation pills.
 */
function bindCitationPills() {
    document.querySelectorAll(".page-citation-pill").forEach(pill => {
        if (pill.getAttribute("data-bound") === "true") return;
        pill.setAttribute("data-bound", "true");

        pill.addEventListener("click", (e) => {
            e.preventDefault();
            const pageNum = pill.getAttribute("data-page");
            if (!pageNum) return;

            const pageAccordion = document.getElementById(`page-item-${pageNum}`);
            if (pageAccordion) {
                pageAccordion.open = true;
                pageAccordion.scrollIntoView({ behavior: "smooth", block: "center" });

                pageAccordion.classList.add("page-accordion-highlight");
                setTimeout(() => {
                    pageAccordion.classList.remove("page-accordion-highlight");
                }, 2000);
            }
        });
    });
}

/**
 * Renders the AI Document Summary report.
 */
function renderMarkdownSummary() {
    const rawContainer = document.getElementById("summary-raw-content");
    const formattedContainer = document.getElementById("formatted-summary");

    if (!rawContainer || !formattedContainer) return;

    const rawMarkdown = (rawContainer.textContent || "").trim();
    if (!rawMarkdown) return;

    if (formattedContainer.getAttribute("data-rendered") === "true" && typeof marked !== "undefined") {
        return;
    }

    const cleanHtml = formatMarkdownToSafeHtml(rawMarkdown);
    formattedContainer.innerHTML = cleanHtml;
    formattedContainer.setAttribute("data-rendered", "true");

    applyMathRendering(formattedContainer);
    enhanceExecutiveOverview(formattedContainer);
    bindCitationPills();
}

/**
 * Renders all AI answers in the Document Q&A conversation thread.
 */
function renderMarkdownQAAnswers() {
    document.querySelectorAll(".qa-answer-markdown").forEach(container => {
        if (container.getAttribute("data-rendered") === "true" && typeof marked !== "undefined") {
            return;
        }

        const rawElement = container.querySelector(".qa-raw-text");
        const formattedElement = container.querySelector(".qa-formatted-answer");

        if (!rawElement || !formattedElement) return;

        const rawText = (rawElement.textContent || "").trim();
        if (!rawText) return;

        const cleanHtml = formatMarkdownToSafeHtml(rawText);
        formattedElement.innerHTML = cleanHtml;
        container.setAttribute("data-rendered", "true");

        applyMathRendering(formattedElement);
    });

    bindCitationPills();
}

/**
 * Master render function for all AI Markdown sections.
 */
function renderAllContent() {
    renderMarkdownSummary();
    renderMarkdownQAAnswers();
}

/**
 * Packages the Executive Overview section into a distinct callout card.
 */
function enhanceExecutiveOverview(container) {
    const headings = container.querySelectorAll("h1, h2, h3, h4");
    headings.forEach(heading => {
        const text = heading.textContent.toLowerCase();
        if (text.includes("executive overview") || text.includes("executive summary")) {
            heading.classList.add("executive-overview-heading");
            const callout = document.createElement("div");
            callout.className = "executive-overview-callout";

            heading.parentNode.insertBefore(callout, heading);
            callout.appendChild(heading);

            while (callout.nextSibling) {
                const sibling = callout.nextSibling;
                if (sibling.nodeType === Node.ELEMENT_NODE && /^H[1-6]$/i.test(sibling.tagName)) {
                    break;
                }
                callout.appendChild(sibling);
            }
        }
    });
}

/**
 * Styles math notation into readable serif typography if KaTeX is not loaded.
 */
function wrapFallbackMath(container) {
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, null, false);
    const nodesToReplace = [];

    while (walker.nextNode()) {
        const node = walker.currentNode;
        if (node.nodeValue && (node.nodeValue.includes("$") || node.nodeValue.includes("\\("))) {
            nodesToReplace.push(node);
        }
    }

    nodesToReplace.forEach(node => {
        const text = node.nodeValue;
        const replaced = text.replace(/(^|[^\$])\$([^\$\n]+?)\$(?!\$)/g, (m, p, math) => {
            return `${p}<span class="math-fallback">${math}</span>`;
        });
        if (replaced !== text) {
            const span = document.createElement("span");
            span.innerHTML = replaced;
            node.parentNode.replaceChild(span, node);
        }
    });
}

document.addEventListener("DOMContentLoaded", () => {
    // 1. Initial Markdown and LaTeX rendering on DOM ready
    renderAllContent();

    // 2. Re-render once external CDN scripts (Marked, KaTeX) finish loading
    window.addEventListener("load", () => {
        renderAllContent();
    });

    // 3. Page accordion global expand / collapse controls
    const expandPagesBtn = document.getElementById("btn-expand-pages");
    const collapsePagesBtn = document.getElementById("btn-collapse-pages");

    if (expandPagesBtn) {
        expandPagesBtn.addEventListener("click", () => {
            document.querySelectorAll("details.page-accordion").forEach(d => d.open = true);
        });
    }

    if (collapsePagesBtn) {
        collapsePagesBtn.addEventListener("click", () => {
            document.querySelectorAll("details.page-accordion").forEach(d => d.open = false);
        });
    }

    // 4. Summarize form loading state
    const summarizeForm = document.getElementById("summarize-form");
    const summarizeBtn = document.getElementById("summarize-btn");
    const summarizeLoading = document.getElementById("summarize-loading");

    if (summarizeForm && summarizeBtn) {
        summarizeForm.addEventListener("submit", () => {
            summarizeBtn.disabled = true;
            summarizeBtn.classList.add("btn-disabled");
            summarizeBtn.textContent = "Processing...";
            if (summarizeLoading) {
                summarizeLoading.style.display = "flex";
            }
        });
    }

    // 5. Q&A form submission & loading state
    const qaForm = document.getElementById("qa-form");
    const qaBtn = document.getElementById("qa-btn");
    const qaLoading = document.getElementById("qa-loading");

    if (qaForm && qaBtn) {
        qaForm.addEventListener("submit", () => {
            const input = qaForm.querySelector('input[name="question"]');
            if (input && input.value.trim()) {
                qaBtn.disabled = true;
                qaBtn.classList.add("btn-disabled");
                qaBtn.textContent = "Consulting...";
                if (qaLoading) {
                    qaLoading.style.display = "flex";
                }
            }
        });
    }

    // 6. Reset button and loading states on pageshow (bfcache or page reload)
    window.addEventListener("pageshow", () => {
        if (summarizeBtn) {
            summarizeBtn.disabled = false;
            summarizeBtn.classList.remove("btn-disabled");
            const isRegen = summarizeForm && summarizeForm.querySelector('input[name="regenerate"]');
            summarizeBtn.textContent = isRegen ? "Regenerate Summary" : "Generate Summary";
        }
        if (summarizeLoading) {
            summarizeLoading.style.display = "none";
        }
        if (qaBtn) {
            qaBtn.disabled = false;
            qaBtn.classList.remove("btn-disabled");
            qaBtn.innerHTML = `
                <svg class="btn-send-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="22" y1="2" x2="11" y2="13"></line>
                    <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                </svg>
                <span>Ask Question</span>
            `;
        }
        if (qaLoading) {
            qaLoading.style.display = "none";
        }
    });
});
