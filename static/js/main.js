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

    // D. Enhance Page Citations [Page X], [Pages X–Y], [Pages X-Y] into interactive jump pills
    rawHtml = rawHtml.replace(/\[Pages?\s+(\d+)\s*[–-]\s*(\d+)\]/gi, '<button type="button" class="page-citation-pill" data-page="$1" title="Jump to Pages $1–$2">Pages $1–$2</button>');
    rawHtml = rawHtml.replace(/\[Pages?\s+(\d+)\]/gi, '<button type="button" class="page-citation-pill" data-page="$1" title="Jump to Page $1">Page $1</button>');

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
    enhanceDocumentSections(formattedContainer);
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
        enhanceQADirectAnswers(formattedElement);
    });

    bindCitationPills();
}

/**
 * Master render function for all AI Markdown sections.
 */
function renderAllContent() {
    renderMarkdownSummary();
    renderMarkdownQAAnswers();
    renderPageExplanations();
}

/**
 * Renders all AI page walkthrough explanations present in the DOM.
 */
function renderPageExplanations() {
    document.querySelectorAll(".page-explanation-content").forEach(container => {
        if (container.getAttribute("data-rendered") === "true" && typeof marked !== "undefined") {
            return;
        }

        const rawElement = container.querySelector(".page-explanation-raw");
        if (!rawElement) return;

        const rawText = (rawElement.textContent || "").trim();
        if (!rawText) return;

        const cleanHtml = formatMarkdownToSafeHtml(rawText);
        container.innerHTML = cleanHtml;
        container.setAttribute("data-rendered", "true");

        applyMathRendering(container);
        enhanceDocumentSections(container);
        bindCitationPills();

        // Update page heading title if a descriptive title exists in explanation
        const h1 = container.querySelector("h1, h2");
        if (h1) {
            const pageIdMatch = container.id.match(/page-explanation-text-(\d+)/);
            if (pageIdMatch) {
                const pageNum = pageIdMatch[1];
                const headingEl = document.getElementById(`page-heading-${pageNum}`);
                if (headingEl) {
                    const cleanTitle = h1.textContent.trim();
                    if (cleanTitle) {
                        headingEl.textContent = cleanTitle;
                    }
                }
            }
        }
    });
}

/**
 * Attaches click handlers for on-demand "Explain Page", "Regenerate", and retry actions.
 */
function bindPageExplanationActions() {
    document.querySelectorAll(".btn-explain-page, .btn-refresh-explanation, .btn-retry-explanation").forEach(btn => {
        if (btn.getAttribute("data-bound") === "true") return;
        btn.setAttribute("data-bound", "true");

        btn.addEventListener("click", async (e) => {
            e.preventDefault();
            e.stopPropagation();

            const pageNum = btn.getAttribute("data-page");
            const storedFilename = btn.getAttribute("data-stored");
            const originalFilename = btn.getAttribute("data-original") || storedFilename;
            const isRegenerate = btn.classList.contains("btn-refresh-explanation") || btn.textContent.toLowerCase().includes("regenerate");

            if (!pageNum || !storedFilename) return;

            const accordion = document.getElementById(`page-item-${pageNum}`);
            if (accordion) accordion.open = true;

            const wrap = document.getElementById(`page-explanation-wrap-${pageNum}`);
            const textContainer = document.getElementById(`page-explanation-text-${pageNum}`);
            const loading = document.getElementById(`page-loading-${pageNum}`);
            const badge = document.getElementById(`page-badge-${pageNum}`);
            const unexplained = document.getElementById(`page-unexplained-${pageNum}`);
            const errorBox = document.getElementById(`page-error-${pageNum}`);

            if (errorBox) errorBox.style.display = "none";
            if (loading) loading.style.display = "flex";
            if (wrap && isRegenerate) wrap.style.opacity = "0.5";
            btn.disabled = true;

            try {
                const csrfInput = document.querySelector('input[name="csrf_token"]');
                const csrfToken = csrfInput ? csrfInput.value : getCSRFToken();

                const formData = new FormData();
                formData.append("csrf_token", csrfToken);
                formData.append("stored_filename", storedFilename);
                formData.append("original_filename", originalFilename);
                formData.append("page_number", pageNum);
                if (isRegenerate) {
                    formData.append("regenerate", "true");
                }

                const response = await fetch("/explain-page", {
                    method: "POST",
                    body: formData,
                    headers: {
                        "X-Requested-With": "XMLHttpRequest"
                    }
                });

                const data = await response.json();

                if (data.success && data.explanation) {
                    if (textContainer) {
                        const cleanHtml = formatMarkdownToSafeHtml(data.explanation);
                        textContainer.innerHTML = cleanHtml;
                        textContainer.setAttribute("data-rendered", "true");

                        applyMathRendering(textContainer);
                        enhanceDocumentSections(textContainer);
                        bindCitationPills();

                        const h1 = textContainer.querySelector("h1, h2");
                        const headingEl = document.getElementById(`page-heading-${pageNum}`);
                        if (h1 && headingEl) {
                            headingEl.textContent = h1.textContent.trim();
                        }
                    }

                    if (wrap) {
                        wrap.style.display = "block";
                        wrap.style.opacity = "1";
                    }

                    if (unexplained) {
                        unexplained.style.display = "none";
                    }

                    if (errorBox) {
                        errorBox.style.display = "none";
                    }

                    if (badge) {
                        badge.style.display = "inline-flex";
                    }

                    // Update action buttons for this page
                    if (accordion) {
                        accordion.querySelectorAll(".btn-explain-page").forEach(b => {
                            b.textContent = "Regenerate Explanation";
                        });
                    }
                } else {
                    const errMsg = data.error || "Failed to generate explanation for this page.";
                    if (errorBox) {
                        const msgEl = errorBox.querySelector(".error-msg");
                        if (msgEl) msgEl.textContent = errMsg;
                        errorBox.style.display = "flex";
                        bindPageExplanationActions();
                    } else {
                        alert(errMsg);
                    }
                }
            } catch (err) {
                console.error("Page explanation request failed:", err);
                const errMsg = "An error occurred while communicating with the server. Please check your connection.";
                if (errorBox) {
                    const msgEl = errorBox.querySelector(".error-msg");
                    if (msgEl) msgEl.textContent = errMsg;
                    errorBox.style.display = "flex";
                    bindPageExplanationActions();
                } else {
                    alert(errMsg);
                }
            } finally {
                if (loading) loading.style.display = "none";
                if (wrap) wrap.style.opacity = "1";
                btn.disabled = false;
            }
        });
    });
}

/**
 * Collects immediate sibling elements following a heading until the next heading of equal or higher level.
 */
function collectSectionSiblings(heading, stopLevels) {
    const levels = stopLevels || ["H1", "H2"];
    const siblings = [];
    let next = heading.nextElementSibling;
    while (next && !levels.includes(next.tagName.toUpperCase())) {
        siblings.push(next);
        next = next.nextElementSibling;
    }
    return siblings;
}

/**
 * Transforms Document Information lists into clean, structured metadata grids.
 */
function enhanceDocumentInformation(container) {
    const headings = container.querySelectorAll("h1, h2, h3, h4");
    headings.forEach(heading => {
        const text = heading.textContent.toLowerCase();
        if ((text.includes("document information") || text.includes("document metadata")) && !heading.closest(".document-meta-card")) {
            heading.classList.add("document-info-heading");
            const siblings = collectSectionSiblings(heading, ["H1", "H2"]);
            const list = siblings.find(el => el.tagName === "UL" || el.tagName === "OL");
            if (list) {
                const metaGrid = document.createElement("div");
                metaGrid.className = "document-meta-card";

                let hasChips = false;
                list.querySelectorAll("li").forEach(li => {
                    const strong = li.querySelector("strong, b");
                    if (strong) {
                        const rawLabel = strong.textContent.replace(/:$/, "").trim();
                        const valText = li.textContent.replace(strong.textContent, "").trim().replace(/^:\s*/, "");

                        const chip = document.createElement("div");
                        chip.className = "meta-chip";

                        const labelEl = document.createElement("span");
                        labelEl.className = "meta-chip-label";
                        labelEl.textContent = rawLabel;

                        const valEl = document.createElement("span");
                        valEl.className = "meta-chip-value";
                        valEl.textContent = valText || "—";

                        chip.appendChild(labelEl);
                        chip.appendChild(valEl);
                        metaGrid.appendChild(chip);
                        hasChips = true;
                    }
                });

                if (hasChips) {
                    list.parentNode.replaceChild(metaGrid, list);
                }
            }
        }
    });
}

/**
 * Packages the Executive Overview or Page Overview section into a distinct callout card.
 */
function enhanceExecutiveOverview(container) {
    const headings = container.querySelectorAll("h1, h2, h3, h4");
    headings.forEach(heading => {
        const text = heading.textContent.toLowerCase();
        if ((text.includes("executive overview") || text.includes("executive summary") || text.includes("page overview") || text === "overview") && !heading.closest(".executive-overview-callout")) {
            heading.classList.add("executive-overview-heading");
            const callout = document.createElement("div");
            callout.className = "executive-overview-callout";

            heading.parentNode.insertBefore(callout, heading);
            callout.appendChild(heading);

            while (callout.nextSibling) {
                const sibling = callout.nextSibling;
                if (sibling.nodeType === Node.ELEMENT_NODE && /^H[1-2]$/i.test(sibling.tagName)) {
                    break;
                }
                callout.appendChild(sibling);
            }
        }
    });
}

/**
 * Packages Key Takeaways sections into an accented callout card.
 */
function enhanceKeyTakeaways(container) {
    const headings = container.querySelectorAll("h1, h2, h3, h4");
    headings.forEach(heading => {
        const text = heading.textContent.toLowerCase();
        if ((text.includes("key takeaways") || text.includes("key findings")) && !heading.closest(".takeaways-callout")) {
            heading.classList.add("takeaways-heading");
            const callout = document.createElement("div");
            callout.className = "takeaways-callout";

            heading.parentNode.insertBefore(callout, heading);
            callout.appendChild(heading);

            while (callout.nextSibling) {
                const sibling = callout.nextSibling;
                if (sibling.nodeType === Node.ELEMENT_NODE && /^H[1-2]$/i.test(sibling.tagName)) {
                    break;
                }
                callout.appendChild(sibling);
            }
        }
    });
}

/**
 * Packages Limitations & Considerations into an alert/callout box.
 */
function enhanceLimitations(container) {
    const headings = container.querySelectorAll("h1, h2, h3, h4");
    headings.forEach(heading => {
        const text = heading.textContent.toLowerCase();
        if (text.includes("limitations") && !heading.closest(".limitations-callout")) {
            heading.classList.add("limitations-heading");
            const callout = document.createElement("div");
            callout.className = "limitations-callout";

            heading.parentNode.insertBefore(callout, heading);
            callout.appendChild(heading);

            while (callout.nextSibling) {
                const sibling = callout.nextSibling;
                if (sibling.nodeType === Node.ELEMENT_NODE && /^H[1-2]$/i.test(sibling.tagName)) {
                    break;
                }
                callout.appendChild(sibling);
            }
        }
    });
}

/**
 * Packages Conclusion & Implications into a distinctive callout card.
 */
function enhanceConclusion(container) {
    const headings = container.querySelectorAll("h1, h2, h3, h4");
    headings.forEach(heading => {
        const text = heading.textContent.toLowerCase();
        if ((text.includes("conclusion") || text.includes("strategic implications")) && !heading.closest(".conclusion-callout")) {
            heading.classList.add("conclusion-heading");
            const callout = document.createElement("div");
            callout.className = "conclusion-callout";

            heading.parentNode.insertBefore(callout, heading);
            callout.appendChild(heading);

            while (callout.nextSibling) {
                const sibling = callout.nextSibling;
                if (sibling.nodeType === Node.ELEMENT_NODE && /^H[1-2]$/i.test(sibling.tagName)) {
                    break;
                }
                callout.appendChild(sibling);
            }
        }
    });
}

/**
 * Packages Q&A Direct Answer sections into a prominent highlight card.
 */
function enhanceQADirectAnswers(container) {
    const headings = container.querySelectorAll("h1, h2, h3, h4");
    headings.forEach(heading => {
        const text = heading.textContent.toLowerCase();
        if (text.includes("direct answer") && !heading.closest(".qa-direct-answer-callout")) {
            heading.classList.add("qa-direct-answer-heading");
            const callout = document.createElement("div");
            callout.className = "qa-direct-answer-callout";

            heading.parentNode.insertBefore(callout, heading);
            callout.appendChild(heading);

            while (callout.nextSibling) {
                const sibling = callout.nextSibling;
                if (sibling.nodeType === Node.ELEMENT_NODE && /^H[1-4]$/i.test(sibling.tagName)) {
                    break;
                }
                callout.appendChild(sibling);
            }
        }
    });
}

/**
 * Master enhancer for all document summary semantic report sections.
 */
function enhanceDocumentSections(container) {
    enhanceDocumentInformation(container);
    enhanceExecutiveOverview(container);
    enhanceKeyTakeaways(container);
    enhanceLimitations(container);
    enhanceConclusion(container);
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
    // 1. Initial Markdown, LaTeX, and page walkthrough rendering on DOM ready
    renderAllContent();
    bindPageExplanationActions();

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
