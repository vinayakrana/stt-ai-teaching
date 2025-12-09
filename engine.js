/**
 * Marp engine with Mermaid.js support
 * Renders ```mermaid code blocks as diagrams
 */

const { Marp } = require('@marp-team/marp-core');

module.exports = (opts) => {
  const marp = new Marp(opts);

  // Get the original fence renderer
  const { fence } = marp.markdown.renderer.rules;

  // Override fence to handle mermaid blocks
  marp.markdown.renderer.rules.fence = (tokens, idx, options, env, slf) => {
    const token = tokens[idx];
    const lang = token.info.trim();

    if (lang === 'mermaid') {
      // Return mermaid div that will be processed by mermaid.js
      return `<pre class="mermaid">${marp.markdown.utils.escapeHtml(token.content)}</pre>\n`;
    }

    // Use default fence renderer for other code blocks
    return fence(tokens, idx, options, env, slf);
  };

  // Inject Mermaid.js library into the HTML
  marp.use((md) => {
    const { html } = md.renderer.rules;

    md.core.ruler.push('inject_mermaid', (state) => {
      // This will be called during rendering
      return true;
    });
  });

  // Hook the afterRender to inject Mermaid script
  const render = marp.render.bind(marp);

  marp.render = (markdown, env) => {
    const result = render(markdown, env);

    // Inject Mermaid.js into the HTML output
    const mermaidScript = `
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({
    startOnLoad: true,
    theme: 'default',
    themeVariables: {
      primaryColor: '#e1f5ff',
      primaryTextColor: '#000',
      primaryBorderColor: '#666',
      lineColor: '#666',
      secondaryColor: '#fff4e1',
      tertiaryColor: '#ffe1f5',
      background: '#fff',
      mainBkg: '#e8f5e9',
      fontSize: '16px'
    }
  });
</script>`;

    // Inject script before closing body tag
    result.html = result.html.replace('</body>', `${mermaidScript}</body>`);
    result.css += `
.mermaid {
  display: flex;
  justify-content: center;
  align-items: center;
  margin: 1em 0;
}
`;

    return result;
  };

  return marp;
};
