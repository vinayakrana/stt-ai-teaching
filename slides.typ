#import "@preview/touying:0.5.5": *
#import themes.metropolis: *

// Custom code block styling with gradient background
#show raw.where(block: true): it => {
  block(
    fill: gradient.linear(
      rgb("#667eea"),
      rgb("#764ba2"),
      angle: 135deg,
    ),
    stroke: 1pt + rgb("#ffffff").transparentize(80%),
    inset: 1.2em,
    radius: 10pt,
    width: 100%,
  )[
    #set text(fill: rgb("#f8f9fa"), font: "Fira Code", size: 0.85em)
    #it
  ]
}

// Inline code with accent styling
#show raw.where(block: false): it => {
  box(
    fill: gradient.linear(
      rgb("#667eea"),
      rgb("#764ba2"),
      angle: 135deg,
    ),
    inset: (x: 0.4em, y: 0.25em),
    outset: (y: 0.2em),
    radius: 4pt,
  )[
    #set text(fill: rgb("#ffffff"), font: "Fira Code", weight: 600, size: 0.9em)
    #it
  ]
}

// Simple title slide with white background and line
#let title-slide(title, subtitle: none, author: none, institution: none) = {
  set page(fill: white)
  align(center + horizon)[
    #block[
      #text(size: 42pt, weight: "bold", fill: rgb("#1e293b"))[#title]
      #v(0.3em)
      #line(length: 60%, stroke: 2pt + rgb("#667eea"))
      #v(0.5em)
      #if subtitle != none [
        #text(size: 24pt, fill: rgb("#64748b"))[#subtitle]
      ]
      #if author != none [
        #v(1em)
        #text(size: 18pt, fill: rgb("#475569"))[#author]
      ]
      #if institution != none [
        #v(0.3em)
        #text(size: 16pt, fill: rgb("#94a3b8"))[#institution]
      ]
    ]
  ]
}

// Helper functions for content
#let columns-layout(left, right) = {
  grid(
    columns: (1fr, 1fr),
    gutter: 2.5em,
    left,
    right
  )
}

// Callout boxes
#let tip-box(body) = {
  block(
    fill: rgb("#eff6ff"),
    stroke: 2pt + rgb("#3b82f6"),
    inset: 1em,
    radius: 8pt,
    width: 100%,
  )[
    #text(fill: rgb("#1e40af"), weight: "bold")[💡 Tip] \
    #body
  ]
}

#let warning-box(body) = {
  block(
    fill: rgb("#fffbeb"),
    stroke: 2pt + rgb("#f59e0b"),
    inset: 1em,
    radius: 8pt,
    width: 100%,
  )[
    #text(fill: rgb("#92400e"), weight: "bold")[⚠️ Warning] \
    #body
  ]
}

#let info-box(body) = {
  block(
    fill: rgb("#f0f9ff"),
    stroke: 2pt + rgb("#0ea5e9"),
    inset: 1em,
    radius: 8pt,
    width: 100%,
  )[
    #text(fill: rgb("#0c4a6e"), weight: "bold")[ℹ️ Note] \
    #body
  ]
}

#let card(body) = {
  block(
    fill: rgb("#f8fafc"),
    inset: 1.2em,
    radius: 10pt,
    width: 100%,
    body
  )
}
